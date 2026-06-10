import gc
import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import torch

from erp_backend.core.config import (
    ADAPTER_DIR,
    BASE_MODEL,
    GGUF_MODEL_PATH,
    LLM_CTX_SIZE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_FLASH_ATTN,
    LLM_N_BATCH,
    LLM_N_GPU_LAYERS,
    LLM_N_THREADS,
    LLM_OFFLOAD_KQV,
    LLM_USE_MLOCK,
    LLM_USE_MMAP,
    SAFETENSORS_MAX_NEW_TOKENS,
    SAFETENSORS_USE_4BIT,
)
from erp_backend.core.feedback import build_retrain_dataset

_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _scalar_text(value, default=""):
    if isinstance(value, dict):
        for key in ("value", "model_id", "path", "id", "name"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return str(default or "").strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _scalar_text(item, default="")
            if text:
                return text
        return str(default or "").strip()
    return str(value or default or "").strip()


def _release_cached_model(entry):
    try:
        model, tokenizer = entry
    except Exception:
        return
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass


def resolve_runtime_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_cuda_device_map() -> str:
    env_device = os.getenv("CUDA_DEVICE")
    if env_device:
        return env_device

    if not torch.cuda.is_available():
        return "cpu"

    best_index = 0
    best_free_bytes = -1
    for index in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(index):
                free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes > best_free_bytes:
                best_free_bytes = free_bytes
                best_index = index
        except Exception:
            continue
    return f"cuda:{best_index}"


def get_model_device(model) -> torch.device:
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            weight = getattr(embeddings, "weight", None)
            if weight is not None:
                return weight.device
    except Exception:
        pass
    try:
        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, dict) and hf_device_map:
            for mapped in hf_device_map.values():
                if mapped in {None, "disk"}:
                    continue
                try:
                    return torch.device(mapped)
                except Exception:
                    continue
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device(resolve_runtime_device())


def is_cuda_inference_error(exc):
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "cuda",
            "cublas",
            "cusparse",
            "cudnn",
            "device-side assert",
            "out of memory",
        )
    )


def recover_from_cuda_error():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


class SafeTensorsChatRuntime:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = str(get_model_device(model))

    def _prepare_inputs(self, prompt):
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def create_chat_completion(self, messages, temperature=0.0, max_tokens=256):
        max_tokens = min(int(max_tokens or 0), int(SAFETENSORS_MAX_NEW_TOKENS))
        if max_tokens <= 0:
            max_tokens = 128
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._prepare_inputs(prompt)
        do_sample = float(temperature or 0.0) > 0.0
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=max(0.1, float(temperature or 0.0)) if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return {"choices": [{"message": {"content": text}}]}

    def create_chat_completion_stream(self, messages, temperature=0.0, max_tokens=256):
        max_tokens = min(int(max_tokens or 0), int(SAFETENSORS_MAX_NEW_TOKENS))
        if max_tokens <= 0:
            max_tokens = 128
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._prepare_inputs(prompt)
        do_sample = float(temperature or 0.0) > 0.0
        try:
            from transformers import TextIteratorStreamer
        except Exception:
            response = self.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if text:
                for chunk in text.split():
                    yield chunk + " "
            return

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "streamer": streamer,
        }
        if do_sample:
            generate_kwargs["temperature"] = max(0.1, float(temperature or 0.0))

        worker = threading.Thread(
            target=self.model.generate,
            kwargs=generate_kwargs,
            daemon=True,
        )
        worker.start()
        for text in streamer:
            if text:
                yield text
        worker.join(timeout=0.2)


def _is_airllm_available() -> bool:
    try:
        import importlib.metadata

        importlib.metadata.version("airllm")
        return True
    except Exception:
        return False


class AirLLMChatRuntime:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = str(get_model_device(model))

    def create_chat_completion(self, messages, temperature=0.0, max_tokens=256):
        max_tokens = min(int(max_tokens or 0), int(SAFETENSORS_MAX_NEW_TOKENS))
        if max_tokens <= 0:
            max_tokens = 128
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=max(512, int(LLM_CTX_SIZE or 2048)),
            padding=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        do_sample = float(temperature or 0.0) > 0.0
        base_kwargs = {
            "max_new_tokens": max_tokens,
            "return_dict_in_generate": True,
            "do_sample": do_sample,
        }
        if attention_mask is not None:
            base_kwargs["attention_mask"] = attention_mask
        if do_sample:
            base_kwargs["temperature"] = max(0.1, float(temperature or 0.0))
            base_kwargs["top_p"] = 0.95

        try:
            output = self.model.generate(input_ids, use_cache=True, **base_kwargs)
        except Exception as exc:
            if "DynamicCache" not in str(exc):
                raise
            output = self.model.generate(input_ids, use_cache=False, **base_kwargs)

        sequence = output.sequences[0] if hasattr(output, "sequences") else output[0]
        prompt_len = input_ids.shape[-1]
        new_tokens = sequence[prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if text:
            return {"choices": [{"message": {"content": text}}]}

        full_text = self.tokenizer.decode(sequence, skip_special_tokens=True).strip()
        if full_text.startswith(prompt):
            full_text = full_text[len(prompt) :].strip()
        return {"choices": [{"message": {"content": full_text}}]}

    def create_chat_completion_stream(self, messages, temperature=0.0, max_tokens=256):
        max_tokens = min(int(max_tokens or 0), int(SAFETENSORS_MAX_NEW_TOKENS))
        if max_tokens <= 0:
            max_tokens = 128
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=max(512, int(LLM_CTX_SIZE or 2048)),
            padding=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        try:
            from transformers import TextIteratorStreamer
        except Exception:
            response = self.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if text:
                for chunk in text.split():
                    yield chunk + " "
            return

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        do_sample = float(temperature or 0.0) > 0.0
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "streamer": streamer,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        if do_sample:
            generate_kwargs["temperature"] = max(0.1, float(temperature or 0.0))
            generate_kwargs["top_p"] = 0.95

        worker = threading.Thread(
            target=self.model.generate,
            kwargs=generate_kwargs,
            daemon=True,
        )
        worker.start()
        for text in streamer:
            if text:
                yield text
        worker.join(timeout=0.2)


def _load_airllm_model(
    force_cpu=False,
    safetensors_model_id=None,
    compute_mode="auto",
):
    if not _is_airllm_available():
        raise RuntimeError(
            "airllm is not installed. Install with `pip install airllm`."
        )

    import importlib

    airllm_pkg = importlib.import_module("airllm")
    model_ref = _scalar_text(safetensors_model_id, default=BASE_MODEL) or BASE_MODEL
    model = airllm_pkg.AutoModel.from_pretrained(model_ref)
    tokenizer = model.tokenizer
    device = str(get_model_device(model))
    mode = str(compute_mode or "auto").strip().lower()
    if force_cpu or mode == "cpu":
        device = "cpu"
    return AirLLMChatRuntime(model, tokenizer, device), tokenizer


def _post_json(url, payload, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaChatRuntime:
    def __init__(self, base_url, model_name):
        self.base_url = str(base_url or OLLAMA_BASE_URL or "").strip().rstrip("/")
        self.model_name = str(model_name or "").strip()
        self.tokenizer = None
        self.device = "ollama"

    def _payload(self, messages, temperature=0.0, max_tokens=256, stream=False):
        return {
            "model": self.model_name,
            "messages": list(messages or []),
            "stream": bool(stream),
            "options": {
                "temperature": float(temperature or 0.0),
                "num_predict": max(1, int(max_tokens or 256)),
                "num_ctx": max(512, int(LLM_CTX_SIZE or 8000)),
            },
        }

    def create_chat_completion(self, messages, temperature=0.0, max_tokens=256, stream=False):
        if stream:
            return self.create_chat_completion_stream(messages, temperature=temperature, max_tokens=max_tokens)
        response = _post_json(
            f"{self.base_url}/api/chat",
            self._payload(messages, temperature=temperature, max_tokens=max_tokens, stream=False),
        )
        text = str(((response.get("message") or {}).get("content")) or "").strip()
        return {"choices": [{"message": {"content": text}}]}

    def create_chat_completion_stream(self, messages, temperature=0.0, max_tokens=256):
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token_text = str(((chunk.get("message") or {}).get("content")) or "")
                if token_text:
                    yield {"choices": [{"delta": {"content": token_text}, "message": {"content": token_text}}]}


def _resolve_default_ollama_model():
    configured = str(OLLAMA_MODEL or "").strip()
    if configured:
        return configured
    candidates = _discover_ollama_models()
    embedding_markers = ("embed", "embedding", "minilm", "bge", "nomic-embed")
    for candidate in candidates:
        lowered = str(candidate).lower()
        if any(marker in lowered for marker in embedding_markers):
            continue
        if candidate:
            return str(candidate)
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return ""


def _discover_ollama_models():
    base_url = str(OLLAMA_BASE_URL or "").strip().rstrip("/")
    if base_url:
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = []
            for item in payload.get("models") or []:
                name = str((item or {}).get("name") or (item or {}).get("model") or "").strip()
                if name:
                    models.append(name)
            if models:
                return sorted(dict.fromkeys(models))
        except Exception:
            pass

    try:
        completed = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []

    models = []
    for line in str(completed.stdout or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("name"):
            continue
        model_name = line.split()[0].strip()
        if model_name:
            models.append(model_name)
    return sorted(dict.fromkeys(models))


def _load_gguf_model(
    force_cpu=False,
    gguf_model_path=None,
    compute_mode="auto",
    hybrid_gpu_layers=None,
):
    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: llama-cpp-python. Install with `pip install llama-cpp-python`."
        ) from exc

    configured_model_path = Path(str(gguf_model_path or GGUF_MODEL_PATH))
    if configured_model_path.is_file():
        model_path = str(configured_model_path)
    else:
        models_dir = Path(__file__).resolve().parents[3] / "models"
        gguf_files = sorted(models_dir.glob("*.gguf"))
        if not gguf_files:
            raise RuntimeError(
                f"Configured GGUF not found: {configured_model_path}. "
                f"No GGUF model found in fallback directory: {models_dir}"
            )
        model_path = str(gguf_files[0])
    mode = str(compute_mode or "auto").strip().lower()
    if force_cpu or mode == "cpu":
        n_gpu_layers = 0
    elif mode == "hybrid":
        default_hybrid = max(1, int(os.getenv("LLM_HYBRID_GPU_LAYERS", "20")))
        n_gpu_layers = int(hybrid_gpu_layers if hybrid_gpu_layers is not None else default_hybrid)
    else:
        n_gpu_layers = int(LLM_N_GPU_LAYERS)
    model = Llama(
        model_path=model_path,
        n_ctx=max(512, int(LLM_CTX_SIZE)),
        n_batch=max(32, int(LLM_N_BATCH)),
        n_threads=max(1, int(LLM_N_THREADS)),
        n_gpu_layers=n_gpu_layers,
        offload_kqv=bool(LLM_OFFLOAD_KQV),
        flash_attn=bool(LLM_FLASH_ATTN),
        use_mmap=bool(LLM_USE_MMAP),
        use_mlock=bool(LLM_USE_MLOCK),
        verbose=False,
    )
    return model, None


def _load_safetensors_model(
    force_cpu=False,
    safetensors_model_id=None,
    compute_mode="auto",
    hybrid_gpu_memory_mb=None,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: transformers. Install with `pip install transformers`."
        ) from exc

    mode = str(compute_mode or "auto").strip().lower()
    force_cpu_mode = bool(force_cpu) or mode == "cpu"
    device = "cpu" if force_cpu_mode else ("cuda" if torch.cuda.is_available() else "cpu")
    model_ref = _scalar_text(safetensors_model_id, default=BASE_MODEL) or BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device == "cuda" else torch.float32
    )

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["attn_implementation"] = "sdpa"
        if mode == "hybrid":
            gpu_mem = int(hybrid_gpu_memory_mb or os.getenv("LLM_HYBRID_GPU_MEMORY_MB", "3072"))
            gpu_mem = max(512, gpu_mem)
            model_kwargs["max_memory"] = {0: f"{gpu_mem}MiB", "cpu": "64GiB"}
            model_kwargs["offload_folder"] = str((Path(__file__).resolve().parents[2] / "offload_cache").resolve())
        if SAFETENSORS_USE_4BIT:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                # bitsandbytes is optional; fallback to normal loading.
                pass

    # Prefer non-deprecated dtype argument; fallback for older transformers versions.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            dtype=dtype,
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype=dtype,
            **model_kwargs,
        )

    if device == "cpu":
        model = model.to(device)
    model.eval()
    return SafeTensorsChatRuntime(model, tokenizer, device), tokenizer


def _load_ollama_model(ollama_model=None):
    model_name = str(ollama_model or _resolve_default_ollama_model() or "").strip()
    if not model_name:
        raise RuntimeError(
            "No Ollama model configured or discovered. Set OLLAMA_MODEL or ensure the Ollama server exposes models."
        )
    runtime = OllamaChatRuntime(OLLAMA_BASE_URL, model_name)
    return runtime, None


def load_model(
    force_cpu=False,
    runtime="gguf",
    gguf_model_path=None,
    safetensors_model_id=None,
    ollama_model=None,
    compute_mode="auto",
    hybrid_gpu_layers=None,
    hybrid_gpu_memory_mb=None,
    cache_namespace="default",
    airllm=None,
):
    runtime_key = str(runtime or "gguf").strip().lower()
    if runtime_key not in {"gguf", "safetensors", "ollama"}:
        runtime_key = "gguf"

    resolved_gguf_path = _scalar_text(gguf_model_path) if runtime_key == "gguf" else ""
    resolved_safetensors_id = _scalar_text(safetensors_model_id, default=BASE_MODEL) if runtime_key == "safetensors" else ""
    resolved_ollama_model = _scalar_text(ollama_model, default=_resolve_default_ollama_model()) if runtime_key == "ollama" else ""
    mode = str(compute_mode or "auto").strip().lower()
    use_airllm = airllm
    if use_airllm is None and runtime_key == "safetensors":
        use_airllm = _is_airllm_available()
    cache_key = (
        str(cache_namespace or "default").strip().lower(),
        bool(force_cpu),
        runtime_key,
        resolved_gguf_path,
        resolved_safetensors_id,
        resolved_ollama_model,
        mode,
        int(hybrid_gpu_layers) if hybrid_gpu_layers is not None else None,
        int(hybrid_gpu_memory_mb) if hybrid_gpu_memory_mb is not None else None,
        bool(use_airllm),
    )
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # Keep one loaded model per namespace to avoid retaining multiple
        # large model instances in memory when users switch model selection.
        namespace = str(cache_namespace or "default").strip().lower()
        stale_keys = [k for k in _MODEL_CACHE.keys() if k and str(k[0]) == namespace and k != cache_key]
        for stale_key in stale_keys:
            stale_entry = _MODEL_CACHE.pop(stale_key, None)
            if stale_entry is not None:
                _release_cached_model(stale_entry)
    recover_from_cuda_error()

    try:
        if runtime_key == "safetensors" and use_airllm:
            result = _load_airllm_model(
                force_cpu=force_cpu,
                safetensors_model_id=resolved_safetensors_id,
                compute_mode=mode,
            )
        elif runtime_key == "safetensors":
            result = _load_safetensors_model(
                force_cpu=force_cpu,
                safetensors_model_id=resolved_safetensors_id,
                compute_mode=mode,
                hybrid_gpu_memory_mb=hybrid_gpu_memory_mb,
            )
        elif runtime_key == "ollama":
            result = _load_ollama_model(
                ollama_model=resolved_ollama_model,
            )
        else:
            result = _load_gguf_model(
                force_cpu=force_cpu,
                gguf_model_path=resolved_gguf_path or None,
                compute_mode=mode,
                hybrid_gpu_layers=hybrid_gpu_layers,
            )
    except Exception as exc:
        # Auto-fallback: prefer GPU, but switch to CPU if CUDA/GPU initialization fails.
        if runtime_key == "ollama" or force_cpu or not is_cuda_inference_error(exc):
            raise
        recover_from_cuda_error()
        if runtime_key == "safetensors" and use_airllm:
            result = _load_airllm_model(
                force_cpu=True,
                safetensors_model_id=resolved_safetensors_id,
                compute_mode="cpu",
            )
        elif runtime_key == "safetensors":
            result = _load_safetensors_model(
                force_cpu=True,
                safetensors_model_id=resolved_safetensors_id,
                compute_mode="cpu",
                hybrid_gpu_memory_mb=hybrid_gpu_memory_mb,
            )
        else:
            result = _load_gguf_model(
                force_cpu=True,
                gguf_model_path=resolved_gguf_path or None,
                compute_mode="cpu",
                hybrid_gpu_layers=hybrid_gpu_layers,
            )

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = result
    return result


def clear_model_cache():
    with _MODEL_CACHE_LOCK:
        entries = list(_MODEL_CACHE.values())
        _MODEL_CACHE.clear()
    for entry in entries:
        _release_cached_model(entry)
    recover_from_cuda_error()


def retrain_model():
    try:
        from train import train as run_train
    except Exception as exc:
        raise RuntimeError(f"Failed to import train.py: {exc}") from exc

    dataset_path, merged_count = build_retrain_dataset()
    if merged_count == 0:
        return {"started": False, "reason": "No training data available.", "rows_merged": 0}

    torch.cuda.empty_cache()
    gc.collect()
    run_train(dataset_path=dataset_path, output_dir=ADAPTER_DIR)
    torch.cuda.empty_cache()
    gc.collect()
    clear_model_cache()
    return {
        "started": True,
        "dataset_path": str(dataset_path),
        "rows_merged": int(merged_count),
        "output_dir": str(ADAPTER_DIR),
    }
