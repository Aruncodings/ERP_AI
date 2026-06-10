import asyncio
import json
import os
import re
from functools import partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from erp_backend.core.config import BASE_MODEL, GGUF_MODEL_PATH, LLM_CTX_SIZE
from erp_backend.core.metrics import set_model_loaded
from erp_backend.llm.runtime import _discover_ollama_models, _resolve_default_ollama_model, load_model
from erp_backend.api.models import QueryRequest


_LLM_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(2, int(os.getenv("ERP_LLM_WORKERS", "2"))),
    thread_name_prefix="erp-llm",
)


async def _run_llm_task(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_LLM_EXECUTOR, partial(func, *args, **kwargs))


async def _iterate_llm_task(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def _worker():
        try:
            for item in func(*args, **kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, ("item", item))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    _LLM_EXECUTOR.submit(_worker)
    while True:
        kind, payload = await queue.get()
        if kind == "item":
            yield payload
            continue
        if kind == "error":
            raise payload
        break


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


def _discover_gguf_models():
    candidates = set()
    configured = str(GGUF_MODEL_PATH or "").strip()
    if configured and Path(configured).exists():
        candidates.add(configured)
    root = Path(__file__).resolve().parents[3]
    models_dir = root / "models"
    if models_dir.exists():
        for file in models_dir.glob("*.gguf"):
            candidates.add(str(file.resolve()))
    return sorted(candidates)


def _default_model_context_limit():
    try:
        return max(512, int(LLM_CTX_SIZE or 8000))
    except Exception:
        return 8000


def _extract_context_limit_hint(text):
    value = str(text or "").strip().lower()
    if not value:
        return None
    compact = re.sub(r"[^a-z0-9]+", " ", value)
    explicit = {
        "256k": 256000,
        "128k": 128000,
        "96k": 96000,
        "64k": 64000,
        "48k": 48000,
        "32k": 32000,
        "24k": 24000,
        "16k": 16000,
        "12k": 12000,
        "8k": 8000,
        "6k": 6000,
        "4k": 4000,
        "2k": 2000,
    }
    for key, limit in explicit.items():
        if key in compact:
            return limit
    match = re.search(r"(?<!\d)(\d{2,3})\s*k(?!\w)", compact)
    if match:
        try:
            return max(512, int(match.group(1)) * 1000)
        except Exception:
            return None
    match = re.search(r"(?<!\d)(\d{5,6})(?!\d)", compact)
    if match:
        try:
            parsed = int(match.group(1))
            if 1000 <= parsed <= 1000000:
                return parsed
        except Exception:
            return None
    return None


def _estimate_model_context_limit(identifier, runtime):
    runtime_key = str(runtime or "").strip().lower()
    text = str(identifier or "").strip()
    if text:
        hint = _extract_context_limit_hint(text)
        if hint:
            return hint
        try:
            hint = _extract_context_limit_hint(Path(text).name)
            if hint:
                return hint
        except Exception:
            pass
    if runtime_key in {"gguf", "safetensors"}:
        return _default_model_context_limit()
    return _default_model_context_limit()


def _build_model_context_limits(gguf_models, safetensors_models, ollama_models=None):
    limits = {}
    for path in gguf_models or []:
        limits[str(path)] = _estimate_model_context_limit(path, "gguf")
    for model_id in safetensors_models or []:
        limits[str(model_id)] = _estimate_model_context_limit(model_id, "safetensors")
    for model_id in ollama_models or []:
        limits[str(model_id)] = _estimate_model_context_limit(model_id, "ollama")
    return limits


def _smallest_gguf_model_path():
    models = _discover_gguf_models()
    if not models:
        return None
    try:
        return min(models, key=lambda p: Path(p).stat().st_size)
    except Exception:
        return models[0]


def _ordered_gguf_candidates(preferred_path: str | None):
    ordered = []
    seen = set()

    def _add(path_like):
        value = str(path_like or "").strip()
        if not value:
            return
        try:
            resolved = str(Path(value).resolve())
        except Exception:
            resolved = value
        if resolved in seen:
            return
        if Path(resolved).is_file():
            seen.add(resolved)
            ordered.append(resolved)

    _add(preferred_path)
    _add(GGUF_MODEL_PATH)
    for candidate in _discover_gguf_models():
        _add(candidate)
    return ordered


def _default_reasoning_gguf_model_path():
    models = _discover_gguf_models()
    if not models:
        configured = str(GGUF_MODEL_PATH or "").strip()
        return configured or None
    preferred_tokens = (
        "qwen3.5-4b(reasoning)",
        "qwen3.5-4b",
        "reasoning",
        "reason",
    )
    for token in preferred_tokens:
        for path in models:
            if token in Path(path).name.lower():
                return path
    return models[0]


def _normalized_model_spec(runtime, gguf_model_path, safetensors_model_id, ollama_model, compute_mode):
    runtime_key = str(runtime or "gguf").strip().lower()
    if runtime_key not in {"gguf", "safetensors", "ollama"}:
        runtime_key = "gguf"
    mode = str(compute_mode or "auto").strip().lower() or "auto"
    gguf_path = _scalar_text(gguf_model_path) if runtime_key == "gguf" else ""
    safe_id = _scalar_text(safetensors_model_id) if runtime_key == "safetensors" else ""
    resolved_ollama_model = _scalar_text(ollama_model, default=_resolve_default_ollama_model()) if runtime_key == "ollama" else ""
    if gguf_path:
        try:
            gguf_path = str(Path(gguf_path).resolve())
        except Exception:
            pass
    return {
        "runtime": runtime_key,
        "compute_mode": mode,
        "gguf_model_path": gguf_path,
        "safetensors_model_id": safe_id,
        "ollama_model": resolved_ollama_model,
    }


def _query_model_spec(payload: QueryRequest):
    return _normalized_model_spec(
        runtime=payload.model_runtime,
        gguf_model_path=payload.gguf_model_path,
        safetensors_model_id=payload.safetensors_model_id,
        ollama_model=payload.ollama_model,
        compute_mode=payload.compute_mode,
    )


def _reasoning_model_spec(payload: QueryRequest):
    runtime = str(payload.reasoning_model_runtime or "gguf").strip().lower()
    if runtime not in {"gguf", "safetensors", "ollama"}:
        runtime = "gguf"
    gguf_path = _scalar_text(payload.reasoning_gguf_model_path)
    safe_id = _scalar_text(payload.reasoning_safetensors_model_id)
    ollama_model = _scalar_text(payload.reasoning_ollama_model)
    if runtime == "gguf" and not gguf_path:
        gguf_path = _default_reasoning_gguf_model_path()
    if runtime == "safetensors" and not safe_id:
        safe_id = str(BASE_MODEL or "").strip()
    if runtime == "ollama" and not ollama_model:
        ollama_model = _resolve_default_ollama_model()
    return _normalized_model_spec(
        runtime=runtime,
        gguf_model_path=gguf_path,
        safetensors_model_id=safe_id,
        ollama_model=ollama_model,
        compute_mode=payload.compute_mode,
    )


def _load_model_from_spec(spec: dict, payload: QueryRequest, cache_namespace: str):
    runtime = str((spec or {}).get("runtime") or "gguf").strip().lower()
    mode = str((spec or {}).get("compute_mode") or "auto").strip().lower()
    gguf_model_path = _scalar_text((spec or {}).get("gguf_model_path")) or None
    safetensors_model_id = _scalar_text((spec or {}).get("safetensors_model_id")) or None
    ollama_model = _scalar_text((spec or {}).get("ollama_model")) or None
    errors = []

    if runtime == "gguf":
        candidates = _ordered_gguf_candidates(gguf_model_path)
        if not candidates:
            raise RuntimeError("No GGUF model files were discovered in configured paths.")
        for path in candidates[:3]:
            try:
                model, tokenizer = load_model(
                    runtime="gguf",
                    gguf_model_path=path,
                    compute_mode=mode,
                    hybrid_gpu_layers=payload.hybrid_gpu_layers,
                    hybrid_gpu_memory_mb=payload.hybrid_gpu_memory_mb,
                    cache_namespace=cache_namespace,
                )
                set_model_loaded("gguf", Path(path).name)
                return model, tokenizer
            except Exception as exc:
                errors.append(f"{Path(path).name} ({mode}): {exc}")
                if mode != "cpu":
                    try:
                        model, tokenizer = load_model(
                            runtime="gguf",
                            gguf_model_path=path,
                            compute_mode="cpu",
                            hybrid_gpu_layers=payload.hybrid_gpu_layers,
                            hybrid_gpu_memory_mb=payload.hybrid_gpu_memory_mb,
                            cache_namespace=cache_namespace,
                        )
                        set_model_loaded("gguf", Path(path).name)
                        return model, tokenizer
                    except Exception as cpu_exc:
                        errors.append(f"{Path(path).name} (cpu): {cpu_exc}")
    elif runtime == "safetensors":
        # Safetensors requested first; fallback to GGUF if unavailable.
        try:
            model, tokenizer = load_model(
                runtime="safetensors",
                safetensors_model_id=safetensors_model_id,
                ollama_model=ollama_model,
                compute_mode=mode,
                hybrid_gpu_layers=payload.hybrid_gpu_layers,
                hybrid_gpu_memory_mb=payload.hybrid_gpu_memory_mb,
                cache_namespace=cache_namespace,
            )
            set_model_loaded("safetensors", safetensors_model_id or "unknown")
            return model, tokenizer
        except Exception as exc:
            errors.append(f"safetensors ({mode}): {exc}")
            if mode != "cpu":
                try:
                    model, tokenizer = load_model(
                        runtime="safetensors",
                        safetensors_model_id=safetensors_model_id,
                        ollama_model=ollama_model,
                        compute_mode="cpu",
                        hybrid_gpu_layers=payload.hybrid_gpu_layers,
                        hybrid_gpu_memory_mb=payload.hybrid_gpu_memory_mb,
                        cache_namespace=cache_namespace,
                    )
                    set_model_loaded("safetensors", safetensors_model_id or "unknown")
                    return model, tokenizer
                except Exception as cpu_exc:
                    errors.append(f"safetensors (cpu): {cpu_exc}")
        candidates = _ordered_gguf_candidates(gguf_model_path)
        for path in candidates[:2]:
            try:
                model, tokenizer = load_model(
                    runtime="gguf",
                    gguf_model_path=path,
                    ollama_model=ollama_model,
                    compute_mode="cpu",
                    hybrid_gpu_layers=payload.hybrid_gpu_layers,
                    hybrid_gpu_memory_mb=payload.hybrid_gpu_memory_mb,
                    cache_namespace=cache_namespace,
                )
                set_model_loaded("gguf", Path(path).name)
                return model, tokenizer
            except Exception as exc:
                errors.append(f"{Path(path).name} (gguf cpu fallback): {exc}")
    else:
        try:
            model, tokenizer = load_model(
                runtime="ollama",
                ollama_model=ollama_model,
                compute_mode=mode,
                hybrid_gpu_layers=payload.hybrid_gpu_layers,
                hybrid_gpu_memory_mb=payload.hybrid_gpu_memory_mb,
                cache_namespace=cache_namespace,
            )
            set_model_loaded("ollama", ollama_model or "unknown")
            return model, tokenizer
        except Exception as exc:
            errors.append(f"ollama ({mode}): {exc}")

    joined = " | ".join(errors[-4:]) if errors else "Unknown model initialization error."
    raise RuntimeError(f"Model initialization failed. {joined}")


def _load_query_model(payload: QueryRequest):
    return _load_model_from_spec(_query_model_spec(payload), payload, cache_namespace="query")


def _load_reasoning_model(payload: QueryRequest):
    return _load_model_from_spec(_reasoning_model_spec(payload), payload, cache_namespace="reasoning")


def _discover_safetensors_models():
    options = set()
    base_model = str(BASE_MODEL or "").strip()
    if base_model:
        options.add(base_model)

    def add_safetensors_path(path_obj):
        try:
            resolved = str(path_obj.resolve())
        except Exception:
            resolved = str(path_obj)
        if resolved:
            options.add(resolved)

    root = Path(__file__).resolve().parents[3]
    models_dir = root / "models"
    if models_dir.exists():
        # Include directories that contain shard files, and direct files.
        for file in models_dir.rglob("*.safetensors"):
            add_safetensors_path(file.parent)

    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        hub_dir = Path(hf_home) / "hub"
    else:
        user_home = Path.home()
        hub_dir = user_home / ".cache" / "huggingface" / "hub"
    if hub_dir.exists():
        # Canonical Hugging Face cache layout:
        #   hub/models--org--repo/snapshots/<revision>/*.safetensors
        for snap in hub_dir.glob("models--*/snapshots/*"):
            if snap.is_dir() and any(snap.glob("*.safetensors")):
                add_safetensors_path(snap)

        # Backward-compatible fallback in case of non-standard nesting.
        for snap in hub_dir.glob("models--*/*/snapshots/*"):
            if snap.is_dir() and any(snap.glob("*.safetensors")):
                add_safetensors_path(snap)

    return sorted(options)
