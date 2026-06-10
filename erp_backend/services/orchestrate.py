from pydantic import aliases
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from functools import partial, wraps
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from bson import ObjectId
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from erp_backend.core.config import (
    DEFAULT_DB_NAME,
    LLM_COLLECTION_CANDIDATES,
    LLM_PROMPT_REWRITE_MODE,
    LLM_SUMMARY_MODE,
    LLM_RAW_MODE,
    LLM_REWRITER_MODE,
    LLM_SINGLE_PASS_QUERY,
    LLM_USE_CHAT_HISTORY_HINTS,
    CHAT_CONTEXT_LIMIT,
    CHAT_CONTEXT_CHARS,
    VECTOR_DB_ENABLED,
    VECTOR_REFRESH_ENABLED,
    VECTOR_REFRESH_INTERVAL_SECONDS,
    VECTOR_DISTINCT_ENABLED,
    VECTOR_DISTINCT_MAX_FIELDS_PER_COLLECTION,
    VECTOR_DISTINCT_MAX_VALUES_PER_FIELD,
    VECTOR_DISTINCT_MAX_VALUE_LENGTH,
    ROLLOUT_DYNAMIC_FIELD_ROLES_ENABLED,
    ROLLOUT_RUNTIME_POLICY_COLLECTION,
    ROLLOUT_RUNTIME_POLICY_ENABLED,
    LLM_CTX_SIZE,
    LLM_USE_CACHE,
    SYSTEM_FIELD_NAMES,
)
from erp_backend.core.observability import add_span_event
from erp_backend.core.security import configure_hidden_field_policy
from erp_backend.core.metrics import observe_perf_log, observe_result_rows, observe_error, observe_vector_refresh, track_active_request
from erp_backend.core.utils import normalize_lookup_text
from erp_backend.services.query import (
    execute_plan,
    generate_clarification_suggestions,
    validate_query_plan,
)
from erp_backend.services.query_rewriter import remap_plan_runtime_fields
from erp_backend.services.schema_indexer import build_schema_index
from erp_backend.services.self_healing import self_heal_user_query
from erp_backend.storage.cache import load_conversation_state, save_conversation_state
from erp_backend.storage.mongo import (
    allowed_collections_for_user,
    get_template_schema,
    infer_schema_from_docs,
    list_collections,
    list_databases,
    load_ai_template_schemas,
    load_runtime_policy,
    load_rbac_users,
    load_table_metadata,
    mongo_client,
    sample_collection,
)
from erp_backend.storage.vector_store import (
    upsert_schema_vectors,
    warm_reverse_lookup_cache,
)

from erp_backend.api.models import QueryRequest, QueryResponse
from erp_backend.services.intent import (
    _infer_collection_field_roles,
)

logger = logging.getLogger(__name__)

_VECTOR_SCHEMA_WARMED = set()
_SUGGESTION_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="erp-suggestions")
_LLM_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(2, int(os.getenv("ERP_LLM_WORKERS", "2"))),
    thread_name_prefix="erp-llm",
)
_VECTOR_REFRESH_TASK = None
_VECTOR_REFRESH_LOCK = asyncio.Lock()
_REQUEST_QUEUE_LOCKS = {}
_REQUEST_QUEUE_LOCKS_GUARD = Lock()
_PERF_LOG_LOCK = Lock()
_PERF_LOG_PATH = Path(os.getenv("ERP_PERF_LOG_PATH", "logs/erp_perf_timing.jsonl"))
STRICT_SCHEMA_SYSTEM_FIELDS = set(SYSTEM_FIELD_NAMES)
_RUNTIME_POLICY_CACHE = {}
_RUNTIME_POLICY_LOCK = asyncio.Lock()
_RUNTIME_SYSTEM_FIELDS = set()

# ── Rewrite cache ──
_REWRITE_CACHE = {}
_REWRITE_CACHE_LOCK = Lock()
_REWRITE_CACHE_TTL = 300
_REWRITE_CACHE_MAX = 256


def _get_cached_rewrite(prompt):
    if not LLM_USE_CACHE:
        return None
    key = _light_normalize_prompt(prompt)
    with _REWRITE_CACHE_LOCK:
        entry = _REWRITE_CACHE.get(key)
        if entry and entry["expires"] > time.time():
            return entry["value"]
        if entry:
            _REWRITE_CACHE.pop(key, None)
    return None


def _set_cached_rewrite(prompt, value):
    if not LLM_USE_CACHE:
        return
    key = _light_normalize_prompt(prompt)
    with _REWRITE_CACHE_LOCK:
        if len(_REWRITE_CACHE) >= _REWRITE_CACHE_MAX:
            oldest = min(_REWRITE_CACHE.keys(), key=lambda k: _REWRITE_CACHE[k]["expires"])
            _REWRITE_CACHE.pop(oldest, None)
        _REWRITE_CACHE[key] = {"value": value, "expires": time.time() + _REWRITE_CACHE_TTL}


# ── Utility helpers (referenced by extracted groups, not in any exclusion category) ──

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


def _normalize_term(text):
    return normalize_lookup_text(text)


def _singular(term):
    if term.endswith("ies") and len(term) > 3:
        return term[:-3] + "y"
    if term.endswith("es") and len(term) > 3:
        return term[:-2]
    if term.endswith("s") and len(term) > 2:
        return term[:-1]
    return term


def _user_name(selected_user):
    if not isinstance(selected_user, dict):
        return ""
    return str(
        selected_user.get("display_name")
        or selected_user.get("name")
        or selected_user.get("user_id")
        or ""
    ).strip()


def _personalize_response(response, selected_user):
    text = str(response or "").strip()
    if not text:
        return text
    name = _user_name(selected_user)
    if not name:
        return text
    lowered_text = text.lower()
    lowered_name = name.lower()
    if lowered_text.startswith(lowered_name):
        return text
    return f"{name}, {text}"


def _is_object_id_like(value):
    text = str(value or "").strip()
    return bool(re.fullmatch(r"[0-9a-fA-F]{24}", text))






    return aliases


def _ensure_plan_collection(plan, default_collection):
    if not isinstance(plan, dict):
        return plan
    if plan.get("needs_clarification"):
        return plan
    collection = str(plan.get("collection") or "").strip()
    fallback = str(default_collection or "").strip()
    if collection or not fallback:
        return plan
    patched = dict(plan)
    patched["collection"] = fallback
    return patched


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1 — Runtime Policy & Perf Logging
# ═══════════════════════════════════════════════════════════════════════════════

def _effective_system_fields():
    return set(STRICT_SCHEMA_SYSTEM_FIELDS) | set(_RUNTIME_SYSTEM_FIELDS)


async def _load_and_apply_runtime_policy(db_name):
    if not ROLLOUT_RUNTIME_POLICY_ENABLED:
        return {}
    async with _RUNTIME_POLICY_LOCK:
        cached = _RUNTIME_POLICY_CACHE.get(db_name)
        if isinstance(cached, dict):
            return cached
        try:
            policy = await load_runtime_policy(
                db_name,
                collection_name=ROLLOUT_RUNTIME_POLICY_COLLECTION,
            )
        except Exception:
            policy = {}

        if not isinstance(policy, dict):
            policy = {}
        configure_hidden_field_policy(
            hidden_names=policy.get("hiddenFieldNames") if isinstance(policy.get("hiddenFieldNames"), list) else [],
            hidden_tokens=policy.get("hiddenFieldTokens") if isinstance(policy.get("hiddenFieldTokens"), list) else [],
        )
        system_fields = []
        if isinstance(policy.get("systemFieldAllowlist"), list):
            system_fields = [str(item).strip() for item in policy["systemFieldAllowlist"] if str(item).strip()]
        _RUNTIME_SYSTEM_FIELDS.clear()
        _RUNTIME_SYSTEM_FIELDS.update(system_fields)
        _RUNTIME_POLICY_CACHE[db_name] = policy
        return policy


def _ms_since(start):
    return round((time.perf_counter() - start) * 1000, 2)


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


def _write_perf_log(payload, endpoint, stage_ms, status, extra=None):
    model_ms = (
        float(stage_ms.get("reasoning_model_load", 0.0))
        + float(stage_ms.get("model_load", 0.0))
        + float(stage_ms.get("prompt_rewrite", 0.0))
        + float(stage_ms.get("scope_gate", 0.0))
        + float(stage_ms.get("plan_generation", 0.0))
        + float(stage_ms.get("verifier", 0.0))
        + float(stage_ms.get("summarize", 0.0))
        + float(stage_ms.get("follow_ups", 0.0))
        + float(stage_ms.get("narrate_stream", 0.0))
    )
    backend_ms = (
        float(stage_ms.get("bootstrap", 0.0))
        + float(stage_ms.get("schema_index", 0.0))
        + float(stage_ms.get("vector_warmup", 0.0))
        + float(stage_ms.get("chat_context", 0.0))
        + float(stage_ms.get("exact_lookup", 0.0))
        + float(stage_ms.get("vector_retrieval", 0.0))
        + float(stage_ms.get("plan_validate", 0.0))
        + float(stage_ms.get("execute", 0.0))
        + float(stage_ms.get("lookup_resolve", 0.0))
        + float(stage_ms.get("post_execute_repairs", 0.0))
        + float(stage_ms.get("conversation_save", 0.0))
    )
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "status": status,
        "db_name": payload.db_name,
        "user_id": payload.user_id,
        "model_runtime": payload.model_runtime,
        "reasoning_model_runtime": payload.reasoning_model_runtime,
        "reasoning_model_enabled": bool(payload.reasoning_model_enabled),
        "compute_mode": payload.compute_mode,
        "rollout_runtime_policy_enabled": bool(ROLLOUT_RUNTIME_POLICY_ENABLED),
        "rollout_dynamic_field_roles_enabled": bool(ROLLOUT_DYNAMIC_FIELD_ROLES_ENABLED),
        "prompt_length": len(str(payload.prompt or "")),
        "stage_ms": stage_ms,
        "model_ms": round(model_ms, 2),
        "backend_ms": round(backend_ms, 2),
        "total_ms": round(model_ms + backend_ms, 2),
    }
    if isinstance(extra, dict) and extra:
        record.update(extra)
    _PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _PERF_LOG_LOCK:
        with _PERF_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Push to Prometheus
    try:
        observe_perf_log(endpoint, status, str(payload.db_name or ""), stage_ms)
    except Exception:
        pass

    if status and str(status).strip().lower() not in ("ok", "success", ""):
        try:
            observe_error(endpoint, str(status))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2 — Request Queue
# ═══════════════════════════════════════════════════════════════════════════════

def _request_queue_key(payload: QueryRequest) -> str:
    db_name = str(getattr(payload, "db_name", "") or "").strip() or "__db__"
    user_id = str(getattr(payload, "user_id", "") or "").strip() or "__user__"
    conversation_id = str(getattr(payload, "conversation_id", "") or "").strip() or "__conversation__"
    return f"{db_name}::{user_id}::{conversation_id}"


def _get_request_queue_lock(queue_key: str) -> asyncio.Lock:
    with _REQUEST_QUEUE_LOCKS_GUARD:
        lock = _REQUEST_QUEUE_LOCKS.get(queue_key)
        if lock is None:
            lock = asyncio.Lock()
            _REQUEST_QUEUE_LOCKS[queue_key] = lock
        return lock


def _with_request_fifo(handler):
    endpoint = f"/{handler.__name__}" if not handler.__name__.startswith("/") else handler.__name__

    @wraps(handler)
    async def wrapped(payload: QueryRequest, *args, **kwargs):
        track_active_request(endpoint, 1)
        lock = _get_request_queue_lock(_request_queue_key(payload))
        try:
            async with lock:
                return await handler(payload, *args, **kwargs)
        finally:
            track_active_request(endpoint, -1)

    return wrapped


def _queue_streaming_response(payload: QueryRequest, stream_response: StreamingResponse) -> StreamingResponse:
    lock = _get_request_queue_lock(_request_queue_key(payload))
    original_iterator = stream_response.body_iterator

    async def queued_iterator():
        async with lock:
            async for chunk in original_iterator:
                yield chunk

    stream_response.body_iterator = queued_iterator()
    return stream_response


# ═══════════════════════════════════════════════════════════════════════════════
# Group 3 — Prompt Rewriting
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_rewritten_prompt(original_prompt, rewritten_prompt):
    original = str(original_prompt or "").strip()
    rewritten = str(rewritten_prompt or "").strip()
    if not rewritten:
        return original
    if len(rewritten) < 2:
        return original
    if not any(ch.isalnum() for ch in rewritten):
        return original
    if len(rewritten) > 800:
        rewritten = rewritten[:800].strip()
    return rewritten or original


def _light_normalize_prompt(prompt):
    text = " ".join(str(prompt or "").strip().split())
    return text


def _should_use_llm_prompt_rewrite(prompt):
    mode = str(LLM_PROMPT_REWRITE_MODE or "adaptive").lower().strip()
    if mode == "off":
        return False
    if mode == "always":
        return True
    text = str(prompt or "").strip()
    lowered = text.lower()
    if len(text) > 90:
        return True
    if any(token in lowered for token in ("this", "that", "him", "her", "them", "same", "above")):
        return True
    if sum(1 for ch in text if not ch.isalnum() and ch not in " -_./@") > 6:
        return True
    return False


def _rewrite_prompt_with_llm(model, original_prompt):
    base = _light_normalize_prompt(original_prompt)
    if not _should_use_llm_prompt_rewrite(base):
        return base
    cached = _get_cached_rewrite(base)
    if cached is not None:
        return cached
    try:
        healed = self_heal_user_query(model, base)
        healed_payload = healed if isinstance(healed, dict) else {}
        rewritten = str(healed_payload.get("normalized_query") or "").strip()
        result = _validate_rewritten_prompt(base, rewritten)
        _set_cached_rewrite(base, result)
        return result
    except Exception:
        return base


def _rewrite_prompt_with_llm_details(model, original_prompt):
    base = _light_normalize_prompt(original_prompt)
    if not _should_use_llm_prompt_rewrite(base):
        return {
            "normalized_query": base,
            "intent_type": "unknown",
            "entity_terms": [],
            "field_terms": [],
            "value_terms": [],
            "confidence": 0.5,
            "needs_clarification": False,
            "hints": {},
        }
    cached = _get_cached_rewrite(base)
    if cached is not None:
        return cached
    try:
        healed = self_heal_user_query(model, base)
        healed_payload = healed if isinstance(healed, dict) else {}
        rewritten = _validate_rewritten_prompt(base, str(healed_payload.get("normalized_query") or "").strip())
        merged = dict(healed_payload)
        merged["normalized_query"] = rewritten
        _set_cached_rewrite(base, merged)
        return merged
    except Exception:
        result = {
            "normalized_query": base,
            "intent_type": "unknown",
            "entity_terms": [],
            "field_terms": [],
            "value_terms": [],
            "confidence": 0.5,
            "needs_clarification": False,
            "hints": {},
        }
        _set_cached_rewrite(base, result)
        return result


def _build_accuracy_planner_prompt(healed_payload):
    payload = healed_payload if isinstance(healed_payload, dict) else {}
    normalized_query = str(payload.get("normalized_query") or "").strip()
    if not normalized_query:
        return ""
    confidence = float(payload.get("confidence") or 0.5)
    entity_terms = [str(item).strip() for item in (payload.get("entity_terms") or []) if str(item).strip()]
    field_terms = [str(item).strip() for item in (payload.get("field_terms") or []) if str(item).strip()]
    value_terms = [str(item).strip() for item in (payload.get("value_terms") or []) if str(item).strip()]
    intent_type = str(payload.get("intent_type") or "unknown").strip()
    hints = payload.get("hints") if isinstance(payload.get("hints"), dict) else {}

    if confidence < 0.35:
        return normalized_query
    parts = [f"Query: {normalized_query}"]
    if intent_type and intent_type != "unknown":
        parts.append(f"Intent: {intent_type}")
    if entity_terms:
        parts.append("Entities: " + ", ".join(entity_terms[:8]))
    if field_terms:
        parts.append("Fields: " + ", ".join(field_terms[:8]))
    if value_terms:
        parts.append("Values: " + ", ".join(value_terms[:8]))
    explicit_collection = str(hints.get("explicit_collection") or "").strip()
    if explicit_collection:
        parts.append(f"Preferred collection: {explicit_collection}")
    if bool(hints.get("needs_aggregation")):
        parts.append("Aggregation required: true")
    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Group 4 — Clarification & Feedback
# ═══════════════════════════════════════════════════════════════════════════════

def _should_append_verifier_note(note, user_prompt, response_text):
    text = str(note or "").strip()
    if not text:
        return False
    if text.endswith("?"):
        return False
    prompt_norm = " ".join(str(user_prompt or "").lower().split())
    note_norm = " ".join(text.lower().split())
    if prompt_norm and (note_norm == prompt_norm or prompt_norm in note_norm):
        return False
    response_norm = " ".join(str(response_text or "").lower().split())
    if response_norm and note_norm in response_norm:
        return False
    return True


def _build_clarification_payload(
    model,
    tokenizer,
    prompt,
    selected_user,
    selected_collection,
    table_choice,
    plan,
    docs,
    total,
    table_metadata,
    accessible_collections,
    reason,
):
    clarification = generate_clarification_suggestions(
        model,
        tokenizer,
        prompt,
        selected_collection,
        table_metadata,
        plan,
        docs,
        reason=reason,
        accessible_collections=accessible_collections,
    )
    response_text = _personalize_response(
        str(clarification.get("message") or "Please confirm your request.").strip(),
        selected_user,
    )
    follow_ups = []
    for item in clarification.get("suggestions") or []:
        text = str(item or "").strip()
        if text and text not in follow_ups:
            follow_ups.append(text[:120])
    if follow_ups:
        response_text = f"{response_text} Pick one of the rewritten prompts below to continue."
    return QueryResponse(
        response=response_text,
        follow_ups=follow_ups[:3],
        needs_clarification=True,
        table_choice=table_choice or {"collection": None, "reason": reason},
        plan=plan or {"operation": "none"},
        docs=docs or [],
        total=int(total or 0),
        collection=str(selected_collection or ""),
        summary="",
    )


def _build_feedback_rewrite_payload(
    model,
    tokenizer,
    prompt,
    selected_user,
    selected_collection,
    table_choice,
    plan,
    docs,
    total,
    table_metadata,
    accessible_collections,
    reason,
):
    clarification = generate_clarification_suggestions(
        model,
        tokenizer,
        prompt,
        selected_collection,
        table_metadata,
        plan,
        docs,
        reason=reason,
        accessible_collections=accessible_collections,
    )
    response_text = _personalize_response(
        str(clarification.get("message") or "Please refine your query.").strip(),
        selected_user,
    )
    follow_ups = []
    for item in clarification.get("suggestions") or []:
        text = str(item or "").strip()
        if text and text not in follow_ups:
            follow_ups.append(text[:120])
    if follow_ups:
        response_text = f"{response_text} Pick one of the rewritten prompts below to continue."
    return QueryResponse(
        response=response_text,
        follow_ups=follow_ups[:3],
        needs_clarification=True,
        table_choice=table_choice or {"collection": None, "reason": reason},
        plan=plan or {"operation": "none"},
        docs=docs or [],
        total=int(total or 0),
        collection=str(selected_collection or ""),
        table_columns=[],
        summary="",
    )


async def _build_clarification_payload_async(*args, **kwargs):
    return await _run_llm_task(_build_clarification_payload, *args, **kwargs)


async def _build_feedback_rewrite_payload_async(*args, **kwargs):
    return await _run_llm_task(_build_feedback_rewrite_payload, *args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Group 5 — Lookup Resolution
# ═══════════════════════════════════════════════════════════════════════════════

def _lookup_display_candidates(field_meta):
    preferred = str((field_meta or {}).get("lookup_display") or "").strip()
    ordered = []
    if preferred:
        ordered.append(preferred)
    for extra in (field_meta or {}).get("lookup_other_display") or []:
        name = str(extra or "").strip()
        if name and name not in ordered:
            ordered.append(name)
    for fallback in ("name", "displayName", "title", "code", "label"):
        if fallback not in ordered:
            ordered.append(fallback)
    return ordered


def _lookup_candidate_collections(field_name, field_meta, table_metadata, accessible_collections=None):
    candidates = []
    lookup_collection = str((field_meta or {}).get("lookup_collection") or "").strip()
    if lookup_collection:
        candidates.append(lookup_collection)

    normalized = _normalize_term(field_name)
    field_tokens = [token for token in normalized.split() if token]
    singular_tokens = [_singular(token) for token in field_tokens]
    search_tokens = list(dict.fromkeys(field_tokens + singular_tokens))

    def _score_collection(name, meta):
        score = 0
        collection_norm = _normalize_term(name)
        template_norm = _normalize_term((meta or {}).get("template_name") or "")
        business_terms = " ".join((meta or {}).get("business_terms") or [])
        haystack = f"{collection_norm} {template_norm} {business_terms}"
        if normalized and normalized in haystack:
            score += 6
        if any(token and token in haystack for token in search_tokens):
            score += 4
        if normalized.endswith("to") and "user" in haystack:
            score += 3
        if normalized.endswith("owner") and "user" in haystack:
            score += 3
        if normalized.endswith("by") and "user" in haystack:
            score += 2
        return score

    scored = []
    for name, meta in (table_metadata or {}).items():
        if accessible_collections and name not in set(accessible_collections):
            continue
        if name in candidates:
            continue
        score = _score_collection(name, meta)
        if score > 0:
            scored.append((score, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    candidates.extend([name for _, name in scored[:3]])
    return [name for name in dict.fromkeys(candidates) if name]


async def _resolve_lookup_ids_to_names(db_name, selected_collection, docs, table_metadata):
    if not docs or not isinstance(docs, list):
        return docs
    metadata = (table_metadata or {}).get(selected_collection) or {}
    fields = list(metadata.get("fields") or [])
    if not fields:
        return docs

    def _compact_text(value):
        if value is None:
            return ""
        if isinstance(value, list):
            parts = []
            for item in value:
                text = _compact_text(item)
                if text:
                    parts.append(text)
            return ", ".join(parts)
        if isinstance(value, dict):
            for key in ("displayName", "display", "name", "title", "label", "code", "value"):
                text = _compact_text(value.get(key))
                if text:
                    return text
            return ""
        if isinstance(value, ObjectId):
            return str(value)
        text = str(value or "").strip()
        return text

    def _candidate_companion_values(row, field_name):
        candidates = []
        base = str(field_name or "").strip()
        if not base:
            return candidates
        for suffix in ("_textMode", "_", "_display", "_label"):
            key = f"{base}{suffix}"
            if key in row:
                text = _compact_text(row.get(key))
                if text and not _is_object_id_like(text):
                    candidates.append(text)
        return candidates

    lookup_fields = []
    for field in fields:
        dtype = str((field or {}).get("type") or "").upper()
        lookup_collection = str((field or {}).get("lookup_collection") or "").strip()
        field_name = str((field or {}).get("field") or "").strip()
        if dtype in {"LOOK_UP", "MULTI_LOOKUP"} and field_name and lookup_collection:
            lookup_fields.append((field_name, lookup_collection, field))
    if not lookup_fields:
        lookup_fields = []

    db = mongo_client()[db_name]
    for field_name, lookup_collection, field_meta in lookup_fields:
        resolved = {}
        try:
            raw_values = set()
            for row in docs:
                if not isinstance(row, dict):
                    continue
                value = row.get(field_name)
                if isinstance(value, list):
                    for item in value:
                        text = _compact_text(item)
                        if text:
                            raw_values.add(text)
                else:
                    text = _compact_text(value)
                    if text:
                        raw_values.add(text)
            if not raw_values:
                continue

            display_candidates = _lookup_display_candidates(field_meta)
            projection = {"_id": 1}
            for key in display_candidates:
                projection[key] = 1

            id_values = []
            string_values = []
            for text in raw_values:
                if _is_object_id_like(text):
                    try:
                        id_values.append(ObjectId(text))
                    except Exception:
                        pass
                else:
                    string_values.append(text)

            or_conditions = []
            if id_values:
                or_conditions.append({"_id": {"$in": id_values}})
            if string_values:
                for candidate in display_candidates:
                    or_conditions.append({candidate: {"$in": string_values}})

            if not or_conditions:
                continue

            filter_query = {"$or": or_conditions} if len(or_conditions) > 1 else or_conditions[0]

            cursor = db[lookup_collection].find(filter_query, projection)
            async for doc in cursor:
                doc_id = str(doc.get("_id"))
                display = ""
                for key in display_candidates:
                    value = doc.get(key)
                    if value is None:
                        continue
                    text = str(value).strip()
                    if text:
                        display = text
                        break
                if display:
                    resolved[doc_id] = display
                    for candidate in display_candidates:
                        code_value = doc.get(candidate)
                        if code_value is not None:
                            code_text = str(code_value).strip()
                            if code_text and code_text != doc_id:
                                resolved[code_text] = display
        except Exception:
            continue

        if not resolved:
            continue

        for row in docs:
            if not isinstance(row, dict):
                continue
            value = row.get(field_name)
            if _is_object_id_like(_compact_text(value)):
                companions = _candidate_companion_values(row, field_name)
                if companions:
                    row[field_name] = companions[0] if len(companions) == 1 else ", ".join(companions)
                    continue
            if isinstance(value, list):
                mapped = []
                for item in value:
                    text = _compact_text(item)
                    mapped_value = resolved.get(text, text)
                    if not mapped_value and _is_object_id_like(text):
                        mapped_value = resolved.get(str(text), text)
                    text_value = _compact_text(mapped_value)
                    if text_value:
                        mapped.append(text_value)
                if mapped:
                    row[field_name] = ", ".join(mapped)
            else:
                text = _compact_text(value)
                if text in resolved:
                    row[field_name] = resolved[text]
                elif _is_object_id_like(text):
                    row[field_name] = resolved.get(text, value)

    for row in docs:
        if not isinstance(row, dict):
            continue
        for field_name, value in list(row.items()):
            if field_name.startswith("_"):
                continue
            if not isinstance(value, list) and not _is_object_id_like(_compact_text(value)):
                continue

            field_meta = next((item for item in fields if str(item.get("field") or "").strip() == field_name), {})
            if str((field_meta or {}).get("type") or "").upper() in {"LOOK_UP", "MULTI_LOOKUP"}:
                continue

            companions = _candidate_companion_values(row, field_name)
            if companions:
                row[field_name] = companions[0] if len(companions) == 1 else ", ".join(companions)
                continue

            candidate_collections = _lookup_candidate_collections(field_name, field_meta, table_metadata)
            if not candidate_collections:
                continue

            resolved_values = {}
            for lookup_collection in candidate_collections:
                try:
                    projection = {"_id": 1}
                    for key in _lookup_display_candidates(field_meta):
                        projection[key] = 1
                    ids = []
                    if isinstance(value, list):
                        for item in value:
                            text = _compact_text(item)
                            if not text:
                                continue
                            ids.append(ObjectId(text) if _is_object_id_like(text) else text)
                    else:
                        text = _compact_text(value)
                        if text:
                            ids.append(ObjectId(text) if _is_object_id_like(text) else text)
                    if not ids:
                        continue
                    cursor = db[lookup_collection].find({"_id": {"$in": ids}}, projection)
                    async for doc in cursor:
                        doc_id = str(doc.get("_id"))
                        display = ""
                        for key in _lookup_display_candidates(field_meta):
                            candidate_value = doc.get(key)
                            if candidate_value is None:
                                continue
                            text = str(candidate_value).strip()
                            if text:
                                display = text
                                break
                        if display and doc_id not in resolved_values:
                            resolved_values[doc_id] = display
                    if resolved_values:
                        break
                except Exception:
                    continue

            if resolved_values:
                if isinstance(value, list):
                    row[field_name] = ", ".join(
                        str(resolved_values.get(str(_compact_text(item)), item)).strip()
                        for item in value
                        if str(resolved_values.get(str(_compact_text(item)), item)).strip()
                    )
                else:
                    text = _compact_text(value)
                    row[field_name] = resolved_values.get(text, value)
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# Group 7 — Response Building
# ═══════════════════════════════════════════════════════════════════════════════

def _response_table_columns(selected_collection, plan, docs, schema_index):
    def is_blank_value(value):
        if value is None:
            return True
        if isinstance(value, str):
            text = value.strip()
            return not text or text in {"-", "—"}
        if isinstance(value, (list, tuple, set)):
            return not any(not is_blank_value(item) for item in value)
        return False

    def doc_driven_columns(rows, limit=12):
        counts = {}
        order = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            for index, key in enumerate(row.keys()):
                key_text = str(key or "").strip()
                if not key_text or key_text == "_id":
                    continue
                order.setdefault(key_text, index)
                counts.setdefault(key_text, 0)
                if not is_blank_value(row.get(key_text)):
                    counts[key_text] += 1
        ranked = sorted(
            counts.keys(),
            key=lambda key: (
                counts.get(key, 0) <= 0,
                -counts.get(key, 0),
                order.get(key, 999),
                key,
            ),
        )
        return ranked[:limit]

    def columns_match_docs(names, rows):
        if not names:
            return False
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            for name in names:
                if not is_blank_value(row.get(name)):
                    return True
        return False

    collection = str(selected_collection or "").strip()
    schema_rows = list((schema_index.get(collection, {}) or {}).get("fields") or [])
    allowed_fields = [
        str((row or {}).get("name") or "").strip()
        for row in schema_rows
        if str((row or {}).get("name") or "").strip()
    ]
    allowed_set = set(allowed_fields)
    columns = []
    populated_doc_columns = doc_driven_columns(docs, limit=12)

    def add(name):
        name = str(name or "").strip()
        if not name or name in columns:
            return
        columns.append(name)

    operation = str((plan or {}).get("operation") or "find").lower().strip()
    if operation == "find":
        projection = plan.get("projection") if isinstance(plan, dict) and isinstance(plan.get("projection"), dict) else {}
        if projection:
            for key, value in projection.items():
                key_text = str(key).strip()
                if not key_text or not value:
                    continue
                if key_text in allowed_set or key_text in _effective_system_fields():
                    add(key_text)
                elif key_text.endswith("_") and key_text[:-1] in allowed_set:
                    add(key_text)
                elif key_text.endswith("_textMode") and key_text[:-9] in allowed_set:
                    add(key_text)
        if columns and not columns_match_docs(columns, docs) and populated_doc_columns:
            columns = []
            for key_text in populated_doc_columns:
                add(key_text)
        if not columns and populated_doc_columns:
            for key_text in populated_doc_columns:
                add(key_text)
        if not columns:
            roles = _infer_collection_field_roles(collection, schema_index)
            candidate_order = []
            candidate_order.extend(roles.get("business_id_fields") or [])
            candidate_order.extend(roles.get("label_fields") or [])
            candidate_order.extend(roles.get("created_time_fields") or [])
            candidate_order.extend([name for name in allowed_fields if "status" in name.lower()])
            candidate_order.extend(roles.get("soft_delete_fields") or [])
            for name in candidate_order:
                if name in allowed_set:
                    add(name)
        if not columns:
            first_doc = next((doc for doc in docs or [] if isinstance(doc, dict)), {})
            for key in first_doc.keys():
                key_text = str(key).strip()
                if key_text and key_text != "_id":
                    add(key_text)
    else:
        pipeline = list((plan or {}).get("pipeline") or [])
        for stage in pipeline:
            if not isinstance(stage, dict) or not stage:
                continue
            stage_name = next(iter(stage.keys()))
            stage_body = stage.get(stage_name)
            if stage_name in {"$project", "$addFields", "$set"} and isinstance(stage_body, dict):
                for key, value in stage_body.items():
                    key_text = str(key).strip()
                    if key_text and not key_text.startswith("$") and value:
                        add(key_text)
            elif stage_name == "$group" and isinstance(stage_body, dict):
                for key, value in stage_body.items():
                    key_text = str(key).strip()
                    if key_text and not key_text.startswith("$") and value is not None:
                        add(key_text)
            elif stage_name == "$lookup" and isinstance(stage_body, dict):
                as_field = str(stage_body.get("as") or "").strip()
                if as_field:
                    add(as_field)
            elif stage_name == "$unwind" and isinstance(stage_body, dict):
                include_array_index = str(stage_body.get("includeArrayIndex") or "").strip()
                if include_array_index:
                    add(include_array_index)
            elif stage_name == "$count":
                add(stage_body)
            elif stage_name == "$sortByCount":
                add("count")
        if columns and not columns_match_docs(columns, docs) and populated_doc_columns:
            columns = []
            for key_text in populated_doc_columns:
                add(key_text)
        if not columns and populated_doc_columns:
            for key_text in populated_doc_columns:
                add(key_text)
        if not columns:
            first_doc = next((doc for doc in docs or [] if isinstance(doc, dict)), {})
            for key in first_doc.keys():
                key_text = str(key).strip()
                if key_text and key_text != "_id":
                    add(key_text)

    return columns[:12]


def build_response_summary(collection_name, plan, docs, total, table_metadata):
    table_label = table_metadata.get(collection_name, {}).get("template_name", collection_name)
    if not docs:
        return f"No rows found in `{table_label}`."
    return (
        f"Found {len(docs)} rows from `{table_label}` "
        f"using `{plan.get('operation', 'find')}` query."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 8 — SSE & Streaming
# ═══════════════════════════════════════════════════════════════════════════════

def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class _ThinkTagStreamParser:
    def __init__(self):
        self.state = "ANSWER"
        self._buffer = ""
        self._open_tag = "<think>"
        self._close_tag = "</think>"

    @staticmethod
    def _suffix_prefix_overlap(text, tag):
        max_len = min(len(text), len(tag) - 1)
        for size in range(max_len, 0, -1):
            if text.endswith(tag[:size]):
                return size
        return 0

    def feed(self, chunk):
        if not chunk:
            return []
        self._buffer += str(chunk)
        events = []

        while self._buffer:
            if self.state == "THINKING":
                close_index = self._buffer.find(self._close_tag)
                if close_index == -1:
                    keep = self._suffix_prefix_overlap(self._buffer, self._close_tag)
                    emit_text = self._buffer[:-keep] if keep else self._buffer
                    if emit_text:
                        events.append(("token", "THINKING", emit_text))
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                if close_index > 0:
                    events.append(("token", "THINKING", self._buffer[:close_index]))
                self._buffer = self._buffer[close_index + len(self._close_tag) :]
                self.state = "ANSWER"
                events.append(("state", "ANSWER", ""))
                continue

            open_index = self._buffer.find(self._open_tag)
            if open_index == -1:
                keep = self._suffix_prefix_overlap(self._buffer, self._open_tag)
                emit_text = self._buffer[:-keep] if keep else self._buffer
                if emit_text:
                    events.append(("token", "ANSWER", emit_text))
                self._buffer = self._buffer[-keep:] if keep else ""
                break
            if open_index > 0:
                events.append(("token", "ANSWER", self._buffer[:open_index]))
            self._buffer = self._buffer[open_index + len(self._open_tag) :]
            self.state = "THINKING"
            events.append(("state", "THINKING", ""))

        return events

    def flush(self):
        if not self._buffer:
            return []
        state = "THINKING" if self.state == "THINKING" else "ANSWER"
        leftover = self._buffer
        self._buffer = ""
        return [("token", state, leftover)]


# ═══════════════════════════════════════════════════════════════════════════════
# Group 9 — Vector Schema Refresh
# ═══════════════════════════════════════════════════════════════════════════════

# Field types that hold human-searchable discrete values — index their distinct values.
_DISTINCT_INCLUDE_TYPES = frozenset({
    "select", "multi_select", "look_up", "multi_lookup",
    "text", "email", "phone", "sequence_number",
})
# Field types that are too long, binary, computed, or non-filterable — never index.
_DISTINCT_EXCLUDE_TYPES = frozenset({
    "password", "single_file", "multi_file", "textarea",
    "formula", "added_time", "modified_time", "checkbox",
    "ip", "date", "datetime", "number", "added_by", "modified_by",
})


def _is_distinct_candidate_field(field_row):
    """Return True if this field's distinct values should be indexed for reverse-lookup.
    Decision is purely type-driven — no hardcoded field-name checks.
    """
    name = str((field_row or {}).get("name") or "").strip()
    if not name:
        return False
    # Companion helper fields are collected separately via _collect_companion_hints
    if name.endswith("_") or name.endswith("_textMode"):
        return False
    ftype = str((field_row or {}).get("type") or "").lower().strip()
    if ftype in _DISTINCT_EXCLUDE_TYPES:
        return False
    if ftype in _DISTINCT_INCLUDE_TYPES:
        return True
    # Fields with options act like SELECT regardless of the raw type string
    options = (field_row or {}).get("options") or []
    return bool(options)


async def _collect_distinct_value_hints(db_name, schema_index):
    if not VECTOR_DISTINCT_ENABLED:
        return {}
    db = mongo_client()[db_name]
    hints = {}
    max_fields = max(1, int(VECTOR_DISTINCT_MAX_FIELDS_PER_COLLECTION or 12))
    max_values = max(1, int(VECTOR_DISTINCT_MAX_VALUES_PER_FIELD or 80))
    max_len = max(8, int(VECTOR_DISTINCT_MAX_VALUE_LENGTH or 64))

    async def _collect_field_distinct(collection, field_name):
        """Return deduplicated text values from a single distinct() call."""
        try:
            values = await db[collection].distinct(
                field_name,
                filter={"isDeleted": {"$ne": True}},
                maxTimeMS=1500,
            )
        except Exception:
            return []
        normalized = []
        seen = set()
        for value in values or []:
            if isinstance(value, (list, dict)) or value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            if len(text) > max_len:
                text = text[:max_len]
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
            if len(normalized) >= max_values:
                break
        return normalized

    for collection, item in (schema_index or {}).items():
        fields = list((item or {}).get("fields") or [])
        # Build a set of all field names present in the schema for companion detection
        all_field_names = {
            str((f or {}).get("name") or "").strip()
            for f in fields
            if str((f or {}).get("name") or "").strip()
        }

        candidates = [row for row in fields if _is_distinct_candidate_field(row)]
        if not candidates:
            continue
        candidates = candidates[:max_fields]
        by_field = {}

        for row in candidates:
            field_name = str((row or {}).get("name") or "").strip()
            if not field_name:
                continue

            # Collect base field distinct values
            base_values = await _collect_field_distinct(collection, field_name)
            if base_values:
                by_field[field_name] = base_values

            # Dynamically collect companion fields if they exist in the schema.
            # These hold human-readable tokens for LOOK_UP/SELECT fields.
            ftype = str((row or {}).get("type") or "").lower().strip()
            if ftype in ("look_up", "multi_lookup", "select", "multi_select"):
                for suffix in ("_", "_textMode"):
                    companion = f"{field_name}{suffix}"
                    if companion in all_field_names:
                        companion_values = await _collect_field_distinct(collection, companion)
                        if companion_values:
                            by_field[companion] = companion_values

        if by_field:
            hints[collection] = by_field
    return hints


async def _refresh_vector_schema_for_db(db_name):
    collections = await list_collections(db_name)
    if not collections:
        return False
    table_metadata = await load_table_metadata(db_name)
    ai_template_schemas = await load_ai_template_schemas(db_name, collections)
    schema_index = build_schema_index(
        table_metadata,
        collections,
        ai_template_schemas=ai_template_schemas,
    )
    distinct_hints = await _collect_distinct_value_hints(db_name, schema_index)
    cache_ready = warm_reverse_lookup_cache(db_name, schema_index, field_value_hints=distinct_hints)
    vector_ready = upsert_schema_vectors(db_name, schema_index, field_value_hints=distinct_hints)
    if cache_ready or vector_ready:
        _VECTOR_SCHEMA_WARMED.add(db_name)
    return bool(cache_ready or vector_ready)


async def _run_vector_refresh_once():
    if not VECTOR_DB_ENABLED:
        return {"enabled": False, "dbs_total": 0, "dbs_refreshed": 0}
    try:
        databases = await list_databases()
    except Exception:
        return {"enabled": True, "dbs_total": 0, "dbs_refreshed": 0}

    refreshed = 0
    for db_name in databases:
        try:
            if await _refresh_vector_schema_for_db(db_name):
                refreshed += 1
        except Exception:
            continue
    return {"enabled": True, "dbs_total": len(databases), "dbs_refreshed": refreshed}


async def _vector_refresh_loop():
    interval = max(60, int(VECTOR_REFRESH_INTERVAL_SECONDS or 900))
    while True:
        try:
            async with _VECTOR_REFRESH_LOCK:
                stats = await _run_vector_refresh_once()
                add_span_event("vector_refresh_tick", stats)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        await asyncio.sleep(interval)


# ═══════════════════════════════════════════════════════════════════════════════
# Group 10 — App Lifecycle Hooks
# ═══════════════════════════════════════════════════════════════════════════════

async def _startup_vector_refresh_scheduler():
    global _VECTOR_REFRESH_TASK
    if not VECTOR_DB_ENABLED or not VECTOR_REFRESH_ENABLED:
        return
    if _VECTOR_REFRESH_TASK is None or _VECTOR_REFRESH_TASK.done():
        _VECTOR_REFRESH_TASK = asyncio.create_task(_vector_refresh_loop())


async def _shutdown_vector_refresh_scheduler():
    global _VECTOR_REFRESH_TASK
    task = _VECTOR_REFRESH_TASK
    _VECTOR_REFRESH_TASK = None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    _SUGGESTION_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    _LLM_EXECUTOR.shutdown(wait=False, cancel_futures=True)


async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
