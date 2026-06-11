import asyncio
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial, wraps
from pathlib import Path
from threading import Lock

from bson import ObjectId
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from erp_backend.api.models import BootstrapResponse, ModelOptionsResponse, QueryRequest, QueryResponse, ResultFeedbackRequest, SuggestionsResponse
from erp_backend.core.config import (
    DEFAULT_DB_NAME,
    DEFAULT_USERS,
    LLM_ENABLE_CLARIFICATION,
    LLM_ENABLE_EMPTY_RESULT_REPAIR,
    LLM_ENABLE_MISMATCH_RESULT_REPAIR,
    LLM_SINGLE_PASS_QUERY,
    LLM_USE_CHAT_HISTORY_HINTS,
    MAX_NEW_TOKENS,
    BASE_MODEL,
    GGUF_MODEL_PATH,
    OLLAMA_MODEL,
    SYSTEM_FIELD_NAMES,
)
from erp_backend.core.observability import add_span_event, init_observability, set_span_attribute, traced_span
from erp_backend.core.feedback import add_feedback
from erp_backend.core.security import configure_hidden_field_policy
from erp_backend.core.utils import lookup_tokens, normalize_lookup_text
from erp_backend.services.continuous_trainer import background_training_loop, trigger_training, get_training_info
from erp_backend.llm.runtime import load_model
from erp_backend.services.field_retriever import retrieve_candidates
from erp_backend.services.query import (
    build_chart_config, build_llm_chart_config,
    build_result_insights,
    execute_plan,
    generate_follow_up_suggestions,
    generate_sidebar_suggestions,
    generate_query_plan,
    generate_single_pass_query_plan,
    generate_table_choice,
    repair_query_plan_on_empty_result,
    repair_query_plan_on_mismatch_result,
    analyze_query_result,
    analyze_query_result_stream,
    summarize_query_result,
    validate_query_plan,
    verify_query_result,
    _fallback_clarification_suggestions,
)

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
    ping_mongo,
    mongo_client,
    sample_collection,
)
from erp_backend.storage.vector_store import (
    retrieve_exact_field_candidates,
    retrieve_exact_field_candidates_with_values,
    retrieve_field_candidates,
    upsert_schema_vectors,
    warm_reverse_lookup_cache,
)
from erp_backend.services.llm import (
    _run_llm_task, _iterate_llm_task,
    _discover_gguf_models, _discover_safetensors_models, _discover_ollama_models,
    _load_query_model, _load_reasoning_model,
    _normalized_model_spec, _query_model_spec, _reasoning_model_spec,
    _load_model_from_spec,
    _default_model_context_limit, _extract_context_limit_hint,
    _estimate_model_context_limit, _build_model_context_limits,
    _smallest_gguf_model_path, _ordered_gguf_candidates, _default_reasoning_gguf_model_path, _resolve_default_ollama_model,
    _scalar_text,
)
from erp_backend.services.orchestrate import (
    _load_and_apply_runtime_policy,
    _build_accuracy_planner_prompt,
    _build_feedback_rewrite_payload_async, _build_clarification_payload_async,
    _collect_distinct_value_hints,
    _should_use_llm_prompt_rewrite,
    _rewrite_prompt_with_llm_details,
    _write_perf_log, _ms_since,
    _queue_streaming_response, _with_request_fifo,
    build_response_summary, _sse, _ThinkTagStreamParser,
    _resolve_lookup_ids_to_names,
    _response_table_columns,
)
from erp_backend.services.intent import (
    _query_route_profile,
    _recent_chat_context,
    _llm_scope_gate,
    _non_data_query_response,
    _personalize_response,
    _selected_user_context,
    _build_user_scoped_retrieval_query,
    _build_hybrid_field_candidates,
    _infer_requested_collections,
    _ensure_plan_collection,
    _field_aliases,
    _humanize_runtime_field_name,
    _light_normalize_prompt,
    _permission_denied_message,
)


logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager


@asynccontextmanager
async def _app_lifespan(app_instance):
    trainer_task = asyncio.create_task(background_training_loop(interval_seconds=300))
    logger.info("Background training loop started.")
    yield
    trainer_task.cancel()
    try:
        await trainer_task
    except asyncio.CancelledError:
        pass
    logger.info("Background training loop stopped.")


app = FastAPI(title="ERP Query Backend", lifespan=_app_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from erp_backend.core.metrics import init_prometheus

init_observability(app, service_name="erp-query-backend")
init_prometheus(app)
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


def _response_artifacts(selected_collection, table_metadata, plan, docs, total, model=None, tokenizer=None, user_input=""):
    chart = build_chart_config(plan, docs)
    if model and tokenizer and user_input and len(docs or []) >= 2:
        try:
            llm_chart = build_llm_chart_config(model, tokenizer, user_input, docs)
            if llm_chart is not None:
                chart = llm_chart
        except Exception:
            pass
    return {
        "insights": build_result_insights(selected_collection, table_metadata, docs, total),
        "chart_config": chart,
        "db_performance": None,
    }





def _augment_schema_index_with_runtime_fields(schema_index, collection_name, runtime_schema):
    collection = str(collection_name or "").strip()
    if not collection or not isinstance(schema_index, dict) or not isinstance(runtime_schema, dict):
        return schema_index
    collection_payload = dict(schema_index.get(collection, {}) or {})
    fields = list(collection_payload.get("fields") or [])
    seen = {
        str((row or {}).get("name") or "").strip()
        for row in fields
        if str((row or {}).get("name") or "").strip()
    }
    runtime_rows = []
    for field_name, types in sorted(runtime_schema.items()):
        name = str(field_name or "").strip()
        if not name or name in seen:
            continue
        runtime_rows.append(
            {
                "name": name,
                "display": _humanize_runtime_field_name(name) or name,
                "type": ", ".join(str(item) for item in (types or []) if str(item).strip()),
                "aliases": _field_aliases(name),
                "lookup_collection": "",
                "options": [],
            }
        )
        seen.add(name)
    if not runtime_rows:
        return schema_index
    collection_payload["fields"] = fields + runtime_rows
    augmented = dict(schema_index)
    augmented[collection] = collection_payload
    return augmented


@app.get("/health")
@app.get("/health/live")
async def health():
    return {"ok": True, "status": "live"}


@app.get("/ready")
@app.get("/health/ready")
async def readiness():
    await ping_mongo()
    return {"ok": True, "status": "ready", "checks": {"mongo": True}}


@app.get("/models/options", response_model=ModelOptionsResponse)
async def model_options():
    default_reasoning_gguf = _default_reasoning_gguf_model_path() or str(GGUF_MODEL_PATH or "")
    gguf_models = _discover_gguf_models()
    safetensors_models = _discover_safetensors_models()
    ollama_models = _discover_ollama_models()
    default_ollama_model = _resolve_default_ollama_model() or str(OLLAMA_MODEL or "")
    context_limits = _build_model_context_limits(gguf_models, safetensors_models, ollama_models)
    default_query_model = str(GGUF_MODEL_PATH or "")
    default_query_context_limit = _estimate_model_context_limit(default_query_model, "gguf")
    default_reasoning_context_limit = _estimate_model_context_limit(default_reasoning_gguf, "gguf")
    return ModelOptionsResponse(
        gguf_models=gguf_models,
        safetensors_models=safetensors_models,
        ollama_models=ollama_models,
        context_limits=context_limits,
        defaults={
            "model_runtime": "gguf",
            "gguf_model_path": str(GGUF_MODEL_PATH or ""),
            "safetensors_model_id": str(BASE_MODEL or ""),
            "ollama_model": default_ollama_model,
            "safetensors_context_token_limit": _default_model_context_limit(),
            "reasoning_model_runtime": "gguf",
            "reasoning_gguf_model_path": str(default_reasoning_gguf),
            "reasoning_safetensors_model_id": str(BASE_MODEL or ""),
            "reasoning_ollama_model": default_ollama_model,
            "reasoning_model_enabled": True,
            "context_token_limit": default_query_context_limit,
            "reasoning_context_token_limit": default_reasoning_context_limit,
        },
    )


@app.get("/bootstrap", response_model=BootstrapResponse)
async def bootstrap(
    db_name: str = DEFAULT_DB_NAME,
    user_id: str | None = None,
    refresh_rbac: bool = False,
):
    await ping_mongo()
    databases = await list_databases()
    collections = await list_collections(db_name)
    table_metadata = await load_table_metadata(db_name)
    users = await load_rbac_users(db_name, force_refresh=refresh_rbac)
    if user_id:
        selected_user = next((user for user in users if str(user.get("user_id")) == str(user_id)), None)
        if selected_user:
            accessible = allowed_collections_for_user(collections, selected_user)
            collections = [name for name in collections if name in accessible]
            table_metadata = {name: meta for name, meta in (table_metadata or {}).items() if name in accessible}
    return BootstrapResponse(
        databases=databases,
        collections=collections,
        table_metadata=table_metadata,
        users=users or DEFAULT_USERS,
    )


@app.get("/suggestions", response_model=SuggestionsResponse)
async def suggestions(
    db_name: str = DEFAULT_DB_NAME,
    user_id: str = "admin",
):
    await ping_mongo()
    collections = await list_collections(db_name)
    table_metadata = await load_table_metadata(db_name)
    users = await load_rbac_users(db_name, force_refresh=False)
    users = users or DEFAULT_USERS
    selected_user = next((user for user in users if str(user.get("user_id")) == str(user_id)), users[0])
    accessible_collections = allowed_collections_for_user(collections, selected_user)
    scoped_metadata = {
        name: meta for name, meta in (table_metadata or {}).items() if name in set(accessible_collections)
    }

    def _suggestion_worker():
        # Keep sidebar AI generation isolated from main query GPU runtime.
        # Use CPU + smallest GGUF to avoid VRAM contention and model-load failures.
        model, tokenizer = load_model(
            runtime="gguf",
            gguf_model_path=_smallest_gguf_model_path(),
            compute_mode="cpu",
            cache_namespace="suggestions",
        )
        return generate_sidebar_suggestions(
            model,
            tokenizer,
            db_name,
            accessible_collections,
            scoped_metadata,
            max_items=12,
        )

    try:
        loop = asyncio.get_running_loop()
        ai_suggestions = await asyncio.wait_for(
            loop.run_in_executor(_SUGGESTION_EXECUTOR, _suggestion_worker),
            timeout=12.0,
        )
        if ai_suggestions:
            return SuggestionsResponse(suggestions=ai_suggestions)
    except Exception:
        pass

    # AI-only mode: do not generate deterministic/rule-based fallback suggestions.
    return SuggestionsResponse(suggestions=[])


@app.post("/query", response_model=QueryResponse)
@_with_request_fifo
async def query(payload: QueryRequest):
    stage_ms = {}
    req_started = time.perf_counter()
    validation_enabled = bool(getattr(payload, "validation_enabled", True))
    direct_llm_mode = not validation_enabled
    with traced_span(
        "erp.query.request",
        {
            "db.name": payload.db_name,
            "user.id": payload.user_id,
            "runtime": payload.model_runtime,
            "reasoning_runtime": payload.reasoning_model_runtime,
            "reasoning_enabled": bool(payload.reasoning_model_enabled),
            "validation_enabled": validation_enabled,
            "compute_mode": payload.compute_mode,
            "prompt.length": len(str(payload.prompt or "")),
        },
    ):
        t_bootstrap = time.perf_counter()
        await ping_mongo()
        collections = await list_collections(payload.db_name)
        table_metadata = await load_table_metadata(payload.db_name)
        users = await load_rbac_users(payload.db_name, force_refresh=False)
        users = users or DEFAULT_USERS
        selected_user = next((user for user in users if user.get("user_id") == payload.user_id), users[0])
        accessible_collections = allowed_collections_for_user(collections, selected_user)
        selected_user_context = _selected_user_context(selected_user, accessible_collections=accessible_collections)
        runtime_policy = await _load_and_apply_runtime_policy(payload.db_name)
        set_span_attribute("collections.total", len(collections))
        set_span_attribute("collections.accessible", len(accessible_collections))
        if not accessible_collections:
            raise HTTPException(status_code=403, detail="Selected user has no table access.")
        requested_collections = _infer_requested_collections(payload.prompt, collections, table_metadata)
        denied_targets = [name for name in requested_collections if name not in accessible_collections]
        allowed_targets = [name for name in requested_collections if name in accessible_collections]
        if denied_targets and not allowed_targets:
            response = _personalize_response(_permission_denied_message(denied_targets, table_metadata), selected_user)
            add_span_event("rbac_denied", {"denied.count": len(denied_targets)})
            stage_ms["bootstrap"] = _ms_since(t_bootstrap)
            stage_ms["total"] = _ms_since(req_started)
            _write_perf_log(payload, "/query", stage_ms, "rbac_denied")
            return QueryResponse(
                response=response,
                follow_ups=[],
                table_choice={"collection": None, "reason": "RBAC denied for requested collection."},
                plan={"operation": "none"},
                docs=[],
                total=0,
                collection="",
            )
        stage_ms["bootstrap"] = _ms_since(t_bootstrap)
        ai_template_schemas = await load_ai_template_schemas(payload.db_name, accessible_collections)
        schema_index = build_schema_index(
            table_metadata,
            accessible_collections,
            ai_template_schemas=ai_template_schemas,
        )
        vector_key = payload.db_name
        if vector_key not in _VECTOR_SCHEMA_WARMED:
            distinct_hints = await _collect_distinct_value_hints(payload.db_name, schema_index)
            cache_ready = warm_reverse_lookup_cache(payload.db_name, schema_index, field_value_hints=distinct_hints)
            vector_ready = upsert_schema_vectors(payload.db_name, schema_index, field_value_hints=distinct_hints)
            if cache_ready or vector_ready:
                _VECTOR_SCHEMA_WARMED.add(vector_key)
                add_span_event(
                    "vector_schema_warm",
                    {"db": payload.db_name, "cache_ready": bool(cache_ready), "vector_ready": bool(vector_ready)},
                )

        effective_chat_context = []
        last_collection = None
        if LLM_USE_CHAT_HISTORY_HINTS:
            conversation_state = await load_conversation_state(
                payload.db_name,
                payload.user_id,
                payload.conversation_id,
            )
            cached_context = conversation_state.get("messages") or []
            last_collection = conversation_state.get("last_collection")
            effective_chat_context = _recent_chat_context(list(cached_context) + list(payload.chat_context or []))
            set_span_attribute("chat.context.messages", len(effective_chat_context))

        t_exact = time.perf_counter()
        exact_candidates, exact_matched_values = retrieve_exact_field_candidates_with_values(
            payload.db_name,
            payload.prompt,
            allowed_collections=accessible_collections,
        )
        stage_ms["exact_lookup"] = _ms_since(t_exact)
        use_vector_fallback = not exact_candidates
        t_vector = time.perf_counter()
        retrieval_query = _build_user_scoped_retrieval_query(payload.prompt, selected_user_context)
        vector_candidates = (
            retrieve_field_candidates(
                payload.db_name,
                retrieval_query,
                allowed_collections=accessible_collections,
            )
            if use_vector_fallback
            else {}
        )
        stage_ms["vector_retrieval"] = _ms_since(t_vector) if use_vector_fallback else 0.0
        hybrid_field_candidates, field_source_tracking = _build_hybrid_field_candidates(
            payload.prompt,
            schema_index,
            exact_candidates,
            vector_candidates,
        )
        route_profile = _query_route_profile(
            payload.prompt,
            accessible_collections,
            table_metadata,
            schema_index,
            exact_candidates,
            vector_candidates,
            last_collection=last_collection,
        )
        use_single_pass_route = direct_llm_mode or bool(LLM_SINGLE_PASS_QUERY) or bool(route_profile.get("use_single_pass"))
        effective_prompt = route_profile.get("normalized_prompt") or _light_normalize_prompt(payload.prompt)

        reasoning_enabled = bool(payload.reasoning_model_enabled)
        if reasoning_enabled:
            query_spec = _query_model_spec(payload)
            reasoning_spec = _reasoning_model_spec(payload)
            same_model_spec = query_spec == reasoning_spec
            t_reasoning_model_load = time.perf_counter()
            with traced_span("erp.query.reasoning_model_load"):
                reasoning_model, reasoning_tokenizer = await _run_llm_task(_load_reasoning_model, payload)
            stage_ms["reasoning_model_load"] = _ms_since(t_reasoning_model_load)
            if use_single_pass_route or same_model_spec:
                model, tokenizer = reasoning_model, reasoning_tokenizer
                stage_ms["model_load"] = 0.0
            else:
                t_model_load = time.perf_counter()
                with traced_span("erp.query.model_load"):
                    model, tokenizer = await _run_llm_task(_load_query_model, payload)
                stage_ms["model_load"] = _ms_since(t_model_load)
        else:
            stage_ms["reasoning_model_load"] = 0.0
            t_model_load = time.perf_counter()
            with traced_span("erp.query.model_load"):
                model, tokenizer = await _run_llm_task(_load_query_model, payload)
            stage_ms["model_load"] = _ms_since(t_model_load)
            reasoning_model, reasoning_tokenizer = model, tokenizer

        if use_single_pass_route:
            healed_prompt = {
                "normalized_query": effective_prompt,
                "intent_type": "direct",
                "entity_terms": [],
                "field_terms": [],
                "value_terms": [],
                "confidence": 0.9,
                "needs_clarification": False,
                "hints": {},
            }
            stage_ms["prompt_rewrite"] = 0.0
            t_scope_gate = time.perf_counter()
            if route_profile.get("skip_scope_gate"):
                scope_gate = _scope_guardrail(
                    effective_prompt,
                    accessible_collections,
                    table_metadata,
                )
            else:
                scope_gate = await _run_llm_task(
                    _llm_scope_gate,
                    reasoning_model,
                    reasoning_tokenizer,
                    effective_prompt,
                    accessible_collections,
                    table_metadata,
                    chat_context=effective_chat_context,
                    db_name=payload.db_name,
                )
            stage_ms["scope_gate"] = _ms_since(t_scope_gate)
            if not scope_gate.get("allow", True):
                response = _personalize_response(scope_gate.get("message") or _non_data_query_response(), selected_user)
                add_span_event("scope_rejected")
                if LLM_USE_CHAT_HISTORY_HINTS:
                    await save_conversation_state(
                        payload.db_name,
                        payload.user_id,
                        payload.conversation_id,
                        effective_chat_context
                        + [{"role": "user", "content": payload.prompt}, {"role": "assistant", "content": response}],
                        last_collection=last_collection,
                    )
                stage_ms["total"] = _ms_since(req_started)
                _write_perf_log(payload, "/query", stage_ms, "scope_rejected")
                return QueryResponse(
                    response=response,
                    follow_ups=[
                        "What is the current organization address?",
                        "List all active users.",
                        "Show branch-wise user count.",
                    ],
                    table_choice={"collection": None, "reason": "AI scope gate rejected prompt."},
                    plan={"operation": "none"},
                    docs=[],
                    total=0,
                    collection="",
                )
        else:
            # Pipeline step 1-2: Prompt -> LLM rewrite -> rewrite validator
            if _should_use_llm_prompt_rewrite(effective_prompt):
                t_prompt_rewrite = time.perf_counter()
                with traced_span("erp.query.prompt_rewrite"):
                    healed_prompt = await _run_llm_task(_rewrite_prompt_with_llm_details, model, payload.prompt)
                    effective_prompt = _build_accuracy_planner_prompt(healed_prompt) or str(payload.prompt or "")
                stage_ms["prompt_rewrite"] = _ms_since(t_prompt_rewrite)
            else:
                healed_prompt = {
                    "normalized_query": effective_prompt,
                    "intent_type": "direct",
                    "entity_terms": [],
                    "field_terms": [],
                    "value_terms": [],
                    "confidence": 0.75,
                    "needs_clarification": False,
                    "hints": {},
                }
                stage_ms["prompt_rewrite"] = 0.0
            t_scope_gate = time.perf_counter()
            scope_gate = await _run_llm_task(
                _llm_scope_gate,
                reasoning_model,
                reasoning_tokenizer,
                effective_prompt,
                accessible_collections,
                table_metadata,
                chat_context=effective_chat_context,
                db_name=payload.db_name,
            )
            stage_ms["scope_gate"] = _ms_since(t_scope_gate)
            if not scope_gate.get("allow", True):
                response = _personalize_response(scope_gate.get("message") or _non_data_query_response(), selected_user)
                add_span_event("scope_rejected")
                if LLM_USE_CHAT_HISTORY_HINTS:
                    await save_conversation_state(
                        payload.db_name,
                        payload.user_id,
                        payload.conversation_id,
                        effective_chat_context
                        + [{"role": "user", "content": payload.prompt}, {"role": "assistant", "content": response}],
                        last_collection=last_collection,
                    )
                stage_ms["total"] = _ms_since(req_started)
                _write_perf_log(payload, "/query", stage_ms, "scope_rejected")
                return QueryResponse(
                    response=response,
                    follow_ups=[
                        "What is the current organization address?",
                        "List all active users.",
                        "Show branch-wise user count.",
                    ],
                    table_choice={"collection": None, "reason": "AI scope gate rejected prompt."},
                    plan={"operation": "none"},
                    docs=[],
                    total=0,
                    collection="",
                )

    table_choice = None

    t_plan_generation = time.perf_counter()
    if use_single_pass_route:
        preferred_col = (route_profile.get("deterministic_choice") or {}).get("collection")
        raw_plan = await _run_llm_task(
            generate_single_pass_query_plan,
            model,
            tokenizer,
            effective_prompt,
            payload.db_name,
            accessible_collections,
            table_metadata,
            chat_context=effective_chat_context,
            preferred_collection=preferred_col,
            exact_field_candidates=exact_candidates,
            vector_field_candidates=vector_candidates,
            user_context=selected_user_context,
            matched_values=exact_matched_values,
            field_source_tracking=field_source_tracking,
        )
        if not isinstance(raw_plan, dict):
            raw_plan = {}
        selected_collection = str(raw_plan.get("collection") or "").strip()
        if selected_collection not in set(accessible_collections or []):
            selected_collection = ""
        if raw_plan.get("needs_clarification") or not selected_collection:
            if LLM_ENABLE_CLARIFICATION:
                stage_ms["plan_generation"] = _ms_since(t_plan_generation)
                stage_ms["total"] = _ms_since(req_started)
                _write_perf_log(payload, "/query", stage_ms, "needs_clarification")
                clarification = await _build_clarification_payload_async(
                    reasoning_model,
                    reasoning_tokenizer,
                    effective_prompt,
                    selected_user,
                    "",
                    table_choice,
                    {"operation": "none"},
                    [],
                    0,
                    table_metadata,
                    accessible_collections,
                    reason=str((raw_plan or {}).get("message") or "Ambiguous request."),
                )
                if LLM_USE_CHAT_HISTORY_HINTS:
                    await save_conversation_state(
                        payload.db_name,
                        payload.user_id,
                        payload.conversation_id,
                        effective_chat_context
                        + [{"role": "user", "content": payload.prompt}, {"role": "assistant", "content": clarification.response}],
                        last_collection=last_collection,
                    )
                return clarification
            raise HTTPException(status_code=422, detail=str((raw_plan or {}).get("message") or "I could not identify the target ERP table."))
        table_choice = {"collection": selected_collection, "reason": "Single-pass LLM routing."}
        table_choice["collection"] = selected_collection
        template_schema = await get_template_schema(payload.db_name, selected_collection)
        sample_docs = await sample_collection(payload.db_name, selected_collection)
        schema = infer_schema_from_docs(sample_docs)
        runtime_field_names = list((schema or {}).keys())
        schema_index = _augment_schema_index_with_runtime_fields(schema_index, selected_collection, schema)
    else:
        if table_choice is None:
            table_choice = await _run_llm_task(
                generate_table_choice,
                reasoning_model,
                reasoning_tokenizer,
                effective_prompt,
                payload.db_name,
                accessible_collections,
                table_metadata,
                chat_context=effective_chat_context,
                user_context=selected_user_context,
                exact_candidates=exact_candidates,
            )
        invalid_collection = str((table_choice or {}).get("collection") or "").strip() not in set(accessible_collections or [])
        if (table_choice or {}).get("needs_clarification") or invalid_collection:
            if LLM_ENABLE_CLARIFICATION:
                stage_ms["plan_generation"] = _ms_since(t_plan_generation)
                stage_ms["total"] = _ms_since(req_started)
                _write_perf_log(payload, "/query", stage_ms, "needs_clarification")
                clarification = await _build_clarification_payload_async(
                    reasoning_model,
                    reasoning_tokenizer,
                    effective_prompt,
                    selected_user,
                    "",
                    table_choice,
                    {"operation": "none"},
                    [],
                    0,
                    table_metadata,
                    accessible_collections,
                    reason=str((table_choice or {}).get("message") or "Ambiguous request."),
                        )
                if LLM_USE_CHAT_HISTORY_HINTS:
                    await save_conversation_state(
                        payload.db_name,
                        payload.user_id,
                        payload.conversation_id,
                        effective_chat_context
                        + [{"role": "user", "content": payload.prompt}, {"role": "assistant", "content": clarification.response}],
                        last_collection=last_collection,
                    )
                return clarification
            raise HTTPException(status_code=422, detail=str((table_choice or {}).get("message") or "I could not identify the target ERP table."))

        selected_collection = str((table_choice or {}).get("collection") or "").strip()
        table_choice["collection"] = selected_collection
        template_schema = await get_template_schema(payload.db_name, selected_collection)
        sample_docs = await sample_collection(payload.db_name, selected_collection)
        schema = infer_schema_from_docs(sample_docs)
        runtime_field_names = list((schema or {}).keys())
        schema_index = _augment_schema_index_with_runtime_fields(schema_index, selected_collection, schema)
        raw_plan = await _run_llm_task(
            generate_query_plan,
            model,
            tokenizer,
            effective_prompt,
            payload.db_name,
            selected_collection,
            schema,
            sample_docs,
            table_metadata,
            template_schema=template_schema,
            chat_context=effective_chat_context,
            exact_field_candidates=exact_candidates,
            vector_field_candidates=vector_candidates,
            matched_values=exact_matched_values,
            user_context=selected_user_context,
            field_source_tracking=field_source_tracking,
        )
    stage_ms["plan_generation"] = _ms_since(t_plan_generation)
    t_plan_validate = time.perf_counter()
    with traced_span("erp.query.plan_validate"):
        from erp_backend.services.intent import _strict_schema_validate_plan
        raw_plan = _ensure_plan_collection(raw_plan, selected_collection)
        _strict_schema_validate_plan(raw_plan, schema_index)
        plan = validate_query_plan(
            raw_plan,
            accessible_collections,
        )
    stage_ms["plan_validate"] = _ms_since(t_plan_validate)
    if plan.get("needs_clarification"):
        if LLM_ENABLE_CLARIFICATION:
            stage_ms["total"] = _ms_since(req_started)
            _write_perf_log(payload, "/query", stage_ms, "needs_clarification")
            clarification = await _build_clarification_payload_async(
                model,
                tokenizer,
                effective_prompt,
                selected_user,
                selected_collection,
                table_choice,
                plan,
                [],
                0,
                table_metadata,
                accessible_collections,
                reason=str(plan.get("message") or "Plan requires clarification."),
            )
            if LLM_USE_CHAT_HISTORY_HINTS:
                await save_conversation_state(
                    payload.db_name,
                    payload.user_id,
                    payload.conversation_id,
                    effective_chat_context
                    + [{"role": "user", "content": payload.prompt}, {"role": "assistant", "content": clarification.response}],
                    last_collection=selected_collection,
                )
            return clarification
        raise HTTPException(status_code=422, detail=str(plan.get("message") or "Plan requires clarification."))
    t_execute = time.perf_counter()
    with traced_span("erp.query.execute"):
        docs, total = await execute_plan(payload.db_name, plan)
    stage_ms["execute"] = _ms_since(t_execute)
    set_span_attribute("rows.returned", len(docs))
    set_span_attribute("rows.total", int(total or 0))
    docs = await _resolve_lookup_ids_to_names(payload.db_name, selected_collection, docs, table_metadata)

    # ── Empty result self-heal ──
    if LLM_ENABLE_EMPTY_RESULT_REPAIR and (total == 0 or not docs) and accessible_collections:
        t_repair = time.perf_counter()
        with traced_span("erp.query.repair_empty"):
            repair_plan = await _run_llm_task(
                repair_query_plan_on_empty_result,
                model,
                tokenizer,
                effective_prompt,
                payload.db_name,
                table_metadata,
                accessible_collections,
                plan,
                exact_field_candidates=exact_candidates,
                vector_field_candidates=vector_candidates,
            )
            if isinstance(repair_plan, dict) and not repair_plan.get("needs_clarification"):
                repair_plan = validate_query_plan(repair_plan, accessible_collections)
                if not repair_plan.get("needs_clarification"):
                    docs, total = await execute_plan(payload.db_name, repair_plan)
                    docs = await _resolve_lookup_ids_to_names(payload.db_name, selected_collection, docs, table_metadata)
                    plan = repair_plan
                    set_span_attribute("rows.returned", len(docs))
                    set_span_attribute("rows.total", int(total or 0))
        stage_ms["repair_empty"] = _ms_since(t_repair)

    # ── Mismatch result self-heal ──
    if LLM_ENABLE_MISMATCH_RESULT_REPAIR and docs and accessible_collections:
        t_verify = time.perf_counter()
        with traced_span("erp.query.verify"):
            verifier_result = await _run_llm_task(
                verify_query_result,
                model,
                tokenizer,
                effective_prompt,
                selected_collection,
                table_metadata,
                plan,
                docs,
                total,
            )
        if isinstance(verifier_result, dict) and str(verifier_result.get("status") or "ok").strip().lower() == "needs_clarification":
            t_repair_mismatch = time.perf_counter()
            with traced_span("erp.query.repair_mismatch"):
                repair_plan = await _run_llm_task(
                    repair_query_plan_on_mismatch_result,
                    model,
                    tokenizer,
                    effective_prompt,
                    payload.db_name,
                    table_metadata,
                    accessible_collections,
                    plan,
                    docs,
                    verifier_message=str(verifier_result.get("message") or ""),
                    exact_field_candidates=exact_candidates,
                    vector_field_candidates=vector_candidates,
                )
                if isinstance(repair_plan, dict) and not repair_plan.get("needs_clarification"):
                    repair_plan = validate_query_plan(repair_plan, accessible_collections)
                    if not repair_plan.get("needs_clarification"):
                        docs, total = await execute_plan(payload.db_name, repair_plan)
                        docs = await _resolve_lookup_ids_to_names(payload.db_name, selected_collection, docs, table_metadata)
                        plan = repair_plan
                        set_span_attribute("rows.returned", len(docs))
                        set_span_attribute("rows.total", int(total or 0))
            stage_ms["repair_mismatch"] = _ms_since(t_repair_mismatch)
        stage_ms["verify"] = _ms_since(t_verify)
    try:
        t_summarize = time.perf_counter()
        with traced_span("erp.query.summarize"):
            response = await _run_llm_task(
                analyze_query_result,
                model,
                tokenizer,
                effective_prompt,
                selected_collection,
                table_metadata,
                plan,
                docs,
                total,
            )
        stage_ms["summarize"] = _ms_since(t_summarize)
    except Exception:
        response = build_response_summary(selected_collection, plan, docs, total, table_metadata)
    response = _personalize_response(response, selected_user)

    try:
        summary = await _run_llm_task(
            summarize_query_result,
            model,
            tokenizer,
            effective_prompt,
            selected_collection,
            table_metadata,
            plan,
            docs,
            total,
        )
    except Exception:
        summary = ""

    follow_ups = []
    try:
        t_followups = time.perf_counter()
        with traced_span("erp.query.followups"):
            follow_ups = await _run_llm_task(
                generate_follow_up_suggestions,
                model,
                tokenizer,
                effective_prompt,
                selected_collection,
                table_metadata,
                plan,
                docs,
                total,
            )
        stage_ms["follow_ups"] = _ms_since(t_followups)
    except Exception:
        follow_ups = []
    table_columns = _response_table_columns(selected_collection, plan, docs, schema_index)

    if LLM_USE_CHAT_HISTORY_HINTS:
        await save_conversation_state(
            payload.db_name,
            payload.user_id,
            payload.conversation_id,
            effective_chat_context
            + [{"role": "user", "content": payload.prompt}, {"role": "assistant", "content": response}],
            last_collection=selected_collection,
        )
    stage_ms["total"] = _ms_since(req_started)
    _write_perf_log(
        payload,
        "/query",
        stage_ms,
        "ok",
        extra={"rows": len(docs), "total_rows": int(total or 0), "collection": selected_collection},
    )
    response_artifacts = _response_artifacts(selected_collection, table_metadata, plan, docs, total, model=model, tokenizer=tokenizer, user_input=effective_prompt)
    return QueryResponse(
        response=response,
        follow_ups=follow_ups,
        insights=response_artifacts["insights"],
        chart_config=response_artifacts["chart_config"],
        db_performance=response_artifacts["db_performance"],
        table_choice=table_choice,
        plan=plan,
        docs=docs,
        total=total,
        collection=selected_collection,
        table_columns=table_columns,
        summary=summary,
    )


@app.post("/query_feedback", response_model=QueryResponse)
async def query_feedback(payload: ResultFeedbackRequest):
    feedback = str(payload.feedback or "").strip().lower()
    is_negative = feedback in {"down", "thumbs_down", "dislike", "incorrect"}

    # Store rich feedback with corrections when available
    wrong_fields_list = None
    correct_fields_list = None
    if payload.plan and isinstance(payload.plan, dict):
        plan_collection = str(payload.plan.get("collection") or "").strip()
        if plan_collection and payload.correct_collection and plan_collection != payload.correct_collection:
            wrong_fields_list = [plan_collection]
            correct_fields_list = [payload.correct_collection]
    if payload.correct_fields and isinstance(payload.correct_fields, dict):
        wrong_fields_list = list(payload.correct_fields.keys())
        correct_fields_list = list(payload.correct_fields.values())

    add_feedback(
        question=payload.prompt,
        positive=not is_negative,
        wrong_collection=str(payload.collection or (payload.plan or {}).get("collection") or "") if is_negative else None,
        correct_collection=payload.correct_collection,
        wrong_fields=wrong_fields_list,
        correct_fields=correct_fields_list,
        plan=payload.plan if is_negative else None,
    )



    if not is_negative:
        return QueryResponse(
            response="Feedback recorded.",
            follow_ups=[],
            needs_clarification=False,
            table_choice=payload.table_choice or {"collection": payload.collection or None, "reason": "feedback"},
            plan=payload.plan or {"operation": "none"},
            docs=payload.docs or [],
            total=int(payload.total or len(payload.docs or []) or 0),
            collection=str(payload.collection or ""),
            table_columns=list(payload.table_columns or []),
            summary="",
        )

    req_started = time.perf_counter()
    stage_ms = {}
    try:
        collections = await list_collections(payload.db_name)
        table_metadata = await load_table_metadata(payload.db_name)
        users = await load_rbac_users(payload.db_name, force_refresh=False) or DEFAULT_USERS
        selected_user = next((user for user in users if user.get("user_id") == payload.user_id), users[0])
        accessible_collections = allowed_collections_for_user(collections, selected_user)
        ai_template_schemas = await load_ai_template_schemas(payload.db_name, accessible_collections)
        schema_index = build_schema_index(
            table_metadata,
            accessible_collections,
            ai_template_schemas=ai_template_schemas,
        )
        exact_candidates, exact_matched_values = retrieve_exact_field_candidates_with_values(
            payload.db_name,
            payload.prompt,
            allowed_collections=accessible_collections,
        )
        use_vector_fallback = not exact_candidates
        retrieval_query = _build_user_scoped_retrieval_query(
            payload.prompt,
            _selected_user_context(selected_user, accessible_collections=accessible_collections),
        )
        vector_candidates = (
            retrieve_field_candidates(
                payload.db_name,
                retrieval_query,
                allowed_collections=accessible_collections,
            )
            if use_vector_fallback
            else {}
        )

        selected_collection = str(payload.collection or "").strip()
        if not selected_collection:
            selected_collection = str((payload.table_choice or {}).get("collection") or "").strip()
        if not selected_collection:
            selected_collection = str((payload.plan or {}).get("collection") or "").strip()
        routed_collection = _query_route_profile(
            payload.prompt,
            accessible_collections,
            table_metadata,
            schema_index,
            exact_candidates,
            vector_candidates,
            last_collection=selected_collection or None,
        ).get("deterministic_choice") or ""
        if routed_collection and routed_collection in accessible_collections:
            selected_collection = routed_collection
        elif selected_collection and selected_collection not in accessible_collections:
            selected_collection = next((name for name in accessible_collections if name), selected_collection)
        if not selected_collection and accessible_collections:
            selected_collection = accessible_collections[0]

        plan = payload.plan or {"operation": "find"}
        docs = list(payload.docs or [])
        total = int(payload.total or len(docs) or 0)
        reason = "User marked the result as incorrect; generate a schema-grounded rewrite using the same subject and fields."

        yield_payload = None
        try:
            if payload.reasoning_model_enabled:
                model, tokenizer = await _run_llm_task(_load_reasoning_model, payload)
            else:
                model, tokenizer = await _run_llm_task(_load_query_model, payload)
            t_rewrite = time.perf_counter()
            yield_payload = await _build_feedback_rewrite_payload_async(
                model,
                tokenizer,
                payload.prompt,
                selected_user,
                selected_collection,
                payload.table_choice,
                plan,
                docs[:12],
                total,
                table_metadata,
                accessible_collections,
                reason=reason,
            )
            stage_ms["rewrite"] = _ms_since(t_rewrite)
        except Exception as exc:
            logger.exception("Failed to build feedback rewrite payload")
            fallback_message, fallback_suggestions = _fallback_clarification_suggestions(
                payload.prompt,
                selected_collection,
                table_metadata,
                docs=docs,
                total=total,
            )
            yield_payload = QueryResponse(
                response=_personalize_response(fallback_message, selected_user),
                follow_ups=fallback_suggestions[:3],
                needs_clarification=bool(LLM_ENABLE_CLARIFICATION),
                table_choice=payload.table_choice or {"collection": selected_collection or None, "reason": reason},
                plan=plan,
                docs=docs,
                total=total,
                collection=selected_collection,
                table_columns=list(payload.table_columns or []),
                summary="",
            )
            stage_ms["rewrite_error"] = _ms_since(req_started)
            _write_perf_log(payload, "/query_feedback", stage_ms, "rewrite_error", extra={"detail": str(exc)})
            return yield_payload

        stage_ms["total"] = _ms_since(req_started)
        _write_perf_log(
            payload,
            "/query_feedback",
            stage_ms,
            "ok",
            extra={"rows": len(docs), "total_rows": total, "collection": selected_collection},
        )
        return yield_payload
    except Exception as exc:
        logger.exception("Unhandled exception in /query_feedback")
        stage_ms["total"] = _ms_since(req_started)
        _write_perf_log(payload, "/query_feedback", stage_ms, "error", extra={"detail": str(exc)})
        raise


@app.get("/train")
async def train_status():
    return get_training_info()


@app.post("/train")
async def trigger_train():
    result = await trigger_training()
    return result


@app.post("/query_stream")
async def query_stream(payload: QueryRequest):
    async def event_generator():
        stage_ms = {}
        table_choice = None
        req_started = time.perf_counter()
        validation_enabled = bool(getattr(payload, "validation_enabled", True))
        direct_llm_mode = not validation_enabled
        try:
            with traced_span(
                "erp.query_stream.request",
                {
                    "db.name": payload.db_name,
                    "user.id": payload.user_id,
                    "runtime": payload.model_runtime,
                    "reasoning_runtime": payload.reasoning_model_runtime,
                    "reasoning_enabled": bool(payload.reasoning_model_enabled),
                    "validation_enabled": validation_enabled,
                    "compute_mode": payload.compute_mode,
                    "prompt.length": len(str(payload.prompt or "")),
                },
            ):
                t_bootstrap = time.perf_counter()
                yield _sse("status", {"stage": "planning_started", "message": "Preparing query pipeline"})
                await ping_mongo()
            collections = await list_collections(payload.db_name)
            table_metadata = await load_table_metadata(payload.db_name)
            users = await load_rbac_users(payload.db_name, force_refresh=False)
            users = users or DEFAULT_USERS
            selected_user = next((user for user in users if user.get("user_id") == payload.user_id), users[0])
            accessible_collections = allowed_collections_for_user(collections, selected_user)
            selected_user_context = _selected_user_context(selected_user, accessible_collections=accessible_collections)
            runtime_policy = await _load_and_apply_runtime_policy(payload.db_name)
            if not accessible_collections:
                raise HTTPException(status_code=403, detail="Selected user has no table access.")
            requested_collections = _infer_requested_collections(payload.prompt, collections, table_metadata)
            denied_targets = [name for name in requested_collections if name not in accessible_collections]
            allowed_targets = [name for name in requested_collections if name in accessible_collections]
            if denied_targets and not allowed_targets:
                response = _personalize_response(_permission_denied_message(denied_targets, table_metadata), selected_user)
                stage_ms["bootstrap"] = _ms_since(t_bootstrap)
                stage_ms["total"] = _ms_since(req_started)
                _write_perf_log(payload, "/query_stream", stage_ms, "rbac_denied")
                final_payload = QueryResponse(
                    response=response,
                    follow_ups=[],
                    table_choice={"collection": None, "reason": "RBAC denied for requested collection."},
                    plan={"operation": "none"},
                    docs=[],
                    total=0,
                    collection="",
                    summary="",
                ).model_dump()
                yield _sse("done", final_payload)
                return
            stage_ms["bootstrap"] = _ms_since(t_bootstrap)
            ai_template_schemas = await load_ai_template_schemas(payload.db_name, accessible_collections)
            schema_index = build_schema_index(
                table_metadata,
                accessible_collections,
                ai_template_schemas=ai_template_schemas,
            )
            vector_key = payload.db_name
            if vector_key not in _VECTOR_SCHEMA_WARMED:
                distinct_hints = await _collect_distinct_value_hints(payload.db_name, schema_index)
                cache_ready = warm_reverse_lookup_cache(payload.db_name, schema_index, field_value_hints=distinct_hints)
                vector_ready = upsert_schema_vectors(payload.db_name, schema_index, field_value_hints=distinct_hints)
                if cache_ready or vector_ready:
                    _VECTOR_SCHEMA_WARMED.add(vector_key)
                    add_span_event(
                        "vector_schema_warm",
                        {"db": payload.db_name, "cache_ready": bool(cache_ready), "vector_ready": bool(vector_ready)},
                    )

            effective_chat_context = []
            last_collection = None
            if LLM_USE_CHAT_HISTORY_HINTS:
                conversation_state = await load_conversation_state(
                    payload.db_name,
                    payload.user_id,
                    payload.conversation_id,
                )
                cached_context = conversation_state.get("messages") or []
                last_collection = conversation_state.get("last_collection")
                effective_chat_context = _recent_chat_context(list(cached_context) + list(payload.chat_context or []))

            yield _sse("status", {"stage": "exact_lookup", "message": "Checking exact value cache"})
            t_exact = time.perf_counter()
            exact_candidates, exact_matched_values = retrieve_exact_field_candidates_with_values(
                payload.db_name,
                payload.prompt,
                allowed_collections=accessible_collections,
            )
            stage_ms["exact_lookup"] = _ms_since(t_exact)
            use_vector_fallback = not exact_candidates
            if use_vector_fallback:
                yield _sse("status", {"stage": "vector_retrieval", "message": "Running vector reverse lookup"})
            t_vector = time.perf_counter()
            retrieval_query = _build_user_scoped_retrieval_query(payload.prompt, selected_user_context)
            vector_candidates = (
                retrieve_field_candidates(
                    payload.db_name,
                    retrieval_query,
                    allowed_collections=accessible_collections,
                )
                if use_vector_fallback
                else {}
            )
            stage_ms["vector_retrieval"] = _ms_since(t_vector) if use_vector_fallback else 0.0
            hybrid_field_candidates, field_source_tracking = _build_hybrid_field_candidates(
                payload.prompt,
                schema_index,
                exact_candidates,
                vector_candidates,
            )
            route_profile = _query_route_profile(
                payload.prompt,
                accessible_collections,
                table_metadata,
                schema_index,
                exact_candidates,
                vector_candidates,
                last_collection=last_collection,
            )
            use_single_pass_route = direct_llm_mode or bool(LLM_SINGLE_PASS_QUERY) or bool(route_profile.get("use_single_pass"))
            effective_prompt = route_profile.get("normalized_prompt") or _light_normalize_prompt(payload.prompt)

            reasoning_enabled = bool(payload.reasoning_model_enabled)
            if reasoning_enabled:
                query_spec = _query_model_spec(payload)
                reasoning_spec = _reasoning_model_spec(payload)
                same_model_spec = query_spec == reasoning_spec
                yield _sse("status", {"stage": "llm_loading", "message": "Loading reasoning model"})
                t_reasoning_model_load = time.perf_counter()
                reasoning_model, reasoning_tokenizer = await _run_llm_task(_load_reasoning_model, payload)
                stage_ms["reasoning_model_load"] = _ms_since(t_reasoning_model_load)
                if use_single_pass_route or same_model_spec:
                    model, tokenizer = reasoning_model, reasoning_tokenizer
                    stage_ms["model_load"] = 0.0
                else:
                    yield _sse("status", {"stage": "llm_loading", "message": "Loading coding model"})
                    t_model_load = time.perf_counter()
                    model, tokenizer = await _run_llm_task(_load_query_model, payload)
                    stage_ms["model_load"] = _ms_since(t_model_load)
            else:
                stage_ms["reasoning_model_load"] = 0.0
                yield _sse("status", {"stage": "llm_loading", "message": "Loading main model"})
                t_model_load = time.perf_counter()
                model, tokenizer = await _run_llm_task(_load_query_model, payload)
                stage_ms["model_load"] = _ms_since(t_model_load)
                reasoning_model, reasoning_tokenizer = model, tokenizer
            if use_single_pass_route:
                yield _sse("status", {"stage": "single_pass_routing", "message": "Using fast single-pass routing"})
                healed_prompt = {
                    "normalized_query": effective_prompt,
                    "intent_type": "direct",
                    "entity_terms": [],
                    "field_terms": [],
                    "value_terms": [],
                    "confidence": 0.9,
                    "needs_clarification": False,
                    "hints": {},
                }
                stage_ms["prompt_rewrite"] = 0.0
                stage_ms["scope_gate"] = 0.0
            else:
                # Pipeline step 1-2: Prompt -> LLM rewrite -> rewrite validator
                yield _sse("status", {"stage": "prompt_rewriting", "message": "Rewriting prompt"})
                t_prompt_rewrite = time.perf_counter()
                healed_prompt = await _run_llm_task(_rewrite_prompt_with_llm_details, model, payload.prompt)
                effective_prompt = _build_accuracy_planner_prompt(healed_prompt) or str(payload.prompt or "")
                stage_ms["prompt_rewrite"] = _ms_since(t_prompt_rewrite)
                t_scope_gate = time.perf_counter()
                scope_gate = await _run_llm_task(
                    _llm_scope_gate,
                    reasoning_model,
                    reasoning_tokenizer,
                    effective_prompt,
                    accessible_collections,
                    table_metadata,
                    chat_context=effective_chat_context,
                    db_name=payload.db_name,
                )
                stage_ms["scope_gate"] = _ms_since(t_scope_gate)
                if not scope_gate.get("allow", True):
                    response = _personalize_response(scope_gate.get("message") or _non_data_query_response(), selected_user)
                    if LLM_USE_CHAT_HISTORY_HINTS:
                        await save_conversation_state(
                            payload.db_name,
                            payload.user_id,
                            payload.conversation_id,
                            effective_chat_context
                            + [{"role": "user", "content": payload.prompt}, {"role": "assistant", "content": response}],
                            last_collection=last_collection,
                        )
                    final_payload = QueryResponse(
                        response=response,
                        follow_ups=[
                            "What is the current organization address?",
                            "List all active users.",
                            "Show branch-wise user count.",
                        ],
                        table_choice={"collection": None, "reason": "AI scope gate rejected prompt."},
                        plan={"operation": "none"},
                        docs=[],
                        total=0,
                        collection="",
                        summary="",
                    ).model_dump()
                    stage_ms["total"] = _ms_since(req_started)
                    _write_perf_log(payload, "/query_stream", stage_ms, "scope_rejected")
                    yield _sse("done", final_payload)
                    return
                yield _sse("status", {"stage": "prompt_rewritten", "message": "Prompt rewritten by LLM"})
                yield _sse("status", {"stage": "prompt_validated", "message": "Rewritten prompt validated"})

            t_plan_generation = time.perf_counter()
            if use_single_pass_route:
                raw_plan = await _run_llm_task(
                    generate_single_pass_query_plan,
                    model,
                    tokenizer,
                    effective_prompt,
                    payload.db_name,
                    accessible_collections,
                    table_metadata,
                    chat_context=effective_chat_context,
                    preferred_collection=None,
                    exact_field_candidates=exact_candidates,
                    vector_field_candidates=vector_candidates,
                    user_context=selected_user_context,
                )
                if not isinstance(raw_plan, dict):
                    raw_plan = {}
                selected_collection = str(raw_plan.get("collection") or "").strip()
                if selected_collection not in set(accessible_collections or []):
                    selected_collection = ""
                if raw_plan.get("needs_clarification") or not selected_collection:
                    if LLM_ENABLE_CLARIFICATION:
                        stage_ms["plan_generation"] = _ms_since(t_plan_generation)
                        stage_ms["total"] = _ms_since(req_started)
                        _write_perf_log(payload, "/query_stream", stage_ms, "needs_clarification")
                        clarification = await _build_clarification_payload_async(
                            reasoning_model,
                            reasoning_tokenizer,
                            effective_prompt,
                            selected_user,
                            "",
                            table_choice,
                            {"operation": "none"},
                            [],
                            0,
                            table_metadata,
                            accessible_collections,
                            reason=str((raw_plan or {}).get("message") or "Ambiguous request."),
                        )
                        if LLM_USE_CHAT_HISTORY_HINTS:
                            await save_conversation_state(
                                payload.db_name,
                                payload.user_id,
                                payload.conversation_id,
                                effective_chat_context
                                + [{"role": "user", "content": payload.prompt}, {"role": "assistant", "content": clarification.response}],
                                last_collection=last_collection,
                            )
                        yield _sse("done", clarification.model_dump())
                        return
                    raise HTTPException(status_code=422, detail=str((raw_plan or {}).get("message") or "I could not identify the target ERP table."))
                table_choice = {"collection": selected_collection, "reason": "Single-pass LLM routing."}
                table_choice["collection"] = selected_collection
                template_schema = await get_template_schema(payload.db_name, selected_collection)
                sample_docs = await sample_collection(payload.db_name, selected_collection)
                schema = infer_schema_from_docs(sample_docs)
                runtime_field_names = list((schema or {}).keys())
                schema_index = _augment_schema_index_with_runtime_fields(schema_index, selected_collection, schema)
            else:
                table_choice = await _run_llm_task(
                    generate_table_choice,
                    reasoning_model,
                    reasoning_tokenizer,
                    effective_prompt,
                    payload.db_name,
                    accessible_collections,
                    table_metadata,
                    chat_context=effective_chat_context,
                    user_context=selected_user_context,
                    exact_candidates=exact_candidates,
                )
                invalid_collection = str((table_choice or {}).get("collection") or "").strip() not in set(accessible_collections or [])
                if (table_choice or {}).get("needs_clarification") or invalid_collection:
                    if LLM_ENABLE_CLARIFICATION:
                        stage_ms["plan_generation"] = _ms_since(t_plan_generation)
                        stage_ms["total"] = _ms_since(req_started)
                        _write_perf_log(payload, "/query_stream", stage_ms, "needs_clarification")
                        clarification = await _build_clarification_payload_async(
                            reasoning_model,
                            reasoning_tokenizer,
                            effective_prompt,
                            selected_user,
                            "",
                            table_choice,
                            {"operation": "none"},
                            [],
                            0,
                            table_metadata,
                            accessible_collections,
                            reason=str((table_choice or {}).get("message") or "Ambiguous request."),
                        )
                        if LLM_USE_CHAT_HISTORY_HINTS:
                            await save_conversation_state(
                                payload.db_name,
                                payload.user_id,
                                payload.conversation_id,
                                effective_chat_context
                                + [{"role": "user", "content": payload.prompt}, {"role": "assistant", "content": clarification.response}],
                                last_collection=last_collection,
                            )
                        yield _sse("done", clarification.model_dump())
                        return
                    raise HTTPException(status_code=422, detail=str((table_choice or {}).get("message") or "I could not identify the target ERP table."))
                selected_collection = str((table_choice or {}).get("collection") or "").strip()
                table_choice["collection"] = selected_collection
                template_schema = await get_template_schema(payload.db_name, selected_collection)
                sample_docs = await sample_collection(payload.db_name, selected_collection)
                schema = infer_schema_from_docs(sample_docs)
                runtime_field_names = list((schema or {}).keys())
                schema_index = _augment_schema_index_with_runtime_fields(schema_index, selected_collection, schema)
                raw_plan = await _run_llm_task(
                    generate_query_plan,
                    model,
                    tokenizer,
                    effective_prompt,
                    payload.db_name,
                    selected_collection,
                    schema,
                    sample_docs,
                    table_metadata,
                    template_schema=template_schema,
                    chat_context=effective_chat_context,
                    exact_field_candidates=exact_candidates,
                    vector_field_candidates=vector_candidates,
                    user_context=selected_user_context,
                )
            stage_ms["plan_generation"] = _ms_since(t_plan_generation)
            if validation_enabled:
                yield _sse("status", {"stage": "backend_validation", "message": "Validating query plan"})
            t_plan_validate = time.perf_counter()
            from erp_backend.services.intent import _strict_schema_validate_plan
            raw_plan = _ensure_plan_collection(raw_plan, selected_collection)
            _strict_schema_validate_plan(raw_plan, schema_index)
            plan = validate_query_plan(
                raw_plan,
                accessible_collections,
            )
            stage_ms["plan_validate"] = _ms_since(t_plan_validate)
            yield _sse("status", {"stage": "query_generated", "plan": plan})
            if validation_enabled and plan.get("needs_clarification"):
                if LLM_ENABLE_CLARIFICATION:
                    clarification = await _build_clarification_payload_async(
                        model,
                        tokenizer,
                        effective_prompt,
                        selected_user,
                        selected_collection,
                        table_choice,
                        plan,
                        [],
                        0,
                        table_metadata,
                        accessible_collections,
                        reason=str(plan.get("message") or "Plan requires clarification."),
                    )
                    if LLM_USE_CHAT_HISTORY_HINTS:
                        await save_conversation_state(
                            payload.db_name,
                            payload.user_id,
                            payload.conversation_id,
                            effective_chat_context
                            + [{"role": "user", "content": payload.prompt}, {"role": "assistant", "content": clarification.response}],
                            last_collection=selected_collection,
                        )
                    stage_ms["total"] = _ms_since(req_started)
                    _write_perf_log(payload, "/query_stream", stage_ms, "needs_clarification")
                    yield _sse("done", clarification.model_dump())
                    return
                raise HTTPException(status_code=422, detail=str(plan.get("message") or "Plan requires clarification."))
            yield _sse("status", {"stage": "query_execute_core", "message": "Executing Mongo query"})
            t_execute = time.perf_counter()
            docs, total = await execute_plan(payload.db_name, plan)
            stage_ms["execute"] = _ms_since(t_execute)
            yield _sse("status", {"stage": "lookup_resolving", "message": "Resolving lookup values"})
            t_lookup_resolve = time.perf_counter()
            docs = await _resolve_lookup_ids_to_names(payload.db_name, selected_collection, docs, table_metadata)
            stage_ms["lookup_resolve"] = _ms_since(t_lookup_resolve)
            yield _sse("status", {"stage": "rows_fetched", "rows": len(docs), "total": total})
            response_tokens = []
            parser = _ThinkTagStreamParser()
            current_stream_state = "ANSWER"
            yield _sse("status", {"stage": "narrating", "message": "Streaming model reasoning and answer"})
            yield _sse("llm_state", {"state": current_stream_state})
            try:
                t_narrate = time.perf_counter()
                async for raw_chunk in _iterate_llm_task(
                    analyze_query_result_stream,
                    model,
                    tokenizer,
                    effective_prompt,
                    selected_collection,
                    table_metadata,
                    plan,
                    docs,
                    total,
                ):
                    for kind, state, text in parser.feed(raw_chunk):
                        if kind == "state":
                            current_stream_state = state
                            yield _sse("llm_state", {"state": state})
                            continue
                        token_text = str(text or "")
                        if not token_text:
                            continue
                        if state == "ANSWER":
                            response_tokens.append(token_text)
                        yield _sse("llm_token", {"state": state, "token": token_text})
                for kind, state, text in parser.flush():
                    token_text = str(text or "")
                    if kind == "token" and token_text:
                        if state == "ANSWER":
                            response_tokens.append(token_text)
                        yield _sse("llm_token", {"state": state, "token": token_text})
                response = "".join(response_tokens).strip()
                response = response.replace("<think>", "").replace("</think>", "").strip()
                if not response:
                    response = await _run_llm_task(
                        analyze_query_result,
                        model,
                        tokenizer,
                        effective_prompt,
                        selected_collection,
                        table_metadata,
                        plan,
                        docs,
                        total,
                    )
                    response = response.replace("<think>", "").replace("</think>", "").strip()
                stage_ms["narrate_stream"] = _ms_since(t_narrate)
            except Exception:
                response = build_response_summary(selected_collection, plan, docs, total, table_metadata)
            response = _personalize_response(response, selected_user)

            try:
                summary = await _run_llm_task(
                    summarize_query_result,
                    model,
                    tokenizer,
                    effective_prompt,
                    selected_collection,
                    table_metadata,
                    plan,
                    docs,
                    total,
                )
            except Exception:
                summary = ""

            follow_ups = []
            try:
                follow_ups = await _run_llm_task(
                    generate_follow_up_suggestions,
                    model,
                    tokenizer,
                    effective_prompt,
                    selected_collection,
                    table_metadata,
                    plan,
                    docs,
                    total,
                )
            except Exception:
                follow_ups = []
            table_columns = _response_table_columns(selected_collection, plan, docs, schema_index)

            if LLM_USE_CHAT_HISTORY_HINTS:
                await save_conversation_state(
                    payload.db_name,
                    payload.user_id,
                    payload.conversation_id,
                    effective_chat_context
                    + [{"role": "user", "content": payload.prompt}, {"role": "assistant", "content": response}],
                    last_collection=selected_collection,
                )

            response_artifacts = _response_artifacts(selected_collection, table_metadata, plan, docs, total, model=model, tokenizer=tokenizer, user_input=effective_prompt)
            final_payload = QueryResponse(
                response=response,
                follow_ups=follow_ups,
                insights=response_artifacts["insights"],
                chart_config=response_artifacts["chart_config"],
                db_performance=response_artifacts["db_performance"],
                table_choice=table_choice,
                plan=plan,
                docs=docs,
                total=total,
                collection=selected_collection,
                table_columns=table_columns,
                summary=summary,
            ).model_dump()
            set_span_attribute("rows.returned", len(docs))
            set_span_attribute("rows.total", int(total or 0))
            stage_ms["total"] = _ms_since(req_started)
            _write_perf_log(
                payload,
                "/query_stream",
                stage_ms,
                "ok",
                extra={"rows": len(docs), "total_rows": int(total or 0), "collection": selected_collection},
            )
            yield _sse("done", final_payload)
        except HTTPException as exc:
            stage_ms["total"] = _ms_since(req_started)
            _write_perf_log(payload, "/query_stream", stage_ms, "http_error", extra={"detail": str(exc.detail)})
            yield _sse("error", {"status_code": exc.status_code, "detail": str(exc.detail)})
        except Exception as exc:
            logger.exception("Unhandled exception in /query_stream")
            stage_ms["total"] = _ms_since(req_started)
            _write_perf_log(payload, "/query_stream", stage_ms, "error", extra={"detail": str(exc)})
            yield _sse("error", {"status_code": 500, "detail": str(exc)})

    stream = StreamingResponse(event_generator(), media_type="text/event-stream")
    return _queue_streaming_response(payload, stream)


