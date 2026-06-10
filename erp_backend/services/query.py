import json
import logging
import re
from collections import Counter
from datetime import datetime

from pymongo import ASCENDING, DESCENDING


logger = logging.getLogger(__name__)

from erp_backend.core.config import (
    BLOCKED_OPERATORS,
    BLOCKED_STAGES,
    CHAT_CONTEXT_CHARS,
    CHAT_CONTEXT_LIMIT,
    COUNT_TOTAL_EXACT,
    LLM_COLLECTION_CANDIDATES,
    LLM_CTX_SIZE,
    LLM_RETRY_MAX_NEW_TOKENS,
    MAX_NEW_TOKENS,
    MAX_RESULT_ROWS,
    TABLE_ROUTER_FIELDS_PER_TABLE,
    TABLE_ROUTER_MAX_NEW_TOKENS,
    TABLE_ROUTER_TERMS_PER_TABLE,
)
from erp_backend.core.security import is_hidden_field, sanitize_doc_for_display
from erp_backend.core.utils import normalize_lookup_text, to_jsonable
from erp_backend.llm.runtime import is_cuda_inference_error, recover_from_cuda_error
from erp_backend.services.query_prompts import (
    TABLE_ROUTER_PROMPT, _ROUTER_STOP_TERMS, QUERY_PLANNER_PROMPT,
    SINGLE_PASS_QUERY_PROMPT, RESULT_VERIFIER_PROMPT, RESULT_SUMMARY_PROMPT,
    RESULT_ANALYSIS_PROMPT,
    FOLLOW_UP_SUGGESTIONS_PROMPT, CLARIFICATION_SUGGESTIONS_PROMPT,
    SIDEBAR_SUGGESTIONS_PROMPT, QUERY_SCOPE_PROMPT, RESULT_CONSTRAINTS_PROMPT,
    EMPTY_RESULT_REPAIR_PROMPT, MISMATCH_REPAIR_PROMPT,
)
from erp_backend.services.query_validate import (
    _has_blocked_operator, _has_hidden_field_reference,
    _normalize_field_path_ref, _normalize_operator_key,
    _coerce_aggregate_stage_key, _normalize_field_key,
    _normalize_plan_value, _normalize_aggregate_pipeline_stages,
    _strip_hidden_field_references, _clean_projection, _clean_sort,
    _validate_collection, _validate_find_plan, _validate_aggregate_plan,
    _sanitize_group_stage, _sanitize_count_stage,
    _infer_sort_by_count_expr, _sanitize_sort_by_count_expr,
    validate_query_plan, execute_plan,
)
from erp_backend.storage.mongo import collection_is_allowed, estimated_count, mongo_client



def recent_chat_context(messages, limit=CHAT_CONTEXT_LIMIT):
    context = []
    for message in messages[-limit:]:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            context.append({"role": role, "content": content[:CHAT_CONTEXT_CHARS]})
    return context


def _parse_json(raw_text):
    text = str(raw_text or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    def _salvage_partial_payload(source_text):
        if not source_text:
            return None
        payload = {}
        bool_keys = {"allow", "needs_clarification"}
        string_keys = {
            "status",
            "message",
            "collection",
            "operation",
            "reason",
            "normalized_query",
        }
        for key in bool_keys:
            match = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false)', source_text, flags=re.IGNORECASE)
            if match:
                payload[key] = match.group(1).lower() == "true"
        for key in string_keys:
            # Closed string value.
            match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"', source_text, flags=re.DOTALL)
            if match:
                payload[key] = bytes(match.group(1), "utf-8").decode("unicode_escape", errors="ignore").strip()
                continue
            # Truncated string value (missing closing quote).
            match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^\r\n}}]*)', source_text, flags=re.DOTALL)
            if match:
                payload[key] = str(match.group(1)).strip(" \t\r\n,")
        return payload or None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        # Try to decode first valid JSON object/array from noisy text.
        for index, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                return value
            except json.JSONDecodeError:
                continue
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text, flags=re.DOTALL)
        if not match:
            salvaged = _salvage_partial_payload(text)
            if salvaged is not None:
                return salvaged
            raise ValueError(f"Model did not return valid JSON: {raw_text}") from None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            salvaged = _salvage_partial_payload(match.group(0))
            if salvaged is not None:
                return salvaged
            raise ValueError(f"Model did not return valid JSON: {raw_text}") from exc


def _compact_messages(messages, char_budget=None):
    if char_budget is None:
        input_token_budget = max(1024, LLM_CTX_SIZE - MAX_NEW_TOKENS - 512)
        char_budget = max(12000, int(input_token_budget * 3.5))

    compact = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        compact.append({"role": role, "content": content})

    total_chars = sum(len(m["content"]) for m in compact)
    if total_chars <= char_budget:
        return compact

    original_total = total_chars
    logger.info("Context budget exceeded: %d chars > %d budget, truncating", total_chars, char_budget)

    def truncate_json_content(content_str, budget):
        try:
            data = json.loads(content_str)
            if not isinstance(data, dict):
                return content_str[:budget]
            
            # Identify fields to trim
            list_keys = ["allowed_tables", "allowed_collections", "collection_schemas", "chat_history", "sample_documents", "schema_fields"]
            for key in list_keys:
                if key in data and isinstance(data[key], list):
                    while len(data[key]) > 1 and len(json.dumps(data, ensure_ascii=False)) > budget:
                        data[key].pop()
                elif key in data and isinstance(data[key], dict):
                    keys = list(data[key].keys())
                    while len(keys) > 1 and len(json.dumps(data, ensure_ascii=False)) > budget:
                        del data[key][keys.pop()]
            
            if len(json.dumps(data, ensure_ascii=False)) > budget:
                if "allowed_tables" in data and isinstance(data["allowed_tables"], list):
                    for tbl in data["allowed_tables"]:
                        if isinstance(tbl, dict) and "fields" in tbl and isinstance(tbl["fields"], list):
                            while len(tbl["fields"]) > 1 and len(json.dumps(data, ensure_ascii=False)) > budget:
                                tbl["fields"].pop()
            
            serialized = json.dumps(data, ensure_ascii=False)
            if len(serialized) <= budget:
                return serialized
            return serialized[:budget]
        except Exception:
            return content_str[:budget]

    # Phase 1: truncate oldest non-system messages first (preserve recent context).
    for msg in compact:
        if msg["role"] != "system" and total_chars > char_budget:
            overflow = total_chars - char_budget
            keep = max(1500, len(msg["content"]) - overflow - 200)
            
            stripped = msg["content"].strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                msg["content"] = truncate_json_content(msg["content"], keep)
            else:
                msg["content"] = msg["content"][:keep]
            total_chars = sum(len(m["content"]) for m in compact)

    # Phase 2: across-the-board trim if still over.
    if total_chars > char_budget:
        ratio = char_budget / max(total_chars, 1)
        for msg in compact:
            if msg["role"] != "system":
                keep = max(800, int(len(msg["content"]) * ratio))
                stripped = msg["content"].strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    msg["content"] = truncate_json_content(msg["content"], keep)
                else:
                    msg["content"] = msg["content"][:keep]
        total_chars = sum(len(m["content"]) for m in compact)

    # Last resort: hard cap all non-system blocks.
    if total_chars > char_budget:
        for msg in compact:
            if msg["role"] != "system":
                msg["content"] = msg["content"][:700]

    logger.warning(
        "Context truncated: %d chars -> %d chars (budget=%d, lost %d chars)",
        original_total, sum(len(m["content"]) for m in compact), char_budget,
        original_total - sum(len(m["content"]) for m in compact),
    )
    return compact


def _context_char_budget(max_output_tokens, result_docs=None, total_rows=None, reserve_tokens=512):
    try:
        output_tokens = max(64, int(max_output_tokens or MAX_NEW_TOKENS))
    except Exception:
        output_tokens = MAX_NEW_TOKENS
    input_token_budget = max(1024, LLM_CTX_SIZE - output_tokens - reserve_tokens)
    base_budget = max(12000, int(input_token_budget * 3.5))
    rows = list(result_docs or [])
    total_rows = max(int(total_rows or 0), len(rows))
    if not rows and total_rows <= 0:
        return base_budget
    serialized_chars = 0
    sample_count = 0
    for row in rows[:10]:
        try:
            serialized_chars += len(json.dumps(row, ensure_ascii=False, default=str))
            sample_count += 1
        except Exception:
            continue
    avg_row_chars = (serialized_chars / sample_count) if sample_count else 0
    density_boost = min(0.35, avg_row_chars / 12000.0)
    volume_boost = min(0.35, total_rows / 60.0)
    scaled_budget = int(base_budget * (0.75 + density_boost + volume_boost))
    hard_cap = max(base_budget, int(input_token_budget * 4.2))
    return max(8000, min(hard_cap, scaled_budget))


def _clip_preview_value(value, max_string_chars=220, max_list_items=6, max_dict_items=24, depth=0):
    if depth >= 4:
        return str(value)[:max_string_chars]
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value).strip()
        return text[:max_string_chars]
    if isinstance(value, list):
        items = [
            _clip_preview_value(item, max_string_chars=max_string_chars, max_list_items=max_list_items, max_dict_items=max_dict_items, depth=depth + 1)
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            items.append(f"...(+{len(value) - max_list_items} more)")
        return items
    if isinstance(value, dict):
        clipped = {}
        for index, (key, child) in enumerate(value.items()):
            if index >= max_dict_items:
                clipped["..."] = f"+{len(value) - max_dict_items} more fields"
                break
            clipped[str(key)] = _clip_preview_value(
                child,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
                depth=depth + 1,
            )
        return clipped
    return value


def _build_rows_preview(docs, total=0, max_output_tokens=320, min_rows=4, max_rows=20):
    rows = list(docs or [])
    if not rows:
        return []
    total_rows = max(int(total or 0), len(rows))
    result_budget = _context_char_budget(max_output_tokens, result_docs=rows, total_rows=total_rows)
    preview_budget = max(1800, min(32000, int(result_budget * 0.55)))
    target_rows = min(
        max_rows,
        max(min_rows, total_rows if total_rows <= max_rows else min_rows + min(max_rows - min_rows, total_rows // 4)),
    )
    if total_rows <= 3:
        target_rows = min(max_rows, total_rows)
    if target_rows <= 6:
        max_string_chars = 320
    elif target_rows <= 12:
        max_string_chars = 220
    else:
        max_string_chars = 160
    preview = []
    used_chars = 0
    for row in rows[:max_rows]:
        clipped = _clip_preview_value(row, max_string_chars=max_string_chars)
        try:
            row_chars = len(json.dumps(clipped, ensure_ascii=False, default=str))
        except Exception:
            row_chars = len(str(clipped))
        if preview and used_chars + row_chars > preview_budget and len(preview) >= min_rows:
            break
        preview.append(clipped)
        used_chars += row_chars
        if len(preview) >= target_rows and used_chars >= preview_budget * 0.6:
            break
    return preview[:max_rows]


def _generate_json_from_messages(model, tokenizer, messages, max_new_tokens, char_budget=None):
    messages = _compact_messages(messages, char_budget=char_budget)
    attempts = [max_new_tokens, min(max_new_tokens, LLM_RETRY_MAX_NEW_TOKENS)]
    last_error = None
    for index, output_tokens in enumerate(attempts):
        try:
            response = model.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=output_tokens,
            )
            text = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            # Strip reasoning tags often emitted by reasoning-tuned models.
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
            if not text:
                raise ValueError("Model returned empty content")
            return _parse_json(text)
        except ValueError as exc:
            # One strict JSON repair attempt before failing.
            try:
                repair_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Return only valid JSON for the required output schema. "
                            "No explanations, no markdown, no <think> tags."
                        ),
                    }
                ]
                repair_response = model.create_chat_completion(
                    messages=_compact_messages(repair_messages, char_budget=char_budget),
                    temperature=0.0,
                    max_tokens=output_tokens,
                )
                repair_text = (
                    repair_response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                repair_text = re.sub(r"<think>[\s\S]*?</think>", "", repair_text, flags=re.IGNORECASE).strip()
                repair_text = re.sub(r"</?think>", "", repair_text, flags=re.IGNORECASE).strip()
                return _parse_json(repair_text)
            except Exception:
                if index == len(attempts) - 1:
                    raise exc
        except Exception as exc:
            if not is_cuda_inference_error(exc) or index == len(attempts) - 1:
                raise
            last_error = exc
            recover_from_cuda_error()
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM generation failed")


def _coerce_plan_object(raw_plan, fallback_collection=None):
    if isinstance(raw_plan, dict):
        return raw_plan
    if isinstance(raw_plan, list):
        # Some local models return only the aggregate pipeline array.
        return {
            "operation": "aggregate",
            "collection": str(fallback_collection or "").strip(),
            "pipeline": raw_plan,
        }
    return raw_plan


def _normalize_generated_plan(raw_plan, fallback_collection=None):
    plan = _coerce_plan_object(raw_plan, fallback_collection=fallback_collection)
    if isinstance(plan, dict):
        return plan
    collection = str(fallback_collection or "").strip()
    return {
        "collection": collection or None,
        "needs_clarification": True,
        "message": "I could not generate a valid query plan. Please rephrase the request.",
    }


def _normalize_text(value):
    return normalize_lookup_text(value)


def _singularize(term):
    if term.endswith("ies") and len(term) > 3:
        return term[:-3] + "y"
    if term.endswith("es") and len(term) > 3:
        return term[:-2]
    if term.endswith("s") and len(term) > 2:
        return term[:-1]
    return term


# Field type tokens that should never be treated as collection aliases.
_FIELD_TYPE_TOKENS = {
    "text", "textarea", "number", "checkbox", "date", "datetime",
    "email", "password", "select", "multi select", "look up",
    "multi lookup", "formula", "sequence number", "single file",
    "ip", "phone", "added by", "added time", "multi_select",
    "look_up", "multi_lookup", "sequence_number", "single_file",
    "added_by", "added_time",
}


def _is_camel_case_field_id(term):
    """Return True if term looks like a raw camelCase field identifier."""
    raw = str(term or "").strip()
    if not raw or "_" in raw or " " in raw:
        return False
    # camelCase: has at least one lowercase followed by uppercase transition
    return bool(re.search(r'[a-z][A-Z]', raw))


def _build_collection_aliases(collection_name, metadata):
    primary_aliases = set()
    primary_aliases.add(_normalize_text(collection_name))
    primary_aliases.add(_normalize_text(collection_name).replace("_", " "))
    primary_aliases.add(_normalize_text(metadata.get("template_name", "")))
    
    secondary_aliases = set()
    for term in metadata.get("business_terms") or []:
        normalized = _normalize_text(term)
        # Skip field type tokens
        if normalized in _FIELD_TYPE_TOKENS:
            continue
        # Skip router stop terms
        if normalized in _ROUTER_STOP_TERMS:
            continue
        # Skip raw camelCase field identifiers (e.g. branchTypeId, statusId)
        if _is_camel_case_field_id(term):
            continue
        # Skip very short tokens (likely abbreviations like "BR", "US")
        if len(normalized) <= 2:
            continue
        secondary_aliases.add(normalized)

    def _expand(aliases):
        expanded = set()
        for alias in aliases:
            if not alias:
                continue
            expanded.add(alias)
            expanded.add(_singularize(alias))
        return {
            item
            for item in expanded
            if item and len(item) > 2 and item not in _ROUTER_STOP_TERMS
            and item not in _FIELD_TYPE_TOKENS
        }
        
    return _expand(primary_aliases), _expand(secondary_aliases)


def _collection_specificity_penalty(user_input, collection_name, metadata):
    normalized_input = _normalize_text(user_input)
    if not normalized_input:
        return 0

    template_name = _normalize_text((metadata or {}).get("template_name") or "")
    if not template_name:
        return 0

    template_tokens = [token for token in template_name.split() if token and token not in _ROUTER_STOP_TERMS]
    if len(template_tokens) <= 1:
        return 0

    if re.search(rf"\b{re.escape(template_name)}\b", normalized_input):
        return 0

    prompt_words = set(normalized_input.split())
    matched_tokens = sum(1 for token in template_tokens if token in prompt_words)
    if matched_tokens <= 0:
        return 0

    return max(0, len(template_tokens) - matched_tokens) * 3


def _deterministic_table_choice(user_input, allowed_collections, table_metadata):
    normalized_input = _normalize_text(user_input)
    if not normalized_input:
        return None

    try:
        from erp_backend.services.intent import _infer_requested_collections as _api_infer_requested_collections
        inferred_collections = _api_infer_requested_collections(user_input, allowed_collections, table_metadata)
    except Exception:
        inferred_collections = []
    candidate_collections = inferred_collections or list(allowed_collections or [])

    best_collection = None
    best_score = -1
    tie = False
    for collection in candidate_collections:
        metadata = table_metadata.get(collection, {})
        score = _collection_match_score(user_input, collection, table_metadata)

        if score > best_score:
            best_collection = collection
            best_score = score
            tie = False
        elif score == best_score and score > 0:
            tie = True

    if best_collection and best_score >= 8 and not tie:
        return {
            "collection": best_collection,
            "reason": "Matched request keywords to template metadata.",
        }
    if best_collection and best_score >= 8 and tie:
        # Resolve ties by preferring the least-modified template name.
        tied_collections = [
            collection
            for collection in candidate_collections
            if _collection_match_score(user_input, collection, table_metadata) == best_score
        ]
        if tied_collections:
            best_collection = min(
                tied_collections,
                key=lambda collection: (
                    len([token for token in _normalize_text((table_metadata.get(collection) or {}).get("template_name") or collection).split() if token]),
                    len(_normalize_text((table_metadata.get(collection) or {}).get("template_name") or collection)),
                    len(collection),
                ),
            )
            return {
                "collection": best_collection,
                "reason": "Matched request keywords to the simplest compatible template metadata.",
            }
    return None


def _collection_match_score(user_input, collection_name, table_metadata):
    normalized_input = _normalize_text(user_input)
    if not normalized_input:
        return 0
    metadata = table_metadata.get(collection_name, {})
    primary_aliases, secondary_aliases = _build_collection_aliases(collection_name, metadata)

    # Track best score per input word to prevent alias-rich collections
    # from inflating their score by matching the same user word many times.
    word_best_scores = {}
    exact_match = False
    
    # Check primary aliases (highest weight)
    for alias in primary_aliases:
        if not alias:
            continue
        alias_root = _singularize(alias)
        if normalized_input == alias or normalized_input == alias_root:
            exact_match = True
            break
        # Check which input words this alias covers
        input_words = normalized_input.split()
        for word in input_words:
            if len(word) < 3:
                continue
            current_best = word_best_scores.get(word, 0)
            if word == alias or word == alias_root:
                word_best_scores[word] = max(current_best, 30)
            elif alias in word or word in alias:
                word_best_scores[word] = max(current_best, 15)
            elif alias_root in word or word in alias_root:
                word_best_scores[word] = max(current_best, 10)

    if exact_match:
        return 50

    # Check secondary aliases (lower weight)
    for alias in secondary_aliases:
        if not alias:
            continue
        alias_root = _singularize(alias)
        input_words = normalized_input.split()
        for word in input_words:
            if len(word) < 3:
                continue
            current_best = word_best_scores.get(word, 0)
            if word == alias or word == alias_root:
                word_best_scores[word] = max(current_best, 12)
            elif alias in word or word in alias:
                word_best_scores[word] = max(current_best, 8)
            elif alias_root in word or word in alias_root:
                word_best_scores[word] = max(current_best, 6)

    score = sum(word_best_scores.values())
    score -= _collection_specificity_penalty(user_input, collection_name, metadata)
    return max(0, score)


def _table_context_rows(user_input, allowed_collections, table_metadata):
    rows = []
    normalized_input = _normalize_text(user_input)
    prompt_terms = set(normalized_input.split()) if normalized_input else set()

    for name in allowed_collections:
        metadata = table_metadata.get(name, {})
        all_fields = metadata.get("fields") or []

        def field_score(field):
            score = 0
            field_name = _normalize_text(field.get("field") or "")
            display = _normalize_text(field.get("display") or "")
            aliases = [_normalize_text(a) for a in field.get("aliases") or []]
            for pt in prompt_terms:
                if len(pt) < 3:
                    continue
                if pt in field_name:
                    score += 3
                if pt in display:
                    score += 2
                for a in aliases:
                    if pt in a:
                        score += 1
            return score

        if prompt_terms:
            ranked_fields_with_index = sorted(enumerate(all_fields), key=lambda x: (-field_score(x[1]), x[0]))
            ranked_fields = [x[1] for x in ranked_fields_with_index]
        else:
            ranked_fields = all_fields

        raw_fields = [
                    {
                        "field": str(field.get("field") or "").strip(),
                        "display": str(field.get("display") or "").strip(),
                        "aliases": field.get("aliases") or [],
                        "type": str(field.get("type") or "").strip() or "TEXT",
                    }
                    for field in ranked_fields[:TABLE_ROUTER_FIELDS_PER_TABLE]
                ]
        rows.append(
            {
                "collection": name,
                "table": _singularize(_normalize_text(metadata.get("template_name") or name)),
                "terms": sorted(list(set(metadata.get("business_terms") or []))),
                "fields": _expand_lookup_companion_fields(raw_fields),
            }
        )
    return rows



def _expand_lookup_companion_fields(fields):
    expanded = []
    seen = set()
    for field in fields:
        field_name = str(field.get("fieldName") or field.get("field") or "")
        if not field_name:
            continue
        if field_name not in seen:
            seen.add(field_name)
            expanded.append(field)
        
        ftype = str(field.get("type") or "").upper().strip()
        if ftype in ("LOOK_UP", "MULTI_LOOKUP"):
            text_mode_field = f"{field_name}_textMode"
            code_field = f"{field_name}_"
            
            display_name = str(field.get("display") or field_name)
            orig_aliases = list(field.get("aliases") or [])
            
            if text_mode_field not in seen:
                seen.add(text_mode_field)
                expanded.append({
                    "field": text_mode_field,
                    "fieldName": text_mode_field,
                    "display": f"{display_name} (Text)",
                    "type": "TEXT",
                    "aliases": sorted(list(set([
                        text_mode_field,
                        f"{display_name} (Text)".lower(),
                        f"{display_name.lower()} text",
                    ] + [f"{alias}_textMode" for alias in orig_aliases]))),
                })
                
            if code_field not in seen:
                seen.add(code_field)
                expanded.append({
                    "field": code_field,
                    "fieldName": code_field,
                    "display": f"{display_name} (Code)",
                    "type": "TEXT",
                    "aliases": sorted(list(set([
                        code_field,
                        f"{code_field}",
                        f"{display_name} (Code)".lower(),
                        f"{display_name.lower()} code",
                    ] + [f"{alias}_" for alias in orig_aliases]))),
                })
    return expanded



def _trimmed_collection_schema(collection_name, table_metadata, max_fields=40, allowed_field_names=None):
    metadata = table_metadata.get(collection_name, {})
    rows = []
    allow = set()
    for item in allowed_field_names or []:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("field") or "").strip()
            if name:
                allow.add(name)
        else:
            text = str(item).strip()
            if text:
                allow.add(text)
    for field in (metadata.get("fields") or [])[:max_fields]:
        if allow and str(field.get("field") or "") not in allow:
            continue
        field_name = str(field.get("field") or "")
        display_name = str(field.get("display") or field_name)
        aliases = sorted({
            field_name,
            display_name,
            display_name.lower(),
            display_name.replace(" ", ""),
        })
        rows.append(
            {
                "field": field_name,
                "fieldName": field_name,
                "display": display_name,
                "aliases": aliases,
                "type": field.get("type"),
                "lookupTargetCollection": field.get("lookupTargetCollection"),
                "options": [
                    {"value": opt.get("value")}
                    for opt in (field.get("options") or [])[:10]
                    if isinstance(opt, dict) and "value" in opt
                ],
            }
        )
    return {
        "templateName": metadata.get("template_name", collection_name),
        "collectionName": collection_name,
        "fields": _expand_lookup_companion_fields(rows),
    }


def _extract_named_collection(user_input):
    text = _normalize_text(user_input)
    if not text:
        return None
    patterns = [
        r"\bin\s+([a-zA-Z0-9_]+)\s+collection\b",
        r"\bfrom\s+([a-zA-Z0-9_]+)\s+collection\b",
        r"\btable\s+([a-zA-Z0-9_]+)\b",
        r"\bcollection\s+([a-zA-Z0-9_]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _normalize_text(match.group(1))
    return None


def _resolve_named_collection(explicit_name, allowed_collections, table_metadata):
    target = _normalize_text(explicit_name)
    if not target:
        return None
    target_root = _singularize(target)

    best = None
    best_score = -1
    for collection in allowed_collections:
        metadata = table_metadata.get(collection, {})
        aliases = _build_collection_aliases(collection, metadata)
        score = 0
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            alias_root = _singularize(alias_norm)
            if target == alias_norm:
                score = max(score, 20)
            elif target_root == alias_root:
                score = max(score, 16)
            elif target in alias_norm or alias_norm in target:
                score = max(score, 12)
        if score > best_score:
            best_score = score
            best = collection
    if best_score >= 12:
        return best
    return None


def _field_match_score(user_input, collection_name, table_metadata):
    normalized_input = _normalize_text(user_input)
    prompt_terms = set(normalized_input.split()) if normalized_input else set()
    if not prompt_terms:
        return 0
    
    metadata = table_metadata.get(collection_name, {})
    all_fields = metadata.get("fields") or []
    
    total_score = 0
    for field in all_fields:
        field_name = _normalize_text(field.get("field") or "")
        display = _normalize_text(field.get("display") or "")
        aliases = [_normalize_text(a) for a in field.get("aliases") or []]
        for pt in prompt_terms:
            if len(pt) < 3:
                continue
            if pt in field_name:
                total_score += 15
            if pt in display:
                total_score += 10
            for a in aliases:
                if pt in a:
                    total_score += 5
    return total_score


def generate_table_choice(model, tokenizer, user_input, db_name, allowed_collections, table_metadata, chat_context=None, user_context=None, exact_candidates=None, preferred_collection=None):
    exact_candidates = exact_candidates or {}
    deterministic_choice = {"collection": preferred_collection} if preferred_collection else _deterministic_table_choice(user_input, allowed_collections, table_metadata)
    
    scored_collections = [
        (collection, _collection_match_score(user_input, collection, table_metadata) * 3 + _field_match_score(user_input, collection, table_metadata))
        for collection in allowed_collections
    ]
    scored_collections.sort(key=lambda item: (
        -item[1],
        len([token for token in _normalize_text((table_metadata.get(item[0]) or {}).get("template_name") or item[0]).split() if token]),
        len(_normalize_text((table_metadata.get(item[0]) or {}).get("template_name") or item[0])),
        len(item[0]),
        item[0]
    ))
    context_collections = [c[0] for c in scored_collections[:LLM_COLLECTION_CANDIDATES]]
    
    if isinstance(deterministic_choice, dict) and deterministic_choice.get("collection"):
        det_col = deterministic_choice["collection"]
        if det_col in context_collections:
            context_collections.remove(det_col)
        context_collections.insert(0, det_col)

    messages = [
        {"role": "system", "content": TABLE_ROUTER_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "database": db_name,
                    "allowed_tables": _table_context_rows(user_input, context_collections, table_metadata),
                    "exact_field_candidates": exact_candidates,
                    "chat_history": (chat_context or [])[-2:],
                    "selected_user": user_context or {},
                    "user_request": str(user_input),
                },
                ensure_ascii=False,
            ),
        },
    ]
    try:
        choice = _generate_json_from_messages(model, tokenizer, messages, TABLE_ROUTER_MAX_NEW_TOKENS)
    except Exception:
        choice = {"needs_clarification": True, "message": "I could not confidently identify the target ERP table."}

    if choice.get("needs_clarification"):
        forced_messages = messages + [
            {
                "role": "user",
                "content": (
                    "You MUST select one collection from allowed_tables. "
                    "Pick the most likely match based on the user's meaningful words. "
                    "Return JSON only with 'collection' and 'reason'. Do NOT return needs_clarification."
                ),
            }
        ]
        try:
            forced_choice = _generate_json_from_messages(model, tokenizer, forced_messages, TABLE_ROUTER_MAX_NEW_TOKENS)
            if isinstance(forced_choice, dict) and not forced_choice.get("needs_clarification"):
                candidate = str(forced_choice.get("collection") or "").strip()
                if candidate and _resolve_named_collection(candidate, allowed_collections, table_metadata):
                    choice = forced_choice
        except Exception:
            pass

    if choice.get("needs_clarification"):
        if isinstance(deterministic_choice, dict) and deterministic_choice.get("collection") in set(allowed_collections or []):
            return {
                "collection": deterministic_choice["collection"],
                "reason": str(deterministic_choice.get("reason") or "Matched request keywords to template metadata."),
            }
        return choice

    collection = str(choice.get("collection") or "").strip()
    resolved_collection = _resolve_named_collection(collection, allowed_collections, table_metadata) if collection else None
    if resolved_collection:
        collection = resolved_collection
    if not collection or collection not in allowed_collections:
        if isinstance(deterministic_choice, dict) and deterministic_choice.get("collection") in set(allowed_collections or []):
            return {
                "collection": str(deterministic_choice.get("collection") or "").strip(),
                "reason": str(deterministic_choice.get("reason") or "Matched request keywords to template metadata."),
            }
        return {
            "needs_clarification": True,
            "message": "I could not identify the target ERP table.",
        }

    return {
        "collection": collection,
        "reason": str(choice.get("reason") or "Selected by LLM router."),
    }


def generate_query_plan(
    model,
    tokenizer,
    user_input,
    db_name,
    selected_collection,
    schema,
    sample_docs,
    table_metadata,
    template_schema=None,
    chat_context=None,
    exact_field_candidates=None,
    vector_field_candidates=None,
    user_context=None,
    matched_values=None,
    field_source_tracking=None,
):
    metadata = table_metadata.get(selected_collection, {})
    exact_field_candidates = exact_field_candidates or {}
    vector_field_candidates = vector_field_candidates or {}
    matched_values = matched_values or {}
    field_source_tracking = field_source_tracking or {}
    template_context = template_schema or {
        "templateName": metadata.get("template_name", selected_collection),
        "collectionName": selected_collection,
        "fields": (metadata.get("fields") or [])[:120],
    }
    if isinstance(template_context, dict):
        wanted = []
        wanted.extend(str(item).strip() for item in (exact_field_candidates.get(selected_collection) or []) if str(item).strip())
        wanted.extend(str(item).strip() for item in (vector_field_candidates.get(selected_collection) or []) if str(item).strip())
        wanted = list(dict.fromkeys(wanted))
        fields = list(template_context.get("fields") or [])
        if wanted and fields:
            ranked = [
                row for row in fields
                if str((row or {}).get("fieldName") or (row or {}).get("field") or "").strip() in wanted
            ]
            remaining = [
                row for row in fields
                if str((row or {}).get("fieldName") or (row or {}).get("field") or "").strip() not in wanted
            ]
            template_context = dict(template_context)
            template_context["fields"] = _expand_lookup_companion_fields((ranked + remaining)[:120])
        else:
            template_context = dict(template_context)
            template_context["fields"] = _expand_lookup_companion_fields(fields[:120])
    context = {
        "database": db_name,
        "selected_collection": selected_collection,
        "selected_table": metadata.get("template_name", selected_collection),
        "schema_fields": [{"field": field, "types": types} for field, types in list((schema or {}).items())[:200]],
        "sample_documents": (sample_docs or [])[:2],
        "chat_history": chat_context or [],
        "selected_user": user_context or {},
        "query_mode_hint": "auto",
        "reverse_lookup_hints": {
            "exact_field_candidates": exact_field_candidates,
            "vector_field_candidates": vector_field_candidates,
            "matched_values": matched_values.get(selected_collection) or [] if isinstance(matched_values, dict) else (matched_values or []),
            "field_sources": {f"{c}::{f}": v for (c, f), v in field_source_tracking.items()} if isinstance(field_source_tracking, dict) else {},
        },
    }
    messages = [
        {
            "role": "system",
            "content": QUERY_PLANNER_PROMPT
            ,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "schema": template_context,
                    "question": str(user_input),
                    "runtime_context": context,
                },
                ensure_ascii=False,
            ),
        },
    ]
    plan = _generate_json_from_messages(model, tokenizer, messages, MAX_NEW_TOKENS)
    plan = _normalize_generated_plan(plan, fallback_collection=selected_collection)
    return plan



def _field_match_score(user_input, collection_name, table_metadata):
    normalized_input = _normalize_text(user_input)
    if not normalized_input:
        return 0
    metadata = table_metadata.get(collection_name, {})
    prompt_terms = set(normalized_input.split())
    total = 0
    for field in (metadata.get("fields") or []):
        field_name = _normalize_text(field.get("field") or "")
        display = _normalize_text(field.get("display") or "")
        aliases = [_normalize_text(a) for a in (field.get("aliases") or []) if a]
        all_labels = {field_name, display} | set(aliases)
        for label in all_labels:
            if not label:
                continue
            if label in prompt_terms:
                total += 15
            elif label in normalized_input:
                total += 10
            else:
                label_tokens = set(label.split())
                overlap = len(label_tokens & prompt_terms)
                if overlap:
                    total += overlap * 5
    return total


def generate_single_pass_query_plan(
    model,
    tokenizer,
    user_input,
    db_name,
    allowed_collections,
    table_metadata,
    chat_context=None,
    preferred_collection=None,
    exact_field_candidates=None,
    vector_field_candidates=None,
    user_context=None,
    matched_values=None,
    field_source_tracking=None,
):
    exact_field_candidates = exact_field_candidates or {}
    vector_field_candidates = vector_field_candidates or {}
    matched_values = matched_values or {}
    field_source_tracking = field_source_tracking or {}
    named_collection = _extract_named_collection(user_input)
    resolved_named_collection = _resolve_named_collection(
        named_collection,
        allowed_collections,
        table_metadata,
    )
    constrained_collections = allowed_collections
    deterministic_choice = {"collection": preferred_collection} if preferred_collection else _deterministic_table_choice(user_input, allowed_collections, table_metadata)
    
    if resolved_named_collection:
        constrained_collections = [resolved_named_collection]
    else:
        scored_collections = [
            (
                collection,
                _collection_match_score(user_input, collection, table_metadata) * 3
                + _field_match_score(user_input, collection, table_metadata),
            )
            for collection in constrained_collections
        ]
        scored_collections.sort(key=lambda item: (
            -item[1],
            len([token for token in _normalize_text((table_metadata.get(item[0]) or {}).get("template_name") or item[0]).split() if token]),
            len(_normalize_text((table_metadata.get(item[0]) or {}).get("template_name") or item[0])),
            len(item[0]),
            item[0]
        ))
        constrained_collections = [c[0] for c in scored_collections]
        
        # Ensure deterministic choice is prioritized
        if isinstance(deterministic_choice, dict) and deterministic_choice.get("collection"):
            det_col = deterministic_choice["collection"]
            if det_col in constrained_collections:
                constrained_collections.remove(det_col)
                constrained_collections.insert(0, det_col)

    sample_documents = {}
    try:
        from erp_backend.storage.mongo import mongo_client
        from erp_backend.core.security import sanitize_doc_for_display
        from erp_backend.core.utils import to_jsonable
        db = mongo_client()[db_name]
        for col in constrained_collections[:2]:
            docs = list(db[col].find({}, {"_id": 0}).sort([("_id", -1)]).limit(2))
            if docs:
                sample_documents[col] = [sanitize_doc_for_display(to_jsonable(d)) for d in docs]
    except Exception:
        pass

    context = {
        "database": db_name,
        "sample_documents": sample_documents,
        "allowed_collections": constrained_collections[:15],
        "collection_schemas": {
            name: _trimmed_collection_schema(
                name,
                table_metadata,
                allowed_field_names=(
                    list(dict.fromkeys(
                        [str(item).strip() for item in (exact_field_candidates or {}).get(name, []) if str(item).strip()]
                        + [str(item).strip() for item in (vector_field_candidates or {}).get(name, []) if str(item).strip()]
                    ))
                ),
            )
            for name in constrained_collections[:15]
        },
        "chat_history": chat_context or [],
        "selected_user": user_context or {},
        "reverse_lookup_hints": {
            "exact_field_candidates": exact_field_candidates,
            "vector_field_candidates": vector_field_candidates,
            "matched_values": matched_values,
            "field_sources": {f"{c}::{f}": v for (c, f), v in field_source_tracking.items()} if isinstance(field_source_tracking, dict) else {},
        },
    }
    messages = [
        {"role": "system", "content": SINGLE_PASS_QUERY_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "context": context,
                    "question": str(user_input),
                },
                ensure_ascii=False,
            ),
        },
    ]
    plan = _generate_json_from_messages(model, tokenizer, messages, MAX_NEW_TOKENS)
    plan = _normalize_generated_plan(plan)
    if not isinstance(plan, dict):
        raise ValueError("query plan must be a JSON object")
    selected = str(plan.get("collection") or "").strip()
    resolved_selected = _resolve_named_collection(selected, allowed_collections, table_metadata) if selected else None
    if resolved_selected:
        plan["collection"] = resolved_selected
    elif selected not in allowed_collections:
        return {
            "needs_clarification": True,
            "message": "Please specify which table/collection you want to query.",
        }
    return plan


def verify_query_result(
    model,
    tokenizer,
    user_input,
    selected_collection,
    table_metadata,
    plan,
    docs,
    total,
):
    metadata = table_metadata.get(selected_collection, {})
    payload = {
        "question": str(user_input),
        "collection": selected_collection,
        "table": metadata.get("template_name", selected_collection),
        "plan": plan,
        "total_rows": int(total or 0),
        "rows_preview": _build_rows_preview(docs, total=total, max_output_tokens=min(MAX_NEW_TOKENS, 320), min_rows=6, max_rows=20),
    }
    messages = [
        {"role": "system", "content": RESULT_VERIFIER_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        verdict = _generate_json_from_messages(
            model,
            tokenizer,
            messages,
            min(MAX_NEW_TOKENS, 320),
            char_budget=_context_char_budget(min(MAX_NEW_TOKENS, 320), result_docs=docs, total_rows=total),
        )
    except Exception:
        return {"status": "ok", "message": "Result validated."}
    status = str(verdict.get("status") or "").strip().lower()
    message = str(verdict.get("message") or "").strip()

    if status not in {"ok", "needs_clarification"}:
        return {"status": "ok", "message": "Result validated."}
    return {
        "status": status,
        "message": message or "Please clarify your request.",
    }


def extract_result_constraints(
    model,
    tokenizer,
    user_input,
    selected_collection,
    table_metadata,
):
    metadata = table_metadata.get(selected_collection, {})
    payload = {
        "question": str(user_input or ""),
        "collection": selected_collection,
        "table": metadata.get("template_name", selected_collection),
    }
    messages = [
        {"role": "system", "content": RESULT_CONSTRAINTS_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        out = _generate_json_from_messages(model, tokenizer, messages, min(MAX_NEW_TOKENS, 180))
    except Exception:
        return {"must_terms": [], "answer_type": "unknown"}

    terms = []
    raw_terms = out.get("must_terms") if isinstance(out, dict) else []
    if isinstance(raw_terms, list):
        for item in raw_terms:
            text = re.sub(r"\s+", " ", str(item or "").strip().lower())
            if not text:
                continue
            if text not in terms:
                terms.append(text)
    answer_type = (
        str(out.get("answer_type") or "unknown").strip().lower()
        if isinstance(out, dict)
        else "unknown"
    )
    if answer_type not in {"date", "count", "status", "text", "unknown"}:
        answer_type = "unknown"
    return {"must_terms": terms[:5], "answer_type": answer_type}


def _build_result_analysis_messages(
    user_input,
    selected_collection,
    metadata,
    plan,
    docs,
    total,
):
    payload = {
        "question": str(user_input),
        "collection": selected_collection,
        "table": metadata.get("template_name", selected_collection),
        "operation": plan.get("operation", "find"),
        "total_rows": int(total or 0),
        "rows_preview": _build_rows_preview(docs, total=total, max_output_tokens=min(MAX_NEW_TOKENS, 512), min_rows=6, max_rows=18),
    }
    char_budget = _context_char_budget(min(MAX_NEW_TOKENS, 512), result_docs=docs, total_rows=total)
    return _compact_messages(
        [
            {"role": "system", "content": RESULT_ANALYSIS_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        char_budget=char_budget,
    )


def _build_result_summary_messages(
    user_input,
    selected_collection,
    metadata,
    plan,
    docs,
    total,
):
    payload = {
        "question": str(user_input),
        "collection": selected_collection,
        "table": metadata.get("template_name", selected_collection),
        "operation": plan.get("operation", "find"),
        "total_rows": int(total or 0),
        "rows_preview": _build_rows_preview(docs, total=total, max_output_tokens=min(MAX_NEW_TOKENS, 180), min_rows=6, max_rows=18),
    }
    char_budget = _context_char_budget(min(MAX_NEW_TOKENS, 180), result_docs=docs, total_rows=total)
    return _compact_messages(
        [
            {"role": "system", "content": RESULT_SUMMARY_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        char_budget=char_budget,
    )


def _clean_result_summary_text(text):
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", str(text or ""), flags=re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    lines = [line.strip() for line in cleaned.splitlines()]
    return "\n".join(lines).strip()


def _summary_text_is_low_quality(text):
    generic_markers = (
        "results are consistent across",
        "with no execution errors",
        "apply a filter or grouping",
        "preview rows",
        "returned 1 row from",
        "returned 2 rows from",
        "returned 3 rows from",
    )
    if not text:
        return True
    lowered = text.lower()
    if len(text) > 2000:
        return True
    return any(marker in lowered for marker in generic_markers)


def _run_llm_result_analysis(model, messages):
    response = model.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=min(MAX_NEW_TOKENS, 512),
    )
    text = _clean_result_summary_text(
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if not _summary_text_is_low_quality(text):
        return text
    repair_messages = messages + [
        {
            "role": "user",
            "content": (
                "Rewrite the analysis using actual values from rows_preview. "
                "Structure it line-by-line using markdown lists. Do not mention preview/profiling language. Return plain text only."
            ),
        }
    ]
    repair_response = model.create_chat_completion(
        messages=repair_messages,
        temperature=0.0,
        max_tokens=min(MAX_NEW_TOKENS, 220),
    )
    repaired = _clean_result_summary_text(
        repair_response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if not _summary_text_is_low_quality(repaired):
        return repaired
    return ""


def analyze_query_result(
    model,
    tokenizer,
    user_input,
    selected_collection,
    table_metadata,
    plan,
    docs,
    total,
):
    metadata = table_metadata.get(selected_collection, {})
    messages = _build_result_analysis_messages(
        user_input,
        selected_collection,
        metadata,
        plan,
        docs,
        total,
    )
    text = _run_llm_result_analysis(model, messages)
    if text:
        return text
    return _deterministic_result_summary(
        selected_collection=selected_collection,
        table_label=metadata.get("template_name", selected_collection),
        docs=docs,
        total=total,
    )


def summarize_query_result(
    model,
    tokenizer,
    user_input,
    selected_collection,
    table_metadata,
    plan,
    docs,
    total,
):
    metadata = table_metadata.get(selected_collection, {})
    messages = _build_result_summary_messages(
        user_input,
        selected_collection,
        metadata,
        plan,
        docs,
        total,
    )
    try:
        response = model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=min(MAX_NEW_TOKENS, 180),
        )
        text = _clean_result_summary_text(
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if text and not _summary_text_is_low_quality(text):
            return text
    except Exception:
        pass
    total_rows = int(total or len(docs or []))
    table_label = metadata.get("template_name", selected_collection)
    if not docs:
        return f"No rows found in `{table_label}`."
    return f"Retrieved {total_rows} row{'s' if total_rows != 1 else ''} from `{table_label}`."


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip().replace(",", "")
        if re.fullmatch(r"-?\d+(\.\d+)?", candidate):
            try:
                return float(candidate)
            except ValueError:
                return None
    return None


def _as_datetime(value):
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        # Handles ISO values, including trailing Z.
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _deterministic_result_summary(selected_collection, table_label, docs, total):
    insights = build_result_insights(selected_collection, {"": {}, selected_collection: {"template_name": table_label}}, docs, total)
    total_rows = int(total or len(docs or []))
    if not docs:
        return (
            f"No rows were returned from {table_label}. "
            f"Try relaxing one filter in {selected_collection} and rerun."
        )
    return (
        f"Returned {total_rows} row{'s' if total_rows != 1 else ''} from {table_label}. "
        + ". ".join(insights[:3])
        + "."
    )


def build_result_insights(selected_collection, table_metadata, docs, total):
    rows = list(docs or [])
    if not rows:
        return []

    # If user asks for expiry/due-end date and it's not present/populated, say that directly.
    # This avoids generic "date missing" language.
    # (Handled via field inference over preview row keys.)
    lower_keys = {
        str(k).strip().lower()
        for row in rows
        if isinstance(row, dict)
        for k in row.keys()
    }
    expiry_like_keys = [
        key for key in lower_keys
        if any(token in key for token in ("expiry", "expire", "due", "enddate", "end_date", "completiondate", "completion_date"))
    ]

    # Collect per-field statistics from preview rows.
    field_counters = {}
    numeric_stats = {}
    date_stats = {}
    null_counts = Counter()

    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if value in (None, "", [], {}):
                null_counts[key] += 1
                continue

            if isinstance(value, list):
                for item in value[:5]:
                    if isinstance(item, (str, int, float, bool)):
                        field_counters.setdefault(key, Counter())[str(item)] += 1
                continue

            if isinstance(value, (str, int, float, bool)):
                field_counters.setdefault(key, Counter())[str(value)] += 1

            num = _as_number(value)
            if num is not None:
                stat = numeric_stats.setdefault(key, {"min": num, "max": num})
                stat["min"] = min(stat["min"], num)
                stat["max"] = max(stat["max"], num)

            dt = _as_datetime(value)
            if dt is not None:
                stat = date_stats.setdefault(key, {"min": dt, "max": dt})
                stat["min"] = min(stat["min"], dt)
                stat["max"] = max(stat["max"], dt)

    insights = []

    # Highest-signal categorical field.
    top_field = None
    top_field_spread = 0
    for key, counter in field_counters.items():
        if not counter:
            continue
        spread = max(counter.values())
        if spread > top_field_spread:
            top_field_spread = spread
            top_field = key
    if top_field:
        top_values = field_counters[top_field].most_common(3)
        formatted = ", ".join(f"{v} ({c})" for v, c in top_values)
        insights.append(f"Top {top_field} values: {formatted}")

    # Numeric range insight.
    if numeric_stats:
        best_num_key = max(
            numeric_stats.keys(),
            key=lambda k: abs(numeric_stats[k]["max"] - numeric_stats[k]["min"]),
        )
        stat = numeric_stats[best_num_key]
        insights.append(f"{best_num_key} ranges from {stat['min']:.2f} to {stat['max']:.2f}")

    # Date range insight.
    if date_stats:
        best_date_key = min(date_stats.keys(), key=lambda k: date_stats[k]["min"])
        stat = date_stats[best_date_key]
        insights.append(
            f"{best_date_key} spans {stat['min'].date().isoformat()} to {stat['max'].date().isoformat()}"
        )

    # Missingness insight.
    if null_counts:
        key, count = null_counts.most_common(1)[0]
        if count > 0:
            # Prefer a business-friendly date-specific message when relevant.
            if key.lower() in {"date", "expirydate", "expirationdate", "dueDate".lower(), "completiondate"} or any(
                token in key.lower() for token in ("expiry", "expire", "due", "date", "completion")
            ):
                insights.append(f"{key} is empty in {count} of {len(rows)} preview rows")
            else:
                insights.append(f"{key} is missing in {count} of {len(rows)} preview rows")

    if not insights:
        sample_keys = sorted({k for row in rows if isinstance(row, dict) for k in row.keys()})[:4]
        field_hint = ", ".join(sample_keys) if sample_keys else "available fields"
        insights.append(f"Preview includes fields: {field_hint}")

    if expiry_like_keys:
        # If expiry-like keys exist but all are null/empty in preview, make it explicit.
        missing_expiry_keys = []
        for k in expiry_like_keys:
            missing = 0
            present = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if k not in {str(x).strip().lower() for x in row.keys()}:
                    continue
                matched_value = None
                for actual_key, actual_value in row.items():
                    if str(actual_key).strip().lower() == k:
                        matched_value = actual_value
                        break
                if matched_value in (None, "", [], {}):
                    missing += 1
                else:
                    present += 1
            if missing > 0 and present == 0:
                missing_expiry_keys.append(k)
        if missing_expiry_keys:
            insights.insert(0, f"Expiry-related field(s) {', '.join(missing_expiry_keys[:2])} are empty in preview")

    return insights[:3]


def _humanize_chart_field_name(field_name):
    text = str(field_name or "").strip()
    if not text:
        return "Value"
    if text == "_id":
        return "Group"
    return text.replace("_", " ").strip().title()


def build_chart_config(plan, docs):
    rows = [row for row in (docs or []) if isinstance(row, dict)]
    if len(rows) < 2:
        return None

    field_names = []
    seen_fields = set()
    for row in rows[:10]:
        for key in row.keys():
            key_text = str(key or "").strip()
            if key_text and key_text not in seen_fields and key_text != "_id":
                field_names.append(key_text)
                seen_fields.add(key_text)
    if "_id" in {str(key).strip() for row in rows[:10] for key in row.keys()}:
        field_names.insert(0, "_id")

    def _series_for(field_name):
        pairs = []
        for row in rows[:10]:
            label_value = row.get(label_field)
            numeric_value = _as_number(row.get(field_name))
            if numeric_value is None or label_value in (None, "", [], {}):
                continue
            pairs.append((str(label_value), numeric_value))
        return pairs

    numeric_fields = []
    for field_name in field_names:
        if field_name == "_id":
            continue
        numeric_count = 0
        for row in rows[:10]:
            if _as_number(row.get(field_name)) is not None:
                numeric_count += 1
        if numeric_count >= 2:
            numeric_fields.append(field_name)
    if not numeric_fields:
        return None

    # Prefer non-numeric (categorical) fields as label; numeric fields belong on the Y-axis.
    # This prevents a count/amount field from being used as both X-axis label and bar height.
    categorical_label_candidates = [
        f for f in field_names if f not in numeric_fields
    ]
    numeric_label_candidates = [f for f in field_names if f in numeric_fields]

    # Build ordered list: categorical first (with _id prioritised), then numeric as fallback
    preferred_label_fields = categorical_label_candidates + numeric_label_candidates

    label_field = None
    for candidate in preferred_label_fields:
        distinct = {
            str(row.get(candidate))
            for row in rows[:10]
            if row.get(candidate) not in (None, "", [], {})
        }
        if len(distinct) >= 2:
            label_field = candidate
            break
    if not label_field:
        return None

    numeric_field = None
    priority_numeric_fields = [
        "count",
        "total",
        "amount",
        "value",
        "sum",
        "qty",
        "quantity",
    ]
    # Pick numeric field that is different from label_field
    for candidate in priority_numeric_fields:
        if candidate in numeric_fields and candidate != label_field:
            numeric_field = candidate
            break
    if not numeric_field:
        for candidate in numeric_fields:
            if candidate != label_field:
                numeric_field = candidate
                break
    if not numeric_field:
        return None

    pairs = _series_for(numeric_field)
    if len(pairs) < 2:
        return None

    # Build a meaningful title: for _id label fields, try to infer a better display name
    # from the actual values (e.g. email domains → "Email Domain").
    if label_field == "_id":
        sample_values = [str(row.get("_id", "")) for row in rows[:5] if row.get("_id") not in (None, "", [], {})]
        if sample_values and all("." in v and "@" not in v for v in sample_values[:3]):
            label_display = "Domain"
        elif sample_values and any("@" in v for v in sample_values[:3]):
            label_display = "Email"
        else:
            label_display = "Group"
    else:
        label_display = _humanize_chart_field_name(label_field)

    labels = [label for label, _ in pairs[:8]]
    values = [value for _, value in pairs[:8]]
    return {
        "type": "bar",
        "title": f"{_humanize_chart_field_name(numeric_field)} by {label_display}",
        "labels": labels,
        "datasets": [
            {
                "label": _humanize_chart_field_name(numeric_field),
                "data": values,
            }
        ],
    }


def _extract_stream_token(chunk):
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
        message = choice.get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _stream_chat_completion_tokens(model, messages, max_tokens):
    # SafeTensors runtime wrapper provides create_chat_completion_stream.
    if hasattr(model, "create_chat_completion_stream"):
        for token in model.create_chat_completion_stream(
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        ):
            token_text = _extract_stream_token(token)
            if token_text:
                yield token_text
        return

    # llama-cpp supports stream=True directly on create_chat_completion.
    try:
        stream = model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            token_text = _extract_stream_token(chunk)
            if token_text:
                yield token_text
        return
    except TypeError:
        pass

    # Fallback: non-streaming completion split into chunks.
    response = model.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if text:
        for chunk in re.split(r"(\s+)", text):
            if chunk:
                yield chunk


def analyze_query_result_stream(
    model,
    tokenizer,
    user_input,
    selected_collection,
    table_metadata,
    plan,
    docs,
    total,
):
    metadata = table_metadata.get(selected_collection, {})
    messages = _build_result_analysis_messages(
        user_input,
        selected_collection,
        metadata,
        plan,
        docs,
        total,
    )
    streamed_tokens = []
    try:
        for token in _stream_chat_completion_tokens(model, messages, min(MAX_NEW_TOKENS, 512)):
            if token:
                streamed_tokens.append(token)
                yield token
    except Exception:
        streamed_tokens = []

    streamed_text = _clean_result_summary_text("".join(streamed_tokens))
    if streamed_tokens and not _summary_text_is_low_quality(streamed_text):
        pass
    else:
        final_text = _run_llm_result_analysis(model, messages)
        if final_text:
            for token in re.split(r"(\s+)", final_text):
                if token:
                    yield token
        else:
            for token in re.split(r"(\s+)", _deterministic_result_summary(
                selected_collection=selected_collection,
                table_label=metadata.get("template_name", selected_collection),
                docs=docs,
                total=total,
            )):
                if token:
                    yield token


def generate_follow_up_suggestions(
    model,
    tokenizer,
    user_input,
    selected_collection,
    table_metadata,
    plan,
    docs,
    total,
):
    metadata = table_metadata.get(selected_collection, {})
    payload = {
        "question": str(user_input),
        "collection": selected_collection,
        "table": metadata.get("template_name", selected_collection),
        "operation": plan.get("operation", "find"),
        "total_rows": int(total or 0),
        "rows_preview": _build_rows_preview(docs, total=total, max_output_tokens=min(MAX_NEW_TOKENS, 220), min_rows=4, max_rows=10),
    }
    messages = [
        {"role": "system", "content": FOLLOW_UP_SUGGESTIONS_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    out = _generate_json_from_messages(
        model,
        tokenizer,
        messages,
        min(MAX_NEW_TOKENS, 220),
        char_budget=_context_char_budget(min(MAX_NEW_TOKENS, 220), result_docs=docs, total_rows=total),
    )
    follow_ups = out.get("follow_ups") or []
    cleaned = []
    for item in follow_ups:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text[:120])
    return cleaned[:3]


def _fallback_field_roles(field_rows):
    def _field_label(row):
        return str((row or {}).get("display") or (row or {}).get("field") or (row or {}).get("name") or "").strip()

    def _field_type(row):
        return str((row or {}).get("type") or "").upper()

    def _field_matches(row, pattern, allowed_types=None):
        label = _field_label(row)
        if not label or not re.search(pattern, label, flags=re.IGNORECASE):
            return False
        if not allowed_types:
            return True
        ftype = _field_type(row)
        return not ftype or ftype in allowed_types

    role_patterns = [
        ("title_role", r"\b(name|title|label|subject|heading|description)\b", None),
        ("ref_role", r"\b(reference|ref|number|no|code|id|serial)\b", None),
        ("status_role", r"\b(status|state|stage|phase|compliance|standing)\b", {"SELECT", "MULTI_SELECT"}),
        ("due_role", r"\b(due|deadline|expiry|expiration|expire|end\s*date)\b", None),
        ("amount_role", r"\b(amount|price|total|cost|fee|tax|charge|value)\b", {"NUMBER", "CURRENCY", "FLOAT", "INT", "DECIMAL"}),
        ("assignee_role", r"\b(assign|assigned|owner|responsible|handler)\b", None),
        ("reviewer_role", r"\b(review|checker|inspector)\b", None),
        ("approver_role", r"\b(approv|authoriz|sanction)\b", None),
        ("branch_role", r"\b(branch|location|office|site|region|zone)\b", None),
        ("category_role", r"\b(type|category|kind|class|mode|group|segment)\b", {"SELECT", "MULTI_SELECT"}),
        ("date_role", r"\b(date|time|datetime|timestamp|when|period)\b", None),
    ]
    roles = {}
    for row in field_rows or []:
        for role_name, pattern, allowed_types in role_patterns:
            if role_name in roles:
                continue
            if _field_matches(row, pattern, allowed_types):
                roles[role_name] = _field_label(row)
    return roles


def _fallback_clarification_suggestions(user_input, selected_collection, table_metadata, docs=None, total=0):
    metadata = table_metadata.get(selected_collection, {}) if selected_collection else {}
    table_label = str(metadata.get("template_name") or selected_collection or "matching records").strip()
    field_rows = list((metadata.get("fields") or [])[:20])

    def _field_label(row):
        return str((row or {}).get("display") or (row or {}).get("field") or (row or {}).get("name") or "").strip()

    def _field_name(row):
        return str((row or {}).get("field") or (row or {}).get("name") or "").strip()

    def _extract_prompt_subject(text):
        raw = str(text or "")
        quoted = re.findall(r"['\"]([^'\"]{2,120})['\"]", raw)
        if quoted:
            return re.sub(r"\s+", " ", quoted[0]).strip()
        norm = " ".join(raw.split())
        match = re.search(
            r"\b(?:show|list|find|get|give|tell|what is|what are|display)\b\s+(?:the\s+)?(.+?)\s+\b(?:record|records|row|rows|item|items|details|info|information)\b",
            norm,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"\b(?:about|for|of)\s+(.+?)(?:\s+\bwith\b|\s+\bwhere\b|\s+\bwhose\b|\s+\bthat\b|$)",
                norm,
                flags=re.IGNORECASE,
            )
        if not match:
            return ""
        phrase = re.sub(r"\b(the|a|an|this|that|all|exact|exactly|specific)\b", " ", match.group(1), flags=re.IGNORECASE)
        phrase = re.sub(r"\s+", " ", phrase).strip(" ,.-")
        return phrase

    def _looks_like_identifier(text):
        norm = str(text or "").strip()
        if not norm:
            return False
        if re.search(r"\b(?:id|code|number|no|ref|reference)\b", norm, flags=re.IGNORECASE):
            return True
        if re.search(r"[A-Za-z]{2,}[-_/][A-Za-z0-9]{2,}", norm):
            return True
        return bool(re.search(r"\b[A-Z]{2,}[A-Z0-9-]{2,}\b", norm))

    def _field_list_phrase(labels):
        items = [str(item or "").strip() for item in labels if str(item or "").strip()]
        if not items:
            return ""
        items = list(dict.fromkeys(items))[:4]
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _field_type(row):
        return str((row or {}).get("type") or "").upper()

    def _field_matches(row, pattern, allowed_types=None):
        label = _field_label(row)
        if not label or not re.search(pattern, label, flags=re.IGNORECASE):
            return False
        if not allowed_types:
            return True
        ftype = _field_type(row)
        return not ftype or ftype in allowed_types

    def _build_dynamic_field_roles(rows):
        role_patterns = [
            ("title_role", r"\b(name|title|label|subject|heading|description)\b", None),
            ("ref_role", r"\b(reference|ref|number|no|code|id|serial)\b", None),
            ("status_role", r"\b(status|state|stage|phase|compliance|standing)\b", {"SELECT", "MULTI_SELECT"}),
            ("due_role", r"\b(due|deadline|expiry|expiration|expire|end\s*date)\b", None),
            ("amount_role", r"\b(amount|price|total|cost|fee|tax|charge|value)\b", {"NUMBER", "CURRENCY", "FLOAT", "INT", "DECIMAL"}),
            ("assignee_role", r"\b(assign|assigned|owner|responsible|handler)\b", None),
            ("reviewer_role", r"\b(review|checker|inspector)\b", None),
            ("approver_role", r"\b(approv|authoriz|sanction)\b", None),
            ("branch_role", r"\b(branch|location|office|site|region|zone)\b", None),
            ("category_role", r"\b(type|category|kind|class|mode|group|segment)\b", {"SELECT", "MULTI_SELECT"}),
            ("date_role", r"\b(date|time|datetime|timestamp|when|period)\b", None),
        ]
        roles = {}
        for row in rows:
            for role_name, pattern, allowed_types in role_patterns:
                if role_name in roles:
                    continue
                if _field_matches(row, pattern, allowed_types):
                    roles[role_name] = _field_label(row)
        return roles

    def _extract_status_hint(rows, user_text):
        normalized = _normalize_term(user_text)
        for row in rows:
            if _field_type(row) not in {"SELECT", "MULTI_SELECT"}:
                continue
            for option in (row or {}).get("options") or []:
                if not isinstance(option, dict):
                    continue
                for candidate in (option.get("value"), option.get("label")):
                    candidate_text = _normalize_term(candidate)
                    if candidate_text and candidate_text in normalized:
                        return str(candidate or "").strip()
        return ""

    roles = _build_dynamic_field_roles(field_rows)
    labels = [
        roles.get(role_name)
        for role_name in (
            "status_role",
            "category_role",
            "branch_role",
            "assignee_role",
            "reviewer_role",
            "approver_role",
            "due_role",
            "amount_role",
            "date_role",
            "ref_role",
            "title_role",
        )
        if roles.get(role_name)
    ]
    if not labels:
        labels = [label for label in (_field_label(row) for row in field_rows) if label]

    text = " ".join(str(user_input or "").lower().split())
    subject = _extract_prompt_subject(user_input)
    if not subject:
        subject = table_label if table_label and table_label != "matching records" else ""
    subject = re.sub(r"\s+", " ", subject).strip()

    status_hint = _extract_status_hint(field_rows, text)
    entity_hint = ""

    def _has_label(token):
        token = str(token or "").strip().lower()
        if not token:
            return False
        return any(token in str(label or "").lower() for label in labels)

    def _first_label(*needles, fallback=""):
        for needle in needles:
            for label in labels:
                if str(needle).lower() in str(label).lower():
                    return label
        return fallback

    requested_fields = []
    for role_name in (
        "title_role",
        "ref_role",
        "due_role",
        "status_role",
        "branch_role",
        "assignee_role",
        "reviewer_role",
        "approver_role",
        "amount_role",
        "date_role",
        "category_role",
    ):
        label = roles.get(role_name)
        if label and label not in requested_fields:
            requested_fields.append(label)
    if not requested_fields:
        requested_fields = [label for label in labels[:4] if label]

    title_field = roles.get("title_role") or (labels[0] if labels else "")
    ref_field = roles.get("ref_role") or title_field
    code_field = roles.get("ref_role") or ref_field
    status_field = roles.get("status_role") or status_hint or ""
    branch_field = roles.get("branch_role") or ""
    assignee_field = roles.get("assignee_role") or ""
    reviewer_field = roles.get("reviewer_role") or ""
    approver_field = roles.get("approver_role") or ""
    due_field = roles.get("due_role") or roles.get("date_role") or ""
    expiry_field = roles.get("due_role") or roles.get("date_role") or ""
    renewal_field = roles.get("status_role") or ""
    sort_field = due_field if _has_label(due_field) else expiry_field
    group_field = branch_field if _has_label(branch_field) else status_field

    base_subject = subject or (table_label if table_label and table_label != "matching records" else "matching records")
    if status_hint and status_hint not in base_subject.lower():
        base_subject = f"{status_hint} {base_subject}"
    if entity_hint and entity_hint not in base_subject.lower():
        base_subject = f"{base_subject}"

    exact_label = title_field or ref_field
    requested_phrase = _field_list_phrase(requested_fields)
    if subject and _looks_like_identifier(subject):
        best_lookup_field = code_field or ref_field or title_field
        best_return_fields = _field_list_phrase(
            [
                best_lookup_field,
                _first_label("type", "branch type", "category", fallback="type"),
                branch_field,
                status_field,
            ]
        )
        suggestions = [
            f"Find the {table_label} record with {best_lookup_field} '{subject}' and return {best_return_fields}.",
            f"Show the record for '{subject}' with {best_lookup_field}, {branch_field}, and {status_field}.",
            f"Return the {table_label} row for '{subject}' and include its type and status fields.",
        ]
    elif subject:
        suggestions = [
            f"Find the exact {exact_label} '{subject}' and return {requested_phrase or f'{due_field} and {expiry_field}'}.",
            f"Show the '{subject}' record with {requested_phrase or f'{due_field} and {expiry_field}'} from the selected table.",
            f"Return the '{subject}' record with {status_field}, {renewal_field}, and {reviewer_field}.",
        ]
    elif status_hint:
        suggestions = [
            f"Find {status_hint} {base_subject} and return {title_field}, {branch_field}, {assignee_field}, and {reviewer_field}.",
            f"Count {status_hint} {base_subject} by {group_field}.",
            f"Return {status_hint} {base_subject} sorted by {sort_field}.",
        ]
    elif entity_hint:
        suggestions = [
            f"Find {entity_hint} records and return {title_field}, {branch_field}, {assignee_field}, and {reviewer_field}.",
            f"Count {entity_hint} records by {group_field}.",
            f"Return {entity_hint} records with {due_field} and {expiry_field}.",
        ]
    else:
        suggestions = [
            f"Find {base_subject} and return {title_field}, {branch_field}, {assignee_field}, and {reviewer_field}.",
            f"Count {base_subject} by {group_field}.",
            f"Return {base_subject} sorted by {sort_field}.",
        ]

    message = "I could not confidently match that request. Use one of the rewritten prompts below."
    if subject:
        if _looks_like_identifier(subject):
            message = f"I found the requested identifier '{subject}' in the selected table. Use one of the rewritten prompts below."
        else:
            message = f"I found the exact subject '{subject}' in the selected table. Use one of the rewritten prompts below."
    elif status_hint:
        message = f"I found a {status_hint} intent in the selected table. Use one of the rewritten prompts below."
    elif entity_hint:
        message = f"I found a {entity_hint} intent in the selected table. Use one of the rewritten prompts below."
    return message, suggestions[:3]


def generate_clarification_suggestions(
    model,
    tokenizer,
    user_input,
    selected_collection,
    table_metadata,
    plan,
    docs,
    reason="",
    accessible_collections=None,
):
    metadata = table_metadata.get(selected_collection, {}) if selected_collection else {}
    field_rows = list((metadata.get("fields") or [])[:20])
    available_rows = []
    for name in (accessible_collections or [])[:12]:
        meta = (table_metadata or {}).get(name, {})
        available_rows.append(
            {
                "collection": name,
                "table": meta.get("template_name", name),
                "business_terms": (meta.get("business_terms") or [])[:8],
                "fields": [
                    str((field or {}).get("display") or (field or {}).get("field") or (field or {}).get("name") or "").strip()
                    for field in (meta.get("fields") or [])[:8]
                    if str((field or {}).get("display") or (field or {}).get("field") or (field or {}).get("name") or "").strip()
                ],
            }
        )
    payload = {
        "question": str(user_input),
        "reason": str(reason or "").strip(),
        "subject": "",
        "collection": selected_collection,
        "table": metadata.get("template_name", selected_collection),
        "operation": (plan or {}).get("operation", "none"),
        "total_rows": int(len(docs or []) or 0),
        "rows_preview": (docs or [])[:8],
        "requested_fields": [],
        "fields": [
            {
                "name": str((field or {}).get("field") or (field or {}).get("name") or "").strip(),
                "display": str((field or {}).get("display") or "").strip(),
                "type": str((field or {}).get("type") or "").strip(),
            }
            for field in field_rows
        ],
        "available_collections": available_rows,
    }
    subject = ""
    if selected_collection:
        # Reuse the same subject extraction as the fallback path so the model
        # sees the user's exact value text when it is present.
        raw = str(user_input or "")
        quoted = re.findall(r"['\"]([^'\"]{2,120})['\"]", raw)
        if quoted:
            subject = re.sub(r"\s+", " ", quoted[0]).strip()
        else:
            norm = " ".join(raw.split())
            match = re.search(
                r"\b(?:show|list|find|get|give|tell|what is|what are|display)\b\s+(?:the\s+)?(.+?)\s+\b(?:record|records|row|rows|item|items|details|info|information)\b",
                norm,
                flags=re.IGNORECASE,
            )
            if not match:
                match = re.search(
                    r"\b(?:about|for|of)\s+(.+?)(?:\s+\bwith\b|\s+\bwhere\b|\s+\bwhose\b|\s+\bthat\b|$)",
                    norm,
                    flags=re.IGNORECASE,
                )
            if match:
                phrase = re.sub(r"\b(the|a|an|this|that|all|exact|exactly|specific)\b", " ", match.group(1), flags=re.IGNORECASE)
                subject = re.sub(r"\s+", " ", phrase).strip(" ,.-")
    payload["subject"] = subject
    roles = _fallback_field_roles(field_rows)
    requested_fields = []
    for role_name in (
        "title_role",
        "ref_role",
        "due_role",
        "status_role",
        "branch_role",
        "assignee_role",
        "reviewer_role",
        "approver_role",
        "amount_role",
        "date_role",
        "category_role",
    ):
        label = roles.get(role_name)
        if label and label not in requested_fields:
            requested_fields.append(label)
    if not requested_fields:
        requested_fields = [
            str((row or {}).get("display") or (row or {}).get("field") or (row or {}).get("name") or "").strip()
            for row in field_rows[:4]
            if str((row or {}).get("display") or (row or {}).get("field") or (row or {}).get("name") or "").strip()
        ]
    payload["requested_fields"] = requested_fields[:6]
    messages = [
        {"role": "system", "content": CLARIFICATION_SUGGESTIONS_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        out = _generate_json_from_messages(model, tokenizer, messages, min(MAX_NEW_TOKENS, 320))
    except Exception:
        out = {}
    message = str(out.get("message") or "").strip() if isinstance(out, dict) else ""
    suggestions = out.get("suggestions") if isinstance(out, dict) else []
    text = " ".join(str(user_input or "").lower().split())
    cleaned = []
    for item in suggestions or []:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text[:120])
    if len(cleaned) < 3:
        fallback_message, fallback_suggestions = _fallback_clarification_suggestions(
            user_input,
            selected_collection,
            table_metadata,
            docs=docs,
            total=payload["total_rows"],
        )
        if not message:
            message = fallback_message
        for item in fallback_suggestions:
            text = str(item or "").strip()
            if text and text not in cleaned:
                cleaned.append(text[:120])
    # Safety filter: keep only suggestions that stay close to the original intent.
    prompt_terms = {term for term in re.split(r"[^a-z0-9]+", str(user_input or "").lower()) if len(term) >= 3}
    subject_terms = {term for term in re.split(r"[^a-z0-9]+", str(subject or "").lower()) if len(term) >= 3}
    requested_terms = {term for term in re.split(r"[^a-z0-9]+", " ".join(requested_fields).lower()) if len(term) >= 3}
    intent_terms = prompt_terms | subject_terms | requested_terms
    anchored = []
    for item in cleaned:
        norm = item.lower()
        if subject and str(subject).lower() in norm:
            anchored.append(item)
            continue
        if requested_fields and any(str(field or "").lower() in norm for field in requested_fields if str(field or "").strip()):
            anchored.append(item)
            continue
        # Require at least one strong overlap with the original prompt or the extracted intent.
        if not prompt_terms:
            anchored.append(item)
            continue
        if any(term in norm for term in intent_terms):
            anchored.append(item)
    if len(anchored) >= 3:
        cleaned = anchored

    def _looks_like_rewrite(text):
        norm = str(text or "").strip().lower()
        if not norm:
            return False
        return norm.startswith(("find ", "show ", "list ", "return ", "get ", "retrieve ", "search "))

    if cleaned and not any(_looks_like_rewrite(item) for item in cleaned):
        _, fallback_suggestions = _fallback_clarification_suggestions(
            user_input,
            selected_collection,
            table_metadata,
            docs=docs,
            total=payload["total_rows"],
        )
        cleaned = list(fallback_suggestions or cleaned)

    safe_message = message.strip()
    fallback_message = "I could not confidently match that request. Use one of the rewritten prompts below."
    if any(term in text for term in prompt_terms):
        fallback_message = "I found matching field hints in the selected table. Use one of the rewritten prompts below."

    if not safe_message:
        safe_message = fallback_message
    else:
        msg_norm = safe_message.lower()
        anchor_terms = set(prompt_terms) | set(subject_terms) | set(requested_terms)
        if anchor_terms and not any(term in msg_norm for term in anchor_terms):
            safe_message = fallback_message
    return {
        "message": safe_message,
        "suggestions": cleaned[:3],
    }


def generate_sidebar_suggestions(
    model,
    tokenizer,
    db_name,
    allowed_collections,
    table_metadata,
    max_items=12,
):
    rows = []
    for name in (allowed_collections or [])[:20]:
        meta = (table_metadata or {}).get(name, {})
        fields = []
        for field in (meta.get("fields") or [])[:12]:
            field_name = str((field or {}).get("field") or "").strip()
            if field_name and not field_name.startswith("_"):
                fields.append(field_name)
        rows.append(
            {
                "collection": name,
                "table": meta.get("template_name", name),
                "business_terms": (meta.get("business_terms") or [])[:10],
                "fields": fields[:8],
            }
        )
    payload = {
        "database": db_name,
        "allowed_collections": rows,
        "target_count": max(4, int(max_items)),
    }
    messages = [
        {"role": "system", "content": SIDEBAR_SUGGESTIONS_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    out = _generate_json_from_messages(model, tokenizer, messages, min(MAX_NEW_TOKENS, 800))
    suggestions = out.get("suggestions") if isinstance(out, dict) else []
    cleaned = []
    for item in suggestions or []:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text[:110])
    return cleaned[: max(1, int(max_items))]


def evaluate_query_scope(
    model,
    tokenizer,
    user_input,
    allowed_collections,
    table_metadata,
    chat_context=None,
):
    rows = []
    for name in (allowed_collections or [])[:24]:
        meta = (table_metadata or {}).get(name, {})
        rows.append(
            {
                "collection": name,
                "table": meta.get("template_name", name),
                "business_terms": (meta.get("business_terms") or [])[:12],
            }
        )
    payload = {
        "question": str(user_input or ""),
        "available_collections": rows,
        "chat_context": recent_chat_context(chat_context or [], limit=4),
    }
    messages = [
        {"role": "system", "content": QUERY_SCOPE_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    out = _generate_json_from_messages(model, tokenizer, messages, min(MAX_NEW_TOKENS, 180))
    allow = bool(out.get("allow", True)) if isinstance(out, dict) else True
    message = ""
    if isinstance(out, dict):
        message = str(out.get("message") or "").strip()
    return {
        "allow": allow,
        "message": message or "I am an ERP Intelligence assistant. I can only help with ERP-related queries.",
    }


def repair_query_plan_on_empty_result(
    model,
    tokenizer,
    user_input,
    db_name,
    table_metadata,
    allowed_collections,
    previous_plan,
    exact_field_candidates=None,
    vector_field_candidates=None,
):
    collection = str(previous_plan.get("collection") or "").strip()
    metadata = table_metadata.get(collection, {})
    context = {
        "database": db_name,
        "allowed_collections": allowed_collections,
        "selected_collection": collection,
        "template_schema": {
            "templateName": metadata.get("template_name", collection),
            "collectionName": collection,
            "fields": (metadata.get("fields") or [])[:120],
        },
        "previous_plan": previous_plan,
        "failure_signal": {"rows_returned": 0},
        "question": str(user_input),
        "reverse_lookup_hints": {
            "exact_field_candidates": exact_field_candidates or {},
            "vector_field_candidates": vector_field_candidates or {},
        },
    }
    messages = [
        {"role": "system", "content": EMPTY_RESULT_REPAIR_PROMPT},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    raw_plan = _generate_json_from_messages(model, tokenizer, messages, MAX_NEW_TOKENS)
    return _normalize_generated_plan(raw_plan, fallback_collection=collection)


def repair_query_plan_on_mismatch_result(
    model,
    tokenizer,
    user_input,
    db_name,
    table_metadata,
    allowed_collections,
    previous_plan,
    docs,
    verifier_message="",
    exact_field_candidates=None,
    vector_field_candidates=None,
):
    collection = str(previous_plan.get("collection") or "").strip()
    metadata = table_metadata.get(collection, {})
    context = {
        "database": db_name,
        "allowed_collections": allowed_collections,
        "selected_collection": collection,
        "template_schema": {
            "templateName": metadata.get("template_name", collection),
            "collectionName": collection,
            "fields": (metadata.get("fields") or [])[:120],
        },
        "question": str(user_input),
        "previous_plan": previous_plan,
        "returned_rows_preview": _build_rows_preview(docs, total=len(docs or []), max_output_tokens=MAX_NEW_TOKENS, min_rows=6, max_rows=20),
        "mismatch_signal": str(verifier_message or "").strip(),
        "reverse_lookup_hints": {
            "exact_field_candidates": exact_field_candidates or {},
            "vector_field_candidates": vector_field_candidates or {},
        },
    }
    messages = [
        {"role": "system", "content": MISMATCH_REPAIR_PROMPT},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    raw_plan = _generate_json_from_messages(
        model,
        tokenizer,
        messages,
        MAX_NEW_TOKENS,
        char_budget=_context_char_budget(MAX_NEW_TOKENS, result_docs=docs, total_rows=len(docs or [])),
    )
    return _normalize_generated_plan(raw_plan, fallback_collection=collection)



from erp_backend.services.query_prompts import (
    TABLE_ROUTER_PROMPT, _ROUTER_STOP_TERMS,
    CLARIFICATION_SUGGESTIONS_PROMPT, EMPTY_RESULT_REPAIR_PROMPT,
    FOLLOW_UP_SUGGESTIONS_PROMPT, MISMATCH_REPAIR_PROMPT, QUERY_PLANNER_PROMPT,
    QUERY_SCOPE_PROMPT, RESULT_CONSTRAINTS_PROMPT, RESULT_SUMMARY_PROMPT,
    RESULT_VERIFIER_PROMPT, SIDEBAR_SUGGESTIONS_PROMPT, SINGLE_PASS_QUERY_PROMPT,
)
from erp_backend.services.query_validate import (
    _has_blocked_operator, _has_hidden_field_reference,
    _normalize_field_path_ref, _normalize_operator_key,
    _coerce_aggregate_stage_key, _normalize_field_key,
    _normalize_plan_value, _normalize_aggregate_pipeline_stages,
    _strip_hidden_field_references, _clean_projection, _clean_sort,
    _validate_collection, _validate_find_plan, _validate_aggregate_plan,
    _sanitize_group_stage, _sanitize_count_stage,
    _infer_sort_by_count_expr, _sanitize_sort_by_count_expr,
    validate_query_plan, execute_plan,
)
