import json
import re

from pymongo import ASCENDING, DESCENDING

from erp_backend.core.config import ALLOWED_AGGREGATE_STAGES, BLOCKED_OPERATORS, BLOCKED_STAGES, COUNT_TOTAL_EXACT, MAX_RESULT_ROWS
try:
    from erp_backend.core.config import QUERY_TIMEOUT_MS
except ImportError:
    QUERY_TIMEOUT_MS = 8000
from erp_backend.core.security import is_hidden_field, sanitize_doc_for_display
from erp_backend.core.utils import to_jsonable, normalize_lookup_text
from erp_backend.storage.mongo import collection_is_allowed, estimated_count, mongo_client


def _has_blocked_operator(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in BLOCKED_OPERATORS:
                return True
            if _has_blocked_operator(child):
                return True
    elif isinstance(value, list):
        return any(_has_blocked_operator(item) for item in value)
    return False


def _has_hidden_field_reference(value, allow_internal_id=False):
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if (
                not key_text.startswith("$")
                and not (allow_internal_id and key_text == "_id")
                and is_hidden_field(key_text)
            ):
                return True
            if _has_hidden_field_reference(child, allow_internal_id=allow_internal_id):
                return True
    elif isinstance(value, list):
        return any(
            _has_hidden_field_reference(item, allow_internal_id=allow_internal_id)
            for item in value
        )
    elif isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
        field_ref = value[1:].split(".")[0]
        if allow_internal_id and field_ref == "_id":
            return False
        return is_hidden_field(field_ref)
    return False


def _normalize_field_path_ref(value):
    text = str(value or "").strip()
    if not text.startswith("$") or text.startswith("$$"):
        return value
    body = text[1:]
    body = body.strip("`\"' \u2018\u2019\u201c\u201d\t\r\n,;:")
    if not body:
        return value
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)", body)
    if not match:
        return value
    return f"${match.group(1)}"


def _normalize_operator_key(key):
    key_text = str(key or "").strip()
    if not key_text.startswith("$") or key_text.startswith("$$"):
        return key_text
    body = key_text[1:]
    body = body.strip("`\"' \u2018\u2019\u201c\u201d\t\r\n,;:")
    if not body:
        return key_text
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)", body)
    if not match:
        return key_text
    return f"${match.group(1)}"


STAGE_NAME_ALIASES = {
    "group": "$group", "groupby": "$group",
    "match": "$match",
    "project": "$project",
    "sort": "$sort",
    "limit": "$limit",
    "skip": "$skip",
    "unwind": "$unwind",
    "count": "$count",
    "lookup": "$lookup",
    "sample": "$sample",
    "facet": "$facet",
    "bucket": "$bucket", "bucketauto": "$bucketAuto",
    "addfields": "$addFields", "set": "$set", "unset": "$unset",
    "sortbycount": "$sortByCount", "replaceroot": "$replaceRoot",
    "replacewith": "$replaceWith", "densify": "$densify",
    "fill": "$fill", "redact": "$redact",
}

def _coerce_aggregate_stage_key(key):
    key_text = str(key or "").strip()
    if not key_text:
        return ""
    if key_text.startswith("$"):
        return _normalize_operator_key(key_text)
    lowered = key_text.lower()
    alias = STAGE_NAME_ALIASES.get(key_text) or STAGE_NAME_ALIASES.get(lowered)
    if alias:
        return alias
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)", key_text)
    if not match:
        return ""
    return f"${match.group(1)}"


def _normalize_field_key(key):
    key_text = str(key or "").strip()
    key_text = key_text.strip("`\"' \u2018\u2019\u201c\u201d\t\r\n,;:")
    key_text = key_text.strip()
    if not key_text:
        return key_text
    parts = []
    for part in key_text.split("."):
        token = str(part).strip("`\"' \u2018\u2019\u201c\u201d\t\r\n,;: ").strip()
        if not token:
            continue
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)", token)
        parts.append(match.group(1) if match else token)
    return ".".join(parts)


def _normalize_plan_value(value):
    if isinstance(value, dict):
        normalized = {}
        for key, child in value.items():
            key_text = str(key).strip()
            if key_text.startswith("$"):
                op = _normalize_operator_key(key_text)
                if op == "$size":
                    child_norm = _normalize_plan_value(child)
                    if isinstance(child_norm, dict) and next(iter(child_norm.keys())) == "$ifNull":
                        normalized[op] = child_norm
                    else:
                        normalized[op] = {"$ifNull": [child_norm, []]}
                else:
                    normalized[op] = _normalize_plan_value(child)
            else:
                key_text = _normalize_field_key(key_text)
                if not key_text:
                    continue
                normalized[key_text] = _normalize_plan_value(child)
        return normalized
    if isinstance(value, list):
        return [_normalize_plan_value(item) for item in value]
    if isinstance(value, str):
        return _normalize_field_path_ref(value)
    return value


def _normalize_aggregate_pipeline_stages(pipeline):
    if not isinstance(pipeline, list):
        return []
    normalized = []
    for stage in pipeline:
        if not isinstance(stage, dict):
            continue
        stage_items = list(stage.items())
        if not stage_items:
            continue
        for stage_name_raw, stage_body in stage_items:
            stage_name = _coerce_aggregate_stage_key(stage_name_raw)
            if not stage_name:
                continue
            normalized.append({stage_name: _normalize_plan_value(stage_body)})
    return normalized


def _strip_hidden_field_references(value, allow_internal_id=False, in_let_scope=False):
    if isinstance(value, dict):
        cleaned = {}
        for key, child in value.items():
            key_text = str(key)
            if key_text.startswith("$"):
                cleaned_child = _strip_hidden_field_references(
                    child,
                    allow_internal_id=allow_internal_id,
                    in_let_scope=in_let_scope,
                )
                cleaned[key_text] = cleaned_child
                continue
            if key_text == "let":
                cleaned[key_text] = _strip_hidden_field_references(
                    child,
                    allow_internal_id=allow_internal_id,
                    in_let_scope=True,
                )
                continue
            if allow_internal_id and key_text == "_id":
                cleaned[key_text] = _strip_hidden_field_references(
                    child,
                    allow_internal_id=allow_internal_id,
                    in_let_scope=in_let_scope,
                )
                continue
            if not in_let_scope and is_hidden_field(key_text):
                continue
            cleaned[key_text] = _strip_hidden_field_references(
                child,
                allow_internal_id=allow_internal_id,
                in_let_scope=in_let_scope,
            )
        return cleaned
    if isinstance(value, list):
        return [
            _strip_hidden_field_references(
                item,
                allow_internal_id=allow_internal_id,
                in_let_scope=in_let_scope,
            )
            for item in value
        ]
    if isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
        field_ref = value[1:].split(".")[0]
        if allow_internal_id and field_ref == "_id":
            return value
        if is_hidden_field(field_ref):
            return None
    return value


def _clean_projection(projection):
    if not isinstance(projection, dict):
        return {}
    cleaned = {}
    for key, value in projection.items():
        if is_hidden_field(str(key)):
            continue
        if value in (0, 1, True, False):
            cleaned[str(key)] = int(value)
    return cleaned


def _clean_sort(sort_value):
    if isinstance(sort_value, dict):
        items = list(sort_value.items())
    elif isinstance(sort_value, list):
        items = sort_value
    else:
        return []

    cleaned = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            field, direction = item
        else:
            continue
        field_name = str(field).strip("`\"' \u2018\u2019\u201c\u201d \t\r\n,;:")
        if not field_name or field_name.startswith("$"):
            continue
        if is_hidden_field(field_name):
            continue
        cleaned.append((field_name, DESCENDING if int(direction) < 0 else ASCENDING))
    return cleaned


def _validate_collection(plan, allowed_collections):
    collection = str(plan.get("collection") or "").strip()
    if not collection:
        raise ValueError("collection is required in query plan")
    if not collection_is_allowed(collection, allowed_collections):
        raise PermissionError(f"collection not allowed: {collection}")
    return collection


def _validate_find_plan(plan, collection):
    filter_doc = _normalize_plan_value(plan.get("filter") or {})
    filter_doc = _strip_hidden_field_references(filter_doc)
    if not isinstance(filter_doc, dict):
        raise ValueError("filter must be an object")
    if _has_blocked_operator(filter_doc):
        raise ValueError("blocked MongoDB operator found in filter")
    if _has_hidden_field_reference(filter_doc):
        raise ValueError("hidden fields cannot be queried")

    projection = _clean_projection(plan.get("projection") or {})
    if _has_hidden_field_reference(projection):
        raise ValueError("hidden fields cannot be projected")

    sort = _clean_sort(plan.get("sort") or [])
    if _has_hidden_field_reference(sort):
        raise ValueError("hidden fields cannot be used in sort")

    return {
        "operation": "find",
        "collection": collection,
        "filter": filter_doc,
        "projection": projection,
        "sort": sort,
    }


def _validate_aggregate_plan(plan, collection, allowed_collections):
    pipeline = _normalize_plan_value(plan.get("pipeline") or [])
    pipeline = _normalize_aggregate_pipeline_stages(pipeline)

    safe_pipeline = []
    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) != 1:
            continue
        stage_name_raw = next(iter(stage))
        stage_name = _normalize_operator_key(stage_name_raw)
        stage_body = stage[stage_name_raw]
        stage = {stage_name: _normalize_plan_value(stage_body)}
        if stage_name == "$lookup" and isinstance(stage_body, dict):
            from_col = stage_body.get("from")
            if from_col:
                resolved_col = _resolve_target_collection(from_col, allowed_collections)
                stage_body["from"] = resolved_col
                stage["$lookup"] = stage_body
        if stage_name == "$group":
            stage["$group"] = _sanitize_group_stage(stage.get("$group"))
        elif stage_name == "$count":
            stage["$count"] = _sanitize_count_stage(stage.get("$count"))
        elif stage_name == "$sortByCount":
            stage["$sortByCount"] = _sanitize_sort_by_count_expr(
                stage.get("$sortByCount"),
                safe_pipeline,
            )
        if stage_name in BLOCKED_STAGES:
            raise ValueError(f"blocked aggregation stage: {stage_name}")
        if stage_name not in ALLOWED_AGGREGATE_STAGES:
            raise ValueError(f"unrecognized aggregation stage: {stage_name}")
        if _has_blocked_operator(stage):
            raise ValueError("blocked MongoDB operator found in pipeline")
        stage = _strip_hidden_field_references(stage, allow_internal_id=True)
        if _has_hidden_field_reference(stage, allow_internal_id=True):
            continue
        if stage_name in {"$skip", "$limit"}:
            continue
        safe_pipeline.append(stage)

    return {
        "operation": "aggregate",
        "collection": collection,
        "pipeline": safe_pipeline,
    }


def _sanitize_group_stage(group_body):
    if not isinstance(group_body, dict):
        raise ValueError("$group stage body must be an object")

    safe = dict(group_body)
    if "_id" not in safe:
        safe["_id"] = None

    def _is_accumulator_object(value):
        if not isinstance(value, dict) or not value:
            return False
        first_key = next(iter(value.keys()))
        return str(first_key).startswith("$")

    for key, value in list(safe.items()):
        if key == "_id":
            continue
        if _is_accumulator_object(value):
            continue
        key_norm = str(key).strip().lower()
        if key_norm in {"count", "total", "qty", "quantity"}:
            safe[key] = {"$sum": 1}
        else:
            safe[key] = {"$first": value}
    return safe


def _sanitize_count_stage(count_field):
    text = str(count_field or "").strip()
    if not text:
        return "count"
    text = text.lstrip("$").strip()
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    text = text.strip("_")
    return text or "count"


def _infer_sort_by_count_expr(prior_pipeline):
    for stage in reversed(prior_pipeline or []):
        if not isinstance(stage, dict) or len(stage) != 1:
            continue
        op = next(iter(stage.keys()))
        body = stage.get(op)
        if op == "$project" and isinstance(body, dict):
            for key in body.keys():
                field = str(key or "").strip()
                if field and field != "_id":
                    return f"${field}"
        if op == "$group" and isinstance(body, dict):
            return "$_id"
    return None


def _sanitize_sort_by_count_expr(expr, prior_pipeline):
    if isinstance(expr, dict) and expr:
        return expr
    if isinstance(expr, str):
        text = expr.strip()
        if text and text != "$":
            return text if text.startswith("$") else f"${text}"
    inferred = _infer_sort_by_count_expr(prior_pipeline)
    return inferred or "$_id"


def _resolve_target_collection(target_name, allowed_collections):
    if not target_name or not allowed_collections:
        return target_name
    
    target_clean = str(target_name).strip().lower()
    for col in allowed_collections:
        if col.lower() == target_clean:
            return col
            
    def clean(s):
        s_norm = normalize_lookup_text(s)
        if s_norm.endswith("s") and len(s_norm) > 2:
            return s_norm[:-1]
        return s_norm

    clean_target = clean(target_clean)
    
    for col in allowed_collections:
        col_clean = col.lower()
        if col_clean.startswith(f"{clean_target}_template_") or col_clean.startswith(f"{target_clean}_template_"):
            return col
            
    for col in allowed_collections:
        col_clean = col.lower()
        if col_clean.startswith(clean_target) or col_clean.startswith(target_clean):
            return col
            
    for col in allowed_collections:
        if clean_target in col.lower() or target_clean in col.lower():
            return col
            
    return target_name


def validate_query_plan(plan, allowed_collections):
    if not isinstance(plan, dict):
        raise ValueError("query plan must be a JSON object")
    if plan.get("needs_clarification"):
        return plan

    collection = _validate_collection(plan, allowed_collections)
    operation = str(plan.get("operation") or "find").lower().strip()
    if operation == "aggregate":
        return _validate_aggregate_plan(plan, collection, allowed_collections)
    return _validate_find_plan(plan, collection)


def _extract_group_fields(pipeline):
    """Extract field names from $group stage _id expression."""
    if not isinstance(pipeline, list):
        return []
    group_fields = []
    for stage in pipeline:
        if not isinstance(stage, dict):
            continue
        for key, value in stage.items():
            normalized_key = key.strip().lstrip("$")
            if normalized_key == "group":
                group_body = value
                if isinstance(group_body, dict) and "_id" in group_body:
                    group_id = group_body["_id"]
                    if isinstance(group_id, str):
                        group_fields.append(group_id.lstrip("$"))
                    elif isinstance(group_id, dict):
                        for field_name in group_id.keys():
                            if field_name != "_id":
                                group_fields.append(field_name)
    return group_fields


def _preserve_group_keys(doc, group_fields):
    """Copy _id value to actual group field names before sanitization."""
    if not isinstance(doc, dict) or not group_fields:
        return doc
    if "_id" not in doc:
        return doc
    group_value = doc["_id"]
    preserved = dict(doc)
    if isinstance(group_value, dict):
        for field in group_fields:
            if field and field not in preserved and field in group_value:
                preserved[field] = group_value[field]
    else:
        for field in group_fields:
            if field and field not in preserved:
                preserved[field] = group_value
    return preserved


def _wrap_size_with_ifnull(value):
    if isinstance(value, dict):
        cleaned = {}
        for k, v in value.items():
            if k == "$size" and isinstance(v, (str, dict)):
                if isinstance(v, dict) and next(iter(v)) == "$ifNull":
                    cleaned[k] = _wrap_size_with_ifnull(v)
                else:
                    cleaned[k] = {"$ifNull": [_wrap_size_with_ifnull(v), []]}
            else:
                cleaned[k] = _wrap_size_with_ifnull(v)
        return cleaned
    if isinstance(value, list):
        return [_wrap_size_with_ifnull(item) for item in value]
    return value


async def execute_plan(db_name, plan):
    collection_name = plan["collection"]
    collection = mongo_client()[db_name][collection_name]

    if plan["operation"] == "aggregate":
        pipeline = list(plan.get("pipeline") or [])
        pipeline = _wrap_size_with_ifnull(pipeline)
        group_fields = _extract_group_fields(pipeline)
        pipeline.append({"$limit": MAX_RESULT_ROWS})
        cursor = collection.aggregate(pipeline, allowDiskUse=False)
        docs = []
        async for doc in cursor:
            doc = to_jsonable(doc)
            doc = _preserve_group_keys(doc, group_fields)
            docs.append(sanitize_doc_for_display(doc))
        return docs, len(docs)

    filter_doc = plan.get("filter") or {}
    projection = plan.get("projection") or None
    cursor = collection.find(filter_doc, projection)
    sort_value = plan.get("sort") or []
    if sort_value:
        cursor = cursor.sort(sort_value)
    cursor = cursor.limit(MAX_RESULT_ROWS)

    docs = []
    async for doc in cursor:
        docs.append(sanitize_doc_for_display(to_jsonable(doc)))

    if len(docs) < MAX_RESULT_ROWS:
        total = len(docs)
    elif COUNT_TOTAL_EXACT:
        total = await estimated_count(db_name, collection_name, filter_doc)
    else:
        total = len(docs)

    return docs, total
