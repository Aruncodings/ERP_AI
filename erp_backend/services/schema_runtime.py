import json
import re

import torch

from erp_backend.core.config import REDIS_SCHEMA_TTL_SECONDS, SYSTEM_FIELD_NAMES
from erp_backend.core.security import is_hidden_field
from erp_backend.core.utils import normalize_lookup_text, similarity, singularize_word
from erp_backend.llm.runtime import get_model_device, is_cuda_inference_error, recover_from_cuda_error
from erp_backend.storage.cache import redis_get_json, redis_set_json


SYSTEM_FIELD_MAPPING_PROMPT = """
You are an ERP schema validator.
Map invalid or business-language field names to real MongoDB fields.
Return only valid JSON. Do not include markdown.

Output schema:
{
  "mappings": {
    "<invalid_field>": "<one available field or empty string>"
  }
}

Rules:
- Choose only from available_fields.
- Use the user request, table name, display labels, and field names.
- Business words can be broad. Example: "person" can map to a name-like display field depending on the available table fields.
- If a field has no safe match, use an empty string.
"""


SYSTEM_FIELDS = set(SYSTEM_FIELD_NAMES) | {"keywords"}


def schema_cache_key(db_name, collection_name):
    return f"erp:schema:{db_name}:{collection_name}"


def build_schema_payload(db_name, collection_name, schema, table_metadata):
    metadata = table_metadata.get(collection_name, {})
    template_fields = metadata.get("fields") or []
    display_by_field = {
        item.get("field"): item.get("display")
        for item in template_fields
        if item.get("field")
    }
    fields = []
    for field, types in schema.items():
        fields.append(
            {
                "field": field,
                "display": display_by_field.get(field, field),
                "types": types,
            }
        )
    for item in template_fields:
        field = item.get("field")
        if field and field not in schema:
            fields.append(
                {
                    "field": field,
                    "display": item.get("display") or field,
                    "type": item.get("type") or "",
                    "lookup_collection": item.get("lookup_collection") or "",
                    "lookup_display": item.get("lookup_display") or "",
                }
            )
    return {
        "database": db_name,
        "collection": collection_name,
        "table": metadata.get("template_name", collection_name),
        "business_terms": metadata.get("business_terms", []),
        "fields": fields,
    }


async def store_collection_schema(db_name, collection_name, schema, table_metadata):
    payload = build_schema_payload(db_name, collection_name, schema, table_metadata)
    await redis_set_json(schema_cache_key(db_name, collection_name), payload, REDIS_SCHEMA_TTL_SECONDS)
    return payload


async def load_cached_collection_schema(db_name, collection_name, schema, table_metadata):
    cached = await redis_get_json(schema_cache_key(db_name, collection_name))
    if cached:
        return cached
    return await store_collection_schema(db_name, collection_name, schema, table_metadata)


def available_field_names(schema_payload, include_hidden=False):
    return [
        item["field"]
        for item in schema_payload.get("fields", [])
        if item.get("field") and (include_hidden or not is_hidden_field(item.get("field")))
    ]


def field_catalog_text(schema_payload):
    lines = []
    for item in schema_payload.get("fields", []):
        field = item.get("field")
        if not field or is_hidden_field(field):
            continue
        display = item.get("display") or field
        field_type = item.get("type") or ", ".join(item.get("types") or [])
        lines.append(f"{field} | {display} | {field_type}")
    return "\n".join(lines[:160])


def normalize_field_reference(field):
    if not isinstance(field, str):
        return field
    if field.startswith("$"):
        return "$" + normalize_field_reference(field[1:])
    return field


def collect_field_references(value):
    references = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key.startswith("$"):
                references.update(collect_field_references(child))
            else:
                references.add(key)
                references.update(collect_field_references(child))
    elif isinstance(value, list):
        for item in value:
            references.update(collect_field_references(item))
    elif isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
        references.add(value[1:].split(".")[0])
    return references


def collect_plan_field_references(plan):
    if not isinstance(plan, dict):
        return set()
    if plan.get("operation") == "aggregate":
        references = set()
        for stage in plan.get("pipeline") or []:
            if not isinstance(stage, dict) or len(stage) != 1:
                continue
            stage_name, stage_body = next(iter(stage.items()))
            if stage_name == "$match":
                references.update(collect_field_references(stage_body))
            elif stage_name in {"$group", "$project", "$addFields", "$set"}:
                references.update(collect_string_field_references(stage_body))
        return references

    references = set()
    references.update(collect_field_references(plan.get("filter") or {}))
    references.update((plan.get("projection") or {}).keys())
    sort_value = plan.get("sort") or []
    if isinstance(sort_value, dict):
        references.update(sort_value.keys())
    elif isinstance(sort_value, list):
        for item in sort_value:
            if isinstance(item, (list, tuple)) and item:
                references.add(str(item[0]))
    return references


def collect_string_field_references(value):
    references = set()
    if isinstance(value, dict):
        for child in value.values():
            references.update(collect_string_field_references(child))
    elif isinstance(value, list):
        for item in value:
            references.update(collect_string_field_references(item))
    elif isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
        references.add(value[1:].split(".")[0])
    return references


def direct_field_match(field, schema_payload):
    available = available_field_names(schema_payload)
    if field in available or field in SYSTEM_FIELDS:
        return field

    normalized_field = normalize_lookup_text(field)
    if not normalized_field:
        return None

    best = None
    best_score = 0
    query_tokens = {
        singularize_word(token)
        for token in normalized_field.split()
        if len(token) >= 2
    }

    for item in schema_payload.get("fields", []):
        candidate = item.get("field")
        if not candidate:
            continue
        haystack = " ".join(
            [
                candidate,
                item.get("display") or "",
                item.get("type") or "",
            ]
        )
        candidate_tokens = {
            singularize_word(token)
            for token in normalize_lookup_text(haystack).split()
            if len(token) >= 2
        }
        score = len(query_tokens & candidate_tokens) * 3
        score += similarity(field, candidate)
        score += similarity(field, item.get("display") or "")
        if normalized_field in normalize_lookup_text(haystack):
            score += 3
        if score > best_score:
            best = candidate
            best_score = score

    return best if best_score >= 2.2 else None


def heuristic_business_field_match(field, user_input, schema_payload):
    text = normalize_lookup_text(f"{field} {user_input}")
    tokens = set(text.split())
    if not (tokens & {"person", "people", "party", "name"}):
        return None

    scored = []
    for item in schema_payload.get("fields", []):
        candidate = item.get("field")
        display = item.get("display") or ""
        haystack = normalize_lookup_text(f"{candidate} {display}")
        score = 0
        candidate_tokens = set(haystack.split())
        if candidate_tokens & {"name", "title", "label"}:
            score += 5
        score += 3 * len(tokens & candidate_tokens)
        if "." not in candidate and candidate not in SYSTEM_FIELDS and score:
            scored.append((score, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def generate_field_mappings(model, tokenizer, user_input, invalid_fields, schema_payload):
    if not invalid_fields:
        return {}
    messages = [
        {"role": "system", "content": SYSTEM_FIELD_MAPPING_PROMPT},
        {
            "role": "user",
            "content": (
                f"User request:\n{user_input}\n\n"
                f"Table:\n{schema_payload.get('table')}\n\n"
                f"Available fields:\n{field_catalog_text(schema_payload)}\n\n"
                f"Invalid fields to map:\n{json.dumps(sorted(invalid_fields), ensure_ascii=False)}"
            ),
        },
    ]
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        response_payload = model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=180,
        )
        response = (
            response_payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    else:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        model_device = get_model_device(model)
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    response = re.sub(r"^```(?:json)?", "", response, flags=re.IGNORECASE).strip()
    response = re.sub(r"```$", "", response).strip()
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        payload = json.loads(match.group(0)) if match else {}
    mappings = payload.get("mappings") if isinstance(payload, dict) else {}
    if not isinstance(mappings, dict):
        return {}
    available = set(available_field_names(schema_payload))
    return {
        str(source): str(target)
        for source, target in mappings.items()
        if source in invalid_fields and target in available
    }


def replace_field_references(value, mappings, replace_keys=True):
    if isinstance(value, dict):
        replaced = {}
        for key, child in value.items():
            new_key = mappings.get(key, key) if replace_keys and not key.startswith("$") else key
            replaced[new_key] = replace_field_references(child, mappings, replace_keys=replace_keys)
        return replaced
    if isinstance(value, list):
        return [replace_field_references(item, mappings, replace_keys=replace_keys) for item in value]
    if isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
        field = value[1:].split(".")[0]
        if field in mappings:
            suffix = value[1 + len(field):]
            return f"${mappings[field]}{suffix}"
    return value


def replace_plan_field_references(plan, mappings):
    if not mappings:
        return plan
    if plan.get("operation") != "aggregate":
        return replace_field_references(plan, mappings, replace_keys=True)

    repaired = dict(plan)
    pipeline = []
    for stage in plan.get("pipeline") or []:
        if not isinstance(stage, dict) or len(stage) != 1:
            pipeline.append(stage)
            continue
        stage_name, stage_body = next(iter(stage.items()))
        if stage_name == "$match":
            pipeline.append({stage_name: replace_field_references(stage_body, mappings, replace_keys=True)})
        elif stage_name in {"$group", "$project", "$addFields", "$set"}:
            pipeline.append({stage_name: replace_field_references(stage_body, mappings, replace_keys=False)})
        else:
            pipeline.append(stage)
    repaired["pipeline"] = pipeline
    return repaired


async def validate_and_repair_schema_plan(
    model,
    tokenizer,
    db_name,
    collection_name,
    plan,
    user_input,
    schema,
    table_metadata,
):
    schema_payload = await load_cached_collection_schema(db_name, collection_name, schema, table_metadata)
    if not isinstance(plan, dict) or plan.get("needs_clarification"):
        return plan, {}

    available = set(available_field_names(schema_payload, include_hidden=True))
    references = collect_plan_field_references(plan)
    invalid = {
        field
        for field in references
        if field not in available
        and field not in SYSTEM_FIELDS
        and not field.startswith("_id.")
    }
    if not invalid:
        return plan, {}

    mappings = {}
    for field in sorted(invalid):
        mappings[field] = (
            direct_field_match(field, schema_payload)
            or heuristic_business_field_match(field, user_input, schema_payload)
            or ""
        )

    unresolved = {field for field, target in mappings.items() if not target}
    if unresolved:
        try:
            llm_mappings = generate_field_mappings(model, tokenizer, user_input, unresolved, schema_payload)
        except RuntimeError as exc:
            if not is_cuda_inference_error(exc):
                raise
            recover_from_cuda_error()
            llm_mappings = {}
        mappings.update(llm_mappings)

    mappings = {field: target for field, target in mappings.items() if target in available}
    repaired = replace_plan_field_references(plan, mappings) if mappings else plan
    return repaired, mappings
