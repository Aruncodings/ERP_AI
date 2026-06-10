import asyncio
import json
import re
from datetime import datetime, timezone

from bson import ObjectId

from erp_backend.core.config import (
    CHAT_CONTEXT_LIMIT,
    CHAT_CONTEXT_CHARS,
    LLM_COLLECTION_CANDIDATES,
    LLM_FIELD_CANDIDATES,
    LLM_SUMMARY_MODE,
    SYSTEM_FIELD_NAMES,
    VECTOR_DISTINCT_ENABLED,
    VECTOR_DISTINCT_MAX_FIELDS_PER_COLLECTION,
    VECTOR_DISTINCT_MAX_VALUES_PER_FIELD,
    VECTOR_DISTINCT_MAX_VALUE_LENGTH,
)
from erp_backend.core.utils import (
    lookup_tokens,
    normalize_lookup_text,
)
from erp_backend.services.field_retriever import retrieve_candidates
from erp_backend.services.query import (
    evaluate_query_scope,
)
from erp_backend.services.query_rewriter import (
    remap_plan_runtime_fields,
)
from erp_backend.storage.mongo import (
    mongo_client,
)


STRICT_SCHEMA_SYSTEM_FIELDS = set(SYSTEM_FIELD_NAMES)
_RUNTIME_SYSTEM_FIELDS = set()


def _effective_system_fields():
    return set(STRICT_SCHEMA_SYSTEM_FIELDS) | set(_RUNTIME_SYSTEM_FIELDS)


def _light_normalize_prompt(prompt):
    text = " ".join(str(prompt or "").strip().split())
    return text


def _normalize_term(text):
    return normalize_lookup_text(text)


def _normalize_field_token(text):
    return re.sub(r"[^a-z0-9]+", "", normalize_lookup_text(text))


def _field_token_parts(text):
    return [part for part in lookup_tokens(text) if part]


def _humanize_runtime_field_name(field_name):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(field_name or "").strip())
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(part.capitalize() for part in text.split())


def _singular(term):
    if term.endswith("ies") and len(term) > 3:
        return term[:-3] + "y"
    if term.endswith("es") and len(term) > 3:
        return term[:-2]
    if term.endswith("s") and len(term) > 2:
        return term[:-1]
    return term


def _tokenize_norm(text):
    return [token for token in _normalize_term(text).split() if token]


def _tokenize(text):
    return lookup_tokens(text)


def _field_type_is_array_like(field_meta):
    ftype = _normalize_term((field_meta or {}).get("type") or "")
    return any(token in ftype for token in ("select", "lookup", "multi"))


def _build_collection_alias_signals(collection_name, meta):
    signals = {}

    def add_signal(raw, kind):
        norm = _normalize_term(raw)
        if not norm:
            return
        parts = [p for p in norm.split() if p]
        if not parts:
            return
        while parts and re.fullmatch(r"[0-9a-f]{6,}", parts[-1]):
            parts = parts[:-1]
        if not parts:
            return
        values = [norm, " ".join(parts)]
        for n in range(1, min(3, len(parts)) + 1):
            values.append(" ".join(parts[:n]))
        for alias in values:
            alias = _normalize_term(alias)
            if len(alias) < 3:
                continue
            singular = _singular(alias)
            candidates = {alias}
            if singular:
                candidates.add(singular)
            for candidate in candidates:
                prev = signals.get(candidate)
                if prev is None or (prev == "business" and kind in {"collection", "template"}):
                    signals[candidate] = kind

    add_signal(collection_name, "collection")
    add_signal((meta or {}).get("template_name") or "", "template")
    for term in (meta or {}).get("business_terms") or []:
        add_signal(term, "business")
    return signals


def _collection_specificity_penalty(prompt, meta):
    normalized = _normalize_term(prompt)
    if not normalized:
        return 0
    template_name = _normalize_term((meta or {}).get("template_name") or "")
    if not template_name:
        return 0
    tokens = [token for token in template_name.split() if token]
    if len(tokens) <= 1:
        return 0
    if re.search(rf"\b{re.escape(template_name)}\b", normalized):
        return 0
    prompt_terms = set(normalized.split())
    matched = sum(1 for token in tokens if token in prompt_terms)
    if matched <= 0:
        return 0
    return max(0, len(tokens) - matched) * 3


def _collection_specificity_rank(meta):
    template_name = _normalize_term((meta or {}).get("template_name") or "")
    if not template_name:
        return (99, 99)
    tokens = [token for token in template_name.split() if token]
    return (len(tokens), len(template_name))


def _mentioned_entities(prompt, accessible_collections, table_metadata=None):
    normalized = _normalize_term(prompt)
    words = set(normalized.split())
    entities = []
    for name in accessible_collections:
        token = _normalize_term(name).replace("_", " ")
        if not token:
            continue
        parts = [token, _singular(token)]
        for part in parts:
            part_words = part.split()
            if len(part_words) == 1:
                if part_words[0] in words:
                    entities.append(name)
                    break
            else:
                if re.search(rf"\b{re.escape(part)}\b", normalized):
                    entities.append(name)
                    break
    # Field display/alias matches: if a field display/alias uniquely identifies a collection
    if table_metadata:
        field_to_collection = {}
        for name in accessible_collections:
            meta = (table_metadata or {}).get(name) or {}
            for field in (meta.get("fields") or []):
                display = _normalize_term(field.get("display") or "")
                for alias in (field.get("aliases") or []):
                    alias_norm = _normalize_term(alias)
                    for label in {display, alias_norm}:
                        if label and len(label) >= 3:
                            field_to_collection.setdefault(label, []).append(name)
        for term in words:
            if term in field_to_collection and len(field_to_collection[term]) == 1:
                unique_col = field_to_collection[term][0]
                if unique_col not in entities:
                    entities.append(unique_col)
    return list(dict.fromkeys(entities))


_VALUE_STOP_WORDS = {
    "current", "show", "list", "get", "all", "the", "a", "an", "of", "for",
    "with", "and", "or", "to", "from", "in", "on", "by", "is", "are",
    "who", "what", "when", "where", "which", "me", "give", "find", "search",
    "how", "many", "detail", "details", "information", "tell", "about",
    "need", "like", "want", "give", "known", "know", "any", "every",
    "could", "would", "should", "can", "do", "does", "did", "has", "have",
    "been", "being", "was", "were", "its", "it", "this", "that", "these",
    "those", "some", "any", "each", "both", "few", "more", "most",
    "other", "into", "over", "such", "only", "own", "same", "than",
    "too", "very", "just", "also",
}


def _normalize_value_token(text):
    return normalize_lookup_text(text)


def _extract_text_value_terms(text, accessible_collections, table_metadata):
    normalized = _normalize_value_token(text)
    if not normalized:
        return []
    tokens = [t for t in normalized.split() if len(t) >= 3]

    known_terms = set()
    for name in accessible_collections or []:
        known_terms.add(_normalize_value_token(name.split("_template_")[0] if "_template_" in name else name))
        meta = (table_metadata or {}).get(name) or {}
        known_terms.add(_normalize_value_token((meta.get("template_name") or "")))
        for field in (meta.get("fields") or []):
            fname = str(field.get("field") or "").strip()
            if fname:
                known_terms.add(_normalize_value_token(fname))
            display = str(field.get("display") or "").strip()
            if display:
                known_terms.add(_normalize_value_token(display))

    value_terms = []
    seen = set()
    for token in tokens:
        if token in _VALUE_STOP_WORDS:
            continue
        if token in known_terms:
            continue
        low = token.lower()
        if low in seen:
            continue
        seen.add(low)
        value_terms.append(token)

    return value_terms


def _extract_identifier_values(text):
    raw = str(text or "").strip().lower()
    if not raw:
        return set()
    values = set()
    for token in re.split(r'[\s,;]+', raw):
        token = token.strip()
        if not token or len(token) < 4:
            continue
        if re.search(r'^[a-z]{2,}[-_/][a-z0-9-]{2,}$', token):
            values.add(token)
    normalized = _normalize_term(text)
    if normalized:
        for token in re.split(r'\s+', normalized):
            if not token:
                continue
            if re.search(r'^[a-z]{2,}[\d]{3,}$', token) and not re.search(r'^[a-z]+$', token):
                values.add(token)
    return values


def _collection_metadata_text(collection, meta):
    parts = [
        _normalize_term(collection),
        _normalize_term((meta or {}).get("template_name") or ""),
    ]
    parts.extend(_normalize_term(t) for t in ((meta or {}).get("business_terms") or []))
    for field in ((meta or {}).get("fields") or []):
        name = str(field.get("field") or "").strip().lower()
        display = str(field.get("display") or "").strip().lower()
        if name:
            parts.append(name)
        if display:
            parts.append(display)
    return " ".join(parts)


def _value_prefix_collection_boost(identifier_values, collections, table_metadata):
    prefixes = set()
    for token in identifier_values:
        prefix = re.split(r'[-_/]', token)[0]
        if len(prefix) >= 2:
            prefixes.add(prefix)
    if not prefixes:
        return {}

    boosts = {}
    for collection in collections or []:
        meta = (table_metadata or {}).get(collection) or {}
        collection_norm = _normalize_term(collection)
        template_norm = _normalize_term((meta or {}).get("template_name") or "")
        for prefix in prefixes:
            if (collection_norm.startswith(prefix) or template_norm.startswith(prefix)):
                boosts[collection] = 25
                break
    return boosts


def _infer_requested_collections(prompt, collections, table_metadata):
    normalized = _normalize_term(prompt)
    if not normalized:
        return []
    prompt_terms = set(normalized.split())
    token_frequency = {}
    alias_signal_map = {}
    for collection in collections or []:
        meta = (table_metadata or {}).get(collection) or {}
        alias_signals = _build_collection_alias_signals(collection, meta)
        alias_signal_map[collection] = alias_signals
        for alias, kind in alias_signals.items():
            if kind not in {"collection", "template"}:
                continue
            tokens = alias.split()
            if len(tokens) == 1:
                token = tokens[0]
                token_frequency[token] = token_frequency.get(token, 0) + 1

    hits = []
    for collection in collections or []:
        meta = (table_metadata or {}).get(collection) or {}
        alias_signals = alias_signal_map.get(collection) or {}
        score = 0
        _prefix_awarded = False
        for alias, kind in alias_signals.items():
            if not alias or len(alias) < 3:
                continue
            is_primary = kind in {"collection", "template"}
            exact_weight = 30 if is_primary else 14
            phrase_weight = 20 if is_primary else 8
            unique_token_weight = 14 if is_primary else 6
            shared_token_weight = 7 if is_primary else 3
            if normalized == alias:
                score += exact_weight
            elif re.search(rf"\b{re.escape(alias)}\b", normalized):
                score += phrase_weight
            else:
                alias_tokens = alias.split()
                if len(alias_tokens) == 1 and alias_tokens[0] in prompt_terms:
                    token = alias_tokens[0]
                    if token_frequency.get(token, 0) == 1:
                        score += unique_token_weight
                    else:
                        score += shared_token_weight
            if is_primary and normalized.startswith(alias + " "):
                score += 4
            if is_primary and len(alias) >= 4 and not _prefix_awarded:
                for pt in prompt_terms:
                    if len(pt) >= 3 and alias.startswith(pt):
                        score += 5
                        _prefix_awarded = True
                        break
        # Field display/alias scoring: field matches override collection name matches
        for field in (meta.get("fields") or []):
            field_name = _normalize_term(field.get("field") or "")
            display = _normalize_term(field.get("display") or "")
            aliases = [_normalize_term(a) for a in (field.get("aliases") or [])]
            all_labels = {field_name, display} | set(aliases)
            all_labels.discard("")
            for label in all_labels:
                if not label:
                    continue
                if label in normalized:
                    score += 25
                elif any(pt in label for pt in prompt_terms if len(pt) >= 3):
                    score += 10
        score = max(0, score - _collection_specificity_penalty(normalized, meta))
        if score > 0:
            hits.append((collection, score))

    identifier_values = _extract_identifier_values(prompt)
    if identifier_values:
        value_boosts = _value_prefix_collection_boost(identifier_values, collections, table_metadata)
        hit_names = {name for name, _ in hits}
        for collection in collections or []:
            boost = value_boosts.get(collection, 0)
            if boost > 0 and collection not in hit_names:
                hits.append((collection, 15))
        hits = [
            (name, score + value_boosts.get(name, 0))
            for name, score in hits
        ]

    hits.sort(key=lambda item: (-item[1], _collection_specificity_rank((table_metadata or {}).get(item[0]) or {})))
    return [name for name, score in hits if score >= 10][:3]


def _explicit_collection_alias_choice(prompt, accessible_collections, table_metadata):
    matches = _infer_requested_collections(prompt, accessible_collections, table_metadata)
    if not matches:
        return None
    return {
        "collection": matches[0],
        "reason": "Selected by explicit template/collection alias match.",
    }


def _expand_alias_tokens(source):
    expanded = set()
    if not source:
        return expanded
    normed = normalize_lookup_text(source)
    if normed:
        expanded.add(normed)
        expanded.add(normed.replace(" ", ""))
        for token in normed.split():
            if len(token) >= 2:
                expanded.add(token)
    return expanded


def _build_schema_alias_index(schema_rows):
    index = {}
    for row in schema_rows or []:
        name = str((row or {}).get("name") or "").strip()
        if not name:
            continue
        aliases = _expand_alias_tokens(name)
        display = str((row or {}).get("display") or "").strip()
        if display:
            aliases.update(_expand_alias_tokens(display))
        for alias in (row or {}).get("aliases") or []:
            text = str(alias or "").strip()
            if text:
                aliases.update(_expand_alias_tokens(text))
        index[name] = aliases
    return index


def _guess_schema_field(field_name, allowed_fields, alias_index):
    target = str(field_name or "").strip()
    if not target:
        return None
    if target in allowed_fields or target in _effective_system_fields():
        return target

    root = target.split(".")[0]
    root_norm = _normalize_field_token(root)
    if not root_norm:
        return None

    best = None
    best_score = 0
    second_best = 0
    root_parts = set(_field_token_parts(root))
    for candidate in allowed_fields:
        cand = str(candidate or "").strip()
        if not cand:
            continue
        score = 0
        for alias in alias_index.get(cand, {cand}):
            alias_norm = _normalize_field_token(alias)
            alias_parts = set(_field_token_parts(alias))
            alias_score = 0
            if alias_norm == root_norm:
                alias_score += 24
            if alias_norm.endswith(root_norm) or root_norm.endswith(alias_norm):
                alias_score += 8
            if root_norm in alias_norm or alias_norm in root_norm:
                alias_score += 6
            overlap = len(root_parts & alias_parts)
            if overlap:
                alias_score += overlap * 5
            if not overlap:
                for rp in root_parts:
                    for ap in alias_parts:
                        if len(rp) >= 2 and len(ap) >= 2 and (rp in ap or ap in rp):
                            alias_score += 3
                            break
            if alias_score > score:
                score = alias_score
        if score > best_score:
            second_best = best_score
            best_score = score
            best = cand
        elif score > second_best:
            second_best = score
    if best_score >= 6 and (best_score - second_best) >= 1:
        return best
    return None


def _align_plan_fields_to_schema(plan, schema_index):
    if not isinstance(plan, dict) or plan.get("needs_clarification"):
        return plan
    collection = str(plan.get("collection") or "").strip()
    schema_rows = list((schema_index.get(collection, {}) or {}).get("fields") or [])
    allowed_fields = {
        str(item.get("name") or "").strip()
        for item in schema_rows
        if str(item.get("name") or "").strip()
    }
    if not allowed_fields:
        return plan
    alias_index = _build_schema_alias_index(schema_rows)

    aggregate_param_keys = {
        "from",
        "localField",
        "foreignField",
        "as",
        "let",
        "pipeline",
        "input",
        "in",
        "cond",
        "if",
        "then",
        "else",
        "vars",
        "branches",
        "case",
        "default",
        "path",
        "preserveNullAndEmptyArrays",
        "includeArrayIndex",
    }

    def remap_field_path(path_text):
        path = str(path_text or "")
        if not path:
            return path
        root = path.split(".")[0]
        mapped = _guess_schema_field(root, allowed_fields, alias_index)
        if not mapped or mapped == root:
            mapped = _broaden_schema_field_match(root, schema_index, collection)
        if not mapped or mapped == root:
            return path
        suffix = path[len(root) :]
        return f"{mapped}{suffix}"

    def walk(value, in_operator_context=False):
        if isinstance(value, dict):
            new_obj = {}
            for key, child in value.items():
                key_text = str(key)
                new_key = key_text
                if not key_text.startswith("$") and not in_operator_context:
                    new_key = remap_field_path(key_text)
                child_in_operator = key_text.startswith("$") or key_text in aggregate_param_keys or in_operator_context
                new_obj[new_key] = walk(child, in_operator_context=child_in_operator)
            return new_obj
        if isinstance(value, list):
            return [walk(item, in_operator_context=in_operator_context) for item in value]
        if isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
            mapped = remap_field_path(value[1:])
            return f"${mapped}"
        return value

    patched = dict(plan)
    operation = str(plan.get("operation") or "find").lower().strip()
    if operation == "aggregate":
        patched["pipeline"] = walk(plan.get("pipeline") or [], in_operator_context=False)
    else:
        patched["filter"] = walk(plan.get("filter") or {}, in_operator_context=False)
        patched["projection"] = walk(plan.get("projection") or {}, in_operator_context=False)
        sort_value = plan.get("sort") or []
        if isinstance(sort_value, list):
            remapped_sort = []
            for item in sort_value:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    remapped_sort.append((remap_field_path(item[0]), item[1]))
                else:
                    remapped_sort.append(item)
            patched["sort"] = remapped_sort
        elif isinstance(sort_value, dict):
            patched["sort"] = {remap_field_path(k): v for k, v in sort_value.items()}
    return patched


def _strict_find_schema_validation(plan, schema_index):
    collection = str(plan.get("collection") or "").strip()
    allowed_fields = {
        str(item.get("name") or "").strip()
        for item in (schema_index.get(collection, {}) or {}).get("fields", [])
        if str(item.get("name") or "").strip()
    }

    def allowed(field_name):
        if field_name in allowed_fields or field_name in _effective_system_fields():
            return True
        if field_name.endswith("_") and field_name[:-1] in allowed_fields:
            return True
        if field_name.endswith("_textMode") and field_name[:-9] in allowed_fields:
            return True
        root = field_name.split(".")[0]
        if root in allowed_fields or root in _effective_system_fields():
            return True
        if root.endswith("_") and root[:-1] in allowed_fields:
            return True
        if root.endswith("_textMode") and root[:-9] in allowed_fields:
            return True
        return False

    def walk_filter(value):
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                if not key_text.startswith("$") and not allowed(key_text):
                    raise ValueError(f"Strict schema validation failed: field '{key_text}' not in selected collection schema.")
                walk_filter(child)
        elif isinstance(value, list):
            for item in value:
                walk_filter(item)
        elif isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
            field_ref = value[1:].split(".")[0]
            if field_ref and not allowed(field_ref):
                if field_ref not in {"_id"}:
                    raise ValueError(f"Strict schema validation failed: field reference '{value}' not in selected collection schema.")

    def walk_projection(value):
        if isinstance(value, dict):
            cleaned = {}
            for key, child in value.items():
                key_text = str(key)
                if key_text.startswith("$"):
                    cleaned[key_text] = child
                    continue
                if allowed(key_text):
                    cleaned[key_text] = child
            value.clear()
            value.update(cleaned)
            for child in value.values():
                walk_projection(child)
        elif isinstance(value, list):
            for item in value:
                walk_projection(item)
        elif isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
            field_ref = value[1:].split(".")[0]
            if field_ref and not allowed(field_ref) and field_ref not in {"_id"}:
                raise ValueError(f"Strict schema validation failed: field reference '{value}' not in selected collection schema.")

    def walk_sort(value):
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    field_name = str(item[0]).strip()
                    if field_name and allowed(field_name):
                        cleaned.append(item)
                elif isinstance(item, dict):
                    cleaned_item = {k: v for k, v in item.items() if allowed(str(k))}
                    if cleaned_item:
                        cleaned.append(cleaned_item)
            value[:] = cleaned
        elif isinstance(value, dict):
            cleaned = {k: v for k, v in value.items() if allowed(str(k))}
            value.clear()
            value.update(cleaned)

    walk_filter(plan.get("filter") or {})
    walk_projection(plan.get("projection") or {})
    walk_sort(plan.get("sort") or [])


def _strict_aggregate_schema_validation(plan, schema_index):
    collection = str(plan.get("collection") or "").strip()
    allowed_fields = {
        str(item.get("name") or "").strip()
        for item in (schema_index.get(collection, {}) or {}).get("fields", [])
        if str(item.get("name") or "").strip()
    }
    stage_aliases = set()

    def allowed_root(root):
        if root in allowed_fields or root in stage_aliases or root in _effective_system_fields() or root == "_id":
            return True
        if root.endswith("_") and root[:-1] in allowed_fields:
            return True
        if root.endswith("_textMode") and root[:-9] in allowed_fields:
            return True
        return False

    aggregate_param_keys = {
        "from",
        "localField",
        "foreignField",
        "as",
        "let",
        "pipeline",
        "input",
        "in",
        "cond",
        "if",
        "then",
        "else",
        "vars",
        "branches",
        "case",
        "default",
        "path",
        "preserveNullAndEmptyArrays",
        "includeArrayIndex",
    }

    def add_stage_alias(name):
        alias = str(name or "").strip()
        if not alias:
            return
        root = alias.split(".")[0]
        if root and not root.startswith("$"):
            stage_aliases.add(root)

    output_alias_stages = {"$project", "$addFields", "$set", "$group"}

    def walk(value, in_operator_context=False):
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                current_is_operator_param = in_operator_context or key_text.startswith("$") or key_text in aggregate_param_keys
                if not key_text.startswith("$") and not current_is_operator_param:
                    root = key_text.split(".")[0]
                    if root and not allowed_root(root):
                        raise ValueError(f"Strict schema validation failed: aggregate field '{key_text}' is outside selected collection schema.")
                walk(child, in_operator_context=current_is_operator_param)
        elif isinstance(value, list):
            for item in value:
                walk(item, in_operator_context=in_operator_context)
        elif isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
            root = value[1:].split(".")[0]
            if root and not allowed_root(root):
                raise ValueError(f"Strict schema validation failed: aggregate reference '{value}' is outside selected collection schema.")

    for stage in plan.get("pipeline") or []:
        if not isinstance(stage, dict):
            continue
        stage_name = next(iter(stage), "")
        stage_body = stage.get(stage_name)
        if stage_name == "$lookup" and isinstance(stage_body, dict):
            add_stage_alias(stage_body.get("as"))
        if stage_name == "$unwind" and isinstance(stage_body, dict):
            add_stage_alias(stage_body.get("includeArrayIndex"))
        if stage_name in output_alias_stages and isinstance(stage_body, dict):
            for key, child in stage_body.items():
                key_text = str(key).strip()
                if key_text and not key_text.startswith("$"):
                    add_stage_alias(key_text)
                    walk(child, in_operator_context=False)
                else:
                    walk(child, in_operator_context=True)
            continue
        walk(stage_body, in_operator_context=stage_name in {"$lookup", "$group", "$facet"})


def _remap_plan_field(plan, old_name, new_name):
    def walk(value, in_operator_context=False):
        if isinstance(value, dict):
            new_obj = {}
            for key, child in value.items():
                key_text = str(key)
                new_key = new_name if (not key_text.startswith("$") and key_text == old_name) else key_text
                new_obj[new_key] = walk(child, in_operator_context=key_text.startswith("$") or in_operator_context)
            return new_obj
        if isinstance(value, list):
            return [walk(item, in_operator_context=in_operator_context) for item in value]
        if isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
            ref = value[1:]
            if ref == old_name or ref.split(".")[0] == old_name:
                suffix = ref[len(old_name):]
                return f"${new_name}{suffix}"
        return value

    operation = str(plan.get("operation") or "find").lower().strip()
    if operation == "aggregate":
        plan["pipeline"] = walk(plan.get("pipeline") or [], in_operator_context=False)
    else:
        plan["filter"] = walk(plan.get("filter") or {}, in_operator_context=False)
        plan["projection"] = walk(plan.get("projection") or {}, in_operator_context=False)
        sort_value = plan.get("sort") or []
        if isinstance(sort_value, list):
            remapped_sort = []
            for item in sort_value:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    remapped_sort.append((new_name if item[0] == old_name else item[0], item[1]))
                else:
                    remapped_sort.append(item)
            plan["sort"] = remapped_sort
        elif isinstance(sort_value, dict):
            plan["sort"] = {new_name if k == old_name else k: v for k, v in sort_value.items()}


def _broaden_schema_field_match(invalid_field, schema_index, collection):
    rows = (schema_index.get(collection, {}) or {}).get("fields", [])
    invalid_norm = _normalize_field_token(invalid_field)
    if not invalid_norm:
        return None
    invalid_parts = set(_field_token_parts(invalid_field))
    best = None
    best_score = 0
    for field in rows:
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        display = str(field.get("display") or "").strip()
        aliases = [str(a).strip() for a in (field.get("aliases") or []) if str(a).strip()]
        candidates = [name, display] + aliases
        for alias in candidates:
            if not alias:
                continue
            alias_norm = _normalize_field_token(alias)
            if alias_norm == invalid_norm:
                return name
            if alias_norm.endswith(invalid_norm) or invalid_norm.endswith(alias_norm):
                return name
            if invalid_norm in alias_norm or alias_norm in invalid_norm:
                return name
            alias_parts = set(_field_token_parts(alias))
            if alias_parts & invalid_parts:
                score = len(alias_parts & invalid_parts)
                if score > best_score:
                    best_score = score
                    best = name
    return best if best_score >= 1 else None


def _strict_schema_validate_plan(plan, schema_index):
    aligned = _align_plan_fields_to_schema(plan, schema_index)
    if isinstance(aligned, dict) and aligned is not plan:
        plan.clear()
        plan.update(aligned)
    operation = str(plan.get("operation") or "find").lower().strip()
    try:
        if operation == "aggregate":
            _strict_aggregate_schema_validation(plan, schema_index)
        else:
            _strict_find_schema_validation(plan, schema_index)
    except ValueError as exc:
        msg = str(exc)
        match = re.search(r"field(?: reference)? '([^']+)'", msg)
        if not match:
            plan["needs_clarification"] = True
            plan["message"] = f"Schema validation failed: {msg}"
            return
        collection = str(plan.get("collection") or "").strip()
        raw_field = match.group(1)
        root_field = raw_field.split(".")[0]
        mapped = _broaden_schema_field_match(root_field, schema_index, collection)
        if mapped and mapped != root_field:
            _remap_plan_field(plan, root_field, mapped)
            try:
                if operation == "aggregate":
                    _strict_aggregate_schema_validation(plan, schema_index)
                else:
                    _strict_find_schema_validation(plan, schema_index)
                return
            except ValueError:
                pass
        plan["needs_clarification"] = True
        plan["message"] = f"Unrecognized field '{raw_field}' in collection '{collection}'."


def _build_hybrid_field_candidates(user_query, schema_index, exact_candidates, vector_candidates):
    keyword = retrieve_candidates(
        user_query,
        schema_index,
        max_collections=LLM_COLLECTION_CANDIDATES,
        max_fields=LLM_FIELD_CANDIDATES,
    )
    keyword_fields = {}
    for collection, rows in (keyword.get("field_candidates") or {}).items():
        names = []
        for row in rows or []:
            name = str((row or {}).get("field") or (row or {}).get("name") or "").strip()
            if name:
                names.append(name)
        keyword_fields[collection] = names

    merged = {}
    source_tracking = {}
    all_collections = set((exact_candidates or {}).keys()) | set((vector_candidates or {}).keys()) | set(keyword_fields.keys())
    for collection in all_collections:
        values = []
        for name in (exact_candidates or {}).get(collection, []):
            field = str(name or "").strip()
            if field and field not in values:
                values.append(field)
                source_tracking[(collection, field)] = "exact"
        for name in (vector_candidates or {}).get(collection, []):
            field = str(name or "").strip()
            if field and field not in values:
                values.append(field)
                source_tracking[(collection, field)] = "vector"
        for name in keyword_fields.get(collection, []):
            field = str(name or "").strip()
            if field and field not in values:
                values.append(field)
                source_tracking[(collection, field)] = "keyword"
        merged[collection] = values[: max(LLM_FIELD_CANDIDATES, 24)]
    return merged, source_tracking


def _dynamic_scope_terms(collections=None, table_metadata=None):
    terms = set()
    for name in collections or []:
        norm = _normalize_term(name)
        if norm:
            terms.update(norm.split())
        meta = (table_metadata or {}).get(name) or {}
        for source in (
            meta.get("template_name"),
            *(meta.get("business_terms") or []),
        ):
            value = _normalize_term(source)
            if value:
                terms.update(value.split())
        for field in (meta.get("fields") or []):
            field_name = _normalize_term((field or {}).get("field") or "")
            display = _normalize_term((field or {}).get("display") or "")
            if field_name:
                terms.update(field_name.split())
            if display:
                terms.update(display.split())
    return {term for term in terms if len(term) >= 3}


def _non_data_query_response():
    return "I am an ERP Intelligence assistant. I can only help with ERP-related queries."


def _looks_like_short_followup(prompt):
    text = str(prompt or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if len(lowered) > 80:
        return False
    if re.match(r"^(and|also|then)\b", lowered):
        return True
    if lowered.startswith("what about") or lowered.startswith("how about"):
        return True
    if re.search(r"\b(this|that|it|its|they|them|their|those|these|same|above|previous|earlier)\b", lowered):
        return True
    tokens = [tok for tok in re.split(r"\s+", lowered) if tok]
    return len(tokens) <= 3 and lowered.endswith("?")


def _is_out_of_scope_prompt(prompt, collections=None, table_metadata=None):
    text = _normalize_term(prompt)
    if not text:
        return False
    if _looks_like_short_followup(prompt):
        return False
    if collections and _infer_requested_collections(prompt, collections, table_metadata or {}):
        return False
    erp_terms = _dynamic_scope_terms(collections, table_metadata)
    if not erp_terms:
        return False
    tokens = set(text.split())
    return not bool(tokens & erp_terms)


def _llm_scope_gate(model, tokenizer, prompt, accessible_collections, table_metadata, chat_context=None):
    try:
        verdict = evaluate_query_scope(
            model,
            tokenizer,
            prompt,
            accessible_collections,
            table_metadata,
            chat_context=chat_context,
        )
        allow = bool(verdict.get("allow", True))
        message = str(verdict.get("message") or "").strip()
        if not allow:
            in_scope_by_rules = not _is_out_of_scope_prompt(
                prompt,
                collections=accessible_collections,
                table_metadata=table_metadata,
            )
            if in_scope_by_rules:
                allow = True
                message = ""
        return {
            "allow": allow,
            "message": message or _non_data_query_response(),
        }
    except Exception:
        return {"allow": True, "message": ""}


def _recent_chat_context(messages, limit=CHAT_CONTEXT_LIMIT):
    items = []
    for message in messages or []:
        role = str((message or {}).get("role", "")).strip()
        content = str((message or {}).get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            items.append({"role": role, "content": content[:CHAT_CONTEXT_CHARS]})
    return items[-max(2, int(limit or 2)) :]


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


def _selected_user_context(selected_user, accessible_collections=None):
    if not isinstance(selected_user, dict):
        return {}

    context = {
        "user_id": str(selected_user.get("user_id") or "").strip(),
        "display_name": str(
            selected_user.get("display_name")
            or selected_user.get("name")
            or selected_user.get("user_id")
            or ""
        ).strip(),
        "email": str(selected_user.get("email") or "").strip(),
        "roles": [str(role).strip() for role in (selected_user.get("roles") or []) if str(role).strip()][:8],
    }

    for key in ("organization", "organisation", "branch", "department", "designation"):
        value = selected_user.get(key)
        if isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
            if values:
                context[key] = values[:8]
        else:
            text = str(value or "").strip()
            if text:
                context[key] = text

    if accessible_collections:
        context["accessible_collections"] = [str(name).strip() for name in accessible_collections[:12] if str(name).strip()]
    return {key: value for key, value in context.items() if value}


def _build_user_scoped_retrieval_query(prompt, user_context):
    base_prompt = str(prompt or "").strip()
    if not base_prompt:
        return base_prompt
    if not isinstance(user_context, dict):
        return base_prompt

    extra_terms = []
    for key in ("roles", "branch", "department", "organization", "organisation", "designation"):
        value = user_context.get(key)
        if isinstance(value, list):
            extra_terms.extend(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value or "").strip()
            if text:
                extra_terms.append(text)

    extra_terms = list(dict.fromkeys(term for term in extra_terms if term))
    if not extra_terms:
        return base_prompt
    return f"{base_prompt} | user context: {', '.join(extra_terms[:6])}"


def _get_field_meta(collection, field_name, table_metadata):
    meta = (table_metadata or {}).get(collection) or {}
    for field in (meta.get("fields") or []):
        fname = str(field.get("field") or "").strip()
        if fname == field_name:
            return field
    return None


def _score_field_candidate_for_prompt(field_name, prompt, field_meta=None):
    text, tokens, token_set = _prompt_scoring_terms(prompt)
    field = str(field_name or "").strip()
    if not field:
        return 0

    aliases = set(_field_aliases(field))
    if isinstance(field_meta, dict):
        display = str(field_meta.get("display") or "").strip()
        if display:
            aliases.add(_normalize_term(display))
        for alias in field_meta.get("aliases") or []:
            alias_text = _normalize_term(alias)
            if alias_text:
                aliases.add(alias_text)

    score = 0
    field_norm = _normalize_term(field)
    field_tokens = set(field_norm.split()) if field_norm else set()

    for alias in aliases:
        alias_tokens = set(alias.split())
        if not alias_tokens:
            continue
        overlap = len(alias_tokens & token_set)
        if overlap:
            score += overlap * 4 + len(alias_tokens) * 2
        if alias in text:
            score += 12 + len(alias_tokens)

    if field_tokens:
        score += len(field_tokens & token_set) * 2

    prompt_intent_terms = {
        "approved": {"approver", "approvedby", "approved_by", "approval"},
        "reviewed": {"reviewer", "reviewedby", "reviewed_by", "review"},
        "assigned": {"assignedto", "assigned_to", "assignee", "owner"},
        "created": {"createdby", "created_by", "creator"},
        "submitted": {"submittedby", "submitted_by", "submitter"},
        "branch": {"branch"},
        "organization": {"organization", "organisation"},
        "title": {"title", "name"},
        "code": {"code", "reference", "number", "id"},
        "status": {"status", "state", "stage"},
        "due": {"due", "deadline"},
        "expiry": {"expiry", "expire", "expiration", "renewal"},
        "amount": {"amount", "fee", "fees", "payment", "tax", "penalty"},
    }
    for prompt_term, field_terms in prompt_intent_terms.items():
        if prompt_term in token_set or prompt_term in text:
            if any(term in field_norm for term in field_terms):
                score += 7

    if field_norm in token_set:
        score += 6
    if any(term in field_norm for term in ("reviewer", "approver", "assignedto", "createdby", "updatedby", "owner", "submitter")):
        score += 2
    return score


def _fallback_collection_choice(
    user_query,
    accessible_collections,
    table_metadata,
    schema_index,
    exact_candidates,
    vector_candidates,
):
    scores = {}
    for collection in accessible_collections:
        scores[collection] = 0

    intent_matches = _infer_requested_collections(user_query, accessible_collections, table_metadata)
    for rank, collection in enumerate(intent_matches):
        if collection in scores:
            scores[collection] += max(0, 60 - rank * 10)

    for collection, fields in (exact_candidates or {}).items():
        if collection in scores:
            field_score = 0
            for field_name in list(fields or [])[:6]:
                field_score += _score_field_candidate_for_prompt(field_name, user_query)
                # Direct field display/alias match bonus
                field_meta = _get_field_meta(collection, field_name, table_metadata)
                if field_meta:
                    display = _normalize_term(field_meta.get("display") or "")
                    aliases = [_normalize_term(a) for a in (field_meta.get("aliases") or [])]
                    prompt_norm = _normalize_term(user_query)
                    if display and display in prompt_norm:
                        field_score += 15
                    for alias in aliases:
                        if alias and alias in prompt_norm:
                            field_score += 10
            scores[collection] += min(45, len(fields) * 15 + field_score)
    vector_multiplier = 10 if not exact_candidates else 3
    for collection, fields in (vector_candidates or {}).items():
        if collection in scores:
            field_score = 0
            for field_name in list(fields or [])[:6]:
                field_score += _score_field_candidate_for_prompt(field_name, user_query)
                # Direct field display/alias match bonus
                field_meta = _get_field_meta(collection, field_name, table_metadata)
                if field_meta:
                    display = _normalize_term(field_meta.get("display") or "")
                    aliases = [_normalize_term(a) for a in (field_meta.get("aliases") or [])]
                    prompt_norm = _normalize_term(user_query)
                    if display and display in prompt_norm:
                        field_score += 15
                    for alias in aliases:
                        if alias and alias in prompt_norm:
                            field_score += 10
            scores[collection] += min(40, len(fields) * vector_multiplier + field_score)

    identifier_values = _extract_identifier_values(user_query)
    if identifier_values:
        value_boosts = _value_prefix_collection_boost(identifier_values, accessible_collections, table_metadata)
        for collection, boost in value_boosts.items():
            if collection in scores:
                scores[collection] += boost + 20

    try:
        retrieved = retrieve_candidates(
            user_query,
            schema_index,
            max_collections=LLM_COLLECTION_CANDIDATES,
            max_fields=LLM_FIELD_CANDIDATES,
        )
        for rank, collection in enumerate(retrieved.get("collections") or []):
            if collection in scores:
                scores[collection] += max(0, 20 - rank * 5)
    except Exception:
        pass

    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    selected, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    margin = best - runner_up
    confidence = round(best / max(best + max(runner_up, 0), 1), 3) if best > 0 else 0.0
    ambiguous = best <= 0 or best < 18 or margin < 8
    if best <= 0:
        return None
    return {
        "collection": selected,
        "reason": "Selected using schema/vector fallback ranking.",
        "score": best,
        "runner_up_score": runner_up,
        "margin": margin,
        "confidence": confidence,
        "ambiguous": ambiguous,
    }


def _collection_choice_is_confident(choice):
    if not isinstance(choice, dict):
        return False
    if choice.get("ambiguous"):
        return False
    confidence = float(choice.get("confidence") or 0.0)
    score = float(choice.get("score") or 0.0)
    margin = float(choice.get("margin") or 0.0)
    return confidence >= 0.62 and score >= 18 and margin >= 8


def _prefer_deterministic_collection_choice(
    user_query,
    selected_collection,
    deterministic_choice,
    accessible_collections,
    table_metadata,
):
    deterministic = str((deterministic_choice or {}).get("collection") or "").strip()
    selected = str(selected_collection or "").strip()
    if not deterministic or not selected or deterministic == selected:
        return selected
    inferred = set(_infer_requested_collections(user_query, accessible_collections, table_metadata))
    if selected not in inferred and deterministic in set(accessible_collections or []):
        return deterministic
    return selected


def _followup_collection_choice(prompt, last_collection, accessible_collections, table_metadata=None):
    last = str(last_collection or "").strip()
    if not last or last not in (accessible_collections or []):
        return None
    if not _looks_like_short_followup(prompt):
        return None
    if _infer_requested_collections(prompt, accessible_collections, table_metadata or {}):
        return None
    if _mentioned_entities(prompt, accessible_collections, table_metadata):
        return None
    return {
        "collection": last,
        "reason": "Selected previous collection from conversation follow-up context.",
    }


def _should_use_llm_summary(plan, docs):
    mode = str(LLM_SUMMARY_MODE or "adaptive").lower().strip()
    if mode == "off":
        return False
    if mode == "always":
        return True
    if not docs:
        return False
    operation = str((plan or {}).get("operation") or "find").lower()
    if operation == "aggregate":
        return True
    if len(docs) <= 8:
        return False
    return True


def _query_route_profile(
    prompt,
    accessible_collections,
    table_metadata,
    schema_index,
    exact_candidates,
    vector_candidates,
    last_collection=None,
):
    normalized_prompt = _light_normalize_prompt(prompt)
    explicit_choice = _explicit_collection_alias_choice(
        normalized_prompt,
        accessible_collections,
        table_metadata,
    )
    followup_choice = _followup_collection_choice(
        normalized_prompt,
        last_collection,
        accessible_collections,
        table_metadata=table_metadata,
    )
    fallback_choice = _fallback_collection_choice(
        normalized_prompt,
        accessible_collections,
        table_metadata,
        schema_index,
        exact_candidates,
        vector_candidates,
    )
    deterministic_choice = explicit_choice or followup_choice or fallback_choice
    inferred_collections = _infer_requested_collections(normalized_prompt, accessible_collections, table_metadata)
    mentioned_entities = _mentioned_entities(normalized_prompt, accessible_collections, table_metadata)
    needs_join, join_targets = _needs_join_shape(normalized_prompt, accessible_collections, table_metadata)
    short_followup = _looks_like_short_followup(normalized_prompt)
    prompt_tokens = [token for token in re.split(r"\s+", normalized_prompt) if token]
    concise = len(prompt_tokens) <= 24 and len(normalized_prompt) <= 160
    clear_single_collection = (
        bool(deterministic_choice)
        and len(inferred_collections) <= 1
        and len(mentioned_entities) <= 1
        and not needs_join
    )
    simple_query = clear_single_collection and concise and not short_followup
    return {
        "normalized_prompt": normalized_prompt,
        "deterministic_choice": deterministic_choice,
        "inferred_collections": inferred_collections,
        "mentioned_entities": mentioned_entities,
        "needs_join": needs_join,
        "join_targets": join_targets,
        "short_followup": short_followup,
        "concise": concise,
        "simple_query": simple_query,
        "use_single_pass": simple_query,
        "skip_rewrite": simple_query,
        "skip_scope_gate": simple_query,
    }


def _looks_like_created_date_question(prompt):
    text = _normalize_term(prompt)
    if not text:
        return False
    asks_when = "when" in text or "date" in text
    asks_created = "created" in text or "creation" in text
    return asks_when and asks_created


def _looks_like_temporal_lookup_question(prompt):
    text = _normalize_term(prompt)
    if not text:
        return False
    has_time_intent = any(
        token in text
        for token in (
            "when",
            "date",
            "expiry",
            "expire",
            "due",
            "deadline",
            "start",
            "end",
            "issue",
            "renewal",
            "review",
        )
    )
    if not has_time_intent:
        return False
    return len(text.split()) >= 3


def _looks_like_upcoming_deadline_question(prompt):
    text = _normalize_term(prompt)
    if not text:
        return False
    has_deadline_intent = any(token in text for token in ("deadline", "deadlines", "due", "expiry", "expire"))
    has_future_sort_intent = any(token in text for token in ("upcoming", "next", "nearest", "soon", "top"))
    return has_deadline_intent and has_future_sort_intent


def _looks_like_non_compliant_question(prompt):
    text = _normalize_term(prompt)
    if not text:
        return False
    if "non compliant" in text or "noncompliant" in text:
        return True
    return "compliance" in text and any(token in text for token in ("status", "marked", "which records", "show"))


def _extract_top_n(prompt, default_value=5, max_value=25):
    text = _normalize_term(prompt)
    if not text:
        return default_value
    match = re.search(r"\btop\s+(\d{1,3})\b", text)
    if not match:
        return default_value
    try:
        value = int(match.group(1))
    except Exception:
        return default_value
    return max(1, min(max_value, value))


def _get_schema_rows(collection_name, schema_index):
    return list((schema_index.get(collection_name, {}) or {}).get("fields") or [])


def _choose_existing_status_fields(candidates, allowed_fields):
    out = []
    seen = set()
    for item in candidates or []:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        if name in allowed_fields:
            out.append(name)
            seen.add(name)
    return out


def _policy_candidates(runtime_policy, key, default_values):
    if isinstance(runtime_policy, dict) and isinstance(runtime_policy.get(key), list):
        values = [str(item).strip() for item in runtime_policy.get(key) or [] if str(item).strip()]
        if values:
            return values
    return list(default_values or [])


def _choose_existing_fields(candidates, allowed_fields):
    out = []
    seen = set()
    for item in candidates or []:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        if name in allowed_fields:
            seen.add(name)
            out.append(name)
    return out


def _created_intent_text_fields(schema_rows):
    picked = []
    for row in schema_rows or []:
        name = str((row or {}).get("name") or "").strip()
        ftype = _normalize_term((row or {}).get("type") or "")
        if not name:
            continue
        if any(token in ftype for token in ("text", "string", "lookup", "select")):
            picked.append(name)
    return picked[:12]


def _infer_collection_field_roles(selected_collection, schema_index, runtime_policy=None):
    schema_rows = list((schema_index.get(selected_collection, {}) or {}).get("fields") or [])
    allowed_fields = {
        str(item.get("name") or "").strip()
        for item in schema_rows
        if str(item.get("name") or "").strip()
    }
    roles = {
        "searchable_text_fields": _created_intent_text_fields(schema_rows),
        "created_time_fields": [],
        "label_fields": [],
        "business_id_fields": [],
        "soft_delete_fields": [],
    }
    if not allowed_fields:
        return roles

    created_time_defaults = sorted(
        name for name in SYSTEM_FIELD_NAMES if str(name).lower().startswith("created")
    )
    created_candidates = _policy_candidates(
        runtime_policy,
        "createdTimeFieldCandidates",
        created_time_defaults,
    )
    label_candidates = _policy_candidates(
        runtime_policy,
        "labelFieldCandidates",
        ["name", "title", "description", "displayName", "label"],
    )
    business_id_candidates = _policy_candidates(
        runtime_policy,
        "businessIdFieldCandidates",
        ["code", "number", "referenceRegNumber", "obligationId"],
    )
    soft_delete_candidates = _policy_candidates(
        runtime_policy,
        "softDeleteFieldCandidates",
        ["isDeleted", "deleted", "is_deleted"],
    )

    roles["created_time_fields"] = _choose_existing_fields(created_candidates, allowed_fields)
    roles["label_fields"] = _choose_existing_fields(label_candidates, allowed_fields)
    roles["business_id_fields"] = _choose_existing_fields(business_id_candidates, allowed_fields)
    roles["soft_delete_fields"] = _choose_existing_fields(soft_delete_candidates, allowed_fields)

    if not roles["created_time_fields"]:
        for row in schema_rows:
            name = str((row or {}).get("name") or "").strip()
            ftype = _normalize_term((row or {}).get("type") or "")
            if name and any(token in ftype for token in ("date", "time", "datetime")):
                roles["created_time_fields"].append(name)
        roles["created_time_fields"] = roles["created_time_fields"][:2]
    if not roles["label_fields"]:
        for row in schema_rows:
            name = str((row or {}).get("name") or "").strip()
            ftype = _normalize_term((row or {}).get("type") or "")
            if name and any(token in ftype for token in ("text", "string")):
                roles["label_fields"].append(name)
        roles["label_fields"] = roles["label_fields"][:3]
    return roles


def _infer_non_compliance_status_fields(collection_name, schema_index, runtime_policy=None):
    schema_rows = _get_schema_rows(collection_name, schema_index)
    allowed_fields = {
        str(item.get("name") or "").strip()
        for item in schema_rows
        if str(item.get("name") or "").strip()
    }
    if not allowed_fields:
        return []
    policy_candidates_list = _policy_candidates(
        runtime_policy,
        "complianceStatusFieldCandidates",
        ["complianceStatus", "obligationStatus_", "obligationStatus", "status", "slaStatus"],
    )
    picked = _choose_existing_status_fields(policy_candidates_list, allowed_fields)
    if picked:
        return picked
    for field_name in allowed_fields:
        lowered = field_name.lower()
        if "status" in lowered:
            picked.append(field_name)
    picked.sort(key=lambda name: (0 if "compliance" in name.lower() or "obligationstatus" in name.lower() else 1, name))
    return picked[:4]


def _collection_non_compliance_score(collection_name, schema_index, table_metadata, runtime_policy=None):
    score = 0
    status_fields = _infer_non_compliance_status_fields(collection_name, schema_index, runtime_policy=runtime_policy)
    for field in status_fields:
        lower = field.lower()
        if lower == "compliancestatus":
            score += 60
        elif "compliance" in lower:
            score += 40
        elif "obligationstatus" in lower:
            score += 24
        elif "status" in lower:
            score += 12
    meta = (table_metadata or {}).get(collection_name) or {}
    text_blob = " ".join(
        [
            _normalize_term(collection_name),
            _normalize_term(meta.get("template_name") or ""),
            " ".join(_normalize_term(term) for term in (meta.get("business_terms") or [])),
        ]
    )
    if "obligation" in text_blob:
        score += 18
    if "compliance" in text_blob:
        score += 18
    return score


def _prefer_non_compliant_collection(
    prompt,
    selected_collection,
    accessible_collections,
    schema_index,
    table_metadata,
    runtime_policy=None,
):
    if not _looks_like_non_compliant_question(prompt):
        return selected_collection
    current = str(selected_collection or "").strip()
    if current not in (accessible_collections or []):
        return current
    current_score = _collection_non_compliance_score(
        current,
        schema_index,
        table_metadata,
        runtime_policy=runtime_policy,
    )
    best_name = current
    best_score = current_score
    for name in accessible_collections or []:
        score = _collection_non_compliance_score(
            name,
            schema_index,
            table_metadata,
            runtime_policy=runtime_policy,
        )
        if score > best_score:
            best_name = name
            best_score = score
    if best_name != current and best_score >= current_score + 15:
        return best_name
    return current


def _pick_deadline_field(schema_rows, runtime_policy=None):
    allowed_fields = {
        str(item.get("name") or "").strip()
        for item in (schema_rows or [])
        if str(item.get("name") or "").strip()
    }
    if not allowed_fields:
        return ""
    policy_candidates_list = _policy_candidates(
        runtime_policy,
        "deadlineFieldCandidates",
        ["dueDate", "expiryDate", "completionDate", "startDate", "issueDate", "endDate"],
    )
    for name in policy_candidates_list:
        if name in allowed_fields:
            return name
    lowered = {name.lower(): name for name in allowed_fields}
    for key in ("duedate", "expirydate", "deadline", "completiondate"):
        if key in lowered:
            return lowered[key]
    return ""


def _build_non_compliant_filter_for_field(field_name, field_meta):
    if not field_name:
        return {}
    ftype = _normalize_term((field_meta or {}).get("type") or "")
    pattern = "non[- ]?compliant"
    if any(token in ftype for token in ("select", "lookup", "multi")):
        return {
            "$or": [
                {field_name: {"$elemMatch": {"$regex": pattern, "$options": "i"}}},
                {f"{field_name}_textMode": {"$regex": pattern, "$options": "i"}},
                {f"{field_name}_": {"$regex": pattern, "$options": "i"}},
            ]
        }
    return {field_name: {"$regex": pattern, "$options": "i"}}


def _apply_non_compliant_guard(plan, prompt, selected_collection, schema_index, runtime_policy=None):
    if not isinstance(plan, dict):
        return plan
    if not _looks_like_non_compliant_question(prompt):
        return plan
    schema_rows = _get_schema_rows(selected_collection, schema_index)
    if not schema_rows:
        return plan
    fields_by_name = {
        str(item.get("name") or "").strip(): item
        for item in schema_rows
        if str(item.get("name") or "").strip()
    }
    status_fields = _infer_non_compliance_status_fields(
        selected_collection,
        schema_index,
        runtime_policy=runtime_policy,
    )
    status_field = next((name for name in status_fields if name in fields_by_name), "")
    if not status_field:
        return plan
    status_filter = _build_non_compliant_filter_for_field(status_field, fields_by_name.get(status_field))
    if not status_filter:
        return plan

    op = str(plan.get("operation") or "find").lower().strip()
    if op == "aggregate":
        pipeline = list(plan.get("pipeline") or [])
        if pipeline and isinstance(pipeline[0], dict) and "$match" in pipeline[0] and isinstance(pipeline[0]["$match"], dict):
            pipeline[0]["$match"] = {"$and": [pipeline[0]["$match"], status_filter]}
        else:
            pipeline = [{"$match": status_filter}] + pipeline
        guarded = dict(plan)
        guarded["pipeline"] = pipeline
        return guarded

    existing_filter = plan.get("filter") if isinstance(plan.get("filter"), dict) else {}
    merged_filter = {"$and": [existing_filter, status_filter]} if existing_filter else status_filter
    guarded = dict(plan)
    guarded["filter"] = merged_filter
    return guarded


def _apply_upcoming_deadline_guard(plan, prompt, selected_collection, schema_index, runtime_policy=None):
    if not isinstance(plan, dict):
        return plan
    if not _looks_like_upcoming_deadline_question(prompt):
        return plan
    schema_rows = list((schema_index.get(selected_collection, {}) or {}).get("fields") or [])
    deadline_field = _pick_deadline_field(schema_rows, runtime_policy=runtime_policy)
    if not deadline_field:
        return plan

    top_n = _extract_top_n(prompt, default_value=5, max_value=25)
    roles = _infer_collection_field_roles(selected_collection, schema_index, runtime_policy=runtime_policy)
    allowed_fields = {
        str(item.get("name") or "").strip()
        for item in schema_rows
        if str(item.get("name") or "").strip()
    }
    projection_fields = [deadline_field]
    projection_fields.extend([name for name in (roles.get("label_fields") or []) if name in allowed_fields])
    projection_fields.extend([name for name in (roles.get("business_id_fields") or []) if name in allowed_fields])
    projection = {name: 1 for name in projection_fields[:5] if name}

    now_utc = datetime.now(timezone.utc)
    filter_doc = {deadline_field: {"$ne": None, "$gte": now_utc}}
    text = _normalize_term(prompt)
    if "active" in text:
        if "isDeleted" in allowed_fields:
            filter_doc["isDeleted"] = {"$ne": True}
        elif "obligationStatus_" in allowed_fields:
            filter_doc["obligationStatus_"] = {"$regex": "^active$", "$options": "i"}
        elif "obligationStatus" in allowed_fields:
            filter_doc["obligationStatus"] = {"$elemMatch": {"$regex": "^active$", "$options": "i"}}

    return {
        "operation": "aggregate",
        "collection": selected_collection,
        "pipeline": [
            {"$match": filter_doc},
            {"$sort": {deadline_field: 1}},
            {"$limit": top_n},
            {"$project": projection if projection else {deadline_field: 1}},
        ],
    }


def _filter_has_unknown_top_level_fields(filter_doc, allowed_fields):
    if not isinstance(filter_doc, dict):
        return False
    for key in filter_doc.keys():
        key_text = str(key)
        if key_text.startswith("$"):
            continue
        root = key_text.split(".")[0]
        if root not in allowed_fields and root not in _effective_system_fields():
            return True
    return False


def _extract_entity_terms_for_created_question(prompt):
    text = _normalize_term(prompt)
    stop = {
        "when", "what", "which", "where", "who", "is", "are", "was", "were", "the", "a", "an",
        "of", "for", "to", "in", "on", "from", "by", "created", "creation", "date", "time",
        "show", "get", "me", "please",
    }
    terms = []
    for token in text.split():
        if len(token) < 3 or token in stop:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:5]


def _apply_created_intent_guard(plan, prompt, selected_collection, schema_index, runtime_policy=None):
    if not isinstance(plan, dict):
        return plan
    if str(plan.get("operation") or "find").lower().strip() != "find":
        return plan
    if not _looks_like_created_date_question(prompt):
        return plan

    schema_rows = list((schema_index.get(selected_collection, {}) or {}).get("fields") or [])
    allowed_fields = {
        str(item.get("name") or "").strip()
        for item in schema_rows
        if str(item.get("name") or "").strip()
    }
    if not allowed_fields:
        return plan

    filter_doc = plan.get("filter") if isinstance(plan.get("filter"), dict) else {}
    if not _filter_has_unknown_top_level_fields(filter_doc, allowed_fields):
        return plan

    terms = _extract_entity_terms_for_created_question(prompt)
    roles = _infer_collection_field_roles(selected_collection, schema_index, runtime_policy=runtime_policy)
    text_fields = list(roles.get("searchable_text_fields") or [])
    if not terms or not text_fields:
        return plan

    term_filters = []
    for term in terms:
        term_filters.append(
            {"$or": [{field: {"$regex": re.escape(term), "$options": "i"}} for field in text_fields]}
        )

    projection = {}
    projection_candidates = []
    projection_candidates.extend(roles.get("created_time_fields") or [])
    projection_candidates.extend(roles.get("label_fields") or [])
    projection_candidates.extend(roles.get("business_id_fields") or [])
    for candidate in projection_candidates:
        if candidate in allowed_fields:
            projection[candidate] = 1

    guarded = dict(plan)
    guarded["filter"] = {"$and": term_filters}
    if projection:
        guarded["projection"] = projection
    created_fields = roles.get("created_time_fields") or []
    if created_fields:
        guarded["sort"] = [[created_fields[0], -1]]
    return guarded


def _find_join_lookup_field(template_schema, target_collection):
    schema_fields = (template_schema or {}).get("fields") or []
    normalized_target = _normalize_term(target_collection)
    target_candidates = {normalized_target, _singular(normalized_target)}
    for field in schema_fields:
        dtype = str(field.get("dataType") or "").upper()
        if dtype not in {"LOOK_UP", "MULTI_LOOKUP"}:
            continue
        lookup_target = str(field.get("lookupTargetCollection") or "")
        lookup_norm = _normalize_term(lookup_target)
        if lookup_norm in target_candidates or any(token and token in lookup_norm for token in target_candidates):
            field_name = str(field.get("fieldName") or "").strip()
            if field_name:
                return field_name, lookup_target
    return None, None


def _build_join_fallback_plan(base_collection, join_collection, join_field):
    as_field = f"{join_field}_details"
    return {
        "operation": "aggregate",
        "collection": base_collection,
        "pipeline": [
            {
                "$lookup": {
                    "from": join_collection,
                    "localField": join_field,
                    "foreignField": "_id",
                    "as": as_field,
                }
            },
            {
                "$project": {
                    "username": 1,
                    "firstName": 1,
                    "lastName": 1,
                    f"{join_field}_name": {
                        "$ifNull": [
                            {"$arrayElemAt": [f"${as_field}.name", 0]},
                            {"$arrayElemAt": [f"${as_field}.displayName", 0]},
                        ]
                    },
                }
            },
        ],
    }


def _needs_join_shape(prompt, accessible_collections, table_metadata=None):
    normalized = _normalize_term(prompt)
    mentioned = _mentioned_entities(prompt, accessible_collections, table_metadata)
    has_join_phrase = any(token in normalized for token in (" with ", " along with ", " and "))
    return has_join_phrase and len(mentioned) >= 2, mentioned


def _null_heavy_docs(docs):
    if not docs:
        return False
    checked = 0
    nullish = 0
    for row in docs[:20]:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            checked += 1
            if value is None or value == "" or value == []:
                nullish += 1
    if checked == 0:
        return False
    return (nullish / checked) >= 0.7


def _looks_like_count_by_intent(prompt):
    text = _normalize_term(prompt)
    if not text:
        return False
    has_count = any(token in text for token in ("count", "how many", "number of", "total"))
    has_group = (" by " in text) or ("group by" in text) or ("grouped by" in text)
    return has_count and has_group


def _extract_group_by_term(prompt):
    text = _normalize_term(prompt)
    if not text:
        return ""
    match = re.search(r"\bby\s+([a-z0-9_ ]{2,80})", text, flags=re.IGNORECASE)
    if not match:
        return ""
    term = str(match.group(1) or "").strip()
    if not term:
        return ""
    term = re.split(r"\b(where|order|limit|for|in|with|and)\b", term, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return term


def _pick_group_field(term, field_rows, vector_field_candidates=None):
    tokens = set(_tokenize_norm(term))
    best = None
    best_score = 0
    for row in field_rows or []:
        field_name = str(row.get("name") or "").strip()
        if not field_name:
            continue
        aliases = [field_name, str(row.get("display") or "")]
        aliases.extend([str(alias) for alias in (row.get("aliases") or []) if str(alias).strip()])
        alias_text = " ".join(aliases)
        alias_tokens = set(_tokenize_norm(alias_text))
        score = len(tokens & alias_tokens) * 3
        norm_alias = _normalize_term(alias_text)
        norm_term = _normalize_term(term)
        if norm_term and norm_term in norm_alias:
            score += 6
        if "status" in tokens and "status" in alias_tokens:
            score += 4
        if score > best_score:
            best_score = score
            best = row
    if best is not None and best_score > 0:
        return best
    if vector_field_candidates:
        vector_set = {str(item).strip() for item in (vector_field_candidates or []) if str(item).strip()}
        for row in field_rows or []:
            if str(row.get("name") or "").strip() in vector_set:
                return row
    return None


def _build_count_by_fallback_plan(prompt, selected_collection, schema_index, vector_candidates, runtime_policy=None):
    if not _looks_like_count_by_intent(prompt):
        return None
    table = (schema_index or {}).get(selected_collection) or {}
    field_rows = list(table.get("fields") or [])
    if not field_rows:
        return None
    term = _extract_group_by_term(prompt)
    vector_fields = (vector_candidates or {}).get(selected_collection) or []
    group_field_meta = _pick_group_field(term, field_rows, vector_field_candidates=vector_fields)
    if not group_field_meta:
        return None
    group_field = str(group_field_meta.get("name") or "").strip()
    if not group_field:
        return None

    allowed_fields = {
        str(item.get("name") or "").strip()
        for item in field_rows
        if str(item.get("name") or "").strip()
    }
    pipeline = []
    roles = _infer_collection_field_roles(selected_collection, schema_index, runtime_policy=runtime_policy)
    soft_delete_fields = roles.get("soft_delete_fields") or []
    if soft_delete_fields:
        pipeline.append({"$match": {soft_delete_fields[0]: {"$ne": True}}})
    if _field_type_is_array_like(group_field_meta):
        pipeline.append(
            {
                "$unwind": {
                    "path": f"${group_field}",
                    "preserveNullAndEmptyArrays": False,
                }
            }
        )
    pipeline.extend(
        [
            {"$group": {"_id": f"${group_field}", "count": {"$sum": 1}}},
            {"$project": {"_id": 0, "groupBy": "$_id", "count": 1}},
            {"$sort": {"count": -1}},
        ]
    )
    return {
        "operation": "aggregate",
        "collection": selected_collection,
        "pipeline": pipeline,
    }


def _enforce_join_plan_if_needed(prompt, selected_collection, template_schema, raw_plan, accessible_collections, table_metadata=None):
    needs_join, mentioned = _needs_join_shape(prompt, accessible_collections, table_metadata)
    if not needs_join:
        return raw_plan

    if not isinstance(raw_plan, dict):
        return raw_plan

    current_op = str(raw_plan.get("operation") or "").lower()
    if current_op == "aggregate":
        pipeline = raw_plan.get("pipeline") or []
        if isinstance(pipeline, list) and any(isinstance(s, dict) and "$lookup" in s for s in pipeline):
            return raw_plan

    join_target = None
    for candidate in mentioned:
        if candidate != selected_collection:
            join_target = candidate
            break
    if not join_target:
        return raw_plan

    join_field, join_collection = _find_join_lookup_field(template_schema, join_target)
    if not join_field or not join_collection:
        return raw_plan
    return _build_join_fallback_plan(selected_collection, join_collection, join_field)


def _permission_denied_message(denied_collections, table_metadata):
    labels = []
    for name in denied_collections[:2]:
        label = str(((table_metadata or {}).get(name) or {}).get("template_name") or name)
        labels.append(label)
    if labels:
        target = ", ".join(labels)
        return f"You are not authorized to access `{target}`. Please contact admin for permission."
    return "You are not authorized for this request. Please contact admin for permission."


def _apply_collection_intent_override(prompt, selected_collection, accessible_collections):
    mentioned = _mentioned_entities(prompt, accessible_collections)
    if len(mentioned) < 2:
        return selected_collection
    normalized = _normalize_term(prompt)
    if "users" in mentioned and re.search(r"\b(users?|usernames?)\b", normalized):
        return "users"
    return selected_collection


def _is_object_id_like(value):
    text = str(value or "").strip()
    return bool(re.fullmatch(r"[0-9a-fA-F]{24}", text))


def _runtime_repair_hint(error_text):
    text = str(error_text or "")
    lowered = text.lower()
    if "sortbycount" in lowered:
        return (
            "Runtime error indicates $sortByCount misuse. "
            "Use $sortByCount only as a standalone aggregation stage "
            "(e.g. {\"$sortByCount\": \"$designation\"}), never as an expression in $project/$addFields/$group."
        )
    return text


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


def _apply_runtime_field_remap(plan, runtime_field_names):
    try:
        return remap_plan_runtime_fields(plan, runtime_field_names or [])
    except Exception:
        return plan


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
    # Companion helper fields are collected separately
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




def _doc_matches_constraints(doc, terms):
    if not isinstance(doc, dict) or not terms:
        return False
    haystack = json.dumps(doc, ensure_ascii=False).lower()
    return all(term in haystack for term in terms)


def _prompt_focus_terms(prompt):
    tokens = []
    seen = set()
    generic_terms = {
        "show",
        "list",
        "get",
        "give",
        "find",
        "search",
        "what",
        "when",
        "where",
        "who",
        "which",
        "how",
        "many",
        "all",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "for",
        "to",
        "in",
        "on",
        "by",
        "with",
        "and",
        "or",
        "from",
        "please",
        "records",
        "record",
        "details",
        "information",
        "data",
        "value",
        "values",
        "result",
        "results",
        "current",
        "latest",
    }
    for token in lookup_tokens(prompt):
        if len(token) < 3 or token in generic_terms:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens[:6]


def _schema_field_names(collection_name, schema_index):
    return [
        str((row or {}).get("name") or "").strip()
        for row in (schema_index.get(collection_name, {}) or {}).get("fields", [])
        if str((row or {}).get("name") or "").strip()
    ]


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


def _field_aliases(field_name):
    raw = str(field_name or "").strip()
    if not raw:
        return []
    variants = {
        raw,
        raw.replace("_textMode", ""),
        raw.rstrip("_"),
        re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw),
        raw.replace("_", " "),
    }
    aliases = []
    for value in variants:
        normalized = _normalize_term(value)
        if normalized and normalized not in aliases:
            aliases.append(normalized)
    return aliases


def _prompt_scoring_terms(prompt):
    text = _normalize_term(prompt)
    tokens = [token for token in _tokenize(text) if token]
    return text, tokens, set(tokens)


def _rank_field_candidates_for_prompt(prompt, collection, candidate_fields, schema_index, top_k=4):
    collection_rows = list((schema_index.get(collection, {}) or {}).get("fields") or [])
    field_meta_map = {
        str(row.get("name") or "").strip(): row
        for row in collection_rows
        if str(row.get("name") or "").strip()
    }
    scored = []
    seen = set()
    for field_name in candidate_fields or []:
        field = str(field_name or "").strip()
        if not field or field in seen:
            continue
        seen.add(field)
        score = _score_field_candidate_for_prompt(field, prompt, field_meta_map.get(field))
        if score > 0:
            scored.append((score, field))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return []
    if len(scored) == 1:
        return [scored[0][1]]
    best_score = scored[0][0]
    second_score = scored[1][0]
    if best_score - second_score <= 2:
        return [name for _, name in scored[: min(max(2, top_k), len(scored))]]
    return [scored[0][1]]


def _best_field_hint_near_text(text, field_names):
    normalized = _normalize_term(text)
    if not normalized:
        return ""
    best = ("", -1, -1)
    for field in field_names:
        for alias in _field_aliases(field):
            if len(alias) < 2:
                continue
            match = list(re.finditer(rf"\b{re.escape(alias)}\b", normalized))
            if not match:
                continue
            score = len(alias.split()) * 4 + len(alias)
            position = match[-1].start()
            if score > best[1] or (score == best[1] and position > best[2]):
                best = (field, score, position)
    return best[0]


def _value_token_patterns(value):
    patterns = []
    for token in lookup_tokens(value):
        if len(token) < 2:
            continue
        compact = re.sub(r"(.)\1+", r"\1", token)
        stems = {token, compact}
        if len(token) > 5:
            stems.add(token[:4])
        if len(compact) > 5:
            stems.add(compact[:4])
        stems = {stem for stem in stems if len(stem) >= 2}
        if not stems:
            continue
        patterns.append("|".join(re.escape(stem) for stem in sorted(stems, key=lambda item: (-len(item), item))))
    return patterns[:6]


def _looks_like_identifier_value(value):
    text = str(value or "").strip()
    if len(text) < 3:
        return False
    if re.search(r"\b(?:id|code|number|no|ref|reference)\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r"[A-Za-z]{2,}[A-Za-z0-9]*[-_/][A-Za-z0-9-]{1,}", text):
        return True
    if re.search(r"\b[A-Za-z]{2,}\s+\d{2,}\b", text):
        return True
    return bool(re.search(r"\b[A-Z]{2,}[A-Z0-9-]{2,}\b", text))


def _identifier_field_names(field_names):
    ranked = []
    for field in field_names:
        score = 0
        field_norm = _normalize_term(field)
        aliases = _field_aliases(field)
        alias_blob = " ".join(aliases)
        if field_norm.endswith("id"):
            score += 6
        if "reference" in field_norm:
            score += 5
        if "number" in field_norm or field_norm == "code":
            score += 4
        if any(term in alias_blob for term in ("id", "code", "number", "reference")):
            score += 7
        if any(term in alias_blob for term in ("code", "number", "reference", "id")):
            score += 2
        if score > 0:
            ranked.append((score, field))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [field for _, field in ranked[:4]]


def _identifier_exact_regex(value):
    parts = [part for part in re.split(r"[-_/\s]+", str(value or "").strip()) if part]
    if not parts:
        return ""
    return "^" + r"[-_/\s]*".join(re.escape(part) for part in parts) + "$"


def _requested_projection_fields(prompt, field_names):
    normalized = _normalize_term(prompt)
    if not normalized:
        return []
    scored = []
    seen = set()
    for field in field_names:
        best = 0
        for alias in _field_aliases(field):
            if len(alias) < 2:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                best = max(best, len(alias.split()) * 4 + len(alias))
        if best > 0 and field not in seen:
            seen.add(field)
            scored.append((best, field))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [field for _, field in scored[:6]]


def _extract_prompt_value_constraints(prompt, selected_collection, schema_index, exact_candidates=None):
    field_names = _schema_field_names(selected_collection, schema_index)
    if not field_names:
        return []
    text = str(prompt or "")
    constraints = []
    seen = set()

    def add(field, value, identifier=False):
        field = str(field or "").strip()
        value = re.sub(r"\s+", " ", str(value or "").strip(" \t\r\n'\"`.,;:"))
        if not field or field not in field_names or len(value) < 2:
            return
        key = (field, value.lower(), bool(identifier))
        if key in seen:
            return
        patterns = _value_token_patterns(value)
        if not patterns:
            return
        seen.add(key)
        payload = {"field": field, "value": value, "patterns": patterns, "identifier": bool(identifier)}
        if identifier:
            exact_regex = _identifier_exact_regex(value)
            if exact_regex:
                payload["exact_regex"] = exact_regex
        constraints.append(payload)

    for match in re.finditer(r"['\"]([^'\"]{2,120})['\"]", text):
        value = match.group(1)
        prefix = text[max(0, match.start() - 100) : match.start()]
        field = _best_field_hint_near_text(prefix, field_names)
        if field:
            add(field, value)

    aliases = []
    for field in field_names:
        for alias in _field_aliases(field):
            if len(alias) >= 2:
                aliases.append((field, alias))
    aliases.sort(key=lambda item: len(item[1]), reverse=True)
    normalized_text = _normalize_term(text)
    identifier_fields = _identifier_field_names(field_names)
    identifier_values = []
    for match in re.finditer(r"\b[A-Za-z]{2,}[A-Za-z0-9]*[-_/][A-Za-z0-9-]{1,}\b", text):
        value = str(match.group(0) or "").strip()
        if _looks_like_identifier_value(value) and value.lower() not in {item.lower() for item in identifier_values}:
            identifier_values.append(value)
    for match in re.finditer(
        r"\b(?:of|for|about|regarding|with|where)\s+([A-Za-z]{2,}[A-Za-z0-9]*[-_/][A-Za-z0-9-]{1,}|[A-Z]{2,}[A-Z0-9-]{2,})\b",
        text,
        flags=re.IGNORECASE,
    ):
        value = str(match.group(1) or "").strip()
        if _looks_like_identifier_value(value) and value.lower() not in {item.lower() for item in identifier_values}:
            identifier_values.append(value)
    for value in identifier_values[:3]:
        for field in identifier_fields[:2]:
            add(field, value, identifier=True)

    stop_words = {
        "and",
        "with",
        "show",
        "list",
        "include",
        "where",
        "sort",
        "by",
        "from",
        "for",
        "details",
        "record",
        "records",
    }
    for field, alias in aliases:
        pattern = rf"\b{re.escape(alias)}\b\s*(?:is|=|as|called|named|like|contains|of)?\s+([a-z0-9][a-z0-9\s._/&-]{{1,80}})"
        match = re.search(pattern, normalized_text)
        if not match:
            continue
        raw_value = match.group(1).strip()
        words = []
        for word in raw_value.split():
            if word in stop_words:
                break
            words.append(word)
            if len(words) >= 8:
                break
        parsed_value = " ".join(words)
        if _looks_like_identifier_value(parsed_value) and field not in identifier_fields:
            continue
        add(field, parsed_value)

    candidate_fields = [
        str(item or "").strip()
        for item in (exact_candidates or {}).get(selected_collection, [])
        if str(item or "").strip() in field_names
    ]
    if candidate_fields:
        ranked_candidate_fields = _rank_field_candidates_for_prompt(
            prompt,
            selected_collection,
            candidate_fields,
            schema_index,
            top_k=4,
        )
        if not ranked_candidate_fields:
            ranked_candidate_fields = candidate_fields[:2]
        for match in re.finditer(r"['\"]([^'\"]{2,120})['\"]", text):
            for field in ranked_candidate_fields[:2]:
                add(field, match.group(1))

    return constraints[:6]


def _constraint_filter_doc(constraints):
    clauses = []
    identifier_groups = {}
    for constraint in constraints or []:
        field = str((constraint or {}).get("field") or "").strip()
        patterns = [pattern for pattern in ((constraint or {}).get("patterns") or []) if field and pattern]
        if not patterns:
            continue
        if (constraint or {}).get("identifier"):
            value_key = str((constraint or {}).get("value") or "").strip().lower()
            if value_key:
                identifier_groups.setdefault(value_key, []).append(
                    {
                        "field": field,
                        "patterns": patterns,
                        "exact_regex": str((constraint or {}).get("exact_regex") or "").strip(),
                    }
                )
            continue
        for pattern in patterns:
            clauses.append({field: {"$regex": pattern, "$options": "i"}})

    for group in identifier_groups.values():
        alternatives = []
        for item in group:
            field = item["field"]
            exact_regex = item.get("exact_regex") or ""
            patterns = item["patterns"]
            if exact_regex:
                alternatives.append({field: {"$regex": exact_regex, "$options": "i"}})
            else:
                field_clauses = [{field: {"$regex": pattern, "$options": "i"}} for pattern in patterns]
                if len(field_clauses) == 1:
                    alternatives.append(field_clauses[0])
                else:
                    alternatives.append({"$and": field_clauses})
        if not alternatives:
            continue
        if len(alternatives) == 1:
            clauses.append(alternatives[0])
        else:
            clauses.append({"$or": alternatives})
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _merge_filter_with_constraint(existing_filter, constraint_filter):
    existing = existing_filter if isinstance(existing_filter, dict) else {}
    extra = constraint_filter if isinstance(constraint_filter, dict) else {}
    if not extra:
        return existing
    if not existing:
        return extra
    return {"$and": [existing, extra]}


def _apply_prompt_value_constraints(plan, prompt, selected_collection, schema_index, exact_candidates=None):
    if not isinstance(plan, dict) or plan.get("needs_clarification"):
        return plan
    constraints = _extract_prompt_value_constraints(
        prompt,
        selected_collection,
        schema_index,
        exact_candidates=exact_candidates,
    )
    constraint_filter = _constraint_filter_doc(constraints)
    if not constraint_filter:
        return plan

    patched = dict(plan)
    operation = str(patched.get("operation") or "find").lower()
    if operation == "find":
        patched["filter"] = _merge_filter_with_constraint(patched.get("filter"), constraint_filter)
        projection = patched.get("projection")
        if isinstance(projection, dict) and projection:
            for constraint in constraints:
                projection.setdefault(constraint["field"], 1)
            patched["projection"] = projection
        return patched

    if operation == "aggregate":
        pipeline = list(patched.get("pipeline") or [])
        if pipeline and isinstance(pipeline[0], dict) and "$match" in pipeline[0]:
            first = dict(pipeline[0])
            first["$match"] = _merge_filter_with_constraint(first.get("$match"), constraint_filter)
            pipeline[0] = first
        else:
            pipeline.insert(0, {"$match": constraint_filter})
        patched["pipeline"] = pipeline
    return patched


def _build_value_constraint_find_plan(prompt, selected_collection, schema_index, exact_candidates=None):
    collection = str(selected_collection or "").strip()
    if not collection:
        return None
    field_names = _schema_field_names(collection, schema_index)
    constraints = _extract_prompt_value_constraints(
        prompt,
        collection,
        schema_index,
        exact_candidates=exact_candidates,
    )
    constraint_filter = _constraint_filter_doc(constraints)
    if not constraint_filter:
        return None
    projection = {}
    projected = set()
    for constraint in constraints:
        field = str((constraint or {}).get("field") or "").strip()
        if field and field not in projected:
            projection[field] = 1
            projected.add(field)
    for field in _requested_projection_fields(prompt, field_names):
        if field and field not in projected:
            projection[field] = 1
            projected.add(field)
    return {
        "operation": "find",
        "collection": collection,
        "filter": constraint_filter,
        "projection": projection,
        "sort": [],
    }


def _code_like_fields_for_collection(collection_name, schema_index):
    rows = list((schema_index.get(collection_name, {}) or {}).get("fields") or [])
    candidates = []
    for row in rows:
        name = str((row or {}).get("name") or "").strip()
        if not name:
            continue
        lowered = name.lower()
        if any(token in lowered for token in ("code", "number", "reference", "reg", "id", "key")):
            candidates.append(name)
        ftype = _normalize_term((row or {}).get("type") or "")
        if any(token in ftype for token in ("code", "reference")):
            if name not in candidates:
                candidates.append(name)
    return candidates[:6]


async def _try_value_across_collections(db_name, prompt, candidate_collections, table_metadata, schema_index, text_value_terms=None):
    if not prompt or not candidate_collections:
        return None, []
    identifier_values = list(_extract_identifier_values(prompt))
    use_text_fallback = bool(text_value_terms)

    db = mongo_client()[db_name]
    for collection_name in candidate_collections:
        if identifier_values:
            code_fields = _code_like_fields_for_collection(collection_name, schema_index)
            if code_fields:
                for value in identifier_values:
                    exact_match = value.replace("-", "").replace("_", "").replace("/", "")
                    or_clauses = []
                    for field in code_fields:
                        or_clauses.append({field: value})
                        or_clauses.append({field: exact_match})
                        or_clauses.append({field: {"$regex": re.escape(value), "$options": "i"}})
                    if not or_clauses:
                        continue
                    filter_doc = {"$or": or_clauses}
                    try:
                        cursor = db[collection_name].find(filter_doc).limit(5)
                        docs = []
                        async for doc in cursor:
                            docs.append(doc)
                        if docs:
                            return collection_name, docs
                    except Exception:
                        continue

        if use_text_fallback:
            all_text_fields = _text_fields_for_collection(collection_name, schema_index)
            if not all_text_fields:
                continue
            and_clauses = []
            for term in text_value_terms:
                term_clauses = []
                for field in all_text_fields:
                    term_clauses.append({field: {"$regex": re.escape(term), "$options": "i"}})
                if term_clauses:
                    and_clauses.append({"$or": term_clauses})
            if not and_clauses:
                continue
            filter_doc = {"$and": and_clauses}
            try:
                cursor = db[collection_name].find(filter_doc).limit(5)
                docs = []
                async for doc in cursor:
                    docs.append(doc)
                if docs:
                    return collection_name, docs
            except Exception:
                continue

    return None, []


def _text_fields_for_collection(collection_name, schema_index):
    rows = list((schema_index.get(collection_name, {}) or {}).get("fields") or [])
    base_names = {
        str((row or {}).get("name") or "").strip()
        for row in rows
        if str((row or {}).get("name") or "").strip()
    }
    candidates = []
    for row in rows:
        name = str((row or {}).get("name") or "").strip()
        if not name:
            continue
        if name.endswith("_textMode"):
            base = name[:-9]
            if base in base_names:
                candidates.append(name)
        elif name.endswith("_"):
            base = name[:-1]
            if base in base_names:
                candidates.append(name)
        else:
            ftype = _normalize_term((row or {}).get("type") or "")
            if ftype in ("text", "textarea", "richtext", "") or not ftype:
                candidates.append(name)
    return candidates[:12]
