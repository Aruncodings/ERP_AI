import re
import hashlib
import threading
from collections import defaultdict

from erp_backend.core.config import (
    VECTOR_DB_ENABLED,
    VECTOR_DB_PATH,
    VECTOR_TOP_K,
    VECTOR_EMBEDDING_MODEL,
    VECTOR_EMBEDDING_DEVICE,
)
from erp_backend.core.utils import lookup_tokens, normalize_lookup_text


_EMBED_FN = None
_EMBED_FN_READY = False
_VALUE_LOOKUP_CACHE = {}
_VALUE_LOOKUP_CACHE_LOCK = threading.Lock()


def _safe_collection_name(name):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(name or "").strip()).lower()


def _normalize_lookup_text(text):
    return normalize_lookup_text(text)


def _lookup_tokens(text):
    return [token for token in lookup_tokens(text) if len(token) >= 2]


def _build_documents(schema_index):
    ids = []
    docs = []
    metas = []
    for collection, item in (schema_index or {}).items():
        table_name = str(item.get("template_name") or collection)
        all_field_names = {
            str(field.get("name") or "").strip()
            for field in (item.get("fields") or [])
            if str(field.get("name") or "").strip()
        }
        for field in item.get("fields") or []:
            field_name = str(field.get("name") or "").strip()
            if not field_name:
                continue
            display = str(field.get("display") or field_name)
            field_type = str(field.get("type") or "")
            aliases = [str(alias) for alias in (field.get("aliases") or []) if str(alias).strip()]
            options = []
            for option in field.get("options") or []:
                if not isinstance(option, dict):
                    continue
                label = str(option.get("label") or "").strip()
                value = str(option.get("value") or "").strip()
                if label:
                    options.append(label)
                if value:
                    options.append(value)
            text = " | ".join(
                [
                    f"collection:{collection}",
                    f"table:{table_name}",
                    f"field:{field_name}",
                    f"display:{display}",
                    f"type:{field_type}",
                    f"aliases:{', '.join(aliases[:12])}",
                    f"options:{', '.join(options[:20])}",
                ]
            )
            ids.append(f"{collection}::{field_name}")
            docs.append(text)
            metas.append(
                {
                    "collection": collection,
                    "field": field_name,
                    "display": display,
                    "type": field_type,
                }
            )
    return ids, docs, metas


def _build_distinct_value_documents(schema_index, field_value_hints):
    ids = []
    docs = []
    metas = []
    for collection, item in (schema_index or {}).items():
        field_map = {
            str(field.get("name") or "").strip(): field
            for field in (item.get("fields") or [])
            if str(field.get("name") or "").strip()
        }
        table_name = str(item.get("template_name") or collection)
        collection_hints = (field_value_hints or {}).get(collection) or {}
        for field_name, values in collection_hints.items():
            field_name = str(field_name or "").strip()
            if not field_name or field_name not in field_map:
                continue
            field_meta = field_map.get(field_name) or {}
            display = str(field_meta.get("display") or field_name)
            field_type = str(field_meta.get("type") or "")
            for raw_value in values or []:
                value = str(raw_value or "").strip()
                if not value:
                    continue
                value_id = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
                ids.append(f"{collection}::{field_name}::value::{value_id}")
                docs.append(
                    " | ".join(
                        [
                            f"collection:{collection}",
                            f"table:{table_name}",
                            f"field:{field_name}",
                            f"display:{display}",
                            f"type:{field_type}",
                            f"value:{value}",
                        ]
                    )
                )
                metas.append(
                    {
                        "collection": collection,
                        "field": field_name,
                        "display": display,
                        "type": field_type,
                        "kind": "distinct_value",
                        "value": value,
                    }
                )
    return ids, docs, metas


def _get_client():
    if not VECTOR_DB_ENABLED:
        return None
    try:
        import chromadb
    except Exception:
        return None
    try:
        return chromadb.PersistentClient(path=VECTOR_DB_PATH)
    except Exception:
        return None


def _get_embedding_function():
    global _EMBED_FN, _EMBED_FN_READY
    if _EMBED_FN_READY:
        return _EMBED_FN

    _EMBED_FN_READY = True
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except Exception:
        _EMBED_FN = None
        return _EMBED_FN

    model_name = str(VECTOR_EMBEDDING_MODEL or "").strip()
    if not model_name:
        _EMBED_FN = None
        return _EMBED_FN

    device = str(VECTOR_EMBEDDING_DEVICE or "auto").strip().lower()
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    try:
        _EMBED_FN = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device,
        )
    except Exception:
        _EMBED_FN = None
    return _EMBED_FN


def _get_collection(client, db_name):
    collection_name = f"erp_schema_{_safe_collection_name(db_name)}"
    embed_fn = _get_embedding_function()
    try:
        if embed_fn is not None:
            return client.get_or_create_collection(
                name=collection_name,
                embedding_function=embed_fn,
            )
        return client.get_or_create_collection(name=collection_name)
    except Exception:
        return None


def upsert_schema_vectors(db_name, schema_index, field_value_hints=None):
    client = _get_client()
    if client is None:
        return False
    ids, docs, metas = _build_documents(schema_index)
    if field_value_hints:
        value_ids, value_docs, value_metas = _build_distinct_value_documents(schema_index, field_value_hints)
        ids.extend(value_ids)
        docs.extend(value_docs)
        metas.extend(value_metas)
    if not ids:
        return False
    col = _get_collection(client, db_name)
    if col is None:
        return False
    try:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        return True
    except Exception:
        return False


def warm_reverse_lookup_cache(db_name, schema_index, field_value_hints=None):
    value_map = defaultdict(list)
    token_map = defaultdict(set)
    for collection, item in (schema_index or {}).items():
        field_map = {
            str(field.get("name") or "").strip(): field
            for field in (item.get("fields") or [])
            if str(field.get("name") or "").strip()
        }
        collection_hints = (field_value_hints or {}).get(collection) or {}
        for field_name, values in collection_hints.items():
            field_name = str(field_name or "").strip()
            if not field_name or field_name not in field_map:
                continue
            field_meta = field_map.get(field_name) or {}
            display = str(field_meta.get("display") or field_name)
            field_type = str(field_meta.get("type") or "")
            for raw_value in values or []:
                value = str(raw_value or "").strip()
                normalized_value = _normalize_lookup_text(value)
                if not normalized_value:
                    continue
                entry = {
                    "collection": collection,
                    "field": field_name,
                    "display": display,
                    "type": field_type,
                    "value": value,
                }
                bucket = value_map[normalized_value]
                if entry not in bucket and len(bucket) < 12:
                    bucket.append(entry)
                for token in _lookup_tokens(normalized_value):
                    token_map[token].add(normalized_value)
    cache_key = _safe_collection_name(db_name)
    with _VALUE_LOOKUP_CACHE_LOCK:
        _VALUE_LOOKUP_CACHE[cache_key] = {
            "values": dict(value_map),
            "tokens": {token: sorted(values) for token, values in token_map.items()},
        }
    return bool(value_map)


def _retrieve_exact_field_candidates_core(db_name, user_query, top_k=VECTOR_TOP_K, allowed_collections=None):
    cache_key = _safe_collection_name(db_name)
    with _VALUE_LOOKUP_CACHE_LOCK:
        cache = _VALUE_LOOKUP_CACHE.get(cache_key) or {}
    values_map = cache.get("values") or {}
    token_map = cache.get("tokens") or {}
    if not values_map:
        return {}, {}

    normalized_query = _normalize_lookup_text(user_query)
    if not normalized_query:
        return {}, {}
    query_tokens = set(_lookup_tokens(normalized_query))
    if not query_tokens and normalized_query not in values_map:
        return {}, {}

    candidate_scores = defaultdict(float)

    def add_candidate(value_key, score):
        if not value_key:
            return
        current = candidate_scores.get(value_key, 0.0)
        if score > current:
            candidate_scores[value_key] = score

    if normalized_query in values_map:
        add_candidate(normalized_query, 100.0)

    for token in query_tokens:
        for value_key in token_map.get(token, []):
            if not value_key:
                continue
            value_tokens = set(_lookup_tokens(value_key))
            overlap = len(query_tokens & value_tokens)
            if overlap <= 0 and value_key not in normalized_query and normalized_query not in value_key:
                continue
            score = float(overlap * 12)
            if value_key == normalized_query:
                score += 100
            elif value_key in normalized_query:
                score += 70 + min(30, len(value_key) // 3)
            elif normalized_query in value_key:
                score += 50
            # Boost identifier/code patterns (e.g. AB-123, BRANCH_01, DOC/2024/001)
            if re.search(r'^[A-Z]{2,}[-_/][A-Z0-9_-]{2,}$', value_key):
                score += 25
            elif re.search(r'^[A-Z0-9]{3,}[-_ /]\d{2,}', value_key):
                score += 15
            add_candidate(value_key, score)

    if not candidate_scores:
        return {}, {}

    ranked_values = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[: max(1, int(top_k or VECTOR_TOP_K))]
    grouped = defaultdict(list)
    matched_values_map = defaultdict(list)  # collection -> [{field, value}]
    allowed = {
        str(item).strip()
        for item in (allowed_collections or [])
        if str(item).strip()
    }
    for value_key, _score in ranked_values:
        for entry in values_map.get(value_key, []):
            collection = str(entry.get("collection") or "").strip()
            field = str(entry.get("field") or "").strip()
            value = str(entry.get("value") or value_key).strip()
            if allowed and collection not in allowed:
                continue
            if collection and field and field not in grouped[collection]:
                grouped[collection].append(field)
            if collection and field and value:
                # Only keep the first (highest-scored) matched value per field
                existing_fields = {mv["field"] for mv in matched_values_map[collection]}
                if field not in existing_fields:
                    matched_values_map[collection].append({"field": field, "value": value})
    return dict(grouped), dict(matched_values_map)


def retrieve_exact_field_candidates(db_name, user_query, top_k=VECTOR_TOP_K, allowed_collections=None):
    """Backward-compatible wrapper — returns only the field names dict."""
    fields, _ = _retrieve_exact_field_candidates_core(db_name, user_query, top_k, allowed_collections)
    return fields


def retrieve_exact_field_candidates_with_values(db_name, user_query, top_k=VECTOR_TOP_K, allowed_collections=None):
    """Returns (field_names_dict, matched_values_dict) for full value injection."""
    return _retrieve_exact_field_candidates_core(db_name, user_query, top_k, allowed_collections)




def retrieve_field_candidates(db_name, user_query, top_k=VECTOR_TOP_K, allowed_collections=None):
    client = _get_client()
    if client is None:
        return {}
    col = _get_collection(client, db_name)
    if col is None:
        return {}
    try:
        result = col.query(query_texts=[str(user_query or "")], n_results=max(1, int(top_k)))
    except Exception:
        return {}

    metadatas = (result or {}).get("metadatas") or []
    rows = metadatas[0] if metadatas else []
    query_norm = _normalize_lookup_text(user_query)
    query_tokens = set(_lookup_tokens(query_norm))
    scored_rows = []
    allowed = {
        str(item).strip()
        for item in (allowed_collections or [])
        if str(item).strip()
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        collection = str(row.get("collection") or "").strip()
        field = str(row.get("field") or "").strip()
        if allowed and collection not in allowed:
            continue
        if not collection or not field:
            continue
        # Re-rank by field display/alias overlap with query tokens
        display = str(row.get("display") or field).strip()
        display_norm = _normalize_lookup_text(display)
        display_tokens = set(_lookup_tokens(display_norm))
        overlap = len(query_tokens & display_tokens)
        score = overlap * 8
        for qt in query_tokens:
            if qt and qt in display_norm:
                score += 6
        scored_rows.append((score, collection, field))
    scored_rows.sort(key=lambda x: -x[0])
    grouped = defaultdict(list)
    for _score, collection, field in scored_rows:
        if field not in grouped[collection]:
            grouped[collection].append(field)
    return dict(grouped)
