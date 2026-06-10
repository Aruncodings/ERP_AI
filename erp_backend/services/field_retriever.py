import re

from erp_backend.core.utils import lookup_tokens, normalize_lookup_text


def _norm(text):
    return normalize_lookup_text(text)


def _tokens(text):
    return set(lookup_tokens(text))


def _collection_profile_score(query_tokens, fields):
    profile = 0
    has_identity_intent = bool(query_tokens & {"who", "person", "employee", "staff", "manager", "designation", "title", "role"})
    if not has_identity_intent:
        return 0

    for field in fields or []:
        aliases = field.get("aliases") or []
        joined = " ".join(str(a).lower() for a in aliases)
        if any(term in joined for term in ("name", "username", "email", "employee", "staff")):
            profile += 2
        if any(term in joined for term in ("designation", "title", "role", "department")):
            profile += 3
    return profile


def retrieve_candidates(user_query, schema_index, max_collections=2, max_fields=24):
    query_norm = _norm(user_query)
    query_tokens = _tokens(user_query)

    collection_rank = []
    for collection, item in schema_index.items():
        score = 0
        for alias in item.get("collection_aliases") or []:
            alias_tokens = _tokens(alias)
            overlap = len(query_tokens & alias_tokens)
            score += overlap * 4
            if alias and alias in query_norm:
                score += 8
        for field in item.get("fields") or []:
            for alias in field.get("aliases") or []:
                alias_tokens = _tokens(alias)
                overlap = len(query_tokens & alias_tokens)
                score += overlap
                if alias and alias in query_norm:
                    score += 2
        score += _collection_profile_score(query_tokens, item.get("fields") or [])
        collection_rank.append((collection, score))
    collection_rank.sort(key=lambda row: row[1], reverse=True)
    # Keep only positively matched collections; zero-score collections introduce noisy routing.
    top_collections = [name for name, score in collection_rank[:max_collections] if score > 0]

    field_candidates = {}
    for collection in top_collections:
        scored_fields = []
        for field in schema_index.get(collection, {}).get("fields") or []:
            score = 0
            for alias in field.get("aliases") or []:
                alias_tokens = _tokens(alias)
                score += len(query_tokens & alias_tokens) * 3
                if alias and alias in query_norm:
                    score += 6
            if score > 0:
                scored_fields.append((field, score))
        scored_fields.sort(key=lambda row: row[1], reverse=True)
        field_candidates[collection] = [field for field, _ in scored_fields[:max_fields]]

    return {
        "collections": top_collections,
        "field_candidates": field_candidates,
    }
