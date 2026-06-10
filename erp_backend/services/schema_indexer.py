import re

from erp_backend.core.utils import lookup_tokens, normalize_lookup_text


def _norm(text):
    return normalize_lookup_text(text)


def _tokens(text):
    return {token for token in lookup_tokens(text)}


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


def _field_aliases(field_row):
    field = str(field_row.get("field") or field_row.get("fieldName") or "")
    display = str(field_row.get("display") or field_row.get("displayName") or "")
    aliases = _expand_alias_tokens(field)
    aliases.update(_expand_alias_tokens(display))
    for option in field_row.get("options") or []:
        if not isinstance(option, dict):
            continue
        aliases.update(_expand_alias_tokens(option.get("label") or ""))
        aliases.update(_expand_alias_tokens(option.get("value") or ""))
    return sorted(item for item in aliases if item)


def build_schema_index(table_metadata, allowed_collections, ai_template_schemas=None):
    template_map = {}
    for item in ai_template_schemas or []:
        if not isinstance(item, dict):
            continue
        collection = str(item.get("collectionName") or "").strip()
        if collection:
            template_map[collection] = item

    index = {}
    for collection in allowed_collections:
        meta = table_metadata.get(collection, {})
        template_schema = template_map.get(collection, {})
        template_fields = template_schema.get("fields") or []
        fields = []
        source_fields = []
        seen_source = set()
        for row in list(template_fields) + list(meta.get("fields") or []):
            field_name = str(row.get("fieldName") or row.get("field") or "").strip()
            if not field_name or field_name in seen_source:
                continue
            seen_source.add(field_name)
            source_fields.append(row)
        for row in source_fields:
            field_name = str(row.get("fieldName") or row.get("field") or "").strip()
            if not field_name:
                continue
            fields.append(
                {
                    "name": field_name,
                    "display": str(row.get("displayName") or row.get("display") or field_name),
                    "type": str(row.get("dataType") or row.get("type") or ""),
                    "aliases": _field_aliases(row),
                    "lookup_collection": str(
                        row.get("lookupTargetCollection")
                        or row.get("lookup_collection")
                        or ""
                    ),
                    "options": row.get("options") or [],
                }
            )
        collection_aliases = {
            _norm(collection),
            _norm(meta.get("template_name", collection)),
            _norm(template_schema.get("templateName") or ""),
        }
        for term in meta.get("business_terms") or []:
            collection_aliases.add(_norm(term))
        index[collection] = {
            "collection": collection,
            "template_name": str(meta.get("template_name", collection)),
            "collection_aliases": sorted(item for item in collection_aliases if item),
            "collection_tokens": set().union(*[_tokens(alias) for alias in collection_aliases if alias]),
            "fields": fields,
        }
    return index
