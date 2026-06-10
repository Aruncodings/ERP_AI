import re


HIDDEN_FIELD_NAMES = {
    "_id",
    "__v",
    "id",
    "keywords",
    "password",
    "hashed_password",
    "hash",
    "salt",
    "token",
    "accessToken",
    "refreshToken",
    "secret",
    "otp",
    "pin",
    "templateId",
    "createdBy",
    "updatedBy",
    "deletedBy",
    "createdAt",
    "updatedAt",
    "deletedAt",
    "isDeleted",
}

HIDDEN_FIELD_TOKENS = (
    "password",
    "hashed",
    "secret",
    "token",
    "otp",
    "pin",
    "salt",
    "keyword",
    "createdby",
    "updatedby",
    "deletedby",
    "createdat",
    "updatedat",
    "deletedat",
)

_POLICY_HIDDEN_FIELD_NAMES = set()
_POLICY_HIDDEN_FIELD_TOKENS = ()
_STATIC_NORMALIZED_HIDDEN = set()
_POLICY_NORMALIZED_HIDDEN = set()


def _refresh_hidden_caches():
    global _STATIC_NORMALIZED_HIDDEN, _POLICY_NORMALIZED_HIDDEN
    _STATIC_NORMALIZED_HIDDEN = {normalized_field_name(item) for item in HIDDEN_FIELD_NAMES}
    _POLICY_NORMALIZED_HIDDEN = {normalized_field_name(item) for item in _POLICY_HIDDEN_FIELD_NAMES}


def configure_hidden_field_policy(hidden_names=None, hidden_tokens=None):
    global _POLICY_HIDDEN_FIELD_NAMES, _POLICY_HIDDEN_FIELD_TOKENS
    names = set()
    for item in hidden_names or []:
        text = str(item or "").strip()
        if text:
            names.add(text)
    tokens = []
    for item in hidden_tokens or []:
        text = normalized_field_name(item)
        if text and text not in tokens:
            tokens.append(text)
    _POLICY_HIDDEN_FIELD_NAMES = names
    _POLICY_HIDDEN_FIELD_TOKENS = tuple(tokens)
    _refresh_hidden_caches()


def normalized_field_name(field):
    return re.sub(r"[^a-z0-9]+", "", str(field).lower())

_refresh_hidden_caches()


def _get_words(text):
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    s2 = re.sub(r"[^a-zA-Z0-9]+", " ", s1)
    return {w.lower() for w in s2.split()}


def is_hidden_field(field):
    field_text = str(field)
    leaf = field_text.split(".")[-1]
    normalized = normalized_field_name(leaf)
    looks_like_identifier = (
        leaf.lower() == "id"
        or leaf.lower() == "_id"
        or leaf.lower().endswith("_id")
        or leaf.lower().endswith("-id")
    )
    leaf_words = _get_words(leaf)
    return (
        field_text in HIDDEN_FIELD_NAMES
        or field_text in _POLICY_HIDDEN_FIELD_NAMES
        or leaf in HIDDEN_FIELD_NAMES
        or leaf in _POLICY_HIDDEN_FIELD_NAMES
        or normalized in _STATIC_NORMALIZED_HIDDEN
        or normalized in _POLICY_NORMALIZED_HIDDEN
        or any(token in leaf_words for token in HIDDEN_FIELD_TOKENS)
        or any(token in leaf_words for token in _POLICY_HIDDEN_FIELD_TOKENS)
        or looks_like_identifier
    )


def sanitize_doc_for_display(doc):
    if not isinstance(doc, dict):
        return doc
    cleaned = {}
    for field, value in doc.items():
        if is_hidden_field(field):
            continue
        if isinstance(value, dict):
            nested = sanitize_doc_for_display(value)
            if nested:
                cleaned[field] = nested
        elif isinstance(value, list):
            cleaned[field] = [
                sanitize_doc_for_display(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[field] = value
    return cleaned


def sanitize_docs_for_display(docs):
    return [sanitize_doc_for_display(doc) for doc in docs]
