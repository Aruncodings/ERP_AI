import json
import re
from datetime import datetime
from difflib import SequenceMatcher


_LOOKUP_TOKEN_CANONICALS = {
    "certification": "certificate",
    "certifications": "certificate",
    "filling": "filing",
    "fillings": "filing",
    "filings": "filing",
    "penality": "penalty",
    "penalities": "penalty",
    "organisation": "organization",
    "organisations": "organization",
    "expiry": "expiry",
    "expire": "expiry",
    "expires": "expiry",
    "expiring": "expiry",
    "expiration": "expiry",
    "expired": "expiry",
    "renewals": "renewal",
}


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__str__") and not isinstance(value, (str, int, float, bool, type(None))):
        return str(value)
    return value


def flatten_for_table(doc, prefix=""):
    row = {}
    for key, value in doc.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            row.update(flatten_for_table(value, name))
        elif isinstance(value, list):
            row[name] = json.dumps(to_jsonable(value), ensure_ascii=False)
        else:
            row[name] = to_jsonable(value)
    return row


def normalize_lookup_text(value):
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    value = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    tokens = []
    for raw_token in value.split():
        token = singularize_word(raw_token)
        token = _LOOKUP_TOKEN_CANONICALS.get(token, token)
        if token:
            tokens.append(token)
    return " ".join(tokens)


def singularize_word(word):
    word = str(word).lower()
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3 and (
        word.endswith("ches")
        or word.endswith("shes")
        or word.endswith("xes")
        or word.endswith("zes")
        or word.endswith("ses")
    ):
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def lookup_tokens(value):
    return [token for token in normalize_lookup_text(value).split() if token]


def tokenize_lookup_text(value):
    return lookup_tokens(value)


def similarity(left, right):
    return SequenceMatcher(None, normalize_lookup_text(left), normalize_lookup_text(right)).ratio()


def humanize_name(value):
    text = str(value or "").strip()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\b[a-f0-9]{12,}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(part.capitalize() for part in text.split())
