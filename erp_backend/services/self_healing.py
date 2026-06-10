import json
import re


SELF_HEAL_PROMPT = """
You normalize ERP query text into a strict structured intent object.
Return JSON only.

Output schema:
{
  "normalized_query": "<clean concise query>",
  "intent_type": "lookup" | "list" | "count" | "aggregate" | "compare" | "unknown",
  "entity_terms": ["..."],
  "field_terms": ["..."],
  "value_terms": ["..."],
  "confidence": 0.0,
  "needs_clarification": false,
  "hints": {
    "explicit_collection": "<optional>",
    "needs_aggregation": true | false
  }
}

Rules:
1. Preserve user meaning; correct only grammar/typos.
2. Keep terms that look like IDs/codes exactly.
3. If unknown, still return best-effort normalized_query.
"""


def _parse_json(raw_text):
    text = str(raw_text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    def _salvage_partial(source_text):
        if not source_text:
            return {}
        payload = {}
        match = re.search(r'"normalized_query"\s*:\s*"((?:[^"\\]|\\.)*)"', source_text, flags=re.DOTALL)
        if not match:
            match = re.search(r'"normalized_query"\s*:\s*"([^\r\n}]*)', source_text, flags=re.DOTALL)
        if match:
            payload["normalized_query"] = str(match.group(1)).strip(" \t\r\n,")
        match = re.search(r'"intent_type"\s*:\s*"((?:[^"\\]|\\.)*)"', source_text, flags=re.DOTALL)
        if match:
            payload["intent_type"] = str(match.group(1)).strip()
        return payload

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return _salvage_partial(text)
        try:
            return json.loads(match.group(0))
        except Exception:
            return _salvage_partial(match.group(0))


def self_heal_user_query(model, user_query):
    messages = [
        {"role": "system", "content": SELF_HEAL_PROMPT},
        {"role": "user", "content": str(user_query or "")},
    ]
    response = model.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=220,
    )
    text = (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    payload = _parse_json(text)
    if not isinstance(payload, dict):
        payload = {}
    normalized_query = str(payload.get("normalized_query") or "").strip()
    if not normalized_query:
        normalized_query = str(user_query or "").strip()
    return {
        "normalized_query": normalized_query,
        "intent_type": str(payload.get("intent_type") or "unknown").strip().lower(),
        "entity_terms": payload.get("entity_terms") if isinstance(payload.get("entity_terms"), list) else [],
        "field_terms": payload.get("field_terms") if isinstance(payload.get("field_terms"), list) else [],
        "value_terms": payload.get("value_terms") if isinstance(payload.get("value_terms"), list) else [],
        "confidence": float(payload.get("confidence")) if isinstance(payload.get("confidence"), (int, float)) else 0.5,
        "needs_clarification": bool(payload.get("needs_clarification", False)),
        "hints": payload.get("hints") if isinstance(payload.get("hints"), dict) else {},
    }
