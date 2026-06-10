import json
import threading
import time

from erp_backend.core.config import REDIS_CHAT_TTL_SECONDS, REDIS_URL
from erp_backend.core.utils import to_jsonable


_MEMORY_CACHE = {}
_REDIS_CLIENT = None
_REDIS_CLIENT_LOCK = threading.Lock()


def get_redis_client():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    if not REDIS_URL:
        return None
    with _REDIS_CLIENT_LOCK:
        if _REDIS_CLIENT is not None:
            return _REDIS_CLIENT
        try:
            from redis.asyncio import from_url

            _REDIS_CLIENT = from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=0.25,
                socket_timeout=0.25,
                retry_on_timeout=False,
            )
        except Exception:
            _REDIS_CLIENT = None
    return _REDIS_CLIENT


async def redis_get_json(key):
    cached = _MEMORY_CACHE.get(key)
    if cached and cached["expires_at"] > time.time():
        return cached["value"]

    redis_client = get_redis_client()
    if redis_client is None:
        return None
    try:
        raw = await redis_client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def redis_set_json(key, value, ttl_seconds):
    _MEMORY_CACHE[key] = {
        "value": to_jsonable(value),
        "expires_at": time.time() + ttl_seconds,
    }

    redis_client = get_redis_client()
    if redis_client is None:
        return False
    try:
        await redis_client.set(
            key,
            json.dumps(to_jsonable(value), ensure_ascii=False),
            ex=ttl_seconds,
        )
        return True
    except Exception:
        return False


async def redis_delete_prefix(prefix):
    for key in list(_MEMORY_CACHE):
        if key.startswith(prefix):
            _MEMORY_CACHE.pop(key, None)

    redis_client = get_redis_client()
    if redis_client is None:
        return 0
    deleted = 0
    try:
        async for key in redis_client.scan_iter(match=f"{prefix}*"):
            deleted += await redis_client.delete(key)
    except Exception:
        return deleted
    return deleted


def conversation_cache_key(db_name, user_id, conversation_id):
    return f"erp:chat:{db_name}:{user_id}:{conversation_id}"


def compact_chat_messages(messages, limit=12, max_chars=320):
    compact = []
    for message in messages[-limit:]:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            compact.append({"role": role, "content": content[:max_chars]})
    return compact


async def load_conversation_context(db_name, user_id, conversation_id):
    if not conversation_id:
        return []
    key = conversation_cache_key(db_name, user_id, conversation_id)
    cached = await redis_get_json(key)
    if isinstance(cached, list):
        return compact_chat_messages(cached)
    return []


async def save_conversation_context(db_name, user_id, conversation_id, messages):
    if not conversation_id:
        return False
    key = conversation_cache_key(db_name, user_id, conversation_id)
    compact = compact_chat_messages(messages)
    return await redis_set_json(key, compact, REDIS_CHAT_TTL_SECONDS)


async def load_conversation_state(db_name, user_id, conversation_id):
    if not conversation_id:
        return {"messages": [], "last_collection": None}
    key = conversation_cache_key(db_name, user_id, conversation_id)
    cached = await redis_get_json(key)
    if isinstance(cached, dict):
        return {
            "messages": compact_chat_messages(cached.get("messages") or []),
            "last_collection": str(cached.get("last_collection") or "").strip() or None,
        }
    if isinstance(cached, list):
        return {"messages": compact_chat_messages(cached), "last_collection": None}
    return {"messages": [], "last_collection": None}


async def save_conversation_state(db_name, user_id, conversation_id, messages, last_collection=None):
    if not conversation_id:
        return False
    key = conversation_cache_key(db_name, user_id, conversation_id)
    payload = {
        "messages": compact_chat_messages(messages),
        "last_collection": str(last_collection or "").strip() or None,
    }
    return await redis_set_json(key, payload, REDIS_CHAT_TTL_SECONDS)
