import asyncio
import threading

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from erp_backend.core.config import (
    BLOCKED_COLLECTIONS,
    BLOCKED_DATABASES,
    DEFAULT_DB_NAME,
    DEFAULT_USERS,
    MAX_TABLE_COUNT,
    MONGO_TIMEOUT_MS,
    MONGO_URI,
    SYSTEM_FIELD_NAMES,
    REDIS_METADATA_TTL_SECONDS,
    REDIS_RBAC_TTL_SECONDS,
    REDIS_SCHEMA_TTL_SECONDS,
    ROLLOUT_POLICY_CACHE_TTL_SECONDS,
    ROLLOUT_RUNTIME_POLICY_COLLECTION,
    SCHEMA_SAMPLE_SIZE,
)
from erp_backend.core.utils import to_jsonable
from erp_backend.storage.cache import redis_get_json, redis_set_json


_MONGO_CLIENT = None
_MONGO_CLIENT_LOCK = threading.Lock()
_ASYNC_LOOP = None
_ASYNC_THREAD = None
_ASYNC_LOOP_LOCK = threading.Lock()


_TEMPLATE_SCHEMA_PIPELINE = [
    {
        "$match": {
            "isActive": True,
            "isDeleted": {"$ne": True},
        }
    },
    {
        "$project": {
            "_id": 0,
            "templateName": 1,
            "collectionName": 1,
            "fields": {
                "$reduce": {
                    "input": {"$ifNull": ["$sections", []]},
                    "initialValue": [],
                    "in": {
                        "$concatArrays": [
                            "$$value",
                            {
                                "$map": {
                                    "input": {
                                        "$cond": {
                                            "if": {"$eq": ["$$this.hasTabs", True]},
                                            "then": {
                                                "$reduce": {
                                                    "input": {"$ifNull": ["$$this.tabs", []]},
                                                    "initialValue": [],
                                                    "in": {
                                                        "$concatArrays": [
                                                            "$$value",
                                                            {"$ifNull": ["$$this.fields", []]},
                                                        ]
                                                    },
                                                }
                                            },
                                            "else": {"$ifNull": ["$$this.layout", []]},
                                        }
                                    },
                                    "as": "field",
                                    "in": {
                                        "fieldName": "$$field.labelName",
                                        "displayName": "$$field.displayName",
                                        "dataType": "$$field.dataType",
                                        "isTabularColumn": {
                                            "$cond": {
                                                "if": {"$eq": ["$$this.layoutType", "TABULAR"]},
                                                "then": True,
                                                "else": "$$REMOVE",
                                            }
                                        },
                                        "options": {
                                            "$cond": {
                                                "if": {
                                                    "$in": [
                                                        "$$field.dataType",
                                                        ["SELECT", "MULTI_SELECT"],
                                                    ]
                                                },
                                                "then": {
                                                    "$map": {
                                                        "input": {
                                                            "$ifNull": [
                                                                "$$field.selectConfig.options",
                                                                [],
                                                            ]
                                                        },
                                                        "as": "opt",
                                                        "in": {
                                                            "label": "$$opt.label",
                                                            "value": "$$opt.value",
                                                        },
                                                    }
                                                },
                                                "else": "$$REMOVE",
                                            }
                                        },
                                        "lookupTarget": {
                                            "$cond": {
                                                "if": {
                                                    "$in": [
                                                        "$$field.dataType",
                                                        ["LOOK_UP", "MULTI_LOOKUP"],
                                                    ]
                                                },
                                                "then": {
                                                    "$ifNull": [
                                                        "$$field.lookupConfig.lookupTemplateName",
                                                        "$$field.lookupConfig.lookupCollectionName",
                                                    ]
                                                },
                                                "else": "$$REMOVE",
                                            }
                                        },
                                        "lookupTargetCollection": {
                                            "$cond": {
                                                "if": {
                                                    "$in": [
                                                        "$$field.dataType",
                                                        ["LOOK_UP", "MULTI_LOOKUP"],
                                                    ]
                                                },
                                                "then": "$$field.lookupConfig.lookupCollectionName",
                                                "else": "$$REMOVE",
                                            }
                                        },
                                    },
                                }
                            },
                        ]
                    },
                }
            },
        }
    },
]


def _iter_template_fields(template):
    for section in template.get("sections") or []:
        layout_type = str(section.get("layoutType") or "")
        if section.get("hasTabs") is True:
            for tab in section.get("tabs") or []:
                for field in tab.get("fields") or []:
                    if isinstance(field, dict):
                        yield field, layout_type
            continue
        for field in section.get("layout") or []:
            if isinstance(field, dict):
                yield field, layout_type


def run_async(coro):
    loop = _ensure_async_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def _ensure_async_loop():
    global _ASYNC_LOOP, _ASYNC_THREAD
    if _ASYNC_LOOP is not None and _ASYNC_THREAD is not None and _ASYNC_THREAD.is_alive():
        return _ASYNC_LOOP

    with _ASYNC_LOOP_LOCK:
        if _ASYNC_LOOP is not None and _ASYNC_THREAD is not None and _ASYNC_THREAD.is_alive():
            return _ASYNC_LOOP

        ready = threading.Event()

        def _loop_worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            global _ASYNC_LOOP
            _ASYNC_LOOP = loop
            ready.set()
            loop.run_forever()

        _ASYNC_THREAD = threading.Thread(target=_loop_worker, name="chat-async-loop", daemon=True)
        _ASYNC_THREAD.start()
        ready.wait(timeout=5)
        if _ASYNC_LOOP is None:
            raise RuntimeError("Failed to start background async loop.")
        return _ASYNC_LOOP


def get_mongo_client():
    global _MONGO_CLIENT
    if _MONGO_CLIENT is not None:
        return _MONGO_CLIENT

    with _MONGO_CLIENT_LOCK:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = AsyncIOMotorClient(
                MONGO_URI,
                serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
                connectTimeoutMS=MONGO_TIMEOUT_MS,
            )
    return _MONGO_CLIENT


def mongo_client():
    return get_mongo_client()


async def ping_mongo():
    await mongo_client().admin.command("ping")


async def list_databases():
    cache_key = "erp:databases"
    cached = await redis_get_json(cache_key)
    if cached is not None:
        return cached
    names = await mongo_client().list_database_names()
    allowed = [name for name in names if name not in BLOCKED_DATABASES]
    result = allowed or [DEFAULT_DB_NAME]
    await redis_set_json(cache_key, result, REDIS_METADATA_TTL_SECONDS)
    return result


async def list_collections(db_name):
    cache_key = f"erp:collections:{db_name}"
    cached = await redis_get_json(cache_key)
    if cached is not None:
        return cached

    db = mongo_client()[db_name]
    names = await db.list_collection_names()
    existing_collections = set(names)

    valid_tables = set()
    if "templates" in existing_collections:
        cursor = db["templates"].find(
            {"collectionName": {"$exists": True, "$ne": ""}},
            {"collectionName": 1, "isDeleted": 1, "isActive": 1},
        )
        async for template in cursor:
            if template.get("isDeleted") is True:
                continue
            if template.get("isActive") is False:
                continue
            collection_name = template.get("collectionName")
            if collection_name in existing_collections:
                valid_tables.add(str(collection_name))

    if not valid_tables:
        valid_tables = set(names)

    result = sorted(
        name
        for name in valid_tables
        if name not in BLOCKED_COLLECTIONS and not name.startswith("system.")
    )[:MAX_TABLE_COUNT]
    await redis_set_json(cache_key, result, REDIS_METADATA_TTL_SECONDS)
    return result


async def load_table_metadata(db_name):
    cache_key = f"erp:table_metadata:{db_name}"
    cached = await redis_get_json(cache_key)
    if cached is not None:
        return cached

    db = mongo_client()[db_name]
    collections = await list_collections(db_name)
    metadata = {
        name: {
            "collection": name,
            "label": name,
            "template_name": name,
        }
        for name in collections
    }

    if "templates" not in await db.list_collection_names():
        return metadata

    cursor = db["templates"].find(
        {"collectionName": {"$in": collections}},
        {
            "collectionName": 1,
            "templateName": 1,
            "name": 1,
            "labelName": 1,
            "shortName": 1,
            "templateSlug": 1,
            "sections": 1,
        },
    )
    async for template in cursor:
        collection_name = template.get("collectionName")
        template_name = template.get("templateName") or template.get("name") or collection_name
        if collection_name in metadata:
            fields = []
            seen_fields = set()
            business_terms = {
                str(template_name),
                str(template.get("labelName") or ""),
                str(template.get("shortName") or ""),
                str(template.get("templateSlug") or ""),
                str(collection_name),
            }
            for item, layout_type in _iter_template_fields(template):
                field_name = item.get("labelName")
                if not field_name:
                    continue
                field_key = str(field_name)
                if field_key in seen_fields:
                    continue
                seen_fields.add(field_key)
                lookup_config = item.get("lookupConfig") or {}
                fields.append(
                    {
                        "field": field_key,
                        "display": str(item.get("displayName") or field_name),
                        "type": str(item.get("dataType") or ""),
                        "is_tabular_column": layout_type == "TABULAR",
                        "lookup_collection": str(lookup_config.get("lookupCollectionName") or ""),
                        "lookup_display": str(lookup_config.get("lookupDisplayField") or ""),
                        "lookup_other_display": lookup_config.get("lookupOtherDisplayFields") or [],
                    }
                )
                business_terms.add(str(field_name))
                business_terms.add(str(item.get("displayName") or field_name))
                business_terms.add(str(item.get("dataType") or ""))
                business_terms.add(str(lookup_config.get("lookupTemplateName") or ""))
                business_terms.add(str(lookup_config.get("lookupDisplayField") or ""))
                for extra_field in lookup_config.get("lookupOtherDisplayFields") or []:
                    business_terms.add(str(extra_field))
            metadata[collection_name] = {
                "collection": collection_name,
                "label": f"{template_name} ({collection_name})",
                "template_name": str(template_name),
                "template_id": str(template.get("_id")),
                "fields": fields,
                "business_terms": sorted(term for term in business_terms if term),
            }
    await redis_set_json(cache_key, metadata, REDIS_METADATA_TTL_SECONDS)
    return metadata


async def sample_collection(db_name, collection_name, limit=SCHEMA_SAMPLE_SIZE):
    cache_key = f"erp:sample:{db_name}:{collection_name}:{limit}"
    cached = await redis_get_json(cache_key)
    if cached is not None:
        return cached

    cursor = mongo_client()[db_name][collection_name].find({}).limit(limit)
    docs = []
    async for doc in cursor:
        docs.append(to_jsonable(doc))
    await redis_set_json(cache_key, docs, REDIS_METADATA_TTL_SECONDS)
    return docs


async def get_template_schema(db_name, collection_name):
    cache_key = f"erp:template_schema:{db_name}:{collection_name}"
    cached = await redis_get_json(cache_key)
    if cached is not None:
        return cached

    db = mongo_client()[db_name]
    names = await db.list_collection_names()
    if "templates" not in names:
        return None

    pipeline = _TEMPLATE_SCHEMA_PIPELINE + [
        {"$match": {"collectionName": collection_name}},
        {"$limit": 1},
    ]
    result = None
    try:
        async for doc in db["templates"].aggregate(pipeline):
            result = doc
            break
    except Exception:
        return None

    if result:
        await redis_set_json(cache_key, result, REDIS_SCHEMA_TTL_SECONDS)
    return result


async def load_ai_template_schemas(db_name, allowed_collections):
    cache_key = f"erp:ai_template_schemas:{db_name}"
    cached = await redis_get_json(cache_key)
    if isinstance(cached, list):
        if allowed_collections:
            allowed = set(allowed_collections)
            return [item for item in cached if str(item.get("collectionName") or "") in allowed]
        return cached

    db = mongo_client()[db_name]
    names = await db.list_collection_names()
    if "templates" not in names:
        return []

    rows = []
    try:
        async for doc in db["templates"].aggregate(_TEMPLATE_SCHEMA_PIPELINE):
            if not isinstance(doc, dict):
                continue
            rows.append(doc)
    except Exception:
        return []

    await redis_set_json(cache_key, rows, REDIS_SCHEMA_TTL_SECONDS)
    if allowed_collections:
        allowed = set(allowed_collections)
        return [item for item in rows if str(item.get("collectionName") or "") in allowed]
    return rows


def _default_runtime_policy():
    created_time_candidates = sorted(
        name for name in SYSTEM_FIELD_NAMES if str(name).lower().startswith("created")
    )
    return {
        "hiddenFieldNames": [],
        "hiddenFieldTokens": [],
        "systemFieldAllowlist": sorted(SYSTEM_FIELD_NAMES),
        "softDeleteFieldCandidates": ["isDeleted", "deleted", "is_deleted"],
        "activeFieldCandidates": ["isActive", "active", "is_active"],
        "createdTimeFieldCandidates": created_time_candidates,
        "labelFieldCandidates": ["name", "title", "description", "displayName", "label"],
        "businessIdFieldCandidates": ["obligationId", "code", "number", "referenceRegNumber"],
        "lookupDisplayFallbackCandidates": ["name", "displayName", "title", "code", "label"],
    }


def _normalize_runtime_policy(raw):
    policy = _default_runtime_policy()
    if not isinstance(raw, dict):
        return policy
    for key, default_value in policy.items():
        value = raw.get(key)
        if isinstance(default_value, list):
            normalized = []
            for item in value or []:
                text = str(item or "").strip()
                if text and text not in normalized:
                    normalized.append(text)
            policy[key] = normalized
        else:
            policy[key] = value if value is not None else default_value
    if isinstance(raw.get("additionalSystemFields"), list):
        combined = list(policy.get("systemFieldAllowlist") or [])
        for item in raw.get("additionalSystemFields") or []:
            text = str(item or "").strip()
            if text and text not in combined:
                combined.append(text)
        policy["systemFieldAllowlist"] = combined
    return policy


async def load_runtime_policy(db_name, collection_name=None, force_refresh=False):
    policy_collection = str(collection_name or ROLLOUT_RUNTIME_POLICY_COLLECTION or "").strip()
    cache_key = f"erp:runtime_policy:{db_name}:{policy_collection or 'default'}"
    if not force_refresh:
        cached = await redis_get_json(cache_key)
        if isinstance(cached, dict):
            return _normalize_runtime_policy(cached)

    if not policy_collection:
        return _default_runtime_policy()

    db = mongo_client()[db_name]
    names = await db.list_collection_names()
    if policy_collection not in names:
        return _default_runtime_policy()

    doc = await db[policy_collection].find_one(
        {"$or": [{"isActive": {"$exists": False}}, {"isActive": True}]},
        {"_id": 0},
    )
    policy = _normalize_runtime_policy(doc)
    await redis_set_json(cache_key, policy, ROLLOUT_POLICY_CACHE_TTL_SECONDS)
    return policy


def infer_schema_from_docs(docs):
    fields: dict[str, set[str]] = {}

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                fields.setdefault(name, set()).add(type(child).__name__)
                walk(child, name)
        elif isinstance(value, list):
            fields.setdefault(prefix, set()).add("list")
            for item in value[:3]:
                walk(item, prefix)

    for doc in docs:
        walk(doc)
    return {field: sorted(types) for field, types in sorted(fields.items())}


async def estimated_count(db_name, collection_name, filter_doc=None):
    collection = mongo_client()[db_name][collection_name]
    if filter_doc:
        return await collection.count_documents(filter_doc, limit=10_000)
    return await collection.estimated_document_count()


def normalize_user_doc(doc):
    display_name = (
        doc.get("display_name")
        or doc.get("name")
        or doc.get("username")
        or doc.get("email")
        or doc.get("user_id")
        or str(doc.get("_id"))
    )
    user_id = str(doc.get("user_id") or doc.get("username") or doc.get("email") or doc.get("_id"))
    allowed = (
        doc.get("allowed_collections")
        or doc.get("allowed_tables")
        or doc.get("collections")
        or doc.get("tables")
        or doc.get("permissions")
        or []
    )
    if isinstance(allowed, dict):
        allowed = [key for key, enabled in allowed.items() if enabled]
    if isinstance(allowed, str):
        allowed = [allowed]
    return {
        "user_id": user_id,
        "display_name": str(display_name),
        "email": str(doc.get("email") or ""),
        "mongo_id": str(doc.get("_id")),
        "allowed_collections": [str(item) for item in allowed],
    }


def mongo_id_candidates(value):
    values = {str(value)}
    try:
        values.add(ObjectId(str(value)))
    except Exception:
        pass
    return list(values)


async def load_template_collection_map(db):
    if "templates" not in await db.list_collection_names():
        return {}

    mapping = {}
    cursor = db["templates"].find({}, {"collectionName": 1, "collection": 1, "templateName": 1})
    async for doc in cursor:
        collection_name = doc.get("collectionName") or doc.get("collection")
        if collection_name:
            mapping[str(doc.get("_id"))] = str(collection_name)
    return mapping


async def load_role_allowed_collections(db, role_ids):
    if not role_ids or "roles" not in await db.list_collection_names():
        return set()

    template_map = await load_template_collection_map(db)
    allowed = set()
    role_id_candidates = []
    for role_id in role_ids:
        role_id_candidates.extend(mongo_id_candidates(role_id))

    cursor = db["roles"].find(
        {"_id": {"$in": role_id_candidates}},
        {"slug": 1, "name": 1, "permissionIds": 1, "entityPermissionRules": 1},
    )
    async for role in cursor:
        if role.get("slug") == "super-admin":
            allowed.add("*")
            continue

        for rule in role.get("entityPermissionRules") or []:
            if not isinstance(rule, dict):
                continue
            if str(rule.get("action", "")).lower() != "read":
                continue
            template_id = rule.get("templateId")
            collection_name = template_map.get(str(template_id))
            if collection_name:
                allowed.add(collection_name)
    return allowed


async def load_role_names(db, role_ids):
    if not role_ids or "roles" not in await db.list_collection_names():
        return []

    role_id_candidates = []
    for role_id in role_ids:
        role_id_candidates.extend(mongo_id_candidates(role_id))

    names = []
    cursor = db["roles"].find(
        {"_id": {"$in": role_id_candidates}},
        {"slug": 1, "name": 1},
    )
    async for role in cursor:
        role_name = role.get("name") or role.get("slug")
        if role_name:
            names.append(str(role_name))
    return sorted(set(names))


async def load_role_permission_details(db, role_ids, collections=None):
    if not role_ids:
        return [], set()

    collections = collections or await db.list_collection_names()
    if "roles" not in collections:
        return [], set()

    template_map = await load_template_collection_map(db)
    role_id_candidates = []
    for role_id in role_ids:
        role_id_candidates.extend(mongo_id_candidates(role_id))

    names = []
    allowed = set()
    cursor = db["roles"].find(
        {"_id": {"$in": role_id_candidates}},
        {"slug": 1, "name": 1, "permissionIds": 1, "entityPermissionRules": 1},
    )
    async for role in cursor:
        role_name = role.get("name") or role.get("slug")
        if role_name:
            names.append(str(role_name))

        if role.get("slug") == "super-admin":
            allowed.add("*")
            continue

        for rule in role.get("entityPermissionRules") or []:
            if not isinstance(rule, dict):
                continue
            if str(rule.get("action", "")).lower() != "read":
                continue
            template_id = rule.get("templateId")
            collection_name = template_map.get(str(template_id))
            if collection_name:
                allowed.add(collection_name)

    return sorted(set(names)), allowed


async def load_user_role_ids(db, user_id):
    if "user_access" not in await db.list_collection_names():
        return set()

    role_ids = set()
    cursor = db["user_access"].find(
        {
            "userId": {"$in": mongo_id_candidates(user_id)},
            "$or": [
                {"isActive": {"$exists": False}},
                {"isActive": True},
            ],
        },
        {"roleId": 1},
    )
    async for access in cursor:
        role_id = access.get("roleId")
        if role_id:
            role_ids.add(role_id)
    return role_ids


async def load_rbac_users(db_name, force_refresh=False):
    cache_key = f"erp:rbac_users:{db_name}"
    if not force_refresh:
        cached = await redis_get_json(cache_key)
        if cached is not None:
            return cached

    db = mongo_client()[db_name]
    collections = await db.list_collection_names()
    if "users" not in collections:
        await redis_set_json(cache_key, DEFAULT_USERS, REDIS_RBAC_TTL_SECONDS)
        return DEFAULT_USERS

    users = []
    cursor = db["users"].find({}, {"password": 0, "hashed_password": 0}).limit(500)
    async for doc in cursor:
        raw_user_id = str(doc.get("_id"))
        user = normalize_user_doc(to_jsonable(doc))
        role_ids = await load_user_role_ids(db, raw_user_id)
        role_names, role_allowed = await load_role_permission_details(db, role_ids, collections)
        user["roles"] = role_names
        if role_allowed:
            user["allowed_collections"] = sorted(role_allowed)
        elif user["user_id"] in {"superadmin", "admin"} or user.get("email") == "admin@yourdomain.com":
            user["allowed_collections"] = ["*"]
        elif "users" in collections:
            user["allowed_collections"] = ["users"]
        users.append(user)
    result = users or DEFAULT_USERS
    await redis_set_json(cache_key, result, REDIS_RBAC_TTL_SECONDS)
    return result


def collection_is_allowed(collection_name, allowed_collections):
    if "*" in allowed_collections:
        return True
    return collection_name in allowed_collections


def allowed_collections_for_user(collections, user):
    role_names = {str(role).strip().lower() for role in (user.get("roles") or []) if str(role).strip()}
    user_id = str(user.get("user_id") or "").strip().lower()
    email = str(user.get("email") or "").strip().lower()
    if (
        "super-admin" in role_names
        or "super admin" in role_names
        or user_id in {"superadmin", "super-admin", "admin"}
        or email == "admin@yourdomain.com"
    ):
        return list(collections)

    allowed = user.get("allowed_collections", [])
    return [name for name in collections if collection_is_allowed(name, allowed)]
