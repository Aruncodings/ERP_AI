import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = SCRIPT_DIR.parent


def _resolve_default_gguf_model_path():
    configured = str(os.getenv("GGUF_MODEL_PATH", "") or "").strip()
    if configured:
        try:
            candidate = Path(configured).expanduser()
            if candidate.is_file():
                return str(candidate.resolve())
        except Exception:
            pass

    models_dir = PROJECT_DIR / "models"
    if models_dir.exists():
        for candidate in sorted(models_dir.glob("*.gguf")):
            try:
                return str(candidate.resolve())
            except Exception:
                return str(candidate)
    return ""


BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GGUF_MODEL_PATH = _resolve_default_gguf_model_path()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip()
ADAPTER_DIR = SCRIPT_DIR / "lora_adapter"

SYSTEM_FIELD_NAMES = frozenset(
    {
        "_id",
        "__v",
        "id",
        "isActive",
        "isDeleted",
        "createdAt",
        "updatedAt",
        "createdDate",
        "updatedDate",
        "createdBy",
        "updatedBy",
        "templateId",
        "organizationId",
        "branchId",
    }
)

FEEDBACK_PATH = PROJECT_DIR / "feedback_data.json"
LOCAL_FEEDBACK_PATH = SCRIPT_DIR / "feedback_data.json"
BASE_DATASET_PATH = SCRIPT_DIR / "dataset.json"
RETRAIN_DATASET_PATH = SCRIPT_DIR / "retrain_dataset.json"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_DB_NAME = os.getenv("DB_NAME", "ECMS_MAY03_COPY")
MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "3000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_SCHEMA_TTL_SECONDS = int(os.getenv("REDIS_SCHEMA_TTL_SECONDS", "3600"))
REDIS_METADATA_TTL_SECONDS = int(os.getenv("REDIS_METADATA_TTL_SECONDS", "300"))
REDIS_RBAC_TTL_SECONDS = int(os.getenv("REDIS_RBAC_TTL_SECONDS", "120"))
REDIS_CHAT_TTL_SECONDS = int(os.getenv("REDIS_CHAT_TTL_SECONDS", "1800"))
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "500"))
ENRICH_MAX_DOCS = int(os.getenv("ENRICH_MAX_DOCS", "120"))
COUNT_TOTAL_EXACT = os.getenv("COUNT_TOTAL_EXACT", "0").strip().lower() in {"1", "true", "yes", "on"}
SCHEMA_SAMPLE_SIZE = 5
MAX_TABLE_COUNT = 200

# LLM inference memory controls
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2000"))
TABLE_ROUTER_MAX_NEW_TOKENS = int(os.getenv("TABLE_ROUTER_MAX_NEW_TOKENS", "160"))
LLM_MAX_INPUT_TOKENS = int(os.getenv("LLM_MAX_INPUT_TOKENS", "2048"))
LLM_CTX_SIZE = int(os.getenv("LLM_CTX_SIZE", "8000"))
LLM_N_BATCH = int(os.getenv("LLM_N_BATCH", "512"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", str(max(1, (os.cpu_count() or 4) - 1))))
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "-1"))
LLM_FLASH_ATTN = os.getenv("LLM_FLASH_ATTN", "1").strip().lower() in {"1", "true", "yes", "on"}
LLM_OFFLOAD_KQV = os.getenv("LLM_OFFLOAD_KQV", "1").strip().lower() in {"1", "true", "yes", "on"}
LLM_USE_MMAP = os.getenv("LLM_USE_MMAP", "1").strip().lower() in {"1", "true", "yes", "on"}
LLM_USE_MLOCK = os.getenv("LLM_USE_MLOCK", "0").strip().lower() in {"1", "true", "yes", "on"}
LLM_SINGLE_PASS_QUERY = os.getenv("LLM_SINGLE_PASS_QUERY", "0").strip().lower() in {"1", "true", "yes", "on"}
LLM_ENABLE_RESULT_VERIFIER = os.getenv("LLM_ENABLE_RESULT_VERIFIER", "1").strip().lower() in {"1", "true", "yes", "on"}
LLM_USE_CHAT_HISTORY_HINTS = os.getenv("LLM_USE_CHAT_HISTORY_HINTS", "1").strip().lower() in {"1", "true", "yes", "on"}
LLM_RAW_MODE = os.getenv("LLM_RAW_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
LLM_COLLECTION_CANDIDATES = int(os.getenv("LLM_COLLECTION_CANDIDATES", "10"))
LLM_FIELD_CANDIDATES = int(os.getenv("LLM_FIELD_CANDIDATES", "96"))
LLM_ENABLE_QUERY_SELF_HEAL = os.getenv("LLM_ENABLE_QUERY_SELF_HEAL", "1").strip().lower() in {"1", "true", "yes", "on"}
LLM_PROMPT_REWRITE_MODE = os.getenv("LLM_PROMPT_REWRITE_MODE", "adaptive").strip().lower()
LLM_ENABLE_EMPTY_RESULT_REPAIR = os.getenv("LLM_ENABLE_EMPTY_RESULT_REPAIR", "1").strip().lower() in {"1", "true", "yes", "on"}
LLM_ENABLE_MISMATCH_RESULT_REPAIR = os.getenv("LLM_ENABLE_MISMATCH_RESULT_REPAIR", "1").strip().lower() in {"1", "true", "yes", "on"}
LLM_REWRITER_MODE = os.getenv("LLM_REWRITER_MODE", "policy").strip().lower()
LLM_ENABLE_CLARIFICATION = os.getenv("LLM_ENABLE_CLARIFICATION", "0").strip().lower() in {"1", "true", "yes", "on"}
TABLE_ROUTER_FIELDS_PER_TABLE = int(os.getenv("TABLE_ROUTER_FIELDS_PER_TABLE", "24"))
TABLE_ROUTER_TERMS_PER_TABLE = int(os.getenv("TABLE_ROUTER_TERMS_PER_TABLE", "32"))
CHAT_CONTEXT_LIMIT = int(os.getenv("CHAT_CONTEXT_LIMIT", "4"))
CHAT_CONTEXT_CHARS = int(os.getenv("CHAT_CONTEXT_CHARS", "4000"))
LLM_RETRY_MAX_INPUT_TOKENS = int(os.getenv("LLM_RETRY_MAX_INPUT_TOKENS", "1024"))
LLM_RETRY_MAX_NEW_TOKENS = int(os.getenv("LLM_RETRY_MAX_NEW_TOKENS", "384"))
LLM_USE_CACHE = os.getenv("LLM_USE_CACHE", "0").strip().lower() in {"1", "true", "yes", "on"}
SAFETENSORS_MAX_NEW_TOKENS = int(os.getenv("SAFETENSORS_MAX_NEW_TOKENS", "256"))
SAFETENSORS_USE_4BIT = os.getenv("SAFETENSORS_USE_4BIT", "1").strip().lower() in {"1", "true", "yes", "on"}
VECTOR_DB_ENABLED = os.getenv("VECTOR_DB_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
VECTOR_DB_PATH = os.getenv(
    "VECTOR_DB_PATH",
    str((PROJECT_DIR / "vector_db").resolve()),
)
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "80"))
VECTOR_REFRESH_ENABLED = os.getenv("VECTOR_REFRESH_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
VECTOR_REFRESH_INTERVAL_SECONDS = int(os.getenv("VECTOR_REFRESH_INTERVAL_SECONDS", "900"))
VECTOR_DISTINCT_ENABLED = os.getenv("VECTOR_DISTINCT_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
VECTOR_DISTINCT_MAX_FIELDS_PER_COLLECTION = int(os.getenv("VECTOR_DISTINCT_MAX_FIELDS_PER_COLLECTION", "12"))
VECTOR_DISTINCT_MAX_VALUES_PER_FIELD = int(os.getenv("VECTOR_DISTINCT_MAX_VALUES_PER_FIELD", "80"))
VECTOR_DISTINCT_MAX_VALUE_LENGTH = int(os.getenv("VECTOR_DISTINCT_MAX_VALUE_LENGTH", "64"))
VECTOR_EMBEDDING_MODEL = os.getenv(
    "VECTOR_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()
VECTOR_EMBEDDING_DEVICE = os.getenv("VECTOR_EMBEDDING_DEVICE", "auto").strip().lower()
LLM_SUMMARY_MODE = os.getenv("LLM_SUMMARY_MODE", "adaptive").strip().lower()

# Safe production rollout flags: dynamic policy + field-role inference.
ROLLOUT_RUNTIME_POLICY_ENABLED = os.getenv("ROLLOUT_RUNTIME_POLICY_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
ROLLOUT_RUNTIME_POLICY_COLLECTION = os.getenv("ROLLOUT_RUNTIME_POLICY_COLLECTION", "ai_runtime_policies").strip()
ROLLOUT_DYNAMIC_FIELD_ROLES_ENABLED = os.getenv("ROLLOUT_DYNAMIC_FIELD_ROLES_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
ROLLOUT_DYNAMIC_CREATED_INTENT_GUARD_ENABLED = os.getenv("ROLLOUT_DYNAMIC_CREATED_INTENT_GUARD_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
ROLLOUT_POLICY_CACHE_TTL_SECONDS = int(os.getenv("ROLLOUT_POLICY_CACHE_TTL_SECONDS", "300"))

# Prometheus metrics
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}

# Continuous training from user feedback
CONTINUOUS_TRAINING_ENABLED = os.getenv("CONTINUOUS_TRAINING_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
FEEDBACK_RETRAIN_THRESHOLD = int(os.getenv("FEEDBACK_RETRAIN_THRESHOLD", "50"))
FEEDBACK_RETRAIN_MIN_SAMPLES = int(os.getenv("FEEDBACK_RETRAIN_MIN_SAMPLES", "10"))
FEEDBACK_COUNTER_PATH = os.getenv(
    "FEEDBACK_COUNTER_PATH",
    str((PROJECT_DIR / "feedback_counter.json").resolve()),
)


TEMPERATURE = 0.1
TOP_P = 0.85
TOP_K = 30

BLOCKED_DATABASES = {"admin", "config", "local"}
BLOCKED_COLLECTIONS = {"system.profile", "system.views"}
BLOCKED_STAGES = {
    "$collStats",
    "$currentOp",
    "$documents",
    "$indexStats",
    "$listLocalSessions",
    "$listSampledQueries",
    "$listSearchIndexes",
    "$merge",
    "$out",
    "$planCacheStats",
    "$querySettings",
    "$setWindowFields",
    "$unionWith",
}
ALLOWED_AGGREGATE_STAGES = {
    "$addFields",
    "$bucket",
    "$bucketAuto",
    "$count",
    "$densify",
    "$facet",
    "$fill",
    "$geoNear",
    "$graphLookup",
    "$group",
    "$limit",
    "$lookup",
    "$match",
    "$project",
    "$redact",
    "$replaceRoot",
    "$replaceWith",
    "$sample",
    "$set",
    "$skip",
    "$sort",
    "$sortByCount",
    "$unset",
    "$unwind",
}
BLOCKED_OPERATORS = {"$where", "$function", "$accumulator"}

DEFAULT_USERS = [
    {
        "user_id": "admin",
        "display_name": "Admin",
        "allowed_collections": ["*"],
    }
]
