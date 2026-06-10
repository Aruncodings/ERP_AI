"""
Prometheus metrics for the ERP Query Backend.

All instruments are gated behind PROMETHEUS_ENABLED.
Import `observe_request`, `observe_stage`, etc. from anywhere in the codebase.
"""
import logging
import time
import subprocess
import shutil
import threading

from erp_backend.core.config import PROMETHEUS_ENABLED

logger = logging.getLogger(__name__)

# ── Metric singletons (lazy-init on first use) ───────────────────────────────

_INITIALIZED = False

# Counters
REQUEST_TOTAL = None
QUERY_ERRORS_TOTAL = None
VECTOR_REFRESH_TOTAL = None

# Histograms
REQUEST_DURATION = None
STAGE_DURATION = None
LLM_INFERENCE_DURATION = None
MONGO_QUERY_DURATION = None
RESULT_ROWS = None

# Gauges
MODEL_LOADED = None
ACTIVE_REQUESTS = None
SYSTEM_CPU_USAGE = None
SYSTEM_MEMORY_USAGE = None
SYSTEM_MEMORY_USED = None
SYSTEM_MEMORY_TOTAL = None
SYSTEM_GPU_USAGE = None
SYSTEM_GPU_MEMORY_USAGE = None
SYSTEM_GPU_MEMORY_USED = None
SYSTEM_GPU_MEMORY_TOTAL = None

_MONITORING_THREAD_STARTED = False


# ── Histogram bucket definitions ─────────────────────────────────────────────

_LATENCY_BUCKETS = (
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
)
_LLM_BUCKETS = (
    0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0,
)
_ROW_BUCKETS = (
    0, 1, 5, 10, 25, 50, 100, 250, 500,
)


def _system_metrics_updater():
    """Background loop to update system and GPU metrics in Gauges."""
    logger.info("Starting background system metrics updater thread")
    while True:
        try:
            # 1. CPU and Memory
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=None)
                virtual_mem = psutil.virtual_memory()
                
                if SYSTEM_CPU_USAGE is not None:
                    SYSTEM_CPU_USAGE.set(cpu_percent)
                if SYSTEM_MEMORY_USAGE is not None:
                    SYSTEM_MEMORY_USAGE.set(virtual_mem.percent)
                if SYSTEM_MEMORY_USED is not None:
                    SYSTEM_MEMORY_USED.set(virtual_mem.used)
                if SYSTEM_MEMORY_TOTAL is not None:
                    SYSTEM_MEMORY_TOTAL.set(virtual_mem.total)
            except Exception as e:
                logger.debug("Failed to read CPU/Memory metrics: %s", e)

            # 2. GPU metrics
            if shutil.which("nvidia-smi"):
                try:
                    res = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
                        timeout=3.0
                    )
                    lines = res.stdout.strip().split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 5:
                            gpu_index = parts[0]
                            gpu_name = parts[1]
                            try:
                                gpu_util = float(parts[2])
                                mem_used_mb = float(parts[3])
                                mem_total_mb = float(parts[4])
                                
                                mem_used_bytes = mem_used_mb * 1024 * 1024
                                mem_total_bytes = mem_total_mb * 1024 * 1024
                                mem_util_percent = (mem_used_bytes / mem_total_bytes * 100) if mem_total_bytes > 0 else 0.0
                                
                                if SYSTEM_GPU_USAGE is not None:
                                    SYSTEM_GPU_USAGE.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(gpu_util)
                                if SYSTEM_GPU_MEMORY_USAGE is not None:
                                    SYSTEM_GPU_MEMORY_USAGE.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(mem_util_percent)
                                if SYSTEM_GPU_MEMORY_USED is not None:
                                    SYSTEM_GPU_MEMORY_USED.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(mem_used_bytes)
                                if SYSTEM_GPU_MEMORY_TOTAL is not None:
                                    SYSTEM_GPU_MEMORY_TOTAL.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(mem_total_bytes)
                            except ValueError as ve:
                                logger.debug("Failed to parse GPU metrics line: %s, error: %s", line, ve)
                except Exception as e:
                    logger.debug("Failed to query NVIDIA GPU metrics: %s", e)
        except Exception as e:
            logger.warning("Error in background system metrics updater: %s", e)
        
        time.sleep(5)


def _init_metrics():
    """Create all Prometheus metric objects. Called once on first use."""
    global _INITIALIZED
    global REQUEST_TOTAL, QUERY_ERRORS_TOTAL, VECTOR_REFRESH_TOTAL
    global REQUEST_DURATION, STAGE_DURATION, LLM_INFERENCE_DURATION
    global MONGO_QUERY_DURATION, RESULT_ROWS
    global MODEL_LOADED, ACTIVE_REQUESTS
    global SYSTEM_CPU_USAGE, SYSTEM_MEMORY_USAGE, SYSTEM_MEMORY_USED, SYSTEM_MEMORY_TOTAL
    global SYSTEM_GPU_USAGE, SYSTEM_GPU_MEMORY_USAGE, SYSTEM_GPU_MEMORY_USED, SYSTEM_GPU_MEMORY_TOTAL

    if _INITIALIZED:
        return
    _INITIALIZED = True

    if not PROMETHEUS_ENABLED:
        return

    try:
        from prometheus_client import Counter, Histogram, Gauge

        REQUEST_TOTAL = Counter(
            "erp_query_requests_total",
            "Total query requests processed",
            ["endpoint", "status", "db_name"],
        )
        QUERY_ERRORS_TOTAL = Counter(
            "erp_query_errors_total",
            "Total query errors",
            ["endpoint", "error_type"],
        )
        VECTOR_REFRESH_TOTAL = Counter(
            "erp_vector_refresh_total",
            "Vector schema refresh events",
            ["db_name", "status"],
        )
        REQUEST_DURATION = Histogram(
            "erp_query_duration_seconds",
            "End-to-end request latency",
            ["endpoint", "status"],
            buckets=_LATENCY_BUCKETS,
        )
        STAGE_DURATION = Histogram(
            "erp_query_stage_duration_seconds",
            "Per-stage latency breakdown",
            ["stage"],
            buckets=_LATENCY_BUCKETS,
        )
        LLM_INFERENCE_DURATION = Histogram(
            "erp_llm_inference_seconds",
            "LLM inference latency by task",
            ["task"],
            buckets=_LLM_BUCKETS,
        )
        MONGO_QUERY_DURATION = Histogram(
            "erp_mongo_query_seconds",
            "MongoDB query execution latency",
            ["collection", "operation"],
            buckets=_LATENCY_BUCKETS,
        )
        RESULT_ROWS = Histogram(
            "erp_query_result_rows",
            "Number of result rows returned",
            ["collection"],
            buckets=_ROW_BUCKETS,
        )
        MODEL_LOADED = Gauge(
            "erp_model_loaded",
            "Currently loaded model info",
            ["runtime", "model_name"],
        )
        ACTIVE_REQUESTS = Gauge(
            "erp_active_requests",
            "Number of currently in-flight requests",
            ["endpoint"],
        )
        SYSTEM_CPU_USAGE = Gauge(
            "erp_system_cpu_usage_percent",
            "System CPU usage percentage"
        )
        SYSTEM_MEMORY_USAGE = Gauge(
            "erp_system_memory_usage_percent",
            "System physical memory usage percentage"
        )
        SYSTEM_MEMORY_USED = Gauge(
            "erp_system_memory_used_bytes",
            "System physical memory used in bytes"
        )
        SYSTEM_MEMORY_TOTAL = Gauge(
            "erp_system_memory_total_bytes",
            "System physical memory total in bytes"
        )
        SYSTEM_GPU_USAGE = Gauge(
            "erp_system_gpu_usage_percent",
            "System GPU usage percentage",
            ["gpu_index", "gpu_name"]
        )
        SYSTEM_GPU_MEMORY_USAGE = Gauge(
            "erp_system_gpu_memory_usage_percent",
            "System GPU memory usage percentage",
            ["gpu_index", "gpu_name"]
        )
        SYSTEM_GPU_MEMORY_USED = Gauge(
            "erp_system_gpu_memory_used_bytes",
            "System GPU memory used in bytes",
            ["gpu_index", "gpu_name"]
        )
        SYSTEM_GPU_MEMORY_TOTAL = Gauge(
            "erp_system_gpu_memory_total_bytes",
            "System GPU memory total in bytes",
            ["gpu_index", "gpu_name"]
        )

        global _MONITORING_THREAD_STARTED
        if not _MONITORING_THREAD_STARTED:
            _MONITORING_THREAD_STARTED = True
            t = threading.Thread(target=_system_metrics_updater, daemon=True, name="erp-system-metrics")
            t.start()

        logger.info("Prometheus metrics initialized")
    except ImportError:
        logger.warning("prometheus_client not installed — metrics disabled")
    except Exception as exc:
        logger.warning("Failed to initialize Prometheus metrics: %s", exc)


# ── Public helpers ────────────────────────────────────────────────────────────

def observe_request(endpoint, status, db_name, duration_seconds):
    """Record a completed request: increment counter + observe latency histogram."""
    _init_metrics()
    if REQUEST_TOTAL is not None:
        REQUEST_TOTAL.labels(endpoint=endpoint, status=status, db_name=db_name).inc()
    if REQUEST_DURATION is not None:
        REQUEST_DURATION.labels(endpoint=endpoint, status=status).observe(duration_seconds)


def observe_stage(stage, duration_seconds):
    """Record a single pipeline stage duration."""
    _init_metrics()
    if STAGE_DURATION is not None and duration_seconds > 0:
        STAGE_DURATION.labels(stage=stage).observe(duration_seconds)


def observe_llm_inference(task, duration_seconds):
    """Record LLM inference time (router, planner, summarizer, etc.)."""
    _init_metrics()
    if LLM_INFERENCE_DURATION is not None and duration_seconds > 0:
        LLM_INFERENCE_DURATION.labels(task=task).observe(duration_seconds)


def observe_mongo_query(collection, operation, duration_seconds):
    """Record MongoDB query execution latency."""
    _init_metrics()
    if MONGO_QUERY_DURATION is not None and duration_seconds > 0:
        MONGO_QUERY_DURATION.labels(collection=collection, operation=operation).observe(duration_seconds)


def observe_result_rows(collection, count):
    """Record the number of result rows returned."""
    _init_metrics()
    if RESULT_ROWS is not None:
        RESULT_ROWS.labels(collection=collection).observe(count)


def observe_error(endpoint, error_type):
    """Increment the error counter."""
    _init_metrics()
    if QUERY_ERRORS_TOTAL is not None:
        QUERY_ERRORS_TOTAL.labels(endpoint=endpoint, error_type=error_type).inc()


def set_model_loaded(runtime, model_name):
    """Set the currently loaded model gauge."""
    _init_metrics()
    if MODEL_LOADED is not None:
        MODEL_LOADED.labels(runtime=runtime, model_name=model_name).set(1)


def track_active_request(endpoint, delta):
    """Increment/decrement active request gauge. delta should be +1 or -1."""
    _init_metrics()
    if ACTIVE_REQUESTS is not None:
        if delta > 0:
            ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()
        else:
            ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()


def observe_vector_refresh(db_name, success):
    """Record a vector schema refresh event."""
    _init_metrics()
    if VECTOR_REFRESH_TOTAL is not None:
        VECTOR_REFRESH_TOTAL.labels(
            db_name=db_name,
            status="success" if success else "failure",
        ).inc()


def observe_perf_log(endpoint, status, db_name, stage_ms):
    """
    Called from _write_perf_log to push all stage timings into Prometheus.
    stage_ms is a dict like {"bootstrap": 12.3, "model_load": 450.0, ...}
    """
    total_seconds = float(stage_ms.get("total", 0)) / 1000.0
    observe_request(endpoint, status, db_name, total_seconds)

    # Push each individual stage as a histogram observation
    for stage_name, ms_value in stage_ms.items():
        if stage_name == "total":
            continue
        seconds = float(ms_value or 0) / 1000.0
        if seconds > 0:
            observe_stage(stage_name, seconds)

    # Categorize LLM-specific stages for the dedicated LLM histogram
    llm_stages = {
        "model_load": "model_load",
        "reasoning_model_load": "reasoning_model_load",
        "plan_generation": "plan_generation",
        "scope_gate": "scope_gate",
        "prompt_rewrite": "prompt_rewrite",
        "summarize": "summarize",
        "narrate_stream": "narrate_stream",
        "follow_ups": "follow_ups",
        "verifier": "verifier",
    }
    for stage_key, task_label in llm_stages.items():
        ms = float(stage_ms.get(stage_key, 0))
        if ms > 0:
            observe_llm_inference(task_label, ms / 1000.0)


def init_prometheus(app):
    """
    Instrument the FastAPI app with prometheus-fastapi-instrumentator
    and expose the /metrics endpoint.
    """
    if not PROMETHEUS_ENABLED:
        logger.info("Prometheus disabled via PROMETHEUS_ENABLED=0")
        return False

    _init_metrics()

    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_instrument_requests_inprogress=True,
            inprogress_name="erp_http_requests_inprogress",
            inprogress_labels=True,
        )
        instrumentator.instrument(app)
        instrumentator.expose(app, include_in_schema=True, tags=["monitoring"])
        logger.info("Prometheus FastAPI instrumentator enabled at /metrics")
        return True
    except ImportError:
        logger.warning("prometheus-fastapi-instrumentator not installed — /metrics endpoint disabled")
        return False
    except Exception as exc:
        logger.warning("Failed to init Prometheus FastAPI instrumentator: %s", exc)
        return False
