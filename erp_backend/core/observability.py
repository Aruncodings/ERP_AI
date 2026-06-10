import os
from contextlib import contextmanager


_OTEL_AVAILABLE = False
_trace = None
_FastAPIInstrumentor = None
_TracerProvider = None
_Resource = None
_BatchSpanProcessor = None
_ConsoleSpanExporter = None

try:
    from opentelemetry import trace as _trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor as _FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource as _Resource
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor as _BatchSpanProcessor,
        ConsoleSpanExporter as _ConsoleSpanExporter,
    )

    _OTEL_AVAILABLE = True
except Exception:
    _OTEL_AVAILABLE = False


def _env_bool(name, default=False):
    raw = os.getenv(name, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _as_attr_value(value):
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def init_observability(app, service_name="erp-query-backend"):
    if not _env_bool("OTEL_ENABLED", True):
        return False
    if not _OTEL_AVAILABLE:
        return False
    if getattr(app.state, "otel_initialized", False):
        return True

    provider = _trace.get_tracer_provider()
    provider_name = provider.__class__.__name__.lower()
    if "tracerprovider" not in provider_name or "proxy" in provider_name:
        resource = _Resource.create({"service.name": str(service_name or "erp-query-backend")})
        provider = _TracerProvider(resource=resource)
        try:
            _trace.set_tracer_provider(provider)
        except Exception:
            # Global provider can only be set once.
            provider = _trace.get_tracer_provider()

    if _env_bool("OTEL_EXPORT_CONSOLE", False):
        try:
            provider.add_span_processor(_BatchSpanProcessor(_ConsoleSpanExporter()))
        except Exception:
            pass

    otlp_endpoint = str(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")).strip()
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(_BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
        except Exception:
            pass

    try:
        _FastAPIInstrumentor.instrument_app(app)
    except Exception:
        pass

    app.state.otel_initialized = True
    return True


def get_tracer(name="erp.query"):
    if not _OTEL_AVAILABLE:
        return None
    try:
        return _trace.get_tracer(name)
    except Exception:
        return None


@contextmanager
def traced_span(name, attributes=None, tracer_name="erp.query"):
    tracer = get_tracer(tracer_name)
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(str(name)) as span:
        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(str(key), _as_attr_value(value))
                except Exception:
                    continue
        yield span


def set_span_attribute(key, value):
    if not _OTEL_AVAILABLE:
        return
    try:
        span = _trace.get_current_span()
        if span is not None and span.is_recording():
            span.set_attribute(str(key), _as_attr_value(value))
    except Exception:
        return


def add_span_event(name, attributes=None):
    if not _OTEL_AVAILABLE:
        return
    try:
        span = _trace.get_current_span()
        if span is not None and span.is_recording():
            payload = {}
            for key, value in (attributes or {}).items():
                payload[str(key)] = _as_attr_value(value)
            span.add_event(str(name), payload or None)
    except Exception:
        return
