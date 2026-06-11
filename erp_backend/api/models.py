from fastapi import FastAPI
from pydantic import BaseModel, Field

from erp_backend.core.config import DEFAULT_DB_NAME


class BootstrapResponse(BaseModel):
    databases: list[str]
    collections: list[str]
    table_metadata: dict
    users: list[dict]


class ModelOptionsResponse(BaseModel):
    gguf_models: list[str]
    safetensors_models: list[str]
    ollama_models: list[str]
    context_limits: dict[str, int] = Field(default_factory=dict)
    defaults: dict


class QueryRequest(BaseModel):
    prompt: str = Field(min_length=1)
    db_name: str = DEFAULT_DB_NAME
    user_id: str = "admin"
    conversation_id: str | None = None
    chat_context: list[dict] = Field(default_factory=list)
    model_runtime: str = "gguf"
    gguf_model_path: str | None = None
    safetensors_model_id: str | None = None
    ollama_model: str | None = None
    reasoning_model_runtime: str = "gguf"
    reasoning_gguf_model_path: str | None = None
    reasoning_safetensors_model_id: str | None = None
    reasoning_ollama_model: str | None = None
    reasoning_model_enabled: bool = True
    validation_enabled: bool = False
    compute_mode: str = "cpu"
    hybrid_gpu_layers: int | None = None
    hybrid_gpu_memory_mb: int | None = None


class QueryResponse(BaseModel):
    response: str
    follow_ups: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    insights: list[str] = Field(default_factory=list)
    chart_config: dict | None = None
    db_performance: dict | None = None
    table_columns: list[str] = Field(default_factory=list)
    table_choice: dict
    plan: dict
    docs: list[dict]
    total: int
    collection: str
    summary: str | None = None


class ResultFeedbackRequest(QueryRequest):
    feedback: str = "down"
    table_choice: dict = Field(default_factory=dict)
    plan: dict = Field(default_factory=dict)
    docs: list[dict] = Field(default_factory=list)
    total: int = 0
    collection: str = ""
    table_columns: list[str] = Field(default_factory=list)
    narrative: str = ""
    correct_collection: str | None = None
    correct_fields: dict[str, str] | None = None


class SuggestionsResponse(BaseModel):
    suggestions: list[str]
