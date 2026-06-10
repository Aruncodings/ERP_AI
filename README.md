# ERP Query Backend

This folder contains the runtime for the ERP natural-language query backend.

It is a FastAPI service that:

- routes ERP questions to MongoDB collections
- uses local GGUF or Hugging Face safetensors models for planning and summarization
- validates generated MongoDB plans before execution
- resolves lookup fields for display
- supports standard JSON and streaming query responses

## Current Structure

```text
train/
|-- erp_backend/
|   |-- api/
|   |   |-- models.py
|   |   `-- server.py
|   |-- cli/
|   |   `-- model_timing.py
|   |-- core/
|   |   |-- config.py
|   |   |-- feedback.py
|   |   |-- observability.py
|   |   |-- permissions.py
|   |   |-- security.py
|   |   `-- utils.py
|   |-- llm/
|   |   `-- runtime.py
|   |-- services/
|   |   |-- field_retriever.py
|   |   |-- intent.py
|   |   |-- llm.py
|   |   |-- orchestrate.py
|   |   |-- query.py
|   |   |-- query_prompts.py
|   |   |-- query_rewriter.py
|   |   |-- query_validate.py
|   |   |-- schema_indexer.py
|   |   |-- schema_runtime.py
|   |   `-- self_healing.py
|   |-- storage/
|   |   |-- cache.py
|   |   |-- mongo.py
|   |   `-- vector_store.py
|-- logs/
|-- graphify-out/
|-- requirements.txt
`-- tests/
```

## Prerequisites

- Python `3.10+`
- MongoDB reachable at `MONGO_URI` (default: `mongodb://localhost:27017`)
- Optional Redis at `REDIS_URL` for cached metadata and chat state
- One or more `.gguf` models in `../models/` if using GGUF runtime

## Setup

From `train/`, install dependencies:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running the API

Start the backend from `train/`:

```powershell
..\.venv\Scripts\python.exe -m uvicorn erp_backend.api.server:app --host 0.0.0.0 --port 8000 --reload
```

For a fixed local bind:

```powershell
..\.venv\Scripts\python.exe -m uvicorn erp_backend.api.server:app --host 127.0.0.1 --port 8000
```

## Health Endpoints

- `GET /health` or `GET /health/live` - liveness only, no Mongo check
- `GET /ready` or `GET /health/ready` - readiness check, includes Mongo ping

## Main API Endpoints

- `GET /models/options` - discover available GGUF and safetensors models
- `GET /bootstrap` - load DB, collection, metadata, and user bootstrap payload
- `GET /suggestions` - generate sidebar suggestions
- `POST /query` - full JSON response
- `POST /query_stream` - streaming SSE response
- `POST /query_feedback` - rewrite/clarification flow for bad answers

## Model Selection

GGUF model selection is now safe by default:

- if `GGUF_MODEL_PATH` points to an existing file, it is used
- otherwise the backend falls back to the first existing `.gguf` file in `../models/`
- if no GGUF file exists, the API reports no GGUF default instead of exposing a dead path

Important variables in `erp_backend/core/config.py`:

- `GGUF_MODEL_PATH`
- `BASE_MODEL`
- `DB_NAME`
- `MONGO_URI`
- `VECTOR_DB_ENABLED`
- `LLM_SINGLE_PASS_QUERY`
- `LLM_ENABLE_RESULT_VERIFIER`
- `LLM_ENABLE_CLARIFICATION` (`0` by default, so clarification prompts stay disabled)

## Smoke Test

Liveness:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```

Readiness:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/ready -UseBasicParsing
```

Query:

```powershell
$body = @{
  prompt = "List 5 active branches"
  db_name = "ECMS_MAY03_COPY"
  user_id = "superadmin"
  model_runtime = "gguf"
  reasoning_model_runtime = "gguf"
  reasoning_model_enabled = $true
  validation_enabled = $true
  compute_mode = "gpu"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/query `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Tests

Run the regression tests from `train/`:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

## Notes

- This folder is not a QLoRA training project anymore.
- All runtime code now lives under `erp_backend/` by domain: `api`, `core`, `llm`, `services`, `storage`, and `cli`.
- The package entrypoint is `erp_backend.api.server`.
- Large prompts can trigger slower reasoning, planning, repair, and verifier stages; those stages are offloaded from the FastAPI event loop.

redis 


C:\Users\sadmin\Downloads\redis\redis-server.exe C:\Users\sadmin\Downloads\redis\redis.windows.conf  