# kuvox-ai

The AI service of **Kuvox** — a graph-augmented retrieval system for intelligent
video editing. This repository implements a modular FastAPI monolith that
handles ingestion, retrieval, planning, rendering, and sandboxed code
execution. It sits behind an ASP.NET business service (separate repo) and a
React/Redux frontend (separate repo); this service is responsible for the ML /
data side only and trusts its caller for authentication.

## Prerequisites

- Python **3.11+** (3.12 recommended; the Docker image pins 3.12-slim).
- Docker + Docker Compose (for local Qdrant / Redis / RabbitMQ / SeaweedFS).
- `make`. On Windows install via Chocolatey (`choco install make`) or run the
  underlying commands directly — every target is a one-liner you can paste.

## Quickstart

```bash
cp .env.example .env
make install      # create .venv, install deps incl. dev extras
make up           # start local infra via docker-compose
make dev          # start FastAPI with auto-reload on http://localhost:8000
```

Then:

- API docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>
- RabbitMQ UI: <http://localhost:15672> (guest / guest)
- SeaweedFS console: <http://localhost:8333> 

Run a worker in a separate shell:

```bash
.venv/bin/python -m kuvox_ai.workers.ingestion_worker
.venv/bin/python -m kuvox_ai.workers.rendering_worker
.venv/bin/python -m kuvox_ai.workers.sandbox_worker
```

## Module map

```
src/kuvox_ai/
├── api/            # FastAPI routes, request/response schemas, middleware
├── config.py       # pydantic-settings — env-driven configuration
├── infrastructure/ # thin async wrappers around external dependencies
├── logging.py      # structlog setup (JSON in prod, console in dev)
├── main.py         # FastAPI app, lifespan, run() entry point
├── modules/        # the five internal modules — each a self-contained package
│   ├── ingestion/  # tiered processing of uploaded videos (RabbitMQ-triggered)
│   ├── retrieval/  # multimodal vector search + Kuzu graph expansion + RRF
│   ├── planning/   # LangGraph-orchestrated plan generation (sync HTTP)
│   ├── rendering/  # MoviePy/FFmpeg execution of plans (RabbitMQ-triggered)
│   └── sandbox/    # dockerized execution of LLM-generated Python
├── schemas/        # cross-module domain models (Shot, Video, Plan, ...)
└── workers/        # one process per heavy workload
```

Each module's `README.md` describes its responsibility in detail. Modules
talk to each other only through the public service class exported from
their `__init__.py`.

## Make targets

| Target          | What it does                                                  |
| --------------- | ------------------------------------------------------------- |
| `make install`  | Create `.venv` and install with `[dev]` extras                |
| `make dev`      | Run FastAPI with auto-reload                                  |
| `make up`       | Start local infra (Qdrant, Redis, RabbitMQ, SeaweedFS)            |
| `make down`     | Stop local infra                                              |
| `make test`     | Run pytest                                                    |
| `make lint`     | Ruff lint + format check                                      |
| `make format`   | Ruff auto-fix + format                                        |
| `make typecheck`| Mypy in strict mode                                           |
| `make clean`    | Drop build artifacts and tool caches                          |

Integration tests are marked `@pytest.mark.integration` and require
`make up` services. Run with `pytest -m integration` once infra is up.

## How to add a new operation to the schema

1. Open [`src/kuvox_ai/schemas/operation.py`](src/kuvox_ai/schemas/operation.py).
2. Define a new pydantic model that subclasses `_OperationBase` and pins
   `op: Literal["your_op_name"] = "your_op_name"`. Add any required fields.
3. Add the new class to the `Operation` discriminated union at the bottom
   of the file.
4. Re-export from [`src/kuvox_ai/schemas/__init__.py`](src/kuvox_ai/schemas/__init__.py).
5. Teach `rendering` how to execute it (a new branch in `RenderingService`)
   and `planning` how to emit it (typically via the structured prompt
   given to the LLM).
6. Add a round-trip test in `tests/unit/test_schemas.py`.

The discriminator field means existing serialized plans remain valid
when you add a new variant — only consumers that explicitly switch on
`op` need updating.

## How to plug in a real LLM provider

The LLM is abstracted behind
[`LLMClient`](src/kuvox_ai/infrastructure/llm_client.py). The scaffold
ships only `StubLLMClient`, which returns canned responses so the service
runs end-to-end without an API key.

To add a real provider (e.g. OpenAI):

1. Create `src/kuvox_ai/infrastructure/openai_client.py` defining
   `OpenAILLMClient(LLMClient)` implementing every abstract method.
   `complete_structured` must call the provider's tool-use / JSON-mode API
   and validate the response against the supplied pydantic schema.
2. Add the provider name to the `LLMProvider` literal in
   [`src/kuvox_ai/config.py`](src/kuvox_ai/config.py).
3. Add a branch in `build_llm_client()` in
   [`llm_client.py`](src/kuvox_ai/infrastructure/llm_client.py) that
   constructs your client from `Settings`.
4. Set `KUVOX_LLM_PROVIDER=openai` (or your provider name) and
   `KUVOX_LLM_API_KEY=...` in `.env`.

No other code needs to change — every caller depends only on the
`LLMClient` interface.

## What's *not* in this scaffold

Intentionally absent: ML model loading, real processing, hard-coded LLM
providers, authentication (handled upstream by the ASP.NET service),
metrics/observability beyond logging, admin UI.
