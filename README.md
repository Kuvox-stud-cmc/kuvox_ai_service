# kuvox-ai

The AI service of **Kuvox** — a graph-augmented retrieval system for intelligent
video editing. This repository implements a modular FastAPI monolith that
handles ingestion, retrieval, planning, rendering, and sandboxed code
execution. It sits behind an ASP.NET business service (separate repo) and a
React/Redux frontend (separate repo); this service is responsible for the ML /
data side only and trusts its caller for authentication.

## Prerequisites

- Python **3.11 or 3.12** (3.12 recommended; the Docker image pins 3.12-slim).
- The ML ingestion dependencies are not declared for newer Python versions yet.
- Docker + Docker Compose (for local Qdrant / Redis / RabbitMQ / SeaweedFS).
- FFmpeg tooling for video/audio media pipelines. Image optimization uses
  Pillow; video/audio optimization and ingestion still need `ffmpeg`/`ffprobe`
  available on PATH.
- `make`. On Windows install via Chocolatey (`choco install make`) or run the
  underlying commands directly — every target is a one-liner you can paste.

## Quickstart

```bash
cp .env.example .env
make install      # create .venv, install deps incl. dev extras
make up           # start local infra via docker-compose
make dev          # start FastAPI plus all RabbitMQ workers
```

For Conda, create the environment with a supported Python first, then install
from the AI service directory:

```bash
conda create -n kuvox-ai python=3.12
conda activate kuvox-ai
python -m pip install -e ".[dev]"
```

`msclap` currently requires NumPy 1.x and `librosa<0.11`, so the project pins
those ranges in `pyproject.toml`. Use a fresh Python 3.11/3.12 environment if
pip reports that no matching NumPy or librosa distributions are available.

## Environment Variables

Copy `.env.example` to `ai-service/.env` for native local development. The same
variable names are used on Linux, macOS, Windows, and Docker, but hostnames and
filesystem paths differ by runtime.

Native Linux with Docker-published infra ports:

```env
KUVOX_QDRANT_HOST=localhost
KUVOX_REDIS_URL=redis://localhost:6379/0
KUVOX_RABBITMQ_URL=amqp://kuvox:kuvox@localhost:5672/
KUVOX_S3_ENDPOINT_URL=http://localhost:8333
KUVOX_MEDIA_WORK_DIR=/tmp/kuvox-media
KUVOX_INGESTION_WORK_DIR=/tmp/kuvox-ingestion
```

Native macOS uses the same values as Linux:

```env
KUVOX_QDRANT_HOST=localhost
KUVOX_REDIS_URL=redis://localhost:6379/0
KUVOX_RABBITMQ_URL=amqp://kuvox:kuvox@localhost:5672/
KUVOX_S3_ENDPOINT_URL=http://localhost:8333
KUVOX_MEDIA_WORK_DIR=/tmp/kuvox-media
KUVOX_INGESTION_WORK_DIR=/tmp/kuvox-ingestion
```

Native Windows with Docker-published infra ports:

```env
KUVOX_QDRANT_HOST=localhost
KUVOX_REDIS_URL=redis://localhost:6379/0
KUVOX_RABBITMQ_URL=amqp://kuvox:kuvox@localhost:5672/
KUVOX_S3_ENDPOINT_URL=http://localhost:8333
KUVOX_MEDIA_WORK_DIR=D:/Kuvox/.codex-tmp/kuvox-media
KUVOX_INGESTION_WORK_DIR=D:/Kuvox/.codex-tmp/kuvox-ingestion
```

AI service running inside Docker Compose:

```env
KUVOX_QDRANT_HOST=qdrant
KUVOX_REDIS_URL=redis://redis:6379/0
KUVOX_RABBITMQ_URL=amqp://kuvox:kuvox@rabbitmq:5672/
KUVOX_S3_ENDPOINT_URL=http://seaweedfs-s3:8333
KUVOX_MEDIA_WORK_DIR=/tmp/kuvox-media
KUVOX_INGESTION_WORK_DIR=/tmp/kuvox-ingestion
```

Do not use `/tmp/...` scratch paths for native Windows workers; use a writable
drive path. If using Conda on Windows, start workers from an activated environment
or ensure `Library/bin` and `Scripts` are on PATH so native tools such as `ffprobe`
can be found. For production, inject secrets as real environment variables or via a
secret manager rather than committing `.env` files.

Then:

- API docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>
- RabbitMQ UI: <http://localhost:15672> (kuvox / kuvox)
- SeaweedFS console: <http://localhost:8333> 

`make dev` runs one FastAPI/Uvicorn process. During app startup, FastAPI wires
the media optimization, ingestion, rendering, and sandbox RabbitMQ consumers into
the same process. Press Ctrl-C once to stop the app and its consumers.

Set `KUVOX_RUN_WORKERS=false` or use `make dev-api` when you intentionally want
the HTTP API without RabbitMQ consumers.

Run a single process directly when debugging one component:

```bash
KUVOX_RUN_WORKERS=false .venv/bin/python -m uvicorn kuvox_ai.main:app --reload --host 0.0.0.0 --port 8000
.venv/bin/python -m kuvox_ai.workers.ingestion_worker
.venv/bin/python -m kuvox_ai.workers.media_optimization_worker
.venv/bin/python -m kuvox_ai.workers.rendering_worker
.venv/bin/python -m kuvox_ai.workers.sandbox_worker
```

If workers are disabled, uploads stay in the API's `Uploaded`/optimizing state
until `kuvox_ai.workers.media_optimization_worker` is running and consuming
`media.optimization.requested`. Videos then move to `Processing` and require
`kuvox_ai.workers.ingestion_worker` to reach `Ready`.

Use the environment-specific paths and hostnames from the Environment Variables
section above before starting workers.

The media optimization worker consumes `media.optimization.requested` from the
`kuvox.events` direct exchange, downloads raw media from SeaweedFS, writes
canonical/proxy/thumbnail objects, and publishes `media.optimization.completed`
or `media.optimization.failed`. These messages are media-library scoped and do not
carry `projectId`; project/media association is handled by the API's Projects module.
Optimization downloads stream objects directly to the local job directory to avoid
Windows S3 temp-file rename failures. Image conversion and metadata use Pillow;
video/audio metadata still comes from FFprobe.

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
└── workers/        # RabbitMQ handlers wired into FastAPI, with standalone entry points
```

Each module's `README.md` describes its responsibility in detail. Modules
talk to each other only through the public service class exported from
their `__init__.py`.

## Make targets

| Target          | What it does                                                  |
| --------------- | ------------------------------------------------------------- |
| `make install`  | Create `.venv` and install with `[dev]` extras                |
| `make dev`      | Run FastAPI with auto-reload and app-wired workers            |
| `make dev-api`  | Run FastAPI with RabbitMQ workers disabled                    |
| `make up`       | Start local infra (Qdrant, Redis, RabbitMQ, SeaweedFS)            |
| `make down`     | Stop local infra                                              |
| `make test`     | Run pytest                                                    |
| `make lint`     | Ruff lint + format check                                      |
| `make format`   | Ruff auto-fix + format                                        |
| `make typecheck`| Mypy in strict mode                                           |
| `make worker-media-optimization` | Run the media optimization worker               |
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
