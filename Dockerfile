# syntax=docker/dockerfile:1.7

# ----- Builder ---------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Install into an isolated prefix so we can copy a clean tree into the final image.
RUN pip install --upgrade pip \
 && pip install --prefix=/install .

# ----- Runtime --------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/usr/local/bin:$PATH

# Non-root user.
RUN groupadd --system kuvox \
 && useradd --system --gid kuvox --home /app --shell /usr/sbin/nologin kuvox

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=kuvox:kuvox src ./src

# Writable data dir for the embedded Kuzu graph DB and model files
# (config defaults: ./data/kuzu, ./data/models). Owned by the non-root user
# so a named volume mounted here inherits kuvox ownership at creation.
RUN mkdir -p /app/data && chown -R kuvox:kuvox /app/data

USER kuvox

EXPOSE 8000

CMD ["uvicorn", "kuvox_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
