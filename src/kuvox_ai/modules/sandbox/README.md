# `sandbox`

Executes language-model-generated Python in isolated Docker containers.

Used by the planning module's code-generation fallback path: when a user's
request falls outside the operation schema, the planner emits Python that
this module runs and whose structured output is then validated as a `Plan`.

Each execution runs in a fresh container configured with:

- No network (`--network none`).
- Read-only root filesystem and read-only mounts for any inputs.
- Strict CPU, memory, and pids limits.
- A hard wall-clock timeout enforced by the runtime.
- A non-root user inside the container.

Triggered by messages on the `kuvox.sandbox` RabbitMQ queue. The worker
entry point lives in `kuvox_ai/workers/sandbox_worker.py`.
