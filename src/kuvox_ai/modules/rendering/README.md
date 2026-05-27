# `rendering`

Executes a validated `Plan` into a finished video file.

Each `Operation` in the plan maps to a MoviePy / FFmpeg invocation. The
service walks the operation list, accumulates clips and transitions, and
writes the encoded output to object storage at the key specified by the
caller.

Triggered by messages on the `kuvox.rendering` RabbitMQ queue; rendering is
CPU-heavy and never runs inside the HTTP request path. The worker entry
point lives in `kuvox_ai/workers/rendering_worker.py`.
