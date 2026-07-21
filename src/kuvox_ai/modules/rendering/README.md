# `rendering`

Executes the exact saved timeline revision from a `rendering.requested`
event into a finished video file.

The current renderer builds manifest schema v3 from the saved document,
downloads canonical media, composites frames with Pillow, encodes with
FFmpeg, mixes explicit and embedded video audio, and writes the result to
the object-storage key supplied by the API. Logical document dimensions are
the coordinate system; requested output dimensions may scale that canvas but
must preserve its aspect ratio. Manifest schemas v1 and v2 remain readable
with neutral style, fade, logical-canvas, and audio-ownership defaults.

Supported demo styling is resolved through the versioned adjustment registry
in `adjustments.py`. Non-default masks, vignette/grain, transitions/effects,
reverse/time-remap/freeze state, entrance/exit animations, advanced color
controls, advanced audio DSP, and AI metadata fail explicitly instead of
silently producing a mismatched export.

Triggered by messages on the `kuvox.rendering` RabbitMQ queue; rendering is
CPU-heavy and never runs inside the HTTP request path. The worker entry
point lives in `kuvox_ai/workers/rendering_worker.py`.
