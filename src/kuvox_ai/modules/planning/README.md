# `planning`

Turns a natural-language editing command into a validated, executable `Plan`.

Orchestrated by LangGraph as a small state machine with three exit paths:

- **Direct path** — concrete commands (e.g. "trim shot 3 to 1.0–4.5 seconds")
  go to a single structured LLM call and out as a `Plan`.
- **Reasoning path** — abstract commands (e.g. "make this more dramatic")
  are decomposed, sent through `retrieval` to gather supporting shots, then
  synthesized into a `Plan`.
- **Code generation fallback** — requests that fall outside the operation
  schema cause the LLM to emit Python that runs in the `sandbox` module; its
  structured output is then validated as a `Plan`.

Every plan is round-tripped through the operation schema before being
returned, so downstream rendering can assume a well-formed input.

Consumed synchronously by the `/planning` HTTP endpoint.
