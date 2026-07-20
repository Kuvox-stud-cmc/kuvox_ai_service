#!/usr/bin/env python3
"""Capture repeatable query-embedding cache before/after evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from prometheus_client.parser import text_string_to_metric_families

_COUNTERS = {
    "kuvox_query_embedding_cache_operations_total",
    "kuvox_query_embedding_encoder_inputs_total",
}


def main() -> None:
    args = _parse_args()
    request = json.loads(args.request_json.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or not isinstance(request.get("query"), str):
        raise SystemExit("request JSON must be an object with a string query field")

    run_id = uuid4().hex
    request["query"] = f"{request['query']} [baseline:{run_id}]"
    endpoint = f"{args.base_url.rstrip('/')}/retrieval/video-editor"
    metrics_endpoint = f"{args.base_url.rstrip('/')}/metrics"

    with httpx.Client(timeout=args.timeout_seconds) as client:
        before = _read_counters(client, metrics_endpoint)
        warmup = _call(client, endpoint, request)
        after_warmup = _read_counters(client, metrics_endpoint)
        repeated = [_call(client, endpoint, request) for _ in range(args.repeat)]
        after_repeats = _read_counters(client, metrics_endpoint)

    warmup_hash = warmup["responseSha256"]
    report = {
        "schemaVersion": 1,
        "capturedAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "runId": run_id,
        "endpoint": endpoint,
        "requestSha256": _json_hash(request),
        "warmup": warmup,
        "repeatedCalls": repeated,
        "responsesEquivalent": all(item["responseSha256"] == warmup_hash for item in repeated),
        "metricDeltas": {
            "warmup": _counter_delta(before, after_warmup),
            "repeatedCalls": _counter_delta(after_warmup, after_repeats),
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{encoded}\n", encoding="utf-8")
    print(encoded)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


def _call(client: httpx.Client, endpoint: str, request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post(endpoint, json=request)
    duration = time.perf_counter() - started
    response.raise_for_status()
    return {
        "durationSeconds": duration,
        "statusCode": response.status_code,
        "responseSha256": _json_hash(response.json()),
    }


def _read_counters(client: httpx.Client, endpoint: str) -> dict[str, float]:
    response = client.get(endpoint)
    response.raise_for_status()
    counters: dict[str, float] = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name not in _COUNTERS:
                continue
            labels = ",".join(f"{key}={value}" for key, value in sorted(sample.labels.items()))
            counters[f"{sample.name}{{{labels}}}"] = float(sample.value)
    return counters


def _counter_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    return {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in sorted(before.keys() | after.keys())
        if after.get(key, 0.0) != before.get(key, 0.0)
    }


def _json_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
