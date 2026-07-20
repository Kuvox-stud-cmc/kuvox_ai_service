from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def test_committed_phase1_phase2_evidence_is_valid_and_redacted() -> None:
    root = Path(__file__).parents[2]
    fixture = json.loads(
        (root / "tests/fixtures/text_embedding_cache_evidence.json").read_text(encoding="utf-8")
    )
    report_path = root / "docs/evidence/cache/phase1-phase2-local.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["schemaVersion"] == 1
    assert report["fixtureSha256"] == sha256_json(fixture)
    assert report["modelId"] == fixture["modelId"]
    assert report["dimension"] == fixture["dimension"]
    assert report["disabled"]["encoderInputs"] == 2
    assert report["disabled"]["equivalent"] is True
    assert len(report["disabled"]["vectorSha256"]) == 64
    assert report["query"]["warmEncoderInputs"] == 1
    assert report["query"]["repeatEncoderInputs"] == 0
    assert report["query"]["repeatEquivalent"] is True
    assert report["ingestion"]["repeatEncoderInputs"] == 0
    assert report["ingestion"]["repeatEquivalent"] is True
    assert report["crossConsumerReuse"]["ingestionWarmInputsSaved"] >= 1
    assert report["legacyPromotion"] == {
        "encoderInputs": 0,
        "equivalent": True,
        "sharedValueMagic": "KTEV",
    }
    assert report["redisOutage"]["encoderInputs"] == 1
    assert report["redisOutage"]["equivalentToDisabled"] is True
    assert report["retrievalDto"]["warmEncoderInputs"] == 1
    assert report["retrievalDto"]["repeatEncoderInputs"] == 0
    assert report["retrievalDto"]["equivalent"] is True
    assert report["query"]["warmRedisCommandDelta"] == {
        "mget:success": 2.0,
        "pipeline_set:success": 1.0,
    }
    assert report["query"]["repeatRedisCommandDelta"] == {"mget:success": 1.0}
    assert report["ingestion"]["warmRedisCommandDelta"] == {
        "mget:success": 2.0,
        "pipeline_set:success": 1.0,
    }
    assert report["ingestion"]["repeatRedisCommandDelta"] == {"mget:success": 1.0}
    assert report["redis"]["keyCardinality"] > 0
    assert report["redis"]["payloadBytes"]["total"] > 0
    assert report["redis"]["memoryBytes"] > report["redis"]["payloadBytes"]["total"]
    for duration in (
        report["disabled"]["firstDurationMs"],
        report["disabled"]["repeatDurationMs"],
        report["query"]["warmDurationMs"],
        report["query"]["repeatDurationMs"],
        report["ingestion"]["warmDurationMs"],
        report["ingestion"]["repeatDurationMs"],
        report["redisOutage"]["durationMs"],
    ):
        assert duration >= 0

    encoded_report = report_path.read_text(encoding="utf-8")
    for text in fixture_texts(fixture):
        assert text not in encoded_report


def fixture_texts(fixture: dict[str, Any]) -> list[str]:
    groups = fixture["ingestionTexts"]
    return [
        fixture["query"],
        fixture["legacyText"],
        *(text for values in groups.values() for text in values),
    ]


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
