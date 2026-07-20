from __future__ import annotations

import json
from pathlib import Path


def test_phase3_evidence_is_redacted_and_complete() -> None:
    path = Path("docs/evidence/cache/phase3-visual-audio-local.json")
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["schemaVersion"] == 1
    assert report["redisUrl"].startswith("redis://")
    assert "@" not in report["redisUrl"]
    assert set(report["modelIdentities"]) == {"visual", "audio"}
    assert set(report["pipelineIdentities"]) == {"visual", "audio"}
    assert len(report["fixtureSha256"]) == 6
    for media, magic in (("visual", "KVEV"), ("audio", "KAEV")):
        evidence = report[media]
        assert evidence["disabledEncoderInputs"] == 2
        assert evidence["coldEncoderInputs"] == 1
        assert evidence["warmEncoderInputs"] == 0
        assert evidence["byteChangeEncoderInputs"] == 1
        assert evidence["corruptEncoderInputs"] == 1
        assert evidence["redisOutageEncoderInputs"] == 1
        assert evidence["disabledEquivalent"] is True
        assert evidence["duplicateEquivalent"] is True
        assert evidence["warmEquivalent"] is True
        assert evidence["byteChangeDifferent"] is True
        assert evidence["corruptEquivalent"] is True
        assert evidence["redisOutageEquivalent"] is True
        assert evidence["repairedMagic"] == magic
        assert evidence["redisCommandDelta"]["cold"] == {
            "mget:success": 1.0,
            "pipeline_set:success": 1.0,
        }
        assert evidence["redisCommandDelta"]["warm"] == {"mget:success": 1.0}
        assert evidence["redis"]["keyCardinality"] == 2
        assert evidence["redis"]["payloadBytes"]["total"] > 0
        assert evidence["redis"]["memoryBytes"] > evidence["redis"]["payloadBytes"]["total"]
        for duration in (
            *evidence["disabledDurationsMs"],
            evidence["coldDurationMs"],
            evidence["warmDurationMs"],
            evidence["byteChangeDurationMs"],
            evidence["corruptDurationMs"],
            evidence["redisOutageDurationMs"],
        ):
            assert duration >= 0

    encoded = path.read_text(encoding="utf-8")
    for forbidden in (
        "password",
        "username",
        "ownerId",
        "userId",
        "mediaId",
        "contentHash",
        "fixture.png",
        "fixture.wav",
        "/Users/",
        "/tmp/",
    ):
        assert forbidden not in encoded
