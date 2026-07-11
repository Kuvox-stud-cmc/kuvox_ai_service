from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_frontend_and_worker_font_catalogs_are_identical() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    frontend_fonts = repository_root / "kuvox_frontend" / "public" / "fonts"
    worker_fonts = repository_root / "kuvox_ai_service" / "src" / "kuvox_ai" / "modules" / "rendering" / "fonts"
    frontend_catalog = json.loads((frontend_fonts / "catalog.json").read_text(encoding="utf-8"))
    worker_catalog = json.loads((worker_fonts / "catalog.json").read_text(encoding="utf-8"))

    assert frontend_catalog == worker_catalog
    filenames = {
        filename
        for entry in frontend_catalog.values()
        for key, filename in entry.items()
        if key != "fallback"
    }
    for filename in filenames:
        assert _sha256(frontend_fonts / filename) == _sha256(worker_fonts / filename)

    frontend_licenses = sorted(path.name for path in frontend_fonts.glob("OFL-*.txt"))
    worker_licenses = sorted(path.name for path in worker_fonts.glob("OFL-*.txt"))
    assert frontend_licenses == worker_licenses
    assert len(frontend_licenses) == 11


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
