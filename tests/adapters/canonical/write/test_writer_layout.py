from __future__ import annotations

import json
from pathlib import Path

from smartrain.adapters.canonical.read.run_adapter import RunAdapter
from smartrain.adapters.canonical.write.layout import canonical_snapshot_dir
from smartrain.adapters.canonical.write.writer import write_canonical_snapshot


def test_write_canonical_snapshot_creates_layout_and_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")

    payload = RunAdapter().read(str(run_dir))
    rep = write_canonical_snapshot(payload, str(run_dir))

    snap = canonical_snapshot_dir(run_dir)
    assert snap.is_dir()
    assert Path(rep.snapshot_path).is_file()
    assert Path(rep.manifest_path).is_file()
    man = json.loads(Path(rep.manifest_path).read_text(encoding="utf-8"))
    assert man["payload_hash_sha256"] == rep.payload_hash_sha256
    assert man["schema_version"] == payload.schema_version
