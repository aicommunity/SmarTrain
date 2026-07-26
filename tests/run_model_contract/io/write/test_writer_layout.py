from __future__ import annotations

import json
from pathlib import Path

from smartrain.run_model_contract.io.read.run_adapter import RunAdapter
from smartrain.run_model_contract.io.write.layout import unified_snapshot_dir
from smartrain.run_model_contract.io.write.writer import write_unified_snapshot


def test_write_unified_snapshot_creates_layout_and_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")

    payload = RunAdapter().read(str(run_dir))
    rep = write_unified_snapshot(payload, str(run_dir))

    snap = unified_snapshot_dir(run_dir)
    assert snap.is_dir()
    assert Path(rep.snapshot_path).is_file()
    assert Path(rep.manifest_path).is_file()
    man = json.loads(Path(rep.manifest_path).read_text(encoding="utf-8"))
    assert man["payload_hash_sha256"] == rep.payload_hash_sha256
    assert man["schema_version"] == payload.schema_version
    assert man["hash_algorithm"] == "sha256"
    assert man["aggregate_artifacts_hash_sha256"] == rep.aggregate_artifacts_hash_sha256
    assert man["artifacts"]["snapshot.json"]["sha256"] == rep.payload_hash_sha256
    assert man["source_run_ref"] == "runs/ds/run1"
    assert "\\" not in man["source_run_ref"]
    assert rep.manifest_hash_sha256
