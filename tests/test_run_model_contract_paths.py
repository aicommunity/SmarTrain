from __future__ import annotations

import json
from pathlib import Path

from smartrain.core.runtime.run_artifacts import write_model_sidecar_metadata
from smartrain.run_model_contract.domain.models import UnifiedModelRef, UnifiedPayload, UnifiedRunRef
from smartrain.run_model_contract.io.read.normalizers import normalize_path
from smartrain.run_model_contract.io.read.run_adapter import RunAdapter
from smartrain.run_model_contract.io.write.writer import write_unified_snapshot


def test_normalize_path_with_anchor(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "ds" / "r1"
    weights = run / "models" / "a.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"pt")
    got = normalize_path("models/a.pt", anchor=run)
    assert Path(got).resolve() == weights.resolve()
    legacy = normalize_path(str(weights.resolve()))
    assert Path(legacy).resolve() == weights.resolve()


def test_run_adapter_stores_relative_weights(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "ds" / "r1"
    models = run / "models"
    models.mkdir(parents=True)
    pt = models / "detect_yolo11n_20260101_000000_640px_1epochs_b1.pt"
    pt.write_bytes(b"pt")
    (run / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {
                    "model": "yolo11n",
                    "dataset": {"name": "ds"},
                    "task_type": "detection",
                }
            }
        ),
        encoding="utf-8",
    )
    payload = RunAdapter().read(str(run))
    wp = payload.models[0].weights_path
    assert "\\" not in wp
    assert not Path(wp).is_absolute()
    assert wp.replace("\\", "/").endswith(pt.name) or wp.startswith("models/")


def test_write_unified_snapshot_source_run_ref_relative(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "ds" / "r1"
    run.mkdir(parents=True)
    payload = UnifiedPayload(
        schema_version="2.0.0",
        generated_at="1970-01-01T00:00:00Z",
        producer="test",
        models=[
            UnifiedModelRef(
                model_id="m",
                format="pt",
                weights_path="models/a.pt",
                config_path=None,
                labels_path=None,
                provenance={},
                task_type="detection",
                backend_type="ultralytics",
            )
        ],
        runs=[
            UnifiedRunRef(
                run_id="r1",
                workspace=".",
                dataset_ref="ds",
                training_ref="runs/ds/r1",
                task_type="detection",
                backend_type="ultralytics",
            )
        ],
    )
    report = write_unified_snapshot(payload, str(run))
    man = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))
    ref = str(man.get("source_run_ref") or "")
    assert "\\" not in ref
    assert ":" not in ref[1:3] if len(ref) > 2 else True
    assert not Path(ref).is_absolute() or ref in {".", "r1", "runs/ds/r1"}


def test_sidecar_metadata_relative_under_workspace(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "ds" / "r1"
    models = run / "models"
    models.mkdir(parents=True)
    pt = models / "a.onnx"
    pt.write_bytes(b"onnx")
    side = write_model_sidecar_metadata(
        pt,
        format_name="onnx",
        run_dir=str(run),
        source_path=str(models / "a.pt"),
        workspace_root=str(tmp_path),
    )
    data = json.loads(side.read_text(encoding="utf-8"))
    assert data["path"] == "models/a.onnx"
    assert "\\" not in data["path"]
    assert data["run_path"] == "runs/ds/r1"
    assert not str(data["path"]).startswith("/")
