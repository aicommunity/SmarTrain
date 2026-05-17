from __future__ import annotations

from pathlib import Path

from smartrain.run_model_contract.io.read.run_adapter import RunAdapter
from smartrain.run_model_contract.io.write.dual_write import run_dual_write


def test_dual_write_best_effort_warns_on_legacy_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    payload = RunAdapter().read(str(run_dir))

    def _boom() -> None:
        raise RuntimeError("legacy")

    rep = run_dual_write(
        payload=payload,
        target_root=str(run_dir),
        mode="dual_write_best_effort",
        legacy_writer=_boom,
    )
    assert rep.unified_status == "ok"
    assert rep.legacy_status == "failed"
    assert rep.write_report is not None
    assert any("legacy write failed" in w for w in rep.warnings)


def test_dual_write_strict_reports_rollback_hint(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run2"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    payload = RunAdapter().read(str(run_dir))

    def _boom() -> None:
        raise RuntimeError("legacy")

    rep = run_dual_write(
        payload=payload,
        target_root=str(run_dir),
        mode="dual_write_strict",
        legacy_writer=_boom,
    )
    assert rep.unified_status == "ok"
    assert rep.legacy_status == "failed"
    assert rep.rollback_hint is not None
    assert Path(rep.write_report.manifest_path).is_file()  # type: ignore[union-attr]
