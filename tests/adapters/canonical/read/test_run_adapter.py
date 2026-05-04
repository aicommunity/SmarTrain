from __future__ import annotations

import json
from pathlib import Path

from smartrain.adapters.canonical.read.run_adapter import RunAdapter


def test_run_adapter_reads_canonical_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds_a" / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {
                    "task_type": "detection",
                    "dataset": {"name": "ds_a"},
                    "provider": {"id": "ultralytics"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = RunAdapter().read(str(run_dir))
    assert payload.models
    assert payload.models[0].format == "pt"
    assert payload.runs[0].run_id == "run_1"

