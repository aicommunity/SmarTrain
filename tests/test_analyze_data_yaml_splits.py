from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from smartrain.services.analyze.data_yaml_splits import (
    collect_split_images_resolved,
    pick_best_data_yaml_candidate,
    resolve_best_split,
    split_dir_exists,
)
from smartrain.workflows.analyze import results_analyzer


def test_collect_split_images_resolved_uses_path_field_and_train_fallback(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "ds_a"
    train_images = dataset_dir / "train" / "images"
    train_images.mkdir(parents=True, exist_ok=True)
    (train_images / "a.jpg").write_bytes(b"img")
    (dataset_dir / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    run_tmp = run_dir / "tmp"
    run_tmp.mkdir(parents=True, exist_ok=True)
    runtime_yaml = run_tmp / "_runtime_data_train.yaml"
    runtime_yaml.write_text(
        f"path: {dataset_dir}\ntrain: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )

    assert split_dir_exists(str(runtime_yaml), "test") is False
    assert split_dir_exists(str(runtime_yaml), "train") is True
    assert resolve_best_split(str(runtime_yaml)) == "train"

    images, used_split = collect_split_images_resolved(str(runtime_yaml), limit=10)
    assert used_split == "train"
    assert len(images) == 1
    assert images[0].endswith("a.jpg")


def test_build_run_data_yaml_map_prefers_workspace_dataset_over_runtime_yaml(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "best.pt").write_bytes(b"model")

    dataset_dir = tmp_path / "datasets" / "ds_a"
    test_images = dataset_dir / "test" / "images"
    test_images.mkdir(parents=True, exist_ok=True)
    (test_images / "t.jpg").write_bytes(b"img")
    (dataset_dir / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )

    runtime_yaml = run_dir / "_runtime_data_train.yaml"
    runtime_yaml.write_text(
        f"path: {tmp_path / 'runs'}\ntrain: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )
    (run_dir / "train" / "args.yaml").write_text(f"data: {runtime_yaml}\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {
                    "dataset": {"name": "ds_a", "path_under_workspace": "datasets/ds_a"},
                }
            }
        ),
        encoding="utf-8",
    )

    run_map, _src_map, unresolved = results_analyzer._build_run_data_yaml_map(
        [str(run_dir)],
        str(tmp_path),
        preferred_split="test",
    )
    assert unresolved == []
    assert run_map[str(run_dir)] == str(dataset_dir / "data.yaml")


def test_benchmark_soft_fail_skips_when_no_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from smartrain.services.analyze import benchmark as benchmark_mod

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: train/images\n", encoding="utf-8")
    args = argparse.Namespace(
        runs_group_dir=str(tmp_path),
        data_yaml=str(data_yaml),
        split="auto",
        frames=5,
        device="cpu",
        half=False,
        soft_fail=True,
        workspace=str(tmp_path),
        out_csv=str(tmp_path / "bench.csv"),
        selected_run_dirs=[],
        reuse_run_cache=False,
        cache_stats_out=str(tmp_path / "cache.json"),
    )
    benchmark_mod.run_inference_benchmark(
        args,
        prompt_text_cb=lambda *_a, **_k: "",
        resolve_workspace_root_cb=lambda _w: str(tmp_path),
        workspace_layout_cls=type("L", (), {"analytics": str(tmp_path / "analytics")}),
        preferred_run_model_path_cb=lambda *_a, **_k: "",
        run_cache_root_cb=lambda *_a, **_k: str(tmp_path),
        compute_fingerprint_cb=lambda *_a, **_k: "fp",
        data_yaml_hash_cb=lambda *_a, **_k: "h",
        weights_hash_cb=lambda *_a, **_k: "w",
        append_cache_entry_cb=lambda *_a, **_k: None,
        ultralytics_sidecar_dir_cb=lambda rd, _n: str(tmp_path / "scratch"),
        clear_gpu_memory_cb=lambda: None,
    )
    assert not (tmp_path / "bench.csv").exists()


def test_pick_best_data_yaml_candidate_ranks_workspace_source_first() -> None:
    candidates = [
        ("/tmp/runtime.yaml", "_runtime_data_train.yaml"),
        ("/w/datasets/ds/data.yaml", "training_metadata.dataset.name -> workspace/datasets"),
    ]
    picked = pick_best_data_yaml_candidate(candidates, preferred_split=None)
    assert picked is not None
    assert picked[0] == "/w/datasets/ds/data.yaml"
