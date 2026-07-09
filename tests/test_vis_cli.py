from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from smartrain.core.runtime.workspace_paths import deploy_workspace
from smartrain.services.visualization.cli_commands import build_vis_arg_parser, run_vis_cli


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 96), color=(120, 120, 120)).save(path)


def _write_label(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def _make_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "datasets" / "ds_a"
    _write_image(ds / "train" / "images" / "a.jpg")
    _write_image(ds / "val" / "images" / "b.jpg")
    _write_image(ds / "test" / "images" / "c.jpg")
    _write_label(ds / "train" / "labels" / "a.txt")
    _write_label(ds / "val" / "labels" / "b.txt")
    _write_label(ds / "test" / "labels" / "c.txt")
    (ds / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: ['obj']\n",
        encoding="utf-8",
    )
    (tmp_path / "datasets" / "datasets_info.json").write_text(
        json.dumps({"ds_a": {"structure": "split", "classes": {"obj": 0}}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ds


def test_vis_parser_subcommands_and_interactive_fallback(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    parser = build_vis_arg_parser()
    args = parser.parse_args([])
    assert args.mode is None
    monkeypatch.setattr("smartrain.services.visualization.cli_commands.is_interactive_allowed", lambda _a: True)
    monkeypatch.setattr("smartrain.services.visualization.cli_commands.prompt_choice", lambda *_a, **_k: "dataset")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    rc = run_vis_cli(["--workspace", str(tmp_path), "dataset", "--dataset", str(_make_dataset(tmp_path))])
    assert rc == 0


def test_vis_dataset_creates_mirrored_visualize_tree(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _make_dataset(tmp_path)
    rc = run_vis_cli(["--workspace", str(tmp_path), "dataset", "--dataset", str(ds)])
    assert rc == 0
    assert (ds / "visualize" / "train" / "a.jpg").is_file()
    assert (ds / "visualize" / "val" / "b.jpg").is_file()
    assert (ds / "visualize" / "test" / "c.jpg").is_file()
    palette = json.loads((tmp_path / "label_colors.json").read_text(encoding="utf-8"))
    assert "obj" in palette


def test_vis_dataset_recreates_after_manual_delete(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _make_dataset(tmp_path)
    run_vis_cli(["--workspace", str(tmp_path), "dataset", "--dataset", str(ds)])
    out = ds / "visualize"
    assert out.is_dir()
    for item in sorted(out.rglob("*"), reverse=True):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
    out.rmdir()
    rc = run_vis_cli(["--workspace", str(tmp_path), "dataset", "--dataset", str(ds)])
    assert rc == 0
    assert (ds / "visualize" / "train" / "a.jpg").is_file()


def test_vis_run_resolves_dataset_splits_from_metadata(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _make_dataset(tmp_path)
    run_dir = tmp_path / "runs" / "ds_a" / "run_1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr("smartrain.services.visualization.pipeline.run_inference_for_split", lambda **_k: {})
    rc = run_vis_cli(["--workspace", str(tmp_path), "run", "--run", str(run_dir)])
    assert rc == 0
    assert (run_dir / "visualize" / "summary.json").is_file()


def test_vis_model_writes_into_run_or_model_artifacts(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _make_dataset(tmp_path)
    run_dir = tmp_path / "runs" / "ds_a" / "run_1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    model_dir = tmp_path / "models" / "m1"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "m1.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "m1.pt", "source_run": str(run_dir), "task_type": "detection"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr("smartrain.services.visualization.pipeline.run_inference_for_split", lambda **_k: {})
    rc = run_vis_cli(["--workspace", str(tmp_path), "model", "--model-name", "m1"])
    assert rc == 0
    assert (run_dir / "visualize" / "summary.json").is_file()


def test_vis_index_and_summary_contract(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _make_dataset(tmp_path)
    rc = run_vis_cli(["--workspace", str(tmp_path), "dataset", "--dataset", str(ds)])
    assert rc == 0
    summary = json.loads((ds / "visualize" / "summary.json").read_text(encoding="utf-8"))
    assert "total_frames" in summary
    assert "ok_frames" in summary
    index_lines = (ds / "visualize" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert index_lines


def test_vis_skip_existing_when_not_overwrite(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _make_dataset(tmp_path)
    run_vis_cli(["--workspace", str(tmp_path), "dataset", "--dataset", str(ds)])
    rc = run_vis_cli(["--workspace", str(tmp_path), "dataset", "--dataset", str(ds)])
    assert rc == 0
    summary = json.loads((ds / "visualize" / "summary.json").read_text(encoding="utf-8"))
    assert summary["skipped_frames"] > 0


def test_vis_interactive_model_prompts_use_catalog(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _make_dataset(tmp_path)
    run_dir = tmp_path / "runs" / "ds_a" / "run_1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    model_dir = tmp_path / "models" / "m1"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "m1.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "m1.pt", "source_run": str(run_dir), "task_type": "detection"}, ensure_ascii=False),
        encoding="utf-8",
    )

    choices = iter(["model", "models", "m1/m1.pt", "no"])
    monkeypatch.setattr("smartrain.services.visualization.cli_commands.is_interactive_allowed", lambda _a: True)
    monkeypatch.setattr("smartrain.services.visualization.cli_commands.prompt_choice", lambda *_a, **_k: next(choices))
    monkeypatch.setattr("smartrain.services.visualization.cli_commands.prompt_multi_choice_csv", lambda *_a, **_k: ["train"])
    monkeypatch.setattr("smartrain.services.visualization.cli_commands.prompt_device_selection", lambda **_k: "cpu")
    monkeypatch.setattr("smartrain.services.visualization.pipeline.run_inference_for_split", lambda **_k: {})
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    rc = run_vis_cli(["--workspace", str(tmp_path)])
    assert rc == 0
    assert (run_dir / "visualize" / "summary.json").is_file()


def test_vis_interactive_run_prompts_use_run_discovery(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _make_dataset(tmp_path)
    run_dir = tmp_path / "runs" / "ds_a" / "run_1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    choices = iter(["run", "1"])
    monkeypatch.setattr("smartrain.services.visualization.cli_commands.is_interactive_allowed", lambda _a: True)
    monkeypatch.setattr("smartrain.services.visualization.cli_commands.prompt_choice", lambda *_a, **_k: next(choices))
    monkeypatch.setattr("smartrain.services.visualization.cli_commands.prompt_multi_choice_csv", lambda *_a, **_k: ["train"])
    monkeypatch.setattr("smartrain.services.visualization.cli_commands.prompt_device_selection", lambda **_k: "cpu")
    monkeypatch.setattr("smartrain.services.visualization.pipeline.run_inference_for_split", lambda **_k: {})
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    rc = run_vis_cli(["--workspace", str(tmp_path)])
    assert rc == 0
    assert (run_dir / "visualize" / "summary.json").is_file()

    deploy_workspace(str(tmp_path))
    ds = _make_dataset(tmp_path)
    run_vis_cli(["--workspace", str(tmp_path), "dataset", "--dataset", str(ds)])
    rc = run_vis_cli(["--workspace", str(tmp_path), "--overwrite", "dataset", "--dataset", str(ds)])
    assert rc == 0
    summary = json.loads((ds / "visualize" / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok_frames"] > 0

