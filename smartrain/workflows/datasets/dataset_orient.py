from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np
from tqdm import tqdm

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.cli_support.cli_prompts import prompt_yes_no
from smartrain.cli_support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.workflows.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.workflows.datasets.dataset_hash import calculate_dataset_hash
from smartrain.workflows.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.workflows.datasets.yolo_labels import read_yolo_labels, rotate_yolo_labels_90cw_k, write_yolo_labels


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ON_UNCERTAIN = ("keep", "skip", "fail")
METHODS = ("reference", "rotnet")


@dataclass(frozen=True)
class _RefSig:
    desc: np.ndarray | None  # ORB descriptors
    grad_vec: tuple[float, float]  # normalized (ex, ey)
    tmpl: np.ndarray  # normalized template vector (flattened)


def build_orient_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Fix rotations 0/90/180/270 in the YOLO dataset (to new datasets/<name>)")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Name of source dataset from datasets_info.json")
    p.add_argument(
        "--method",
        choices=METHODS,
        default="reference",
        help="Orientation determination method: reference (standards) or rotnet (trained classifier).",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name of output dataset (default <dataset>_oriented)",
    )
    p.add_argument(
        "--reference",
        action="append",
        default=None,
        help="Path to reference image in correct orientation (can be repeated).",
    )
    p.add_argument(
        "--min-score",
        type=int,
        default=8,
        help="Minimum confidence score (after combining ORB+gradient fallback).",
    )
    p.add_argument(
        "--on-uncertain",
        choices=ON_UNCERTAIN,
        default="keep",
        help="What to do if confidence is lower --min-score: keep|skip|fail.",
    )
    p.add_argument("--report-only", action="store_true", help="Only a report on the distribution of angles, without recording a dataset.")
    p.add_argument("--dry-run", action="store_true", help="Not write files, but perform the calculation and print the result.")
    p.add_argument("--no-legend", action="store_true", help="Disable progress bar.")
    p.add_argument("--rotnet-model-path", type=str, default=None, help="Path to the .pt RotNet model (inside the dataset by default).")
    p.add_argument("--rotnet-epochs", type=int, default=4, help="Number of RotNet training/retraining epochs.")
    p.add_argument("--rotnet-batch-size", type=int, default=64, help="Batch size for RotNet.")
    p.add_argument("--rotnet-lr", type=float, default=1e-3, help="Learning rate for RotNet.")
    p.add_argument("--rotnet-image-size", type=int, default=96, help="RotNet input size (square).")
    p.add_argument(
        "--rotnet-device",
        type=str,
        default="auto",
        help="Device for RotNet: auto|cpu|cuda",
    )
    p.add_argument("--rotnet-finetune", action="store_true", help="Retrain an existing RotNet model from the dataset.")
    p.add_argument(
        "--rotnet-anchor-mode",
        choices=("reference", "majority"),
        default="majority",
        help="Absolute angle calibration for RotNet: reference (by --reference) or majority (by dataset mode).",
    )
    p.add_argument(
        "--rotnet-anchor-samples",
        type=int,
        default=256,
        help="How many images to use for majority-calibration of RotNet.",
    )
    p.add_argument(
        "--rotnet-pretrained",
        type=str,
        default=None,
        help="Path to external checkpoint for initialization/additional training of RotNet.",
    )
    return p


def _load_catalog(layout: WorkspaceLayout) -> dict:
    p = layout.work_datasets_info_path()
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _interactive_fill(args, *, dataset_names: list[str]) -> None:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter
    from smartrain.cli_support.cli_prompts import prompt_choice, prompt_text

    print("[INFO] Interactive mode orient")
    args.dataset = prompt_choice("Dataset", dataset_names, default=dataset_names[0])
    args.output_name = prompt_text("Output dataset name (empty=auto)", default=(args.output_name or "")) or None
    args.method = prompt_choice("Method", list(METHODS), default=str(getattr(args, "method", "reference")))

    if args.method == "reference":
        refs: list[str] = []
        print("[INFO] References (correct orientation). Enter paths, empty line = end.")
        while True:
            r = prompt("Path to reference image (empty=finish): ", default="").strip()
            if not r:
                break
            refs.append(r)
        if refs:
            args.reference = refs
    else:
        raw_epochs = prompt_text("Number of RotNet epochs (--rotnet-epochs)", default=str(getattr(args, "rotnet_epochs", 4))).strip()
        if raw_epochs:
            try:
                args.rotnet_epochs = int(raw_epochs)
            except ValueError:
                pass
        raw_size = prompt_text("RotNet input size (--rotnet-image-size)", default=str(getattr(args, "rotnet_image_size", 96))).strip()
        if raw_size:
            try:
                args.rotnet_image_size = int(raw_size)
            except ValueError:
                pass
        raw_bs = prompt_text("RotNet batch size (--rotnet-batch-size)", default=str(getattr(args, "rotnet_batch_size", 64))).strip()
        if raw_bs:
            try:
                args.rotnet_batch_size = int(raw_bs)
            except ValueError:
                pass
        raw_lr = prompt_text("Learning rate RotNet (--rotnet-lr)", default=str(getattr(args, "rotnet_lr", 1e-3))).strip()
        if raw_lr:
            try:
                args.rotnet_lr = float(raw_lr)
            except ValueError:
                pass
        args.rotnet_device = prompt_choice(
            "RotNet device (--rotnet-device)",
            ["auto", "cpu", "cuda"],
            default=str(getattr(args, "rotnet_device", "auto")),
        )
        args.rotnet_model_path = prompt_text(
            "Path to the RotNet model (--rotnet-model-path, empty=inside the dataset)",
            default=str(getattr(args, "rotnet_model_path", "") or ""),
        ).strip() or None
        args.rotnet_pretrained = prompt_text(
            "Path to pretrained RotNet (--rotnet-pretrained, empty=no)",
            default=str(getattr(args, "rotnet_pretrained", "") or ""),
        ).strip() or None
        args.rotnet_anchor_mode = prompt_choice(
            "RotNet Anchor Mode (--rotnet-anchor-mode)",
            ["reference", "majority"],
            default=str(getattr(args, "rotnet_anchor_mode", "majority")),
        )
        raw_anchor_samples = prompt_text(
            "Number of RotNet anchor samples (--rotnet-anchor-samples)",
            default=str(getattr(args, "rotnet_anchor_samples", 256)),
        ).strip()
        if raw_anchor_samples:
            try:
                args.rotnet_anchor_samples = int(raw_anchor_samples)
            except ValueError:
                pass
        args.rotnet_finetune = prompt_yes_no("Enable RotNet retraining (--rotnet-finetune)?", default=bool(args.rotnet_finetune))

    raw = prompt_text("Minimum score (--min-score)", default=str(getattr(args, "min_score", 8))).strip()
    if raw:
        try:
            args.min_score = int(raw)
        except ValueError:
            pass
    args.on_uncertain = prompt_choice(
        "Behavior during uncertainty (--on-uncertain)",
        list(ON_UNCERTAIN),
        default=str(getattr(args, "on_uncertain", "keep")),
    )
    args.report_only = prompt_yes_no("Only a report without recording a dataset (--report-only)?", default=bool(args.report_only))
    args.dry_run = prompt_yes_no("Do dry-run (--dry-run)?", default=bool(args.dry_run))


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {path}")
    return img


def _rot_img_k(img: np.ndarray, k: int) -> np.ndarray:
    kk = int(k) % 4
    if kk == 0:
        return img
    if kk == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if kk == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _orb() -> cv2.ORB:
    # nfeatures improved: scenes with mesh/noise may produce few key points at defaults
    return cv2.ORB_create(nfeatures=1200, fastThreshold=7)


def _compute_grad_vec(gray: np.ndarray) -> tuple[float, float]:
    g = gray.astype(np.float32) / 255.0
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ex = float(np.mean(np.abs(sx)))
    ey = float(np.mean(np.abs(sy)))
    s = ex + ey + 1e-9
    return ex / s, ey / s


def _template_vec(gray: np.ndarray, *, size: int = 64) -> np.ndarray:
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = small.reshape(-1)
    v = v - float(np.mean(v))
    n = float(np.linalg.norm(v) + 1e-9)
    return v / n


def _load_references(paths: list[str]) -> list[_RefSig]:
    o = _orb()
    out: list[_RefSig] = []
    for p in paths:
        gray = _read_gray(p)
        kp, desc = o.detectAndCompute(gray, None)
        _ = kp  # for symmetry, unused
        out.append(_RefSig(desc=desc, grad_vec=_compute_grad_vec(gray), tmpl=_template_vec(gray)))
    return out


def _score_orb(desc: np.ndarray | None, refs: list[_RefSig]) -> int:
    if desc is None or len(desc) == 0:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    score = 0
    for r in refs:
        if r.desc is None or len(r.desc) == 0:
            continue
        try:
            matches = bf.match(desc, r.desc)
        except cv2.error:
            continue
        if not matches:
            continue
        # Good matches by distance threshold
        good = [m for m in matches if m.distance <= 48.0]
        score += int(len(good))
    return int(score)


def _grad_distance(vec: tuple[float, float], refs: list[_RefSig]) -> float:
    if not refs:
        return 1.0
    rv = np.array([np.mean([r.grad_vec[0] for r in refs]), np.mean([r.grad_vec[1] for r in refs])], dtype=np.float32)
    v = np.array([vec[0], vec[1]], dtype=np.float32)
    return float(np.linalg.norm(v - rv, ord=2))


def _template_similarity(vec: np.ndarray, refs: list[_RefSig]) -> float:
    if not refs:
        return 0.0
    sims = [float(np.dot(vec, r.tmpl)) for r in refs]
    return float(max(sims)) if sims else 0.0


def _combined_score(orb_score: int, grad_dist: float, tmpl_sim: float) -> int:
    # grad_dist in [0..~1.4], map to [0..100] where 0 dist => 100
    grad_score = int(max(0.0, 100.0 * (1.0 - min(1.0, grad_dist))))
    tmpl_score = int(max(0.0, min(1.0, (tmpl_sim + 1.0) / 2.0)) * 120.0)  # [-1..1] -> [0..120]
    return int(orb_score * 2 + grad_score + tmpl_score)


def _choose_rotation(gray: np.ndarray, refs: list[_RefSig]) -> tuple[int, dict[int, int]]:
    o = _orb()
    scores: dict[int, int] = {}
    for k in (0, 1, 2, 3):
        g = _rot_img_k(gray, k)
        _kp, desc = o.detectAndCompute(g, None)
        orb_s = _score_orb(desc, refs)
        gd = _grad_distance(_compute_grad_vec(g), refs)
        ts = _template_similarity(_template_vec(g), refs)
        scores[k] = _combined_score(orb_s, gd, ts)
    best_k = max(scores.items(), key=lambda kv: kv[1])[0]
    return int(best_k), {int(k): int(v) for k, v in scores.items()}


def _choose_device(name: str):
    import torch

    nm = (name or "auto").strip().lower()
    if nm == "cpu":
        return torch.device("cpu")
    if nm == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_rotnet_paths(src_root: str) -> tuple[str, str]:
    base = os.path.join(src_root, ".orient_rotnet")
    return os.path.join(base, "model.pt"), os.path.join(base, "train_meta.json")


def _iter_dataset_images(buckets: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for images_path, _labels_path in buckets:
        for name in os.listdir(images_path):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                out.append(os.path.join(images_path, name))
    return out


def _make_rotnet_model():
    import torch.nn as nn

    class _TinyRotNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(64, 4)

        def forward(self, x):
            z = self.features(x).flatten(1)
            return self.head(z)

    return _TinyRotNet()


def _load_gray_resized(path: str, size: int) -> np.ndarray:
    g = _read_gray(path)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return g


def _train_or_load_rotnet(args, *, src_root: str, image_paths: list[str]):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    model_path_default, meta_path = _default_rotnet_paths(src_root)
    model_path = os.path.abspath(args.rotnet_model_path or model_path_default)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    device = _choose_device(args.rotnet_device)
    image_size = max(32, int(args.rotnet_image_size))

    class _RotDataset(Dataset):
        def __init__(self, paths: list[str], img_size: int):
            self.paths = paths
            self.img_size = img_size

        def __len__(self):
            return len(self.paths) * 4

        def __getitem__(self, idx: int):
            p = self.paths[idx // 4]
            r = idx % 4
            g = _load_gray_resized(p, self.img_size)
            g = _rot_img_k(g, r)
            x = torch.from_numpy(g).unsqueeze(0)
            y = torch.tensor(r, dtype=torch.long)
            return x, y

    model = _make_rotnet_model().to(device)

    init_path = None
    if args.rotnet_pretrained:
        init_path = os.path.abspath(args.rotnet_pretrained)
    elif args.rotnet_finetune and os.path.isfile(model_path):
        init_path = model_path
    elif os.path.isfile(model_path) and int(args.rotnet_epochs) <= 0:
        init_path = model_path
    if init_path and os.path.isfile(init_path):
        ckpt = torch.load(init_path, map_location=device)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)

    need_train = bool(args.rotnet_finetune or not os.path.isfile(model_path) or int(args.rotnet_epochs) > 0)
    epochs = max(0, int(args.rotnet_epochs))
    if need_train and epochs > 0:
        print(
            f"[INFO] RotNet: training, epochs={epochs}, images={len(image_paths)}, "
            f"batch={max(4, int(args.rotnet_batch_size))}, device={device}"
        )
        ds = _RotDataset(image_paths, image_size)
        dl = DataLoader(ds, batch_size=max(4, int(args.rotnet_batch_size)), shuffle=True, num_workers=0)
        opt = torch.optim.Adam(model.parameters(), lr=float(args.rotnet_lr))
        model.train()
        epoch_bar = tqdm(range(epochs), desc="orient:rotnet:epochs", disable=bool(args.no_legend))
        for _ in epoch_bar:
            running_loss = 0.0
            seen = 0
            batch_bar = tqdm(dl, desc="orient:rotnet:batches", leave=False, disable=bool(args.no_legend))
            for xb, yb in batch_bar:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                bs = int(yb.shape[0])
                running_loss += float(loss.detach().cpu().item()) * bs
                seen += bs
                if seen > 0:
                    batch_bar.set_postfix(loss=f"{running_loss / seen:.4f}", refresh=False)
            batch_bar.close()
            if seen > 0:
                epoch_bar.set_postfix(loss=f"{running_loss / seen:.4f}", refresh=False)
        epoch_bar.close()
        torch.save(
            {
                "model_state": model.state_dict(),
                "image_size": int(image_size),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "source_images": len(image_paths),
                "epochs": int(epochs),
            },
            model_path,
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_path": model_path,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "source_images": len(image_paths),
                    "image_size": int(image_size),
                    "epochs": int(epochs),
                    "dataset_root": os.path.abspath(src_root),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    elif os.path.isfile(model_path):
        print(f"[INFO] RotNet: loading saved model {model_path}")
        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        if isinstance(ckpt, dict):
            image_size = int(ckpt.get("image_size", image_size))

    model.eval()
    return model, device, image_size, model_path


def _choose_rotation_rotnet(gray: np.ndarray, model, device, image_size: int) -> tuple[int, dict[int, int]]:
    import torch
    import torch.nn.functional as F

    scores: dict[int, int] = {}
    with torch.no_grad():
        for k in (0, 1, 2, 3):
            g = _rot_img_k(gray, k)
            g = cv2.resize(g, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            x = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(device)
            probs = F.softmax(model(x), dim=1).squeeze(0).detach().cpu().numpy()
            # score = "how much does the frame look like 0° after rotation k"
            scores[k] = int(float(probs[0]) * 1000.0)
    best_k = max(scores.items(), key=lambda kv: kv[1])[0]
    return int(best_k), {int(k): int(v) for k, v in scores.items()}


def _calibrate_rotnet_offset(
    args,
    *,
    model,
    device,
    image_size: int,
    dataset_image_paths: list[str],
) -> int:
    """
    Returns offset c in the formula corrected_k = (raw_k - c) mod 4.
    """
    preds: list[int] = []
    if args.rotnet_anchor_mode == "reference" and args.reference:
        print(f"[INFO] RotNet: calibration by reference ({len(args.reference)} pcs.)")
        for rp in args.reference:
            try:
                gray = _read_gray(str(rp))
            except Exception as e:
                print(f"[WARNING] Reference not available: {rp} ({e})")
                continue
            raw_k, _scores = _choose_rotation_rotnet(gray, model, device, image_size)
            preds.append(int(raw_k))
    if not preds:
        # majority fallback: assume that the majority of frames in the dataset are in canonical orientation
        sample_n = max(8, int(getattr(args, "rotnet_anchor_samples", 256)))
        pool = list(dataset_image_paths)
        random.shuffle(pool)
        pool = pool[: min(sample_n, len(pool))]
        print(f"[INFO] RotNet: calibration by majority ({len(pool)} images)")
        for p in pool:
            try:
                gray = _read_gray(p)
            except Exception:
                continue
            raw_k, _scores = _choose_rotation_rotnet(gray, model, device, image_size)
            preds.append(int(raw_k))
    if not preds:
        print("[WARNING] RotNet: failed to calibrate offset, using offset=0")
        return 0
    # mode by counts
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for p in preds:
        counts[int(p) % 4] += 1
    offset = max(counts.items(), key=lambda kv: kv[1])[0]
    print(f"[INFO] RotNet: offset={offset} (raw_k mode), counts={counts}")
    return int(offset)


def _copy_tree_structure_if_exists(src_root: str, dst_root: str) -> None:
    # copy data.yaml if present at top-level
    for name in ("data.yaml", "data.yml"):
        sp = os.path.join(src_root, name)
        if os.path.isfile(sp):
            os.makedirs(dst_root, exist_ok=True)
            shutil.copy2(sp, os.path.join(dst_root, name))


def _update_datasets_sidecar(
    layout: WorkspaceLayout,
    output_key: str,
    entry: dict,
    target_dir: str,
    output_hash: str,
) -> None:
    os.makedirs(layout.datasets, exist_ok=True)
    rel = os.path.relpath(os.path.abspath(target_dir), layout.root)
    prev: dict = {}
    info_path = layout.work_datasets_info_path()
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            prev = loaded
    new_entry = dict(entry) if isinstance(entry, dict) else {}
    new_entry["data_path"] = rel
    new_entry["dataset_hash"] = output_hash
    new_entry["modified"] = False
    prev[output_key] = new_entry
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=4)


def _write_orient_stats_csv(out_dir: str, rows: list[dict]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "orient_stats.csv")
    fields = [
        "image_path",
        "label_path",
        "method",
        "raw_k",
        "offset_k",
        "final_k",
        "rotated",
        "score_best",
        "scores_json",
        "uncertain",
        "on_uncertain",
        "action",
        "output_image_path",
        "output_label_path",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return out_path


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_orient_arg_parser()
    args = parser.parse_args(argv)
    interactive_allowed = is_interactive_allowed(argv)
    if args.dataset is None and not interactive_allowed:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    interactive_used = False
    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    catalog = _load_catalog(layout)
    if not catalog:
        print("[ERROR] datasets_info.json was not found or is empty.")
        return

    if args.dataset is None and interactive_allowed and sys.stdin.isatty():
        _interactive_fill(args, dataset_names=sorted(catalog.keys()))
        interactive_used = True
    if not args.dataset:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    if args.dataset not in catalog:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        return
    if args.method == "reference" and not args.reference:
        print("[ERROR] At least one --reference is needed.")
        return
    replay_cmd = None
    if interactive_used:
        replay_cmd = build_non_interactive_command("orient", parser, args)
        print_replay_command("before launch", replay_cmd)

    entry = catalog[args.dataset]
    structure = str(entry.get("structure", "split"))
    if structure == "cvat11":
        print("[ERROR] structure=cvat11 is not supported for orient (by agreement we only work with datasets/without cvat11).")
        return

    src_root = resolve_dataset_root_for_entry(
        args.dataset,
        entry,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )
    buckets = iter_image_label_buckets(
        src_root,
        structure,
        entry,
        dataset_name=args.dataset,
        temp_root=os.path.join(layout.root, "tmp"),
        exclude_test=False,
    )
    if not buckets:
        print("[ERROR] No images/labels pairs were found in the dataset.")
        return
    refs = _load_references([str(p) for p in args.reference]) if args.method == "reference" else []
    rotnet_model = None
    rotnet_device = None
    rotnet_image_size = None
    rotnet_model_path = None
    rotnet_offset = 0
    if args.method == "rotnet":
        all_images = _iter_dataset_images(buckets)
        if not all_images:
            print("[ERROR] There are no images in the dataset for RotNet training/inference.")
            return
        rotnet_model, rotnet_device, rotnet_image_size, rotnet_model_path = _train_or_load_rotnet(
            args,
            src_root=src_root,
            image_paths=all_images,
        )
        rotnet_offset = _calibrate_rotnet_offset(
            args,
            model=rotnet_model,
            device=rotnet_device,
            image_size=int(rotnet_image_size or 96),
            dataset_image_paths=all_images,
        )

    out_base = args.output_name or f"{args.dataset}_oriented"
    out_name = next_dataset_name(layout.datasets, out_base)
    out_dir = os.path.join(layout.datasets, out_name)

    all_images_total = len(_iter_dataset_images(buckets))
    print(f"[INFO] orient: image processing: {all_images_total}")

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    uncertain = 0
    total = 0
    decision_rows: list[dict] = []

    progress = tqdm(total=all_images_total, desc="orient:images", unit="img", disable=bool(args.no_legend))
    for images_path, labels_path in buckets:
        # Preserve bucket-relative layout
        rel_images = os.path.relpath(os.path.abspath(images_path), os.path.abspath(src_root))
        rel_labels = os.path.relpath(os.path.abspath(labels_path), os.path.abspath(src_root))
        # If layout is split-like, keep split folders under output root too.
        dst_images = os.path.join(out_dir, rel_images)
        dst_labels = os.path.join(out_dir, rel_labels)
        os.makedirs(dst_images, exist_ok=True)
        os.makedirs(dst_labels, exist_ok=True)

        for name in os.listdir(images_path):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            img_path = os.path.join(images_path, name)
            lbl_path = os.path.join(labels_path, f"{stem}.txt")
            try:
                gray = _read_gray(img_path)
            except Exception as e:
                print(f"[WARNING] Skip image: {img_path} ({e})")
                progress.update(1)
                continue

            if args.method == "reference":
                k, scores = _choose_rotation(gray, refs)
                raw_k = int(k)
                score_best = scores.get(int(k), 0)
            else:
                raw_k, scores = _choose_rotation_rotnet(
                    gray,
                    rotnet_model,
                    rotnet_device,
                    int(rotnet_image_size or 96),
                )
                k = int((int(raw_k) - int(rotnet_offset)) % 4)
                score_best = scores.get(int(raw_k), 0)
            total += 1
            counts[int(k)] = counts.get(int(k), 0) + 1

            is_uncertain = score_best < int(args.min_score)
            action = "apply"
            if is_uncertain:
                uncertain += 1
                if args.on_uncertain == "skip":
                    action = "skip_uncertain"
                    decision_rows.append(
                        {
                            "image_path": img_path,
                            "label_path": lbl_path,
                            "method": str(args.method),
                            "raw_k": int(raw_k),
                            "offset_k": int(rotnet_offset if args.method == "rotnet" else 0),
                            "final_k": int(k),
                            "rotated": int(k != 0),
                            "score_best": int(score_best),
                            "scores_json": json.dumps(scores, ensure_ascii=False),
                            "uncertain": int(is_uncertain),
                            "on_uncertain": str(args.on_uncertain),
                            "action": action,
                            "output_image_path": "",
                            "output_label_path": "",
                        }
                    )
                    progress.update(1)
                    continue
                if args.on_uncertain == "fail":
                    raise RuntimeError(f"Unsure turn for {img_path}: scores={scores}, chosen={k}")

            if args.report_only or args.dry_run:
                action = "report_only" if args.report_only else "dry_run"
                decision_rows.append(
                    {
                        "image_path": img_path,
                        "label_path": lbl_path,
                        "method": str(args.method),
                        "raw_k": int(raw_k),
                        "offset_k": int(rotnet_offset if args.method == "rotnet" else 0),
                        "final_k": int(k),
                        "rotated": int(k != 0),
                        "score_best": int(score_best),
                        "scores_json": json.dumps(scores, ensure_ascii=False),
                        "uncertain": int(is_uncertain),
                        "on_uncertain": str(args.on_uncertain),
                        "action": action,
                        "output_image_path": "",
                        "output_label_path": "",
                    }
                )
                progress.update(1)
                continue

            # rotate image and labels
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARNING] Failed to read image (color): {img_path}")
                progress.update(1)
                continue
            rbgr = _rot_img_k(bgr, k)
            labels = read_yolo_labels(lbl_path)
            rotated_labels, new_w, new_h = rotate_yolo_labels_90cw_k(labels, w=int(bgr.shape[1]), h=int(bgr.shape[0]), k=int(k))
            # sanity: dimensions should match rotated image
            ih, iw = int(rbgr.shape[0]), int(rbgr.shape[1])
            if (iw, ih) != (new_w, new_h):
                # Should not happen, but keep consistent with actual image
                rotated_labels, _, _ = rotate_yolo_labels_90cw_k(
                    labels,
                    w=int(bgr.shape[1]),
                    h=int(bgr.shape[0]),
                    k=int(k),
                )

            out_img = os.path.join(dst_images, name)
            out_lbl = os.path.join(dst_labels, f"{stem}.txt")
            cv2.imwrite(out_img, rbgr)
            write_yolo_labels(out_lbl, rotated_labels)
            decision_rows.append(
                {
                    "image_path": img_path,
                    "label_path": lbl_path,
                    "method": str(args.method),
                    "raw_k": int(raw_k),
                    "offset_k": int(rotnet_offset if args.method == "rotnet" else 0),
                    "final_k": int(k),
                    "rotated": int(k != 0),
                    "score_best": int(score_best),
                    "scores_json": json.dumps(scores, ensure_ascii=False),
                    "uncertain": int(is_uncertain),
                    "on_uncertain": str(args.on_uncertain),
                    "action": action,
                    "output_image_path": out_img,
                    "output_label_path": out_lbl,
                }
            )
            progress.update(1)
            progress.set_postfix(
                rot0=counts[0],
                rot90=counts[1],
                rot180=counts[2],
                rot270=counts[3],
                uncertain=uncertain,
                refresh=False,
            )

    progress.close()

    print("[OK] orient report")
    print(f"  dataset: {args.dataset}")
    print(f"  output: {out_name}" if not args.report_only else "  output: <report-only>")
    print(f"  total_images: {total}")
    print(f"  rotations(k*90cw): {counts}")
    print(f"  uncertain: {uncertain} (min_score={int(args.min_score)}, on_uncertain={args.on_uncertain})")
    if args.method == "rotnet":
        print(f"  rotnet_model: {rotnet_model_path}")

    if args.report_only or args.dry_run:
        if replay_cmd:
            print_replay_command("after execution", replay_cmd)
        return

    _copy_tree_structure_if_exists(src_root, out_dir)
    stats_path = _write_orient_stats_csv(out_dir, decision_rows)
    out_hash = calculate_dataset_hash(out_dir)
    _update_datasets_sidecar(layout, out_name, entry if isinstance(entry, dict) else {}, out_dir, out_hash)
    write_dataset_passport(
        output_dataset_dir=out_dir,
        command="orient",
        source_datasets=[{"name": args.dataset, "dataset_hash": entry.get("dataset_hash") if isinstance(entry, dict) else None}],
        workspace_root=layout.root,
        parameters={
            "dataset": args.dataset,
            "output_name": out_name,
            "method": str(args.method),
            "reference": list(args.reference or []),
            "min_score": int(args.min_score),
            "on_uncertain": str(args.on_uncertain),
            "rotnet_model_path": rotnet_model_path,
            "rotnet_epochs": int(getattr(args, "rotnet_epochs", 0)),
            "rotnet_finetune": bool(getattr(args, "rotnet_finetune", False)),
            "stats_csv": stats_path,
        },
        transformations=[
            {
                "type": "discrete_rotation_90",
                "angles": [0, 90, 180, 270],
                "method": ("orb+grad_reference" if args.method == "reference" else "rotnet"),
            }
        ],
        random_seed=None,
    )
    if replay_cmd:
        print_replay_command("after execution", replay_cmd)

