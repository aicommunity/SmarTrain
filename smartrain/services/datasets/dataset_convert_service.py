from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.services.datasets.cvat11_converter import (
    YOLO_IMAGE_EXTS,
    export_yolo_to_cvat11_zip,
    generate_temp_yolo_labels_from_cvat11_extracted,
    import_cvat11_zip_to_yolo,
)
from smartrain.services.datasets.cvsdcldet_converter import (
    _pack_cvat11_zip,
    convert_cvsdcldet_to_cvat11,
)
from smartrain.services.datasets.data_yaml_writer import write_flat_yolo_data_yaml
from smartrain.services.datasets.dataset_access import find_dataset_paths, resolve_dataset_root_for_entry
from smartrain.services.datasets.dataset_cli_common import load_dataset_catalog
from smartrain.services.datasets.dataset_scan import find_obj_names_file, find_yaml_file
from smartrain.services.datasets.datasets_json_cvat11_normalize_service import convert_cvat11_folder_to_yolo_flat
from smartrain.services.datasets.datasets_json_scan_core_service import (
    _find_cvat_annotations_xml,
    _load_cvat11_label_names,
    detect_structure,
    load_yaml,
    yolo_flat_image_label_buckets,
)

TARGET_YOLO = "yolo"
TARGET_CVAT11 = "cvat11"
TARGET_CVAT11_ZIP = "cvat11_zip"

ALL_TARGETS = (TARGET_YOLO, TARGET_CVAT11, TARGET_CVAT11_ZIP)

STRUCTURE_CVAT11_ZIP = "cvat11_zip"

YOLO_SOURCE_STRUCTURES = frozenset({"flat", "subset_flat", "split", "nested_split", "darknet"})

CONVERSION_TARGETS: dict[str, list[str]] = {
    STRUCTURE_CVAT11_ZIP: [TARGET_YOLO],
    "cvat11": [TARGET_YOLO, TARGET_CVAT11_ZIP],
    "cvsdcldet": [TARGET_CVAT11, TARGET_YOLO, TARGET_CVAT11_ZIP],
    "flat": [TARGET_CVAT11, TARGET_CVAT11_ZIP],
    "subset_flat": [TARGET_CVAT11, TARGET_CVAT11_ZIP],
    "split": [TARGET_CVAT11, TARGET_CVAT11_ZIP],
    "nested_split": [TARGET_CVAT11, TARGET_CVAT11_ZIP],
    "darknet": [TARGET_YOLO, TARGET_CVAT11, TARGET_CVAT11_ZIP],
}

STRUCTURE_DISPLAY_NAMES: dict[str, str] = {
    STRUCTURE_CVAT11_ZIP: "CVAT for images 1.1 (zip archive)",
    "cvat11": "CVAT for images 1.1 (folder)",
    "cvsdcldet": "CvsDclDet detection export",
    "flat": "YOLO flat paired directories layout",
    "subset_flat": "YOLO flat with subset subfolders",
    "split": "YOLO split directories layout",
    "nested_split": "YOLO nested split under images/labels",
    "darknet": "Darknet YOLO dataset layout",
    "unknown": "unknown",
}

TARGET_DISPLAY_NAMES: dict[str, str] = {
    TARGET_YOLO: "YOLO flat dataset (images/ + labels/ + data.yaml)",
    TARGET_CVAT11: "CVAT for images 1.1 (folder: annotations.xml + images/)",
    TARGET_CVAT11_ZIP: "CVAT for images 1.1 (zip archive)",
}


@dataclass(frozen=True)
class ConvertTarget:
    target_id: str
    label: str


@dataclass
class DatasetSource:
    path: Path
    structure: str
    name: str
    dataset_key: str | None = None
    source_zip: Path | None = None
    source_archive: Path | None = None
    structures: list[str] = field(default_factory=list)

    @property
    def all_structures(self) -> list[str]:
        if self.structures:
            return list(self.structures)
        return [self.structure]

    @property
    def display_structure(self) -> str:
        return structures_display_name(self.all_structures)


@dataclass
class ConvertOptions:
    task_name: str | None = None
    names: list[str] = field(default_factory=list)
    class_rename: dict[str, str] | None = None
    force: bool = False
    tmp_base_dir: Path | None = None
    create_zip: bool = False
    delete_after_zip: bool = True
    zip_path: Path | None = None


@dataclass
class ConvertResult:
    target: str
    output_dir: Path | None
    zip_path: Path | None
    info: dict[str, Any]
    is_folder_output: bool


def structure_display_name(structure_id: str) -> str:
    return STRUCTURE_DISPLAY_NAMES.get(structure_id, structure_id)


def structures_display_name(structure_ids: list[str]) -> str:
    if not structure_ids:
        return STRUCTURE_DISPLAY_NAMES.get("unknown", "unknown")
    return "; ".join(STRUCTURE_DISPLAY_NAMES.get(sid, sid) for sid in structure_ids)


def pick_structure_for_target(structures: list[str], target: str) -> str:
    """Pick the structure ID best suited for a conversion target."""
    if not structures:
        raise ValueError("structures is empty")
    if target == TARGET_YOLO:
        preference = [
            STRUCTURE_CVAT11_ZIP,
            "cvat11",
            "cvsdcldet",
            "flat",
            "subset_flat",
            "split",
            "nested_split",
            "darknet",
        ]
    else:
        preference = [
            "cvsdcldet",
            "flat",
            "subset_flat",
            "split",
            "nested_split",
            "darknet",
            "cvat11",
            STRUCTURE_CVAT11_ZIP,
        ]
    for sid in preference:
        if sid in structures:
            return sid
    return structures[0]


def target_display_name(target_id: str) -> str:
    return TARGET_DISPLAY_NAMES.get(target_id, target_id)


def list_available_targets(source_structure: str | list[str]) -> list[ConvertTarget]:
    if isinstance(source_structure, str):
        structure_ids = [source_structure]
    else:
        structure_ids = list(source_structure)
    seen: set[str] = set()
    ids: list[str] = []
    for sid in structure_ids:
        for tid in CONVERSION_TARGETS.get(sid, []):
            if tid not in seen:
                seen.add(tid)
                ids.append(tid)
    return [ConvertTarget(target_id=tid, label=target_display_name(tid)) for tid in ids]


def detect_source_structure(path: Path, *, workspace_root: str | None = None) -> str:
    from smartrain.core.runtime.workspace_paths import is_dataset_archive_path
    from smartrain.services.datasets.dataset_source_resolver import (
        CVAT11_ZIP_STRUCTURE,
        detect_path_structure,
        peek_archive_structure,
    )

    if path.is_file() and is_dataset_archive_path(path):
        peeked = peek_archive_structure(path)
        if peeked == CVAT11_ZIP_STRUCTURE:
            return STRUCTURE_CVAT11_ZIP
        if peeked:
            return peeked
        return detect_path_structure(path, workspace_root=workspace_root)
    if path.is_file() and path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path, "r") as zf:
                if any(n.endswith("annotations.xml") for n in zf.namelist()):
                    return STRUCTURE_CVAT11_ZIP
        except zipfile.BadZipFile:
            pass
    return detect_structure(str(path))


def resolve_source(
    *,
    workspace_root: str | None,
    dataset_key: str | None = None,
    source_dir: str | Path | None = None,
    source_zip: str | Path | None = None,
    source: str | Path | None = None,
) -> DatasetSource:
    from smartrain.services.datasets.dataset_source_resolver import (
        resolved_to_dataset_source,
        resolve_dataset_source,
        structures_for_workspace_dataset,
    )

    direct_path = source if source is not None else source_dir
    if sum(bool(x) for x in (dataset_key, direct_path, source_zip)) != 1:
        raise ValueError("Specify exactly one of --dataset, --source/--source-dir, or --source-zip.")

    if source_zip is not None:
        zip_path = Path(source_zip).expanduser().resolve()
        if not zip_path.is_file():
            raise FileNotFoundError(f"Source zip not found: {zip_path}")
        structure = detect_source_structure(zip_path, workspace_root=workspace_root)
        if structure != STRUCTURE_CVAT11_ZIP:
            resolved = resolve_dataset_source(workspace_root, zip_path)
            return resolved_to_dataset_source(resolved)
        return DatasetSource(
            path=zip_path,
            structure=structure,
            name=zip_path.stem,
            source_zip=zip_path,
            structures=[STRUCTURE_CVAT11_ZIP],
        )

    if dataset_key is not None:
        if workspace_root is None:
            raise ValueError("Workspace is required when using --dataset.")
        layout = WorkspaceLayout(workspace_root)
        catalog = load_dataset_catalog(layout)
        entry = catalog.get(dataset_key)
        if not isinstance(entry, dict):
            raise KeyError(f"Dataset not found in catalog: {dataset_key!r}")
        structure = str(entry.get("structure") or "unknown")
        root = resolve_dataset_root_for_entry(
            dataset_key,
            entry,
            workspace_root=workspace_root,
            source_catalog_dir=layout.datasets,
            legacy_source_parent=layout.datasets,
        )
        return DatasetSource(
            path=Path(root).resolve(),
            structure=structure if structure != "unknown" else detect_structure(root),
            name=dataset_key,
            dataset_key=dataset_key,
            structures=structures_for_workspace_dataset(workspace_root, dataset_key, entry),
        )

    if direct_path is None:
        raise ValueError("source path is required for direct conversion")
    root = Path(direct_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Source not found: {root}")
    resolved = resolve_dataset_source(workspace_root, root)
    return resolved_to_dataset_source(resolved)


def _load_class_names(dataset_dir: Path, structure: str) -> list[str]:
    yaml_path = find_yaml_file(str(dataset_dir))
    if yaml_path:
        data = load_yaml(yaml_path)
        if isinstance(data, dict):
            names = data.get("names")
            if isinstance(names, list):
                return [str(x) for x in names]
            if isinstance(names, dict):
                try:
                    return [str(v) for _k, v in sorted(names.items())]
                except Exception:
                    pass

    if structure == "cvat11":
        xml_path = _find_cvat_annotations_xml(str(dataset_dir))
        if xml_path:
            names = _load_cvat11_label_names(xml_path)
            if names:
                return list(names)

    obj_names = find_obj_names_file(str(dataset_dir))
    if obj_names and os.path.isfile(obj_names):
        lines = Path(obj_names).read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip()]

    return []


def _unique_dest_name(stem: str, ext: str, used: set[str]) -> str:
    candidate = f"{stem}{ext}"
    if candidate not in used:
        used.add(candidate)
        return candidate
    idx = 2
    while True:
        candidate = f"{stem}_{idx}{ext}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


def _copy_image_label_pair(img_path: Path, lbl_path: Path, images_out: Path, labels_out: Path, used: set[str]) -> None:
    ext = img_path.suffix.lower()
    if ext not in YOLO_IMAGE_EXTS:
        ext = ".jpg"
    dest_name = _unique_dest_name(img_path.stem, ext, used)
    shutil.copy2(img_path, images_out / dest_name)
    shutil.copy2(lbl_path, labels_out / Path(dest_name).with_suffix(".txt"))


def _iter_yolo_image_label_pairs(source_root: Path, structure: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if structure in YOLO_SOURCE_STRUCTURES:
        buckets = find_dataset_paths(str(source_root), structure)
        if not buckets and structure in ("flat", "subset_flat"):
            buckets = yolo_flat_image_label_buckets(str(source_root))
        for img_dir, lbl_dir in buckets:
            img_root = Path(img_dir)
            lbl_root = Path(lbl_dir)
            if structure == "darknet":
                for ext in YOLO_IMAGE_EXTS:
                    for img in sorted(img_root.glob(f"*{ext}")):
                        lbl = lbl_root / f"{img.stem}.txt"
                        if lbl.is_file():
                            pairs.append((img, lbl))
                continue
            for lbl in sorted(lbl_root.rglob("*.txt")):
                rel = lbl.relative_to(lbl_root)
                stem = lbl.stem
                img = None
                for ext in YOLO_IMAGE_EXTS:
                    cand = img_root / rel.with_suffix(ext)
                    if cand.is_file():
                        img = cand
                        break
                    cand_flat = img_root / f"{stem}{ext}"
                    if cand_flat.is_file():
                        img = cand_flat
                        break
                if img is not None:
                    pairs.append((img, lbl))
    return pairs


def stage_flat_yolo_snapshot(
    source_root: Path,
    structure: str,
    output_dir: Path,
    *,
    force: bool = False,
) -> list[str]:
    """Materialize a flat YOLO dataset under output_dir. Returns class names."""
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --force to overwrite.")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    names = _load_class_names(source_root, structure)
    used: set[str] = set()
    pairs = _iter_yolo_image_label_pairs(source_root, structure)
    if not pairs and structure == "flat":
        pairs = _iter_yolo_image_label_pairs(source_root, "subset_flat")
    if not pairs:
        raise ValueError(f"No image/label pairs found in {source_root} (structure={structure})")

    for img_path, lbl_path in pairs:
        _copy_image_label_pair(img_path, lbl_path, images_out, labels_out, used)

    if not names:
        raise ValueError(f"Could not determine class names for {source_root}")

    write_flat_yolo_data_yaml(str(output_dir), list(names))
    return names


def _needs_yolo_staging(structure: str) -> bool:
    return structure in {"split", "nested_split", "subset_flat", "darknet"}


def _resolve_yolo_workdir(
    source: DatasetSource,
    *,
    tmp_base_dir: Path | None,
    force: bool,
) -> tuple[Path, Path | None]:
    """Return (yolo_dataset_dir, temp_dir_to_cleanup_or_none)."""
    if source.structure == "flat":
        return source.path, None
    if source.structure in YOLO_SOURCE_STRUCTURES and not _needs_yolo_staging(source.structure):
        return source.path, None

    tmp_parent = tmp_base_dir or Path.cwd()
    tmp_parent.mkdir(parents=True, exist_ok=True)
    td = tempfile.mkdtemp(prefix="dataset_convert_yolo_", dir=str(tmp_parent))
    flat_dir = Path(td) / "flat"
    stage_flat_yolo_snapshot(source.path, source.structure, flat_dir, force=True)
    return flat_dir, Path(td)


def _extract_cvat11_zip_to_folder(zip_path: Path, output_dir: Path, *, force: bool) -> str:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --force to overwrite.")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cvat11_unzip_") as td:
        td_path = Path(td)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(td_path)
        task_dirs = [p for p in td_path.iterdir() if p.is_dir() and (p / "annotations.xml").is_file()]
        if not task_dirs:
            raise ValueError(f"No CVAT task folder found in zip: {zip_path}")
        task_dir = task_dirs[0]
        task_name = task_dir.name
        for item in task_dir.iterdir():
            dest = output_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    return task_name


def pack_directory_zip(source_dir: Path, zip_path: Path, *, force: bool) -> None:
    if zip_path.exists():
        if not force:
            raise FileExistsError(f"Zip already exists: {zip_path}. Use --force to overwrite.")
        zip_path.unlink()
    ann = source_dir / "annotations.xml"
    if ann.is_file():
        task_name = source_dir.name
        _pack_cvat11_zip(output_dir=source_dir, task_name=task_name, zip_path=zip_path, force=True)
        return
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(source_dir)))


def apply_zip_postprocess(
    output_dir: Path,
    *,
    opts: ConvertOptions,
    task_name: str | None = None,
) -> Path | None:
    if not opts.create_zip:
        return None
    if (output_dir / "annotations.xml").is_file():
        zip_path = opts.zip_path or Path(str(output_dir) + ".cvat11.zip")
    else:
        zip_path = opts.zip_path or Path(str(output_dir) + ".zip")
    if (output_dir / "annotations.xml").is_file():
        effective_task = task_name or output_dir.name
        _pack_cvat11_zip(
            output_dir=output_dir,
            task_name=effective_task,
            zip_path=zip_path,
            force=opts.force or True,
        )
    else:
        pack_directory_zip(output_dir, zip_path, force=opts.force or True)
    if opts.delete_after_zip and output_dir.is_dir():
        try:
            shutil.rmtree(output_dir)
        except OSError as exc:
            raise RuntimeError(f"Failed to delete temporary output directory: {output_dir}") from exc
    return zip_path


def run_conversion(
    source: DatasetSource,
    target: str,
    output_path: Path,
    *,
    opts: ConvertOptions | None = None,
) -> ConvertResult:
    opts = opts or ConvertOptions()
    target = str(target).strip()
    if target not in ALL_TARGETS:
        raise ValueError(f"Unsupported target format: {target!r}")

    effective_structure = pick_structure_for_target(source.all_structures, target)
    available = {t.target_id for t in list_available_targets(source.all_structures)}
    if target not in available:
        raise ValueError(
            f"Cannot convert {source.display_structure} to {target_display_name(target)}."
        )

    source = DatasetSource(
        path=source.path,
        structure=effective_structure,
        name=source.name,
        dataset_key=source.dataset_key,
        source_zip=source.source_zip,
        source_archive=source.source_archive,
        structures=source.all_structures,
    )

    tmp_base = opts.tmp_base_dir
    info: dict[str, Any] = {}
    output_path = output_path.expanduser().resolve()

    if target == TARGET_YOLO:
        if source.structure == STRUCTURE_CVAT11_ZIP:
            if source.source_zip is None:
                raise ValueError("source_zip is required for cvat11_zip conversion")
            info = import_cvat11_zip_to_yolo(
                cvat_zip_path=source.source_zip,
                output_dir=output_path,
                task_name=opts.task_name,
                force=opts.force,
                tmp_base_dir=tmp_base,
            )
            result_dir = Path(str(info["output_dir"]))
            zip_path = apply_zip_postprocess(result_dir, opts=opts)
            return ConvertResult(
                target=target,
                output_dir=None if zip_path and opts.delete_after_zip else result_dir,
                zip_path=zip_path,
                info=info,
                is_folder_output=True,
            )

        if source.structure == "cvat11":
            info = convert_cvat11_folder_to_yolo_flat(
                source.path,
                output_path,
                detect_structure_cb=detect_structure,
                find_cvat_annotations_xml_cb=_find_cvat_annotations_xml,
                load_cvat11_label_names_cb=_load_cvat11_label_names,
                generate_temp_yolo_labels_cb=generate_temp_yolo_labels_from_cvat11_extracted,
                force=opts.force,
            )
            result_dir = Path(str(info["output_dir"]))
            zip_path = apply_zip_postprocess(result_dir, opts=opts)
            return ConvertResult(
                target=target,
                output_dir=None if zip_path and opts.delete_after_zip else result_dir,
                zip_path=zip_path,
                info=info,
                is_folder_output=True,
            )

        if source.structure == "cvsdcldet":
            with tempfile.TemporaryDirectory(prefix="cvsdcldet_cvat_", dir=str(tmp_base or Path.cwd())) as td:
                cvat_tmp = Path(td) / "cvat11"
                convert_cvsdcldet_to_cvat11(
                    source_dir=source.path,
                    output_dir=cvat_tmp,
                    task_name=opts.task_name,
                    class_rename=opts.class_rename,
                    force=True,
                    create_zip=False,
                )
                info = convert_cvat11_folder_to_yolo_flat(
                    cvat_tmp,
                    output_path,
                    detect_structure_cb=detect_structure,
                    find_cvat_annotations_xml_cb=_find_cvat_annotations_xml,
                    load_cvat11_label_names_cb=_load_cvat11_label_names,
                    generate_temp_yolo_labels_cb=generate_temp_yolo_labels_from_cvat11_extracted,
                    force=opts.force,
                )
            result_dir = Path(str(info["output_dir"]))
            zip_path = apply_zip_postprocess(result_dir, opts=opts)
            return ConvertResult(
                target=target,
                output_dir=None if zip_path and opts.delete_after_zip else result_dir,
                zip_path=zip_path,
                info=info,
                is_folder_output=True,
            )

        if source.structure in YOLO_SOURCE_STRUCTURES | {"flat"}:
            names = stage_flat_yolo_snapshot(source.path, source.structure, output_path, force=opts.force)
            info = {
                "output_dir": str(output_path),
                "nc": len(names),
                "names": names,
            }
            zip_path = apply_zip_postprocess(output_path, opts=opts)
            return ConvertResult(
                target=target,
                output_dir=None if zip_path and opts.delete_after_zip else output_path,
                zip_path=zip_path,
                info=info,
                is_folder_output=True,
            )

        raise ValueError(f"Unsupported source structure for YOLO target: {source.structure}")

    if target == TARGET_CVAT11:
        if source.structure == "cvsdcldet":
            info = convert_cvsdcldet_to_cvat11(
                source_dir=source.path,
                output_dir=output_path,
                task_name=opts.task_name,
                class_rename=opts.class_rename,
                force=opts.force,
                create_zip=False,
            )
            result_dir = Path(str(info["output_dir"]))
            zip_path = apply_zip_postprocess(result_dir, opts=opts, task_name=str(info.get("task_name")))
            return ConvertResult(
                target=target,
                output_dir=None if zip_path and opts.delete_after_zip else result_dir,
                zip_path=zip_path,
                info=info,
                is_folder_output=True,
            )

        yolo_dir, tmp_cleanup = _resolve_yolo_workdir(source, tmp_base_dir=tmp_base, force=opts.force)
        try:
            names = opts.names or _load_class_names(yolo_dir, "flat")
            if not names:
                raise ValueError("Could not determine class names: specify --names or provide data.yaml.")
            with tempfile.TemporaryDirectory(prefix="yolo_cvat11_", dir=str(tmp_base or Path.cwd())) as td:
                zip_tmp = Path(td) / "export.cvat11.zip"
                export_info = export_yolo_to_cvat11_zip(
                    dataset_dir=yolo_dir,
                    task_name=opts.task_name,
                    output_zip_path=zip_tmp,
                    names=names,
                    force=True,
                    tmp_base_dir=tmp_base,
                )
                task_name = str(export_info.get("task_name") or output_path.name)
                _extract_cvat11_zip_to_folder(zip_tmp, output_path, force=opts.force)
            info = {**export_info, "output_dir": str(output_path), "task_name": task_name}
            zip_path = apply_zip_postprocess(output_path, opts=opts, task_name=task_name)
            return ConvertResult(
                target=target,
                output_dir=None if zip_path and opts.delete_after_zip else output_path,
                zip_path=zip_path,
                info=info,
                is_folder_output=True,
            )
        finally:
            if tmp_cleanup is not None:
                try:
                    shutil.rmtree(tmp_cleanup)
                except OSError:
                    pass

    if target == TARGET_CVAT11_ZIP:
        zip_path = output_path
        if zip_path.suffix.lower() != ".zip":
            zip_path = Path(str(output_path) + ".cvat11.zip")

        if source.structure == "cvsdcldet":
            out_dir = zip_path.parent / zip_path.stem.replace(".cvat11", "")
            info = convert_cvsdcldet_to_cvat11(
                source_dir=source.path,
                output_dir=out_dir,
                task_name=opts.task_name,
                class_rename=opts.class_rename,
                force=opts.force,
                create_zip=True,
                zip_path=zip_path,
            )
            if out_dir.is_dir() and opts.delete_after_zip:
                try:
                    shutil.rmtree(out_dir)
                except OSError as exc:
                    raise RuntimeError(f"Failed to delete temporary output directory: {out_dir}") from exc
            return ConvertResult(
                target=target,
                output_dir=None,
                zip_path=Path(str(info["zip_path"])),
                info=info,
                is_folder_output=False,
            )

        if source.structure == "cvat11":
            task_name = opts.task_name or source.name
            _pack_cvat11_zip(
                output_dir=source.path,
                task_name=task_name,
                zip_path=zip_path,
                force=opts.force,
            )
            info = {
                "zip_path": str(zip_path),
                "task_name": task_name,
                "images_count": len(list((source.path / "images").glob("*"))) if (source.path / "images").is_dir() else 0,
            }
            return ConvertResult(
                target=target,
                output_dir=None,
                zip_path=zip_path,
                info=info,
                is_folder_output=False,
            )

        yolo_dir, tmp_cleanup = _resolve_yolo_workdir(source, tmp_base_dir=tmp_base, force=opts.force)
        try:
            names = opts.names or _load_class_names(yolo_dir, "flat")
            if not names:
                raise ValueError("Could not determine class names: specify --names or provide data.yaml.")
            info = export_yolo_to_cvat11_zip(
                dataset_dir=yolo_dir,
                task_name=opts.task_name,
                output_zip_path=zip_path,
                names=names,
                force=opts.force,
                tmp_base_dir=tmp_base,
            )
            return ConvertResult(
                target=target,
                output_dir=None,
                zip_path=Path(str(info["zip_path"])),
                info=info,
                is_folder_output=False,
            )
        finally:
            if tmp_cleanup is not None:
                try:
                    shutil.rmtree(tmp_cleanup)
                except OSError:
                    pass

    raise ValueError(f"Unhandled conversion: {source.structure} -> {target}")
