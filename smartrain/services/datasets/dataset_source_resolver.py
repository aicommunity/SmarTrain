from __future__ import annotations

import io
import json
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from smartrain.core.runtime.workspace_paths import (
    WorkspaceLayout,
    extract_dataset_archive_to_cache,
    is_dataset_archive_path,
)
from smartrain.services.datasets.cvsdcldet_converter import _json_has_cvsdcldet_detections
from smartrain.services.datasets.cvat11_converter import YOLO_IMAGE_EXTS
from smartrain.services.datasets.dataset_cli_common import load_dataset_catalog
from smartrain.services.datasets.datasets_json_scan_core_service import detect_structure, detect_structures
from smartrain.services.datasets.datasets_json_scan_index_service import _load_datasets_list_file

MANUAL_SOURCE_OPTION = "<enter path to directory or archive>"
DATASETS_LIST_FILE = "datasets_list.txt"
CVAT11_ZIP_STRUCTURE = "cvat11_zip"

SourceGroup = Literal["datasets", "raw_data", "external", "manual"]


@dataclass(frozen=True)
class SourceCandidate:
    option_id: str
    label: str
    path: Path
    group: SourceGroup
    structure_hint: str | None = None
    dataset_key: str | None = None


@dataclass
class ResolvedDatasetSource:
    working_path: Path
    structure: str
    display_name: str
    source_archive: Path | None = None
    is_cvat11_zip: bool = False
    dataset_key: str | None = None
    structures: list[str] | None = None

    @property
    def all_structures(self) -> list[str]:
        if self.structures:
            return list(self.structures)
        return [self.structure]


def _archive_member_stem(name: str) -> str:
    base = os.path.basename(name)
    stem, _ext = os.path.splitext(base)
    if stem.lower().endswith(".tar"):
        stem, _ = os.path.splitext(stem)
    return stem


def _archive_has_image_for_stem(members: set[str], stem: str) -> bool:
    for ext in YOLO_IMAGE_EXTS:
        for member in members:
            if _archive_member_stem(member) == stem and member.lower().endswith(ext.lower()):
                return True
    return False


def _peek_zip_structure(zip_path: Path) -> str | None:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            members = set(names)
            if any(n.endswith("annotations.xml") for n in names):
                return CVAT11_ZIP_STRUCTURE
            json_names = [n for n in names if n.lower().endswith(".json") and not n.endswith("/")]
            for json_name in json_names[:8]:
                try:
                    data = json.loads(zf.read(json_name))
                except Exception:
                    continue
                if not _json_has_cvsdcldet_detections(data):
                    continue
                if _archive_has_image_for_stem(members, _archive_member_stem(json_name)):
                    return "cvsdcldet"
    except (zipfile.BadZipFile, OSError):
        return None
    return None


def _peek_tar_structure(archive_path: Path) -> str | None:
    try:
        from smartrain.core.runtime.workspace_paths import archive_kind

        kind = archive_kind(archive_path)
        mode = "r:gz" if kind in ("tar.gz", "tgz") else "r:"
        with tarfile.open(archive_path, mode) as tf:
            members = [m.name for m in tf.getmembers() if m.isfile()]
            member_set = set(members)
            if any(n.endswith("annotations.xml") for n in members):
                return CVAT11_ZIP_STRUCTURE
            json_names = [n for n in members if n.lower().endswith(".json")]
            for json_name in json_names[:8]:
                member = tf.getmember(json_name)
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                try:
                    data = json.loads(extracted.read())
                except Exception:
                    continue
                if not _json_has_cvsdcldet_detections(data):
                    continue
                if _archive_has_image_for_stem(member_set, _archive_member_stem(json_name)):
                    return "cvsdcldet"
    except (tarfile.TarError, OSError):
        return None
    return None


def peek_archive_structures(archive_path: Path) -> list[str]:
    """Best-effort structure IDs inside an archive without full extraction."""
    peeked = peek_archive_structure(archive_path)
    return [peeked] if peeked else []


def peek_archive_structure(archive_path: Path) -> str | None:
    """Best-effort structure detection inside an archive without full extraction."""
    if not archive_path.is_file() or not is_dataset_archive_path(archive_path):
        return None
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        return _peek_zip_structure(archive_path)
    return _peek_tar_structure(archive_path)


def detect_path_structure(path: Path, *, workspace_root: str | None = None) -> str:
    """Detect dataset structure for a directory or archive path."""
    if path.is_dir():
        return detect_structure(str(path))
    if path.is_file() and is_dataset_archive_path(path):
        peeked = peek_archive_structure(path)
        if peeked == CVAT11_ZIP_STRUCTURE:
            return CVAT11_ZIP_STRUCTURE
        if peeked:
            return peeked
        if workspace_root is None:
            raise ValueError(
                f"Could not detect dataset structure in archive: {path}. "
                "Specify --workspace to allow extraction to cache."
            )
        print(f"[INFO] Extracting archive to cache: {path}")
        extracted = Path(extract_dataset_archive_to_cache(workspace_root, str(path)))
        return detect_structure(str(extracted))
    return detect_structure(str(path))


def _display_name_for_path(path: Path) -> str:
    if path.is_dir():
        return path.name
    if is_dataset_archive_path(path):
        name = path.name
        for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
            if name.lower().endswith(suffix):
                return name[: -len(suffix)]
        return path.stem
    return path.name


def _structures_label(structures: list[str]) -> str:
    if not structures:
        return "unknown"
    from smartrain.services.datasets.dataset_convert_service import structures_display_name

    return structures_display_name(structures)


def _structure_label(structure: str | None) -> str:
    if not structure or structure == "unknown":
        return "unknown"
    return _structures_label([structure])


def _merge_structures(primary: list[str], extra: list[str]) -> list[str]:
    out = list(primary)
    for item in extra:
        if item and item != "unknown" and item not in out:
            out.append(item)
    return out


def _structures_for_dir(path: Path) -> list[str]:
    return detect_structures(str(path))


def structures_for_workspace_dataset(workspace_root: str, name: str, entry: dict) -> list[str]:
    return _structures_for_workspace_dataset(workspace_root, name, entry)


def _structures_for_workspace_dataset(workspace_root: str, name: str, entry: dict) -> list[str]:
    from smartrain.services.datasets.dataset_access import resolve_dataset_root_for_entry

    catalog_structure = str(entry.get("structure") or "unknown")
    layout = WorkspaceLayout(workspace_root)
    try:
        root = resolve_dataset_root_for_entry(
            name,
            entry,
            workspace_root=workspace_root,
            source_catalog_dir=layout.datasets,
            legacy_source_parent=layout.datasets,
        )
        detected = _structures_for_dir(Path(root))
        if detected:
            return _merge_structures(detected, [catalog_structure] if catalog_structure != "unknown" else [])
    except Exception:
        pass
    if catalog_structure != "unknown":
        return [catalog_structure]
    return []


def _raw_data_display_name(path: Path) -> str:
    if path.is_dir():
        return f"{path.name}/"
    return path.name


def list_workspace_dataset_candidates(workspace_root: str) -> list[SourceCandidate]:
    layout = WorkspaceLayout(workspace_root)
    catalog = load_dataset_catalog(layout)
    out: list[SourceCandidate] = []
    for name in sorted(catalog.keys()):
        entry = catalog[name]
        structures = _structures_for_workspace_dataset(workspace_root, name, entry)
        if not structures:
            continue
        label = f"[datasets] {name} ({_structures_label(structures)})"
        out.append(
            SourceCandidate(
                option_id=f"datasets:{name}",
                label=label,
                path=Path("."),
                group="datasets",
                structure_hint=structures[0],
                dataset_key=name,
            )
        )
    return out


def list_raw_data_candidates(workspace_root: str) -> list[SourceCandidate]:
    layout = WorkspaceLayout(workspace_root)
    root = Path(layout.raw_data)
    if not root.is_dir():
        return []
    out: list[SourceCandidate] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if entry.name in (DATASETS_LIST_FILE, "datasets_info.json", "datasets_list.txt"):
            continue
        if entry.is_dir():
            structures = _structures_for_dir(entry)
            if not structures:
                continue
            label = f"[raw_data] {_raw_data_display_name(entry)} ({_structures_label(structures)})"
            out.append(
                SourceCandidate(
                    option_id=f"raw_data:{entry.name}",
                    label=label,
                    path=entry.resolve(),
                    group="raw_data",
                    structure_hint=structures[0],
                )
            )
            continue
        if entry.is_file() and is_dataset_archive_path(entry):
            structures = peek_archive_structures(entry)
            if not structures:
                continue
            label = f"[raw_data] {_raw_data_display_name(entry)} ({_structures_label(structures)})"
            out.append(
                SourceCandidate(
                    option_id=f"raw_data:{entry.name}",
                    label=label,
                    path=entry.resolve(),
                    group="raw_data",
                    structure_hint=structures[0],
                )
            )
    return out


def list_external_candidates(workspace_root: str) -> list[SourceCandidate]:
    layout = WorkspaceLayout(workspace_root)
    list_path = Path(layout.raw_data) / DATASETS_LIST_FILE
    if not list_path.is_file():
        return []
    out: list[SourceCandidate] = []
    for src_path in _load_datasets_list_file(str(list_path)):
        path = Path(src_path)
        if not path.exists():
            continue
        if path.is_dir():
            structures = _structures_for_dir(path)
        elif is_dataset_archive_path(path):
            structures = peek_archive_structures(path)
        else:
            continue
        if not structures:
            continue
        label = f"[external] {path} ({_structures_label(structures)})"
        out.append(
            SourceCandidate(
                option_id=f"external:{path}",
                label=label,
                path=path.resolve(),
                group="external",
                structure_hint=structures[0],
            )
        )
    return out


def build_interactive_source_options(workspace_root: str) -> tuple[list[SourceCandidate], str]:
    candidates: list[SourceCandidate] = []
    candidates.extend(list_workspace_dataset_candidates(workspace_root))
    candidates.extend(list_raw_data_candidates(workspace_root))
    candidates.extend(list_external_candidates(workspace_root))
    manual = SourceCandidate(
        option_id="manual",
        label=MANUAL_SOURCE_OPTION,
        path=Path("."),
        group="manual",
    )
    candidates.append(manual)
    return candidates, MANUAL_SOURCE_OPTION


def resolve_manual_source_token(workspace_root: str, token: str) -> Path:
    """Resolve a user token to an existing directory or archive path."""
    raw = (token or "").strip()
    if not raw:
        raise ValueError("Source path is required.")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Source path not found: {candidate}")

    layout = WorkspaceLayout(workspace_root)
    relative_ws = (Path(workspace_root) / candidate).resolve()
    if relative_ws.exists():
        return relative_ws

    abs_candidate = candidate.resolve()
    if abs_candidate.exists():
        return abs_candidate

    name = raw[:-4] if raw.lower().endswith(".zip") else raw
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        if raw.lower().endswith(suffix):
            name = raw[: -len(suffix)]
            break
    dir_candidate = Path(layout.raw_data) / name
    if dir_candidate.is_dir():
        return dir_candidate.resolve()
    for ext in ("", ".zip", ".tar", ".tar.gz", ".tgz"):
        archive_candidate = Path(layout.raw_data) / f"{name}{ext}"
        if archive_candidate.is_file() and is_dataset_archive_path(archive_candidate):
            return archive_candidate.resolve()

    raise FileNotFoundError(f"Source path not found: {raw}")


def resolve_dataset_source(
    workspace_root: str | None,
    path: Path,
    *,
    dataset_key: str | None = None,
) -> ResolvedDatasetSource:
    """Resolve a directory or archive path into a working dataset source."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Source not found: {resolved}")

    if dataset_key:
        from smartrain.services.datasets.dataset_convert_service import resolve_source

        source = resolve_source(workspace_root=workspace_root, dataset_key=dataset_key)
        return ResolvedDatasetSource(
            working_path=source.path,
            structure=source.structure,
            display_name=source.name,
            dataset_key=source.dataset_key,
            structures=source.all_structures,
        )

    if resolved.is_dir():
        structures = detect_structures(str(resolved))
        if not structures:
            raise ValueError(f"Unsupported dataset structure: {resolved}")
        return ResolvedDatasetSource(
            working_path=resolved,
            structure=structures[0],
            display_name=_display_name_for_path(resolved),
            structures=structures,
        )

    if resolved.is_file() and is_dataset_archive_path(resolved):
        peeked_list = peek_archive_structures(resolved)
        if CVAT11_ZIP_STRUCTURE in peeked_list:
            return ResolvedDatasetSource(
                working_path=resolved,
                structure=CVAT11_ZIP_STRUCTURE,
                display_name=_display_name_for_path(resolved),
                is_cvat11_zip=True,
                structures=[CVAT11_ZIP_STRUCTURE],
            )
        if workspace_root is None:
            raise ValueError("Workspace is required when source is a dataset archive container.")
        print(f"[INFO] Extracting archive to cache: {resolved}")
        extracted = Path(extract_dataset_archive_to_cache(workspace_root, str(resolved)))
        structures = detect_structures(str(extracted))
        if peeked_list:
            structures = _merge_structures(structures, peeked_list)
        if not structures:
            raise ValueError(f"Unsupported dataset structure in archive: {resolved}")
        return ResolvedDatasetSource(
            working_path=extracted,
            structure=structures[0],
            display_name=_display_name_for_path(resolved),
            source_archive=resolved,
            structures=structures,
        )

    raise ValueError(
        f"Unsupported source path: {resolved} "
        f"(expected directory or archive: .zip, .tar, .tar.gz, .tgz)"
    )


def resolved_to_dataset_source(resolved: ResolvedDatasetSource):
    from smartrain.services.datasets.dataset_convert_service import DatasetSource

    if resolved.is_cvat11_zip:
        return DatasetSource(
            path=resolved.working_path,
            structure=CVAT11_ZIP_STRUCTURE,
            name=resolved.display_name,
            source_zip=resolved.working_path,
            dataset_key=resolved.dataset_key,
            structures=resolved.all_structures,
        )
    return DatasetSource(
        path=resolved.working_path,
        structure=resolved.structure,
        name=resolved.display_name,
        dataset_key=resolved.dataset_key,
        source_archive=resolved.source_archive,
        structures=resolved.all_structures,
    )


def replay_source_path(source) -> str:
    """Return the CLI path to replay for a resolved DatasetSource."""
    if source.dataset_key:
        return ""
    if source.source_zip is not None and source.structure == CVAT11_ZIP_STRUCTURE:
        return str(source.source_zip)
    if source.source_archive is not None:
        return str(source.source_archive)
    return str(source.path)
