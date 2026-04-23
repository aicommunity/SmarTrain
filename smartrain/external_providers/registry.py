from __future__ import annotations

from smartrain.external_providers.base import ExternalProviderSpec


_SPECS: tuple[ExternalProviderSpec, ...] = (
    ExternalProviderSpec(
        id="dr-yolo",
        display_name="DR-YOLO",
        repo_url="https://github.com/DRdairuiDR/DR-YOLO.git",
        branch="master",
        train_entry="train.py",
        infer_entry="detect.py",
    ),
    ExternalProviderSpec(
        id="leaf-yolo",
        display_name="LEAF-YOLO",
        repo_url="https://github.com/highquanglity/LEAF-YOLO.git",
        branch="main",
        train_entry="train.py",
        infer_entry="detect.py",
    ),
    ExternalProviderSpec(
        id="mfel-yolo",
        display_name="MFEL-YOLO",
        repo_url="https://github.com/kyxh1095/MFEL-YOLO-main.git",
        branch="master",
        train_entry="train.py",
        infer_entry="val.py",
    ),
    ExternalProviderSpec(
        id="mp-yolo",
        display_name="MP-YOLO",
        repo_url="https://github.com/Wang-jj-zs/MP-YOLO.git",
        branch="main",
        train_entry="train.py",
        infer_entry="detect.py",
    ),
    ExternalProviderSpec(
        id="ssdm-yolo",
        display_name="SSDM-YOLO",
        repo_url="https://github.com/liuliuliu2002/SSDM-YOLO.git",
        branch="main",
        train_entry="train.py",
        infer_entry="detect.py",
        ready=True,
        note="Repository ships archive in root; installer auto-unpacks and resolves runnable root.",
    ),
    ExternalProviderSpec(
        id="enhanced-yolov8",
        display_name="Enhanced YOLOv8",
        repo_url="https://github.com/GuccIceCream/yolov8.git",
        branch="master",
        train_entry="train.py",
        infer_entry="detect.py",
        ready=True,
        note="Use master branch; runnable scripts are in nested folder and resolved automatically.",
    ),
)


def list_provider_specs() -> tuple[ExternalProviderSpec, ...]:
    return _SPECS


def get_provider_spec(provider_id: str) -> ExternalProviderSpec:
    key = str(provider_id).strip().lower()
    for spec in _SPECS:
        if spec.id == key:
            return spec
    known = ", ".join(s.id for s in _SPECS)
    raise ValueError(f"Unknown provider: {provider_id!r}. Known providers: {known}")

