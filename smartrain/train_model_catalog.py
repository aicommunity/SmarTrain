from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smartrain.train_backend_registry import default_train_provider, get_train_backend_spec


_EXTERNAL_PROVIDER_FALLBACK_ALIASES: dict[str, tuple[str, ...]] = {
    # Integration-safe aliases supported by current provider launchers/runtime.
    "dr-yolo": ("yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"),
    "leaf-yolo": ("yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"),
    "mp-yolo": ("yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"),
    "ssdm-yolo": ("yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"),
    "enhanced-yolov8": (
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov8l",
        "yolov8x",
    ),
    # MFEL models can be read from ultralytics/cfg/MFEL-YOLO/*.yaml when repo is available.
    "mfel-yolo": ("mfel-yolo", "e_pan+"),
}


def _normalize_mfel_alias_from_yaml_name(name: str) -> str:
    alias = name.strip().lower().replace(".yaml", "")
    if alias == "mfel-yolo":
        return "mfel-yolo"
    if alias in {"e_pan+", "e-pan+", "e_pan_plus", "e-pan-plus"}:
        return "e_pan+"
    return alias


def _discover_external_provider_aliases(
    provider: str,
    *,
    repo_path: str | None = None,
) -> tuple[str, ...]:
    pid = (provider or "").strip().lower()
    if pid == "mfel-yolo" and repo_path:
        root = Path(repo_path).expanduser().resolve()
        cfg_root = root / "ultralytics" / "cfg" / "MFEL-YOLO"
        if cfg_root.is_dir():
            alias_set = {
                _normalize_mfel_alias_from_yaml_name(p.name)
                for p in cfg_root.glob("*.yaml")
                if p.is_file()
            }
            aliases = sorted(alias_set)
            if "mfel-yolo" in aliases:
                aliases = ["mfel-yolo"] + [a for a in aliases if a != "mfel-yolo"]
            if aliases:
                return tuple(aliases)
    return _EXTERNAL_PROVIDER_FALLBACK_ALIASES.get(pid, ())


def normalize_external_provider_model_ref(value: str) -> str:
    """
    Normalize provider model token for alias lookup.
    """
    token = str(value or "").strip().lower()
    if not token:
        return token
    token = Path(token).name
    if token.endswith(".pt"):
        token = token[:-3]
    if token.endswith(".yaml"):
        token = token[:-5]
    if token in {"mfel_yolo"}:
        return "mfel-yolo"
    if token in {"e-pan+", "e_pan_plus", "e-pan-plus"}:
        return "e_pan+"
    return token


def is_supported_external_provider_model(
    provider: str,
    model_ref: str,
    *,
    provider_repo_path: str | None = None,
) -> bool:
    pid = (provider or "").strip().lower()
    if pid not in _EXTERNAL_PROVIDER_FALLBACK_ALIASES:
        return True
    normalized = normalize_external_provider_model_ref(model_ref)
    aliases = TrainModelCatalog(provider=pid, provider_repo_path=provider_repo_path).supported_aliases()
    return normalized in {normalize_external_provider_model_ref(a) for a in aliases}


@dataclass(frozen=True)
class TrainModelCatalog:
    """Catalog of model aliases supported by a training backend provider."""

    provider: str = default_train_provider()
    provider_repo_path: str | None = None

    def supported_aliases(self) -> tuple[str, ...]:
        # Keep aliases copy-paste friendly for `--model`.
        key = (self.provider or "").strip().lower()
        if key in _EXTERNAL_PROVIDER_FALLBACK_ALIASES:
            return _discover_external_provider_aliases(key, repo_path=self.provider_repo_path)
        return get_train_backend_spec(key).supported_aliases

