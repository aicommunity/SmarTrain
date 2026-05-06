from __future__ import annotations

from dataclasses import dataclass
import re

from smartrain.cli_support.cli_prompts import print_numbered_options, prompt_text


@dataclass(frozen=True)
class DeviceOption:
    label: str
    value: str
    is_cuda: bool


def discover_device_options() -> list[DeviceOption]:
    options: list[DeviceOption] = []
    try:
        import torch

        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            for idx in range(count):
                try:
                    gpu_name = str(torch.cuda.get_device_name(idx))
                except Exception:
                    gpu_name = f"GPU {idx}"
                options.append(DeviceOption(label=f"GPU {idx}: {gpu_name}", value=str(idx), is_cuda=True))
    except Exception:
        pass
    options.append(DeviceOption(label="CPU", value="cpu", is_cuda=False))
    return options


def default_device_value() -> str:
    for option in discover_device_options():
        if option.is_cuda and option.value == "0":
            return "0"
    return "cpu"


def is_cuda_device(device: str | None) -> bool:
    if device is None:
        return False
    token = str(device).strip().lower()
    if not token:
        return False
    if token == "cpu":
        return False
    if token.startswith("cuda"):
        return True
    return any(ch.isdigit() for ch in token)


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip().lower()


def _normalize_name(value: str | None) -> str:
    token = _normalize_token(value)
    return re.sub(r"[^a-z0-9]+", "", token)


def _parse_cuda_index_token(token: str) -> int | None:
    if not token:
        return None
    if token.isdigit():
        return int(token)
    m = re.fullmatch(r"(?:cuda|gpu)\s*:?\s*(\d+)", token)
    if m:
        return int(m.group(1))
    return None


def resolve_device_request(request: str | None, options: list[DeviceOption] | None = None) -> str:
    device_options = list(options or discover_device_options())
    if not device_options:
        return "cpu"
    by_value = {o.value: o for o in device_options}
    request_token = _normalize_token(request)
    if not request_token:
        return default_device_value()
    if request_token == "cpu":
        return "cpu"
    parsed_idx = _parse_cuda_index_token(request_token)
    if parsed_idx is not None:
        parsed_value = str(parsed_idx)
        if parsed_value in by_value and by_value[parsed_value].is_cuda:
            return parsed_value
        return default_device_value()
    exact_name = [
        o for o in device_options if o.is_cuda and _normalize_name(o.label).find(_normalize_name(request_token)) != -1
    ]
    if len(exact_name) == 1:
        return exact_name[0].value
    return default_device_value()


def resolve_device_candidates_by_name(name: str, options: list[DeviceOption] | None = None) -> list[DeviceOption]:
    device_options = list(options or discover_device_options())
    needle = _normalize_name(name)
    if not needle:
        return []
    return [o for o in device_options if o.is_cuda and needle in _normalize_name(o.label)]


def validate_device_available(device: str | None) -> None:
    raw = _normalize_token(device)
    if not raw:
        token = resolve_device_request(None)
    elif raw == "cpu":
        token = "cpu"
    else:
        idx = _parse_cuda_index_token(raw)
        if idx is not None:
            token = str(idx)
        else:
            token = resolve_device_request(device)
    if not is_cuda_device(token):
        return
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"CUDA device requested ({token}), but torch is unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({token}), but torch.cuda.is_available()=False. "
            f"torch={getattr(torch, '__version__', 'unknown')} cuda_runtime={getattr(torch.version, 'cuda', 'unknown')}"
        )
    idx = _parse_cuda_index_token(_normalize_token(token))
    if idx is None:
        return
    count = int(torch.cuda.device_count())
    if idx >= count:
        raise RuntimeError(f"CUDA device requested ({token}), but available GPU indices are 0..{max(0, count - 1)}")


def device_display_name(device: str | None, options: list[DeviceOption] | None = None) -> str:
    resolved = resolve_device_request(device, options=options)
    device_options = list(options or discover_device_options())
    for item in device_options:
        if item.value == resolved:
            return item.label
    return "CPU" if resolved == "cpu" else f"GPU {resolved}"


def prompt_device_selection(*, title: str = "devices", default_device: str | None = None) -> str:
    options = discover_device_options()
    labels = [o.label for o in options]
    if not options:
        return "cpu"
    default_value = resolve_device_request(default_device, options=options)
    default_label = next((o.label for o in options if o.value == default_value), labels[0])
    print_numbered_options(title, labels)
    raw = prompt_text("Select device (number/name/value)", default=default_label).strip()
    if not raw:
        return default_value
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1].value
    resolved_direct = resolve_device_request(raw, options=options)
    if _normalize_token(raw) in {"cpu", resolved_direct, f"cuda:{resolved_direct}", f"gpu:{resolved_direct}"}:
        return resolved_direct
    name_matches = resolve_device_candidates_by_name(raw, options=options)
    if len(name_matches) == 1:
        return name_matches[0].value
    if len(name_matches) > 1:
        numbered = [m.label for m in name_matches]
        print_numbered_options("Matching devices", numbered)
        chosen = prompt_text("Multiple matches, select number", default="1").strip()
        if chosen.isdigit():
            n = int(chosen)
            if 1 <= n <= len(name_matches):
                return name_matches[n - 1].value
    return default_value
