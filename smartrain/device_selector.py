from __future__ import annotations

from dataclasses import dataclass


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
