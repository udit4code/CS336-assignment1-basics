"""Shared device and dtype policy for training and inference."""

import torch


SUPPORTED_DTYPES = ("float32", "float64", "float16", "bfloat16")


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` once so every model tensor uses the same backend."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(requested)
    except RuntimeError as error:
        raise ValueError(f"invalid device: {requested}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    """Map a CLI dtype name to its PyTorch dtype."""
    try:
        return {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


__all__ = ["SUPPORTED_DTYPES", "resolve_device", "resolve_dtype"]
