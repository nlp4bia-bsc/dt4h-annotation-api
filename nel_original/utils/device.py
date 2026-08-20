from __future__ import annotations

import importlib
import importlib.util
from typing import Any


def optional_module(name: str) -> Any | None:
    """Return an optional module if installed without making it a hard dependency."""
    try:
        if importlib.util.find_spec(name) is None:
            return None
    except ModuleNotFoundError:
        return None
    return importlib.import_module(name)


def resolve_torch_device(device: str = "auto") -> str:
    """Resolve a GPU-first torch device string, falling back to CPU when unavailable."""
    if device != "auto":
        return device
    torch = optional_module("torch")
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_faiss_device(device: str = "auto") -> str:
    """Resolve a GPU-first FAISS device string, falling back to CPU when unavailable."""
    if device != "auto":
        return device
    faiss = optional_module("faiss")
    if faiss is not None and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        return "cuda"
    return "cpu"
