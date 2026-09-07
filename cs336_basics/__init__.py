import importlib.metadata

from .generation import GenerationResult, generate, sample_next_token

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass


__all__ = ["GenerationResult", "generate", "sample_next_token"]
