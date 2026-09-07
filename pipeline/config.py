"""Typed configuration shared by the CLI, trainer, and artifact loader.

Keeping configuration in one place prevents a common production failure mode:
the model is reconstructed with slightly different dimensions than the ones
used during training.  Dataclasses also make the configuration easy to store
inside a checkpoint for reproducibility.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    train_path: Path
    valid_path: Path | None
    encoding: str
    context_length: int
    validation_fraction: float

    def validate(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not self.train_path.is_file():
            raise FileNotFoundError(f"training text file does not exist: {self.train_path}")
        if self.valid_path is not None and not self.valid_path.is_file():
            raise FileNotFoundError(f"validation text file does not exist: {self.valid_path}")


@dataclass(frozen=True)
class ModelConfig:
    context_length: int
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float

    def validate(self) -> None:
        if self.vocab_size <= 0 or self.d_model <= 0 or self.num_layers <= 0 or self.num_heads <= 0 or self.d_ff <= 0:
            raise ValueError("model dimensions must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    max_steps: int
    weight_decay: float
    max_grad_norm: float

    def validate(self) -> None:
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be non-negative, with learning_rate positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be at least one")
        if self.max_steps <= self.warmup_steps:
            raise ValueError("max_steps must be greater than warmup_steps")
        if self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("weight_decay must be non-negative and max_grad_norm positive")


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int
    seed: int
    device: str
    dtype: str
    eval_interval: int
    checkpoint_interval: int
    eval_batches: int
    log_interval: int

    def validate(self) -> None:
        if self.batch_size <= 0 or self.eval_batches <= 0:
            raise ValueError("batch_size and eval_batches must be positive")
        if self.eval_interval <= 0 or self.checkpoint_interval <= 0 or self.log_interval <= 0:
            raise ValueError("eval_interval, checkpoint_interval, and log_interval must be positive")
        if self.dtype not in {"float32", "float64", "float16", "bfloat16"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")


@dataclass(frozen=True)
class InferenceConfig:
    """Validated user controls for one text-generation request."""

    artifact_path: Path
    prompt: str
    max_new_tokens: int
    min_words: int
    temperature: float
    top_p: float
    seed: int | None
    device: str
    dtype: str

    def validate(self) -> None:
        artifact_file = self.artifact_path / "artifact.pt" if self.artifact_path.is_dir() else self.artifact_path
        if not artifact_file.is_file():
            raise FileNotFoundError(f"model artifact does not exist: {artifact_file}")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.min_words < 0:
            raise ValueError("min_words must be non-negative")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be finite and in the interval (0, 1]")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.dtype not in {"float32", "float64", "float16", "bfloat16"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")


def dataclass_dict(value: Any) -> dict[str, Any]:
    """Convert a configuration dataclass to JSON-friendly primitive values."""
    result = asdict(value)
    for key, item in result.items():
        if isinstance(item, Path):
            result[key] = str(item)
    return result
