"""Versioned, atomic checkpoint and final-artifact persistence."""

import os
from pathlib import Path
import tempfile
import random
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


SCHEMA_VERSION = 1


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG used by the pipeline so resume is reproducible."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    config: dict[str, Any],
    data_metadata: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> None:
    """Save all state needed to continue training, not just model weights."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config,
        "data": data_metadata,
        "metrics": metrics,
        "rng_state": capture_rng_state(),
    }
    _atomic_torch_save(payload, path)


def load_training_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer) -> dict[str, Any]:
    """Restore a checkpoint into already-constructed compatible objects."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')}")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    restore_rng_state(payload["rng_state"])
    return payload


def load_final_artifact(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a final artifact for inference.

    A run directory is accepted as a convenience and resolves to its
    ``artifact.pt`` file. The restricted weights-only loader accepts the
    tensors and primitive metadata produced by :func:`save_final_artifact`
    without enabling arbitrary Python object reconstruction.
    """
    artifact_path = path / "artifact.pt" if path.is_dir() else path
    if not artifact_path.is_file():
        raise FileNotFoundError(f"model artifact does not exist: {artifact_path}")

    payload = torch.load(artifact_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("model artifact payload must be a dictionary")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported artifact schema: {payload.get('schema_version')}")

    required = {"step", "model_state_dict", "config", "tokenizer"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"model artifact is missing required fields: {', '.join(missing)}")
    if not isinstance(payload["model_state_dict"], Mapping) or not payload["model_state_dict"]:
        raise ValueError("model artifact contains an invalid model_state_dict")
    if not isinstance(payload["config"], Mapping) or not isinstance(payload["config"].get("model"), Mapping):
        raise ValueError("model artifact contains an invalid model configuration")
    if not isinstance(payload["tokenizer"], Mapping):
        raise ValueError("model artifact contains invalid tokenizer metadata")
    if not isinstance(payload["tokenizer"].get("name"), str):
        raise ValueError("model artifact tokenizer metadata is missing its name")
    if not isinstance(payload["tokenizer"].get("vocab_size"), int):
        raise ValueError("model artifact tokenizer metadata is missing its vocabulary size")
    return payload


def save_final_artifact(
    run_dir: Path,
    model: nn.Module,
    config: dict[str, Any],
    tokenizer: dict[str, Any],
    data: dict[str, Any],
    metrics: list[dict[str, Any]],
    step: int,
) -> Path:
    """Write a self-contained artifact that future inference can reconstruct."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config,
        "tokenizer": tokenizer,
        "data": data,
        "metrics": metrics,
    }
    artifact_path = run_dir / "artifact.pt"
    _atomic_torch_save(payload, artifact_path)
    (run_dir / "config.json").write_text(json_dumps(config), encoding="utf-8")
    metric_fields = sorted({field for record in metrics for field in record})
    metric_summary = _summarize_metrics(metrics)
    metrics_document = {
        "schema_version": SCHEMA_VERSION,
        "description": "One record per optimizer step; update_norm is sampled at log intervals.",
        "fields": metric_fields,
        "summary": metric_summary,
        "records": metrics,
    }
    (run_dir / "metrics.json").write_text(json_dumps(metrics_document), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json_dumps({"schema_version": SCHEMA_VERSION, "artifact": str(artifact_path), "step": step}),
        encoding="utf-8",
    )
    return artifact_path


def json_dumps(value: object) -> str:
    return json.dumps(value, indent=2, default=str) + "\n"


def _summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute useful plotting summaries without requiring pandas or NumPy."""
    if not metrics:
        return {}
    summary: dict[str, Any] = {"num_records": len(metrics), "final_step": metrics[-1].get("step")}
    for field in ("train_loss", "valid_loss"):
        values = [float(record[field]) for record in metrics if record.get(field) is not None]
        if values:
            summary[f"final_{field}"] = values[-1]
            summary[f"best_{field}"] = min(values)
    valid_records = [record for record in metrics if record.get("valid_loss") is not None]
    if valid_records:
        best = min(valid_records, key=lambda record: float(record["valid_loss"]))
        summary["best_valid_step"] = best["step"]
    for field in ("tokens_seen", "tokens_per_second", "wall_time_seconds"):
        if field in metrics[-1]:
            summary[f"final_{field}"] = metrics[-1][field]
    return summary
