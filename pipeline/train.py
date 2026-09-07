"""End-to-end language-model training built on the existing CS336 modules."""

from datetime import UTC, datetime
from collections.abc import Callable
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import tiktoken
import torch

from cs336_basics.batching import get_batch
from cs336_basics.data import LanguageModelDataset
from cs336_basics.nn import AdamW, TransformerLM, cross_entropy, get_lr_cosine_schedule, gradient_clipping

from .artifacts import load_training_checkpoint, save_final_artifact, save_training_checkpoint
from .config import DataConfig, ModelConfig, OptimizerConfig, RuntimeConfig, dataclass_dict
from .prepare_data import encode_text_file, save_token_array, split_tokens


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` once, so every tensor uses the same device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def seed_everything(seed: int) -> None:
    """Seed all random sources used by sampling and model initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_datasets(data: DataConfig, run_dir: Path) -> tuple[LanguageModelDataset, LanguageModelDataset, dict[str, Any]]:
    """Prepare train/validation arrays and return the existing dataset adapter."""
    data.validate()
    token_dir = run_dir / "tokens"
    train_path = token_dir / "train.npy"
    train_meta = encode_text_file(data.train_path, train_path, data.encoding)

    if data.valid_path is not None:
        valid_path = token_dir / "valid.npy"
        valid_meta = encode_text_file(data.valid_path, valid_path, data.encoding)
    else:
        all_tokens = np.load(train_path, mmap_mode="r")
        train_tokens, valid_tokens = split_tokens(all_tokens, data.validation_fraction, data.context_length)
        # The split arrays are persisted independently: validation windows can
        # never reach across the train/validation boundary.
        # Copy before replacing the source file. A copy closes the read-only
        # memmap dependency, which matters on Windows where an open mapped
        # file cannot be atomically replaced.
        train_copy = np.array(train_tokens, dtype=np.uint32, copy=True)
        valid_copy = np.array(valid_tokens, dtype=np.uint32, copy=True)
        save_token_array(train_path, train_copy)
        valid_path = token_dir / "valid.npy"
        save_token_array(valid_path, valid_copy)
        valid_meta = {
            "source_path": "deterministic suffix split",
            "encoding": data.encoding,
            "vocab_size": train_meta["vocab_size"],
            "token_count": int(valid_copy.size),
            "dtype": "uint32",
            "token_file": str(valid_path),
        }
        train_meta["token_count"] = int(train_copy.size)
        # Keep sidecar metadata synchronized with the persisted split arrays.
        (train_path.with_suffix(".json")).write_text(json.dumps(train_meta, indent=2) + "\n", encoding="utf-8")
        (valid_path.with_suffix(".json")).write_text(json.dumps(valid_meta, indent=2) + "\n", encoding="utf-8")

    train_dataset = LanguageModelDataset(train_path, data.context_length)
    valid_dataset = LanguageModelDataset(valid_path, data.context_length)
    return train_dataset, valid_dataset, {"train": train_meta, "valid": valid_meta}


def _gradient_l2_norm(model: TransformerLM) -> float:
    """Return the global L2 norm of currently populated gradient buffers."""
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(torch.sum(parameter.grad.detach() ** 2).item())
    return squared**0.5


def _parameter_l2_norm(model: TransformerLM) -> float:
    """Return the global L2 norm of model parameters after an update."""
    squared = sum(float(torch.sum(parameter.detach() ** 2).item()) for parameter in model.parameters())
    return squared**0.5


def _snapshot_parameters(model: TransformerLM) -> list[torch.Tensor]:
    """Copy parameters temporarily when update-norm telemetry is requested."""
    return [parameter.detach().clone() for parameter in model.parameters()]


def _update_l2_norm(model: TransformerLM, before: list[torch.Tensor]) -> float:
    """Measure the actual AdamW parameter displacement for this optimizer step."""
    squared = sum(float(torch.sum((parameter.detach() - old) ** 2).item()) for parameter, old in zip(model.parameters(), before, strict=True))
    return squared**0.5


def evaluate(model: TransformerLM, dataset: LanguageModelDataset, runtime: RuntimeConfig, device: torch.device) -> float:
    """Estimate validation loss with independent random windows."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(runtime.eval_batches):
            inputs, targets = get_batch(dataset, runtime.batch_size, str(device))
            losses.append(float(cross_entropy(model(inputs), targets).item()))
    model.train()
    return float(np.mean(losses))


def run_training(
    data: DataConfig,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    runtime: RuntimeConfig,
    artifacts_dir: Path,
    run_name: str | None = None,
    resume: Path | None = None,
    on_persist: Callable[[], None] | None = None,
) -> Path:
    """Run training and return the path to the final self-contained artifact."""
    data.validate()
    model_config.validate()
    optimizer_config.validate()
    runtime.validate()
    if model_config.context_length != data.context_length:
        raise ValueError("model and data context_length must match")

    seed_everything(runtime.seed)
    device = resolve_device(runtime.device)
    dtype = resolve_dtype(runtime.dtype)
    run_id = run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, valid_dataset, data_metadata = prepare_datasets(data, run_dir)
    if model_config.vocab_size != data_metadata["train"]["vocab_size"]:
        raise ValueError("model vocab_size must equal the selected tokenizer vocabulary size")

    model = TransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        theta=model_config.rope_theta,
        dtype=dtype,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=optimizer_config.learning_rate, weight_decay=optimizer_config.weight_decay)
    step = 0
    metrics: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    if resume is not None:
        payload = load_training_checkpoint(resume, model, optimizer)
        step = int(payload["step"])
        metrics = list(payload.get("metrics", []))

    config_payload = {
        "data": dataclass_dict(data),
        "model": dataclass_dict(model_config),
        "optimizer": dataclass_dict(optimizer_config),
        "runtime": dataclass_dict(runtime),
    }
    if resume is not None:
        saved_config = payload.get("config", {})
        for section in ("model", "data"):
            if saved_config.get(section) != config_payload[section]:
                raise ValueError(f"resume checkpoint {resume} is incompatible with current {section} configuration")
    while step < optimizer_config.max_steps:
        inputs, targets = get_batch(train_dataset, runtime.batch_size, str(device))
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        gradient_norm_before_clip = _gradient_l2_norm(model)
        gradient_clipping(model.parameters(), optimizer_config.max_grad_norm)
        gradient_norm_after_clip = _gradient_l2_norm(model)
        # ``step`` is zero-based, while a schedule describes the learning rate
        # for the update being performed.  Using step + 1 avoids a completely
        # wasted first optimizer update when warmup_steps is positive.
        update_step = step + 1
        lr = get_lr_cosine_schedule(
            update_step,
            optimizer_config.learning_rate,
            optimizer_config.min_learning_rate,
            optimizer_config.warmup_steps,
            optimizer_config.max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        parameter_snapshot = _snapshot_parameters(model) if (step + 1) % runtime.log_interval == 0 else None
        optimizer.step()
        step += 1

        elapsed = time.perf_counter() - started_at
        record: dict[str, Any] = {
            "step": step,
            "train_loss": float(loss.item()),
            "learning_rate": float(lr),
            "gradient_norm_before_clip": gradient_norm_before_clip,
            "gradient_norm_after_clip": gradient_norm_after_clip,
            "parameter_norm": _parameter_l2_norm(model),
            "tokens_seen": step * runtime.batch_size * data.context_length,
            "tokens_per_second": step * runtime.batch_size * data.context_length / max(elapsed, 1e-9),
            "wall_time_seconds": elapsed,
            # Cloning a full model every step is wasteful. We measure the exact
            # AdamW displacement at log intervals, which is dense enough for
            # optimizer diagnostics without doubling training memory usage.
            "update_norm": _update_l2_norm(model, parameter_snapshot) if parameter_snapshot is not None else None,
        }
        if step % runtime.eval_interval == 0 or step == optimizer_config.max_steps:
            record["valid_loss"] = evaluate(model, valid_dataset, runtime, device)
        metrics.append(record)
        if step % runtime.log_interval == 0 or step == 1:
            message = f"step={step:06d} train_loss={record['train_loss']:.4f} lr={lr:.3e}"
            if "valid_loss" in record:
                message += f" valid_loss={record['valid_loss']:.4f}"
            print(message)
        if step % runtime.checkpoint_interval == 0:
            save_training_checkpoint(run_dir / "checkpoints" / f"step_{step:08d}.pt", model, optimizer, step, config_payload, data_metadata, metrics)
            if on_persist is not None:
                on_persist()

    encoding = tiktoken.get_encoding(data.encoding)
    artifact_path = save_final_artifact(
        run_dir,
        model,
        config_payload,
        {"name": data.encoding, "vocab_size": encoding.n_vocab},
        data_metadata,
        metrics,
        step,
    )
    if on_persist is not None:
        on_persist()
    return artifact_path
