"""Modal entry point for running the repository's training pipeline on CUDA.

This adapter intentionally contains no model or optimization logic.  Modal
only supplies the container, GPU, and persistent filesystem; ``run_training``
continues to be the single source of truth for training behavior.

Install the optional local client with ``uv add --dev modal`` and authenticate
with ``modal setup`` before running this file.
"""

from pathlib import Path

import modal  # ty: ignore[unresolved-import]  # Optional dependency; installed in the Modal image/local client.


APP_NAME = "cs336-training"
DATA_MOUNT = "/mnt/data"
ARTIFACT_MOUNT = "/mnt/artifacts"
RUN_NAME = "tinystories-gpt2"

app = modal.App(APP_NAME)

# ``uv_sync`` installs the dependencies declared in pyproject.toml.  The two
# local Python sources are added explicitly so the remote container imports the
# exact code that was checked out on the developer's machine.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .add_local_python_source("pipeline", "cs336_basics")
)

data_volume = modal.Volume.from_name("cs336-training-data", create_if_missing=True)
artifact_volume = modal.Volume.from_name("cs336-training-artifacts", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=86400,
    retries=2,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def train_remote(
    train_filename: str = "TinyStoriesV2-GPT4-train.txt",
    valid_filename: str = "TinyStoriesV2-GPT4-valid.txt",
    run_name: str = RUN_NAME,
    max_steps: int = 10_000,
) -> str:
    """Train on an L4 and return the persistent artifact path.

    Data and outputs are under mounted Volume paths.  Writing elsewhere would
    use ephemeral container storage and would not be available after exit.
    """
    from pipeline.config import DataConfig, ModelConfig, OptimizerConfig, RuntimeConfig
    from pipeline.train import run_training

    train_path = Path(DATA_MOUNT) / train_filename
    valid_path = Path(DATA_MOUNT) / valid_filename
    artifact = run_training(
        data=DataConfig(
            train_path=train_path,
            valid_path=valid_path,
            encoding="gpt2",
            context_length=128,
            validation_fraction=0.1,
        ),
        model_config=ModelConfig(
            context_length=128,
            vocab_size=50257,
            d_model=256,
            num_layers=4,
            num_heads=8,
            d_ff=1024,
            rope_theta=10000.0,
        ),
        optimizer_config=OptimizerConfig(
            learning_rate=3e-4,
            min_learning_rate=3e-5,
            warmup_steps=1000,
            max_steps=max_steps,
            weight_decay=0.1,
            max_grad_norm=1.0,
        ),
        runtime=RuntimeConfig(
            batch_size=8,
            seed=0,
            device="cuda",
            dtype="float32",
            eval_interval=100,
            checkpoint_interval=1000,
            eval_batches=10,
            log_interval=10,
        ),
        artifacts_dir=Path(ARTIFACT_MOUNT),
        run_name=run_name,
        # Persist every checkpoint and the final artifact to the Volume.  This
        # is important because Modal may retry a preempted Function.
        on_persist=artifact_volume.commit,
    )
    return str(artifact)


@app.local_entrypoint()
def main(
    train_filename: str = "TinyStoriesV2-GPT4-train.txt",
    valid_filename: str = "TinyStoriesV2-GPT4-valid.txt",
    run_name: str = RUN_NAME,
    max_steps: int = 10_000,
) -> None:
    artifact = train_remote.remote(train_filename, valid_filename, run_name, max_steps)
    print(f"Remote artifact: {artifact}")
