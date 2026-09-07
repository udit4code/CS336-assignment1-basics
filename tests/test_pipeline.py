import json
from pathlib import Path

import numpy as np
import tiktoken
import torch

from pipeline.config import DataConfig, ModelConfig, OptimizerConfig, RuntimeConfig
from pipeline.prepare_data import encode_text_file
from pipeline.train import run_training


def test_endoftext_is_one_token(tmp_path: Path):
    source = tmp_path / "corpus.txt"
    source.write_text("before <|endoftext|> after", encoding="utf-8")
    metadata = encode_text_file(source, tmp_path / "tokens.npy", "gpt2")

    encoding = tiktoken.get_encoding("gpt2")
    token_ids = np.load(metadata["token_file"])
    assert int(encoding.eot_token) in token_ids
    assert int(np.count_nonzero(token_ids == encoding.eot_token)) == 1
    assert metadata["endoftext_id"] == encoding.eot_token


def test_encode_text_file_is_reproducible(tmp_path: Path):
    source = tmp_path / "corpus.txt"
    source.write_text("A small corpus.\n", encoding="utf-8")
    first = encode_text_file(source, tmp_path / "first.npy", "gpt2")
    second = encode_text_file(source, tmp_path / "second.npy", "gpt2")

    np.testing.assert_array_equal(np.load(first["token_file"]), np.load(second["token_file"]))
    assert first["source_sha256"] == second["source_sha256"]
    assert first["token_count"] > 0


def test_tiny_training_writes_loadable_artifact(tmp_path: Path):
    source = tmp_path / "corpus.txt"
    source.write_text("A tiny language-model corpus. " * 20, encoding="utf-8")
    data = DataConfig(source, None, "gpt2", 8, 0.2)
    model = ModelConfig(8, 50257, 16, 1, 1, 32, 10000.0)
    optimizer = OptimizerConfig(3e-4, 3e-5, 1, 2, 0.1, 1.0)
    runtime = RuntimeConfig(1, 0, "cpu", "float32", 1, 2, 1, 1)

    artifact = run_training(data, model, optimizer, runtime, tmp_path / "artifacts", "test-run")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 1
    assert payload["step"] == 2
    assert (artifact.parent / "manifest.json").is_file()
    metrics = json.loads((artifact.parent / "metrics.json").read_text())
    assert {"train_loss", "learning_rate", "gradient_norm_before_clip", "parameter_norm", "tokens_seen"} <= set(metrics["fields"])
    assert len(metrics["records"]) == 2
