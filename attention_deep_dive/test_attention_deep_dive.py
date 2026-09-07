from __future__ import annotations

import torch

from .attention_probe import probe_attention
from .demo import run_demo
from .model import build_random_model
from .tokenize import tokenize_sentence


def test_token_bytes_reconstruct_input() -> None:
    tokenized = tokenize_sentence("While waiting for the bus, Sam.")
    assert b"".join(tokenized.token_bytes) == tokenized.text.encode("utf-8")
    assert len(tokenized.token_ids) == len(tokenized.labels)


def test_probe_matches_mha_and_preserves_causality() -> None:
    tokenized = tokenize_sentence("While waiting for the bus")
    model = build_random_model(50_257, d_model=32, num_heads=4, max_seq_len=32, seed=3)
    ids = torch.tensor([tokenized.token_ids], dtype=torch.long)
    positions = torch.arange(ids.shape[1], dtype=torch.long).unsqueeze(0)
    _, hidden = model.hidden_states(ids)

    trace = probe_attention(model.attention, hidden, positions)
    probabilities = trace.probabilities
    assert probabilities.shape == (1, 4, ids.shape[1], ids.shape[1])
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones_like(probabilities.sum(dim=-1)))
    assert torch.count_nonzero(torch.triu(probabilities, diagonal=1)) == 0


def test_probe_supports_batch_size_different_from_head_count() -> None:
    model = build_random_model(100, d_model=24, num_heads=6, max_seq_len=8, seed=11)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    positions = torch.arange(3, dtype=torch.long).expand(2, 3)
    _, hidden = model.hidden_states(ids)
    trace = probe_attention(model.attention, hidden, positions)
    assert trace.probabilities.shape == (2, 6, 3, 3)


def test_cli_runner_writes_token_level_artifacts(tmp_path) -> None:
    html_file, json_file = run_demo(
        "The cat sat.",
        output_dir=tmp_path,
        d_model=32,
        num_heads=4,
        seed=9,
        device="cpu",
        dtype="float32",
    )
    assert html_file.name == "token_attention.html"
    assert json_file.name == "token_attention.json"
    assert html_file.stat().st_size > 0
    assert json_file.stat().st_size > 0
