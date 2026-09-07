from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import tiktoken
import torch

from .attention_probe import probe_attention
from .model import build_model
from .render_html import write_attention_artifacts
from .tokenize import tokenize_sentence


DEFAULT_SENTENCE = "While waiting for the bus, Sam got frustrated, as it was too late"


def run_demo(
    sentence: str,
    *,
    output_dir: str | Path = "attention_deep_dive/outputs",
    d_model: int = 128,
    num_heads: int = 4,
    theta: float = 10_000.0,
    context_length: int = 128,
    seed: int = 0,
    mode: Literal["random", "checkpoint"] = "random",
) -> tuple[Path, Path]:
    """Run the token-level random-attention demonstration."""
    tokenized = tokenize_sentence(sentence)
    if len(tokenized.token_ids) > context_length:
        raise ValueError(
            f"sentence produced {len(tokenized.token_ids)} tokens, exceeding context_length={context_length}"
        )

    model = build_model(
        tiktoken.get_encoding("gpt2").n_vocab,
        mode=mode,
        d_model=d_model,
        num_heads=num_heads,
        theta=theta,
        max_seq_len=context_length,
        seed=seed,
    )
    token_ids = torch.tensor([tokenized.token_ids], dtype=torch.long)
    token_positions = torch.arange(token_ids.shape[1], dtype=torch.long).unsqueeze(0)

    with torch.inference_mode():
        _, normalized = model.hidden_states(token_ids)
        trace = probe_attention(model.attention, normalized, token_positions)

    return write_attention_artifacts(output_dir, tokenized, trace)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a token-level causal attention map")
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE)
    parser.add_argument("--output-dir", type=Path, default=Path("attention_deep_dive/outputs"))
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--theta", type=float, default=10_000.0)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("random", "checkpoint"), default="random")
    args = parser.parse_args()

    html_file, json_file = run_demo(
        args.sentence,
        output_dir=args.output_dir,
        d_model=args.d_model,
        num_heads=args.num_heads,
        theta=args.theta,
        context_length=args.context_length,
        seed=args.seed,
        mode=args.mode,
    )
    print(f"Wrote {html_file}")
    print(f"Wrote {json_file}")


if __name__ == "__main__":
    main()
