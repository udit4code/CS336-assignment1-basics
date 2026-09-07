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

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


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
    encoding_name: str = "gpt2",
    device: str | torch.device = "auto",
    dtype: torch.dtype | str = torch.float32,
    use_rope: bool = True,
    checkpoint_path: str | None = None,
) -> tuple[Path, Path]:
    """Run the token-level attention demonstration and write two artifacts."""
    tokenized = tokenize_sentence(sentence, encoding_name=encoding_name)
    if len(tokenized.token_ids) > context_length:
        raise ValueError(
            f"sentence produced {len(tokenized.token_ids)} tokens, exceeding context_length={context_length}"
        )

    resolved_device = _resolve_device(str(device)) if isinstance(device, str) else device
    if isinstance(dtype, str):
        try:
            resolved_dtype = _DTYPES[dtype]
        except KeyError as error:
            raise ValueError(f"unsupported dtype: {dtype!r}; choose from {tuple(_DTYPES)}") from error
    else:
        resolved_dtype = dtype
    encoding = tiktoken.get_encoding(encoding_name)

    model = build_model(
        encoding.n_vocab,
        mode=mode,
        checkpoint_path=checkpoint_path,
        d_model=d_model,
        num_heads=num_heads,
        theta=theta,
        max_seq_len=context_length,
        seed=seed,
        use_rope=use_rope,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    token_ids = torch.tensor([tokenized.token_ids], dtype=torch.long, device=resolved_device)
    token_positions = torch.arange(token_ids.shape[1], dtype=torch.long, device=resolved_device).unsqueeze(0)

    with torch.inference_mode():
        _, normalized = model.hidden_states(token_ids)
        trace = probe_attention(model.attention, normalized, token_positions)

    return write_attention_artifacts(
        output_dir,
        tokenized,
        trace,
        metadata={
            "mode": mode,
            "encoding_name": encoding_name,
            "d_model": d_model,
            "num_heads": num_heads,
            "theta": theta,
            "context_length": context_length,
            "seed": seed,
            "device": str(resolved_device),
            "dtype": str(resolved_dtype),
            "use_rope": use_rope,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a token-level causal attention map using the repository MHA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE, help="Text to tokenize and inspect")
    parser.add_argument("--encoding-name", default="gpt2", help="Encoding name accepted by tiktoken")
    parser.add_argument("--output-dir", type=Path, default=Path("attention_deep_dive/outputs"))
    parser.add_argument("--d-model", type=int, default=128, help="Embedding and MHA model width")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of equal-width attention heads")
    parser.add_argument("--theta", type=float, default=10_000.0, help="RoPE base theta")
    parser.add_argument("--context-length", type=int, default=128, help="Maximum cached RoPE position")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible weights")
    parser.add_argument("--mode", choices=("random", "checkpoint"), default="random")
    parser.add_argument("--checkpoint-path", default=None, help="Reserved for future checkpoint mode")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument("--no-rope", dest="use_rope", action="store_false", help="Disable RoPE in MHA")
    args = parser.parse_args()

    try:
        html_file, json_file = run_demo(
            args.sentence,
            output_dir=args.output_dir,
            d_model=args.d_model,
            num_heads=args.num_heads,
            theta=args.theta,
            context_length=args.context_length,
            seed=args.seed,
            mode=args.mode,
            encoding_name=args.encoding_name,
            device=args.device,
            dtype=args.dtype,
            use_rope=args.use_rope,
            checkpoint_path=args.checkpoint_path,
        )
    except (NotImplementedError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    labels = tokenized_labels(args.sentence, args.encoding_name)
    print(f"Tokenized {len(labels)} tokens: {' | '.join(labels)}")
    print(f"Wrote {html_file}")
    print(f"Wrote {json_file}")


def tokenized_labels(sentence: str, encoding_name: str) -> tuple[str, ...]:
    """Return labels for concise CLI output without exposing model internals."""
    return tokenize_sentence(sentence, encoding_name=encoding_name).labels


if __name__ == "__main__":
    main()
