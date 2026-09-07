from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from cs336_basics.nn import Embedding, MultiHeadSelfAttention, RMSNorm


class RandomAttentionModel(nn.Module):
    """A minimal pre-norm embedding plus one repository MHA layer.

    This model is intentionally not a language model. Its random attention
    patterns demonstrate tensor mechanics, not learned linguistic behaviour.
    """

    embedding: Embedding
    norm: RMSNorm
    attention: MultiHeadSelfAttention

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        theta: float = 10_000.0,
        max_seq_len: int = 128,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def hidden_states(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return embeddings and the pre-attention normalized states."""
        embeddings = self.embedding(token_ids)
        return embeddings, self.norm(embeddings)


def build_random_model(
    vocab_size: int,
    *,
    d_model: int = 128,
    num_heads: int = 4,
    theta: float = 10_000.0,
    max_seq_len: int = 128,
    seed: int = 0,
) -> RandomAttentionModel:
    """Build a deterministic random model for the educational visualization."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if d_model <= 0 or num_heads <= 0:
        raise ValueError("d_model and num_heads must be positive")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    torch.manual_seed(seed)
    model = RandomAttentionModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
    )
    model.eval()
    return model


def build_model(
    vocab_size: int,
    *,
    mode: Literal["random", "checkpoint"] = "random",
    checkpoint_path: str | None = None,
    d_model: int = 128,
    num_heads: int = 4,
    theta: float = 10_000.0,
    max_seq_len: int = 128,
    seed: int = 0,
) -> RandomAttentionModel:
    """Model factory with a reserved, intentionally unimplemented checkpoint mode."""
    if mode == "checkpoint":
        raise NotImplementedError(
            "checkpoint mode is reserved for a future compatible-checkpoint loader; "
            "the current implementation supports mode='random' only"
        )
    if checkpoint_path is not None:
        raise ValueError("checkpoint_path is only valid when checkpoint mode is implemented")
    if mode != "random":
        raise ValueError(f"unsupported mode: {mode!r}")
    return build_random_model(
        vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
        seed=seed,
    )
