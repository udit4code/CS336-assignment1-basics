from __future__ import annotations

from dataclasses import dataclass

import torch

from cs336_basics.nn import MultiHeadSelfAttention, softmax


@dataclass(frozen=True, slots=True)
class AttentionTrace:
    """Intermediate tensors needed for inspection and visualization.

    Shapes use ``B`` for batch, ``H`` for heads, ``S`` for sequence length, and
    ``K`` for head width. The demo uses ``B=1`` but the probe is batch-capable.
    """

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    raw_scores: torch.Tensor
    masked_scores: torch.Tensor
    probabilities: torch.Tensor
    output: torch.Tensor


def _validate_inputs(mha: MultiHeadSelfAttention, x: torch.Tensor, token_positions: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape (batch, sequence, d_model), got {tuple(x.shape)}")
    if x.shape[-1] != mha.d_model:
        raise ValueError(f"x has width {x.shape[-1]}, expected {mha.d_model}")
    if token_positions.shape != x.shape[:2]:
        raise ValueError(
            "token_positions must have shape (batch, sequence), "
            f"got {tuple(token_positions.shape)} for x {tuple(x.shape)}"
        )
    if token_positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("token_positions must be int32 or int64")


def probe_attention(
    mha: MultiHeadSelfAttention,
    x: torch.Tensor,
    token_positions: torch.Tensor,
    *,
    verify_output: bool = True,
) -> AttentionTrace:
    """Extract per-head causal attention while reusing repository components.

    The projection, head layout, RoPE, mask, scaling, and custom softmax mirror
    ``MultiHeadSelfAttention.forward``. ``verify_output`` reconstructs the MHA
    output and compares it with the module's public forward pass, protecting the
    visualization path from silently diverging from production code.
    """
    _validate_inputs(mha, x, token_positions)
    batch_size, seq_len, _ = x.shape

    query = mha.q_proj(x).view(batch_size, seq_len, mha.num_heads, mha.d_k).transpose(1, 2)
    key = mha.k_proj(x).view(batch_size, seq_len, mha.num_heads, mha.d_k).transpose(1, 2)
    value = mha.v_proj(x).view(batch_size, seq_len, mha.num_heads, mha.d_k).transpose(1, 2)

    if mha.use_rope:
        query = mha.rope(query, token_positions)
        key = mha.rope(key, token_positions)

    raw_scores = torch.matmul(query, key.transpose(-2, -1)) / (mha.d_k**0.5)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
    )
    masked_scores = raw_scores.masked_fill(~causal_mask, float("-inf"))
    probabilities = softmax(masked_scores, dim=-1)

    head_output = torch.matmul(probabilities, value)
    merged = head_output.transpose(1, 2).contiguous().view(batch_size, seq_len, mha.d_model)
    output = mha.out_proj(merged)

    if verify_output:
        reference = mha(x, token_positions)
        torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)

    return AttentionTrace(
        query=query,
        key=key,
        value=value,
        raw_scores=raw_scores,
        masked_scores=masked_scores,
        probabilities=probabilities,
        output=output,
    )
