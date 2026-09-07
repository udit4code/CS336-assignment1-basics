import math
import torch

from .softmax import softmax

# torch.matmul and the @ operator have the same batched matrix-multiplication
# semantics. They contract the final dimension of the left operand with the
# penultimate dimension of the right operand and broadcast leading dimensions.


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention and return only its output."""
    output, _, _, _ = scaled_dot_product_attention_with_weights(query, key, value, mask)
    return output


def scaled_dot_product_attention_with_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute attention and expose its diagnostic tensors.

    Args:
        query: Shape ``(..., query_length, d_k)``.
        key: Shape ``(..., key_length, d_k)``.
        value: Shape ``(..., key_length, d_v)``.
        mask: Boolean mask broadcastable to ``(..., query_length, key_length)``.
            True  -> allow attention
            False -> block attention

    Returns:
        ``(output, probabilities, raw_scores, masked_scores)``. The output has
        shape ``(..., query_length, d_v)`` and both score tensors have shape
        ``(..., query_length, key_length)``.
    """

    # QK^T forms every query-key dot product:
    # (..., Q, d_k) @ (..., d_k, K) -> (..., Q, K).
    raw_scores = torch.matmul(
        query,
        key.transpose(-2, -1),
    )

    # Dividing by sqrt(d_k) keeps score variance roughly independent of head
    # width, reducing softmax saturation when query/key components have unit variance.
    d_k = query.shape[-1]
    raw_scores = raw_scores / math.sqrt(d_k)
    scores = raw_scores

    # False means disallowed. Replacing those logits by -inf makes their
    # softmax probabilities zero, unless an entire row is masked (which yields NaNs).
    if mask is not None:
        # masked_fill replaces positions where its mask argument is True; the
        # inversion is therefore required for this API's True-means-allowed mask.
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    # Normalize over keys so each non-fully-masked query row sums to one.
    attention = softmax(
        scores,
        dim=-1,
    )

    # Weighted value sum: (..., Q, K) @ (..., K, d_v) -> (..., Q, d_v).
    output = torch.matmul(
        attention,
        value,
    )

    return output, attention, raw_scores, scores
