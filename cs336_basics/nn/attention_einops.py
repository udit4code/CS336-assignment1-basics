import math
import torch
from einops import einsum

from .softmax import softmax


def scaled_dot_product_attention_with_einops(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Args:
        query: Shape ``(..., query_length, d_k)``.
        key: Shape ``(..., key_length, d_k)``.
        value: Shape ``(..., key_length, d_v)``.
        mask: Boolean mask broadcastable to ``(..., query_length, key_length)``.
            True  -> allow attention
            False -> block attention

    Returns: Tensor of shape ``(..., query_length, d_v)``.
    """

    # Step 1 :
    # Compute every query-key similarity.
    # Mathematically: Scores = Q @ K.T
    #
    # Shape:
    # Query: (..., seq_len_q, d_k)
    # Key: (..., seq_len_k, d_k)
    #
    # Scores: (..., seq_len_q, seq_len_k)
    scores = einsum(
        query,
        key,
        "... q d_k, ... k d_k -> ... q k",
    )

    # Step 2 : Scale the scores.
    # Without scaling,
    # dot-products become very large as d_k grows,
    # causing Softmax to become extremely peaky.
    d_k = query.shape[-1]

    scores = scores / math.sqrt(d_k)

    # Step 3 : Apply the optional attention mask.
    if mask is not None:
        # This API uses True for allowed positions, while masked_fill replaces
        # True positions; invert the mask before filling disallowed logits.
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    # Step 4 : Convert scores into probabilities.
    # Every row now sums to one. Shape: (..., seq_len_q, seq_len_k)
    attention = softmax(
        scores,
        dim=-1,
    )

    # Step 5 : Compute Weighted sum of Value vectors.
    # Each query uses its attention probabilities
    # to compute a weighted average over all values.
    # Shape:
    # Attention: (..., seq_len_q, seq_len_k)
    # Value:(..., seq_len_k, d_v)
    # Output: (..., seq_len_q, d_v)
    output = einsum(
        attention,
        value,
        "... q k, ... k d_v -> ... q d_v",
    )

    return output
