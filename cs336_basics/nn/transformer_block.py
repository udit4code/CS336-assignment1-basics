import torch
import torch.nn as nn

from .feed_forward import SwiGLU
from .multihead_attention import MultiHeadSelfAttention
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        max_seq_len: int = 4096,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # This is a pre-norm block: normalize before each residual branch.
        self.attention_norm = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        self.ffn_norm = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.feed_forward = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    # x -> RMSNorm -> attention -> residual add -> RMSNorm -> SwiGLU -> residual add.
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        # Step 1 : Apply RMSNorm on input x. Initially, input x has shape = (batch_size, sequence_length, d_model) = (B, S, D)
        # RMSNorm does not change the shape of x. So, shape of normalized_x is still (B, S, D)
        normalized_x = self.attention_norm(x)

        # Attention preserves (B, S, D): projections split D into H heads of
        # width d_k=D/H, attention operates on (B,H,S,d_k), then heads merge.
        attention_output = self.attention(
            normalized_x,
            token_positions,
        )

        # Step 3 : x has shape (B, S, D) and attention_output has shape (B, S, D). So, we do a residual after attention.
        residual_after_attention = x + attention_output

        # Step 4 : We apply RMSNorm on residual_after_attention. This preserves the same Shape (B, S, D).
        normalized_residual = self.ffn_norm(residual_after_attention)

        # Step 5 : Now, we pass normalized_residual through a FFN layer.
        ffn_output = self.feed_forward(normalized_residual)

        # The second residual path preserves a direct gradient/information route.
        output = residual_after_attention + ffn_output

        return output
