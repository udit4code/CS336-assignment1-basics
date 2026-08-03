import torch
import torch.nn as nn

from cs336_basics.TransformerImplementation.RMSNormModule.RMSNormLayer import RMSNorm
from cs336_basics.TransformerImplementation.MultiHeadSelfAttentionModule.MultiHeadSelfAttention import (
    MultiHeadSelfAttention,
)
from cs336_basics.TransformerImplementation.PositionWiseFeedForwardModule.SwiGLULayer import (
    SwiGLU,
)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        max_seq_len: int = 4096,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        ####################################################################
        # First RMSNorm
        ####################################################################
        self.ln1 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        ####################################################################
        # Multi-Head Self Attention
        ####################################################################
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        ####################################################################
        # Second RMSNorm
        ####################################################################
        self.ln2 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        ####################################################################
        # Feed Forward
        ####################################################################
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        ###############################################################
        # First sub-layer
        #
        # x = x + Attention(LN(x))
        ###############################################################
        x = x + self.attn(
            self.ln1(x),
            token_positions,
        )

        ###############################################################
        # Second sub-layer
        #
        # x = x + FFN(LN(x))
        ###############################################################
        x = x + self.ffn(
            self.ln2(x),
        )

        return x