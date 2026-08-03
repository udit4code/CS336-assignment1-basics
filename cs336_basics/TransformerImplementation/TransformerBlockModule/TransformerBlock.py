import torch
import torch.nn as nn

from cs336_basics.TransformerImplementation.MultiHeadSelfAttentionModule.MultiHeadSelfAttention import (
    MultiHeadSelfAttention,
)
from cs336_basics.TransformerImplementation.RMSNormModule.RMSNormLayer import RMSNorm
from cs336_basics.TransformerImplementation.PositionWiseFeedForwardModule.SwiGLULayer import SwiGLU


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

        # In both cases, be attention norm or ffn_norm, we use RMSNorm.
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

    # We can think of the following computational graph, something like:
    # x -> RMSNorm -> MultiHeadSelfAttention (QKV -> RoPE -> Attention -> Output Projection) -> Add(x) -> RMSNorm -> SwiGLU -> Add(residual) -> output 
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        
        # Step 1 : Apply RMSNorm on input x. Initially, input x has shape = (batch_size, sequence_length, d_model) = (B, S, D)
        # RMSNorm does not change the shape of x. So, shape of normalized_x is still (B, S, D)
        normalized_x = self.attention_norm(x)
        
        # Step 2 : Then, we apply multi-head self-attention on normalized_x 
        # Under the hood, normalized_x with shape (B, S, D) -> Q, K, V Projections, each having shape (B, S, D) -> Then, we reshape into heads, so shape is now (B, S, H, d_k),
        # where, H = num_of_heads and d_k = dimension of each head. After that, we transpose on dimensions 1 and 2 (that is, S, H) so that we go from (B, S, H, d_k) to (B, H, S, d_k) 
        # After that, we apply RoPE, which doesn't change shape (B, H, S, d_k). After RoPE, we compute attention scores via Q @ K.T (we take transpose over last 2 dimensions). So,
        # (B, H, S, d_k) @ (B, H, d_k, S) = (B, H, S, S) is the shape for Q @ K.T . Then, we apply causal mask on Q @ K.T and after that softmax on Q @ K.T. Causal mask and attention 
        # does not change shape, so shape is still (B, H, S, S). Now, we do Attention score x Value = Softmax(Q @ K.T) @ V, whose shape is (B, H, S, S) @ (B, H, S, d_k) = (B, H, S, d_k). 
        # Now, we transpose dimensions 1 and 2, and hence, we go back from (B, H, S, d_k) to (B, S, H, d_k). Then, we merge heads and go from shape (B, S, H, d_k) to (B, S, D), Finally,
        # we do output projection via W_o, which doesn't alter shape. So, finally, shape of attention output is (B, S, D) 
        attention_output = self.attention(
            normalized_x,
            token_positions,
        )

        # Step 3 : x has shape (B, S, D) and attention_output has shape (B, S, D). So, we do a residual after attention. 
        residual_after_attention = x + attention_output

        # Step 4 : We apply RMSNorm on residual_after_attention. This preserves the same Shape (B, S, D).
        normalized_residual = self.ffn_norm(
            residual_after_attention
        )

        # Step 5 : Now, we pass normalized_residual through a FFN layer. 
        ffn_output = self.feed_forward(
            normalized_residual
        )

        # Step 6 : Finally, add ffn_output to residual_after_attention
        output = residual_after_attention + ffn_output

        return output