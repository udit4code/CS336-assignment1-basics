import torch
import torch.nn as nn
from einops import rearrange


class RotaryPositionalEmbeddingWithReduce(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()

        assert d_k % 2 == 0

        # Compute one inverse frequency per 2D rotation block. Shape: (d_k/2,)
        freq_seq = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )

        inv_freq = 1.0 / (
            theta ** (freq_seq / d_k)
        )

        # Position indices. Shape: (max_seq_len,)
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        # Compute every rotation angle.
        # angle(position, frequency)
        # Shape: (max_seq_len, d_k/2)
        angles = torch.outer(
            positions,
            inv_freq,
        )

        self.register_buffer(
            "cos_cached",
            torch.cos(angles),
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            torch.sin(angles),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        # x has Shape: (..., seq_len, d_k)
        # Select the cosine/sine values corresponding to the token positions.
        # cos_cached: (max_seq_len, d_k/2) becomes (..., seq_len, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]


        # Rearrange the last dimension into pairs.
        # Before: (..., seq_len, d_k)
        # Example: [x0 x1 x2 x3 x4 x5 x6 x7]
        # After: (..., seq_len, d_k/2, 2)
        # [
        #   [x0 x1]
        #   [x2 x3]
        #   [x4 x5]
        #   [x6 x7]
        # ]
        # Every row is one 2D vector that will be rotated.
        # The below rearrange(...) tells that "Treat the last dimension of length d_k as d_k/2 groups, where each group contains exactly 2 numbers."
        # As a result, a tensor with shape (..., seq_len, 8) is reinterpreted as (..., seq_len, 4, 2), where each [x, y] pair is exactly the 2D vector that RoPE rotates. 
        # In einops, parentheses mean split or combine. For example, rearrange(x, "(h w) -> h w", h = 4) mean that if x.shape is (12, ) (meaning a 1 x 12 row vector), then, 12 = 4 x 3. 
        # So, the output is a (4, 3) tensor. So, (pair two) splits d_k  into pair x two , such that pair x two = d_k . We need to specifically mention two = 2 because einops knows d_k, but not pair or two.
        # Once two is specified as 2, then, it can deduce pair as pair x two = d_k => pair = d_k / two = 8 / 2 = 4. So, our tensor is no longer a flat embedding after rearrange operation.
        x = rearrange(
            x,
            "... seq (pair two) -> ... seq pair two",
            two=2,
        )

        # Split every 2D vector into its x- and y-components.
        # Shapes: (..., seq_len, d_k/2)
        x1 = x[..., 0]
        x2 = x[..., 1]

        # Apply the 2D rotation
        # [ cos -sin ]
        # [ sin  cos ]
        #
        # x' = x cos - y sin
        # y' = x sin + y cos
        #
        # These are elementwise operations.
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        # Stack the rotated coordinates back together.
        # Shape: (..., seq_len, d_k/2, 2)
        out = torch.stack(
            (y1, y2),
            dim=-1,
        )
        
        # Flatten the pairs back into the embedding dimension.
        # Before: (..., seq_len, d_k/2, 2)
        # After:  (..., seq_len, d_k)
        out = rearrange(
            out,
            "... seq pair two -> ... seq (pair two)",
        )

        return out