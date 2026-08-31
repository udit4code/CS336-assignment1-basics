import math
from numbers import Integral, Real

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

        if not isinstance(theta, Real) or isinstance(theta, bool):
            raise TypeError(f"theta must be a real number, got {type(theta).__name__}")
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError(f"theta must be finite and positive, got {theta}")

        if not isinstance(d_k, Integral) or isinstance(d_k, bool):
            raise TypeError(f"d_k must be an integer, got {type(d_k).__name__}")
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError(f"d_k must be a positive even integer, got {d_k}")

        if not isinstance(max_seq_len, Integral) or isinstance(max_seq_len, bool):
            raise TypeError(
                f"max_seq_len must be an integer, got {type(max_seq_len).__name__}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

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

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim < 2:
            raise ValueError(
                f"x must have shape (..., seq_len, d_k), got shape {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(f"x must be floating point, got dtype {x.dtype}")
        if x.shape[-1] != self.d_k:
            raise ValueError(
                f"Expected x.shape[-1] == d_k == {self.d_k}, got {x.shape[-1]}"
            )

        if not isinstance(token_positions, torch.Tensor):
            raise TypeError(
                "token_positions must be a torch.Tensor, "
                f"got {type(token_positions).__name__}"
            )
        if token_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "token_positions must have dtype torch.int32 or torch.int64, "
                f"got {token_positions.dtype}"
            )
        if token_positions.ndim < 1:
            raise ValueError("token_positions must have at least one dimension")
        if token_positions.ndim > x.ndim - 1:
            raise ValueError(
                "token_positions has too many dimensions for x: "
                f"got {token_positions.ndim} and {x.ndim}, respectively"
            )
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                "token_positions and x must have the same sequence length, got "
                f"{token_positions.shape[-1]} and {x.shape[-2]}"
            )
        if x.device != self.cos_cached.device:
            raise ValueError(
                f"x is on {x.device}, but the RoPE cache is on {self.cos_cached.device}"
            )
        if token_positions.device != self.cos_cached.device:
            raise ValueError(
                "token_positions and the RoPE cache must be on the same device, got "
                f"{token_positions.device} and {self.cos_cached.device}"
            )
        if token_positions.numel() > 0:
            min_position = int(token_positions.min().item())
            max_position = int(token_positions.max().item())
            if min_position < 0 or max_position >= self.max_seq_len:
                raise ValueError(
                    "token_positions must lie in "
                    f"[0, {self.max_seq_len}), got range [{min_position}, {max_position}]"
                )

        # x has Shape: (..., seq_len, d_k)
        # Select the cosine/sine values corresponding to the token positions.
        # cos_cached: (max_seq_len, d_k/2) becomes (..., seq_len, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        rotary_input_shape = (*x.shape[:-1], self.d_k // 2)
        try:
            torch.broadcast_shapes(rotary_input_shape, cos.shape)
        except RuntimeError as error:
            raise ValueError(
                "token_positions leading dimensions are not broadcastable with x: "
                f"got gathered cache shape {tuple(cos.shape)} and "
                f"rotary input shape {rotary_input_shape}"
            ) from error


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
