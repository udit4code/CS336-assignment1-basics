import math
from numbers import Integral, Real

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

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
            raise TypeError(f"max_seq_len must be an integer, got {type(max_seq_len).__name__}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # RoPE groups adjacent coordinates (0,1), (2,3), ... into d_k/2
        # independent 2D planes. The even indices 0,2,... supply the exponent
        # 2i/d_k for each plane's inverse frequency.
        freq_seq = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )

        # inv_freq[i] = theta^(-2i/d_k). For theta=10000 and d_k=8 this is
        # [1, 0.1, 0.01, 0.001] radians per position. Earlier coordinate pairs
        # therefore rotate faster than later pairs.
        theta_tensor = torch.tensor(theta, device=device, dtype=torch.float32)
        inv_freq = torch.reciprocal(torch.pow(theta_tensor, freq_seq / d_k))

        # Cache positions [0, max_seq_len); position zero has angle zero.
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        # The outer product evaluates angle[p,i] = p * inv_freq[i] for every
        # position and coordinate pair, yielding (max_seq_len, d_k/2).
        angles = torch.outer(positions, inv_freq)

        # Buffers are non-trainable module state and follow model device moves.
        # These deterministic lookup tables are non-persistent so checkpoints do
        # not store data that __init__ can reconstruct.
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

    # Apply each 2D rotation directly instead of materializing a block-diagonal matrix.
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim < 2:
            raise ValueError(f"x must have shape (..., seq_len, d_k), got shape {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(f"x must be floating point, got dtype {x.dtype}")
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected x.shape[-1] == d_k == {self.d_k}, got {x.shape[-1]}")

        if not isinstance(token_positions, torch.Tensor):
            raise TypeError(f"token_positions must be a torch.Tensor, got {type(token_positions).__name__}")
        if token_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"token_positions must have dtype torch.int32 or torch.int64, got {token_positions.dtype}")
        if token_positions.ndim < 1:
            raise ValueError("token_positions must have at least one dimension")
        if token_positions.ndim > x.ndim - 1:
            raise ValueError(
                f"token_positions has too many dimensions for x: got {token_positions.ndim} and {x.ndim}, respectively"
            )
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                "token_positions and x must have the same sequence length, got "
                f"{token_positions.shape[-1]} and {x.shape[-2]}"
            )
        if x.device != self.cos_cached.device:
            raise ValueError(f"x is on {x.device}, but the RoPE cache is on {self.cos_cached.device}")
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
                    f"token_positions must lie in [0, {self.max_seq_len}), got range [{min_position}, {max_position}]"
                )

        # Advanced indexing gathers one row per requested position. Positions
        # shaped (B,S) therefore produce trigonometric tables shaped (B,S,d_k/2).
        seq_cos = self.cos_cached[token_positions]
        seq_sin = self.sin_cached[token_positions]

        # Insert singleton axes immediately before (S,d_k/2). In the common
        # multi-head case, (B,S,d_k/2) becomes (B,1,S,d_k/2) and broadcasts over H.
        while seq_cos.ndim < x.ndim:
            seq_cos = seq_cos.unsqueeze(-3)
            seq_sin = seq_sin.unsqueeze(-3)

        try:
            torch.broadcast_shapes(x[..., 0::2].shape, seq_cos.shape)
        except RuntimeError as error:
            raise ValueError(
                "token_positions leading dimensions are not broadcastable with x: "
                f"got gathered cache shape {tuple(seq_cos.shape)} and "
                f"rotary input shape {tuple(x[..., 0::2].shape)}"
            ) from error

        # Split adjacent coordinate pairs into even and odd components.
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # For each pair (x,y), multiply by [[cos,-sin],[sin,cos]]. Direct
        # elementwise evaluation is O(d_k) per token; a dense d_k-by-d_k
        # block-diagonal matrix would waste O(d_k^2) storage and arithmetic.
        rotated_even = x_even * seq_cos - x_odd * seq_sin
        rotated_odd = x_even * seq_sin + x_odd * seq_cos

        # The output still costs O(sequence_length*d_k). For max_seq_len=4096
        # and d_k=128 in float32, the two caches use 2 MiB total, versus
        # 256 MiB for one dense 128x128 matrix per cached position.
        out = torch.empty_like(x)

        out[..., 0::2] = rotated_even
        out[..., 1::2] = rotated_odd

        return out
