import math

import torch
import torch.nn as nn
from einops import einsum


class LinearEinops(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            self.weight,
            "... d_in, d_out d_in -> ... d_out",
        )


# In ``... d_in, d_out d_in -> ... d_out``, d_in appears in both
# inputs but not the output, so it is summed. d_out remains, and ``...``
# preserves every leading axis. For x shaped (B,S,d_in) and W shaped
# (d_out,d_in), the result is (B,S,d_out):
#
#     y[..., o] = sum_i x[..., i] * W[o, i]
#
# This equals ``x @ W.T``. Named axes avoid an explicit transpose but do not
# imply a performance advantage; backend lowering determines performance.
