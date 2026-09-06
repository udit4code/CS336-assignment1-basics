import torch
import torch.nn as nn
from einops import reduce


class RMSNormReduce(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # The only structural difference from the torch.mean version is the
        # explicit einops reduction pattern. It preserves leading axes and
        # reduces d_model to a singleton axis for broadcasting.
        # So, say, initially, the shape of x = [32, 128, 768], meaning batch_size = 32, seq_len = 128, d_model = 768.
        # Now, x * x will also have the same shape [32, 128, 768].
        # The mean square is sum_i(x_i^2) / d_model. An equivalent einsum
        # computes the sum, after which division by d_model is still required.
        # Here, we can simply use reduce instead, as : reduce(x * x, "... d -> ... 1", "mean"). So, the pattern "...d -> ...1" means "remove the index named d using the mean operation and replace it with a singleton dimension (1)".
        # As a result, (batch_size, seq_len, d_model) -> (batch_size, seq_len, 1)

        # Key takeaway: einsum implies a sum over omitted repeated axes, while
        # einops.reduce names both the reduced axis and operation explicitly.
        rms = torch.sqrt(
            reduce(
                x * x,
                "... d_model -> ... 1",
                "mean",
            )
            + self.eps
        )

        x = x / rms
        x = x * self.weight

        return x.to(in_dtype)
