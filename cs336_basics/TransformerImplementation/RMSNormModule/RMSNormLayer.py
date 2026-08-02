import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Learnable gain parameter g
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original dtype (float16/bfloat16/etc.)
        in_dtype = x.dtype

        # Upcast for numerical stability
        x = x.to(torch.float32)

        # Compute RMS over the last dimension
        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True)
            + self.eps
        )

        # Normalize
        x = x / rms

        # Apply learnable gain
        x = x * self.weight

        # Downcast back to original dtype
        return x.to(in_dtype)