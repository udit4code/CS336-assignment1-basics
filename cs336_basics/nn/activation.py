import torch
import torch.nn as nn


# SiLU(x) = x * sigmoid(x). This explicit form is useful for revision and
# produces the same mathematical result as torch.nn.functional.silu. Backend,
# device, dtype, and compiler determine whether either form is fused, so a
# particular kernel count or peak-memory reduction is not guaranteed.
class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the definition visible rather than delegating to the built-in.
        return x * torch.sigmoid(x)
