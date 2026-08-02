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
        
        # The only difference with respect to the non-reduce solution is :  reduce(x * x, "... d_model -> ... 1", "mean") + s 
        # What does it mean under the hood ? Literally, it says : take the mean over the d_model dimension, while keeping all leading dimensions unchanged, and then, replace the last dimension with a singleton dimension (1). 
        # So, say, initially, the shape of x = [32, 128, 768], meaning batch_size = 32, seq_len = 128, d_model = 768. 
        # Now, x * x will also have the same shape [32, 128, 768]. 
        # Here, x_i * x_i = Summation(i) x_i^2 / d_model, where i = 0, 1, 2, ..., d_model - 1. So, i is repeated index. 
        # So, sum_sq = einsum(x, x, "... d, ... d -> ...") and mean_sq = sum_sq / D . 
        # Here, we can simply use reduce instead, as : reduce(x * x, "... d -> ... 1", "mean"). So, the pattern "...d -> ...1" means "remove the index named d using the mean operation and replace it with a singleton dimension (1)".
        # As a result, (batch_size, seq_len, d_model) -> (batch_size, seq_len, 1)
        
        # Key Takeaway : einsum expresses reductions through repeated indices, while einops.reduce expresses reductions by explicitly naming the axis to eliminate and the operation ("mean", "sum", etc.). T
        rms = torch.sqrt(
            reduce(
                x * x,
                "... d_model -> ... 1",
                "mean",
            ) + self.eps
        )

        x = x / rms
        x = x * self.weight

        return x.to(in_dtype)