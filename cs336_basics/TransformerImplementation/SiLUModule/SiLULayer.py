import torch
import torch.nn as nn


# DOUBT : Why use F.silu(x) when we can use x * torch.sigmoid(x) ? What are the tradeoffs ? 
# When we use F.silu(x), then, PyTorch internally invokes its dedicated SiLU kernel, which is highly optimized.
# Under the hood, PyTorch uses fused operations, where no large intermediate tensor has to be materialized in Python, and hence, peak memory consumption is much lower. 
# Conceptually, x -> Optimized kernel -> output. The kernel computes x x sigmoid(x) without exposing the intermediate tensor to python.

# CPU : The impact is distinct, when we use a very heavy tensor. Eg : x is a 1 GB Tensor. 
# In manual implementation, we do : x -> sigmoid(x) , which creates another temporary 1 GB intermediate tensor -> multiply x with intermediate sigmoid(x) ->  output. 
# So, in manual implementation, our peak memory is 1 + 1 + 1 = 3 GB, when we maintain memory for x, intermediate sigmoid(x) and multiply operation.

# GPU : In GPU, x * torch.sigmoid(x) typically launches multiple kernels : a kernel for sigmoid and a kernel for multiply. Each kernel has its own overhead, especially synchronization and kernel launch. 
# F.silu() can use a single fused kernel, reducing : kernel launches, memory traffic and synchronization
class SiLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # We can also use F.silu(x), but for now, we will stick to manual implementation.
        return x * torch.sigmoid(x)