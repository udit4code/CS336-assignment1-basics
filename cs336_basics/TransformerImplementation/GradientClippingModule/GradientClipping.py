import torch
from torch import nn


# During training, we can sometimes hit training examples that yield large gradients, which can destabilize training.
# In order to mitigate it, one technique we often employ in practice is called gradient clipping. The idea is to enforce a limit on
# the norm of the gradient after each backward pass before taking an optimizer step. 
def gradient_clipping(
    parameters: list[nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:

    # Step 1 : Compute total_squared_norm for all parameters. 
    total_squared_norm = 0.0

    for p in parameters:
        if p.grad is None:
            continue
        total_squared_norm += torch.sum(p.grad ** 2)
    total_norm = torch.sqrt(total_squared_norm)
    
    # Step 2 : If total_l2_norm <= max_l2_norm, then, we exit. Because, in this case, it will scale up the gradient if applied, which we want to avoid in the first place.
    if total_norm <= max_l2_norm:
        return

    # Step 3 : Otherwise scale-down 
    scale = max_l2_norm / (total_norm + eps)
    for p in parameters:
        if p.grad is None:
            continue
        p.grad = p.grad * scale