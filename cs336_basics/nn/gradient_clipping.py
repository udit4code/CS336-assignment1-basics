from collections.abc import Iterable

import torch
from torch import nn


# During training, we can sometimes hit training examples that yield large gradients, which can destabilize training.
# In order to mitigate it, one technique we often employ in practice is called gradient clipping. The idea is to enforce a limit on
# the norm of the gradient after each backward pass before taking an optimizer step. 
def gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    parameters = tuple(parameters)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    # Step 1 : Compute total_squared_norm for all parameters. 
    total_squared_norm = torch.zeros((), device=gradients[0].device)

    for p in parameters:
        if p.grad is None:
            continue
        total_squared_norm += torch.sum(p.grad.detach() ** 2).to(total_squared_norm.device)
    total_norm = torch.sqrt(total_squared_norm)
    
    # Step 2 : If total_l2_norm <= max_l2_norm, then, we exit. Because, in this case, it will scale up the gradient if applied, which we want to avoid in the first place.
    if total_norm <= max_l2_norm:
        return

    # Step 3 : Otherwise scale-down 
    scale = max_l2_norm / (total_norm + eps)
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
