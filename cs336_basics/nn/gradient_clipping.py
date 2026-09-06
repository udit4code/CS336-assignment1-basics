from collections.abc import Iterable

import torch
from torch import nn


# Global-norm clipping rescales all available gradients by the same factor so
# their concatenated L2 norm is at most max_l2_norm. Apply it after backward()
# and before optimizer.step(); shared scaling preserves the gradient direction.
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

    # ||g||_2 = sqrt(sum over parameters and tensor elements of g_i^2).
    total_squared_norm = torch.zeros((), device=gradients[0].device)

    for p in parameters:
        if p.grad is None:
            continue
        total_squared_norm += torch.sum(p.grad.detach() ** 2).to(total_squared_norm.device)
    total_norm = torch.sqrt(total_squared_norm)

    # Clipping must never enlarge a gradient already below the threshold.
    if total_norm <= max_l2_norm:
        return

    # Detach the scale from autograd and update gradient buffers in place.
    scale = max_l2_norm / (total_norm + eps)
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
