import math
from typing import Callable, Optional

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

    def step(
        self,
        closure: Optional[Callable] = None,
    ):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]
                step = state["step"] + 1

                alpha_t = (
                    lr
                    * math.sqrt(1 - beta2 ** step)
                    / (1 - beta1 ** step)
                )

                with torch.no_grad():
                    p -= lr * weight_decay * p
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * grad * grad
                    p -= alpha_t * m / (torch.sqrt(v) + eps)

                state["m"] = m
                state["v"] = v

                state["step"] = step

        return loss