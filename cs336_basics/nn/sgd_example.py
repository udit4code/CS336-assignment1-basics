import torch

from .sgd import StochasticGradientDescentOptimizer

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
optimizer = StochasticGradientDescentOptimizer(params=[weights], lr=1)

for t in range(10):
    # Gradients accumulate by default, so clear the previous iteration first.
    optimizer.zero_grad()
    # A scalar loss makes backward() seed its derivative with 1.
    loss = (weights**2).mean()
    print(f"loss value in iteration {t + 1} : {loss.item()}\n")
    # Populate weights.grad through reverse-mode automatic differentiation.
    loss.backward()
    # Apply one parameter update using the stored gradient.
    optimizer.step()
