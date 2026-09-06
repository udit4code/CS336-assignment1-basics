from collections.abc import Callable
import torch
import math


# A torch optimizer subclasses Optimizer, passes parameter groups and defaults
# to super().__init__, and implements step(). Gradients are populated by
# backward(); step updates parameters without recording those updates in autograd.


class StochasticGradientDescentOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        # The base constructor normalizes an iterable of Parameters into one
        # parameter group and attaches the default hyperparameters. Callers can
        # instead provide multiple group dictionaries with distinct settings.
        # This educational variant stores a per-parameter step because its
        # effective learning rate is lr/sqrt(t+1); ordinary constant-rate SGD
        # does not require per-parameter state.
        # Optimizer.step conventionally accepts an optional loss closure.
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")
        defaults = {"lr": lr}
        super().__init__(params=params, defaults=defaults)

    # The step(self, ...) method implements one optimization step.
    # It updates every registered parameter that currently has a gradient.
    # Broadly speaking, training looks like this :
    # Step 1: Forward pass
    # Step 2: Compute loss
    # Step 3: backward pass computes/accumulates gradients
    # Step 4: optimizer.step() updates parameters
    def step(self, closure: Callable | None = None):
        # One call applies one update to every eligible parameter.
        # A closure optionally reevaluates and returns the loss. Optimizers such
        # as LBFGS may call it repeatedly; this implementation calls it once.
        loss = None if closure is None else closure()
        # Parameter groups allow different subsets to use different hyperparameters.
        # SGD(model.parameters(), lr=0.01) stores a group similar to:
        # param_groups = [
        #     {
        #         "params" : [...],
        #         "lr" : 0.01,
        #     }
        # ]
        # Iterate through groups, then their parameters.
        for group in self.param_groups:
            # Get the learning rate for the given group
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    # Skip parameters unused by this loss or configured without gradients.
                    continue
                # Get state associated with p.
                # Optimizers may keep additional per-parameter bookkeeping.
                # Optimizer state is algorithm-specific: this class stores t,
                # while Adam-style methods store first and second moments.
                state = self.state[p]
                # Get iteration number from the state, or 0.
                t = state.get("t", 0)
                # Read the accumulated gradient of the loss with respect to p.
                grad = p.grad
                # Update weight tensor in-place
                # Here, learning rate shrinks over time, as t grows.  This idea is called learning rate decay schedule.
                # p is an nn.Parameter tracked by Autograd engine of PyTorch. The optimizer is not part of the computation graph.
                # The optimizer should modify the weights directly, not create new graph nodes to keep memory consumption minimal.
                # torch.no_grad keeps the update out of the autograd graph while
                # retaining PyTorch's normal in-place safety/version tracking.
                with torch.no_grad():
                    p -= lr / math.sqrt(t + 1) * grad
                # Increment iteration number
                state["t"] = t + 1
        return loss
