from collections.abc import Callable, Iterable 
from typing import Optional
import torch  
import math 


# How do we implement Optimizers ? 
# To implement optimizers, we will use the PyTorch torch.optim.Optimizer as base class and extend it further. An Optimizer subclass must implement 2 methods: 
# 1. __int__(self, params) that will initialize our optimizer. 
# 2. step(self) that should make one update of the parameters. During the training loop, this method will be called after the backward pass, so we have access to the gradients of the last batch. 
# The step(self) method should iterate through each parameter tensor p and modify them in place, like setting p.data which holds the tensor associated with that parameter based on the gradient p.grad (if it exists). 
# p.grad is the tensor representing the gradient of the loss with respect to that parameter. 

class StochasticGradientDescentOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        # Here, we pas the parameters as well as hyperparameters (eg : learning rate) to the base constructor. 
        # In case the parameters are just a single collection of torch.nn.Parameter objects, the base constructor will create a single group and 
        # assign it the default hyperparameters. Then, in step, we iterate over each parameter group, then over 
        # each parameter in that group, and apply parameter update equation. 
        # Here, we keep the iteration number as a state associated with each parameter: we first read this value, use it in the gradient update, and then update it. 
        # The API specifies that the user might pass in a callable closure to re-compute the loss before the  optimizer step. 
        # We won’t need this for the optimizers we’ll use, but we add it to comply with the API.
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")
        defaults = {"lr" : lr}
        super().__init__(params=params, defaults=defaults) 
        
    # The step(self, ...) method implements one optimization step.
    # It updates every learnable parameter using its computed gradient. 
    # Broadly speaking, training looks like this : 
    # Step 1: Forward pass 
    # Step 2: Compute loss
    # Step 3: Backward pass
    # Step 4: Compute gradients
    # Step 5: optimizer.step() to update parameters 
    def step(self, closure: Optional[Callable] = None):
        # The function signature tells the optimizer : "Update all parameters once"
        # DOUBT : What is a closure ? 
        loss = None if closure is None else closure()
        # DOUBT : Why do we iterate over self.param_groups ? 
        # Because when we create an optimizer via optimizer = SGD(model.parameters(), lr=0.01), PyTorch internally stores something like : 
        # param_groups = [
        #     {
        #         "params" : [...],
        #         "lr" : 0.01,
        #     }
        # ]
        # We need parameter groups, because, it is possible that different parameters may need different learning rates.
        # This loop means that : iterate over each parameter group. 
        for group in self.param_groups:
            # Get the learning rate for the given group
            lr = group["lr"] 
            for p in group["params"]:
                if p.grad is None:
                    # Skip Tensors without gradients, typically when we set p.requires_grad=False
                    continue
                # Get state associated with p.
                # Every optimizer keeps some additional information for book-keeping. 
                # For instance, vanilla SGD's weight only has iteration number.
                # But, Adam Optimizer's weight has first moment, second moment as well -> all these stored in self.state. 
                state = self.state[p]  
                # Get iteration number from the state, or 0.
                t = state.get("t", 0)
                # Get the gradient of loss with respect to p.  
                grad = p.grad.data  
                # Update weight tensor in-place
                # Here, learning rate shrinks over time, as t grows.  This idea is called learning rate decay schedule.
                # p is an nn.Parameter tracked by Autograd engine of PyTorch. The optimizer is not part of the computation graph.
                # The optimizer should modify the weights directly, not create new graph nodes to keep memory consumption minimal. 
                # Updating p.data performs an in-place change to the underlying tensor values without recording that update for gradient computation. 
                # We want to update parameters without tracking gradients. 
                # In Modern PyTorch, we use the block "torch.no_grad()", which is a way to tell PyTorch Autograd that "I am intentionally modifying parameters, but do not record these operations"
                with torch.no_grad():
                    # This no_grad() block updates the parameter tensor p in-place, avoids creating new graph nodes and preserves Autograd's version counters and consistency checks.
                    p.data -= lr / math.sqrt(t + 1) * grad 
                # Increment iteration number 
                state["t"] = t + 1 
        return loss 