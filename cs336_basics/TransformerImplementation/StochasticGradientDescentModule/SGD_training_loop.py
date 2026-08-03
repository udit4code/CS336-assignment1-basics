import torch 

from cs336_basics.TransformerImplementation.StochasticGradientDescentModule.SGD import StochasticGradientDescentOptimizer
 
weights = torch.nn.Parameter(5 * torch.randn((10, 10))) 
optimizer = StochasticGradientDescentOptimizer(
    params=[weights], 
    lr=1
) 

for t in range(10):
    # We Reset the gradients for all learnable parameters 
    optimizer.zero_grad() 
    # Compute scalar loss 
    loss = (weights**2).mean()
    print(f"loss value in iteration {t + 1} : {loss.item()}\n")
    # Run backward pass 
    loss.backward()
    # Run Optimizer step
    optimizer.step()