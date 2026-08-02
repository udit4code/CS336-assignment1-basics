import math

import torch
import torch.nn as nn
from einops import einsum


class LinearEinops(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            self.weight,
            "... d_in, d_out d_in -> ... d_out",
        )
        
# Why did we use einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out") instead of x @ self.weight.T ?
# 1. The einsum function provides a more flexible and readable way to express tensor operations
# 2. The einsum function allows us to specify the desired output shape using the ellipsis notation, which can be more intuitive than using the @ operator and transposing the weight matrix. 

# For example : x = [1, 2, 3] which is a 1 x 3 row vector and it can be interpreted as a (3, ) tensor. 
# Similarly, self.weight = [[w11, w12, w13], [w21, w22, w23]] which is a 2 x 3 matrix (as, in_features = 3 and out_features = 2) and it can be interpreted as a (2, 3) tensor.  
# Now, row points to output neurons and column points to input neurons. We want the output to be a 1 x 2 row vector, which is a (2, ) tensor. 
# This is exactly, what is y = x @ self.weights.T 
# Now, Let us ignore numbers and instead, perceive x and W based on their indices. 
# For x, we can index each element via i and for W, we can index each element via o and i (where, o = output neuron index and i = input neuron index) 
# Then, output is y = Summation (i) of (x[i] * W[o, i]) for each o. Here, we find that index i appears twice in the expression, which means we are summing over it. 
# So, it translates to einsum(x, W, "i, oi -> o"). 
# But, here we have einsum(x, W, "... d_in, d_out d_in -> ... d_out"). It seems to be slightly different though. Why ? 
# Because, we have to consider the scenario when we have batches. 
# For example, if we have a batch of 4 samples, then x will be a 4 x 3 matrix (4 samples, each with 3 input features).
# So, we can index each element of x via b and i (where, b = batch index and i = input neuron index) and for W, we can index each element via o and i (where, o = output neuron index and i = input neuron index). 
# In this case, the batch dimension should simply pass through, without being touched at all. Via broadcasting, (4, 3) @ (2, 3) = (4, 3) 
# So, we write "... d_in", where "..." means that "we don't care how many leading dimensions there are." 
# Thus, "... d_in, d_out d_in -> ... d_out" means : 
# for each leading dimension (batch dimension), we want to sum over the input neuron index (d_in) and produce an output tensor with the same leading dimensions and the output neuron index (d_out). 
# So, einsum(x, W, "... d_in, d_out d_in -> ... d_out") is a more general and flexible way to express the same operation as x @ self.weight.T, especially when dealing with batches of data. 
# So, if x is a batched input as (batch_size, seq_len, in_features) and self.weight is (out_features, in_features), then einsum will correctly compute the output as (batch_size, seq_len, out_features).

# Unlike the explicit matrix multiplication using @ operator where we use transpose, einsum doesn't need an explicit transpose because the index names already tell it how to align dimensions. 