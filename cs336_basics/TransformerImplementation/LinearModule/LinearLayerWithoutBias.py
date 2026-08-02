import math
import torch
import torch.nn as nn

# Why do we implement Linear using nn.Module as a base class ? 
# Because nn.Module is the base class for all neural network modules in PyTorch. It provides a convenient way to define and manage the parameters of the module, as well as to implement the forward pass. 
# By inheriting from nn.Module, we can easily integrate our Linear layer into larger neural network architectures and take advantage of PyTorch's automatic differentiation and optimization features. 
# By using nn.Module as base class, we imply that "Any object that inherits from me is part of a neural network". 
# Now, PyTorch can automatically walk through your model, as it can perceive the model as a graph/tree. 
# With nn.Module and nn.Paramater, every parameter of the model is registered and can be accessed using model.parameters() or model.named_parameters(). 
# Also, during Hardware acceleration when we write model.to(device), all the parameters of the model are moved to the specified device (CPU or GPU) as PyTorch walks through the module tree.
# Moreover, when we want to save a model's checkpoint, we write torch.save(model.state_dict(), PATH) and when we want to load a model's checkpoint, we write model.load_state_dict(torch.load(PATH)). 
# How does it work ? PyTorch recursively walks through the module tree and saves/loads the state_dict of each module.
# nn.Module allows us to construct models as nested modules. For example : 
# GPT contains 12 Transformer blocks, each Transformer block contains a MultiHeadAttention module and a FeedForward module, and each of these modules can contain other sub-modules.
# PyTorch can recursively traverse this tree to perform operations like collecting parameters, moving tensors to devices, switching training/evaluation modes, and saving/loading state.
class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Why did we use nn.Parameter here and use it to wrap torch.empty(...) ? 
        # 1. torch.empty(...) creates a tensor with uninitialized values. 
        # We made it output_features x input_features because we want to multiply the input with the weight matrix to get the output.
        # 2. nn.Parameter is a special kind of tensor that is automatically registered as a parameter of the module. This means that when we call model.parameters(), this weight tensor will be included in the list of parameters that will be optimized during training. 
        # If we didn't use nn.Parameter, the weight tensor would not be considered a parameter of the module and would not be updated during training.   
        # nn.Parameter also has the property that it will be moved to the appropriate device (CPU or GPU) when we call model.to(device). 
        # With the sign (nn.Parameter), PyTorch knows this tensor is one of the model's learnable parameters and includes it in gradient computation, checkpointing, and optimization.
        self.weight = nn.Parameter(
            # torch.empty(...) just creates a tensor with garbage values. We will initialize it properly in the next step.
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        sigma = math.sqrt(2.0 / (in_features + out_features))
        # nn.init.trunc_normal_ initializes the weight tensor with values drawn from a truncated normal distribution. 
        # Why not initialize with all zeros ? Because if we initialize all weights to zero, all neurons in the layer will learn the same features during training, leading to poor performance.
        # We want to break the symmetry and allow different neurons to learn different features. Random initialization helps achieve this by giving each neuron a different starting point in the weight space.
        # Our goal is to fill every weight in this layer with a small random number drawn from a zero-centered Gaussian, but reject unusually large values so the network starts training in a stable state.
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Why x @ self.weight.T ? 
        # Say, in_features = 3 and out_features = 2. 
        # Then, it means that we have 3 input features (3 input neurons : x1, x2, x3) and we want to transform them into 2 output features (2 output neurons : y1, y2).
        # So, y1 = x1 * w11 + x2 * w12 + x3 * w13 and y2 = x1 * w21 + x2 * w22 + x3 * w23.
        # Here, w11, w12, w13 are the weights connecting input neurons to the first output neuron, and w21, w22, w23 are the weights connecting input neurons to the second output neuron. 
        # So, organizing y1 and y2 as matrix-vector multiplication with y = [y1, y2] , which is 1 x 2 row vector and x = [x1, x2, x3] which is 1 x 3 row vector, 
        # we can write y = x @ W^T, where W is the weight matrix of shape (out_features, in_features).   
        # So, W^T is the weight matrix of shape (in_features, out_features).
        return x @ self.weight.T