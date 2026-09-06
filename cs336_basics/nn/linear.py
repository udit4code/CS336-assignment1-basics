import math
import torch
import torch.nn as nn


# nn.Module registers Parameters and child Modules assigned as attributes. That
# registration powers model.parameters(), device/dtype moves, train/eval mode
# propagation, and state_dict serialization. It is a module hierarchy, distinct
# from the dynamic autograd graph built while forward executes.
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

        # Weight rows correspond to output features. Wrapping the uninitialized
        # tensor in Parameter registers it as trainable module state.
        self.weight = nn.Parameter(
            # torch.empty is safe here because initialization follows immediately.
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        sigma = math.sqrt(2.0 / (in_features + out_features))
        # The assignment uses a zero-mean truncated normal with a fan-aware
        # standard deviation. Random rows break output-unit symmetry; truncation
        # limits extreme initial weights. Bounds a and b are absolute values.
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
