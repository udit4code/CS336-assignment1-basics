import torch
import torch.nn as nn

from cs336_basics.TransformerImplementation.LinearModule.LinearLayerWithoutBias import Linear


# In Standard Transformer by Vaswani : we used : FFN(x) = W_2 (ReLU (W_1 @ x)) 
# Modern LLMs like LLaMa and PaLM use SwiGLU. 
# SwiGLU(x) = W_down (SiLU(W_gate) hadamard_product (W_up @ x)) 
# where SiLU(z) = z * sigmoid(z) 
# So, now, there are 3 layers : 

#               x
#               │
#      ┌────────┴────────┐
#      │                 │
#  gate_proj         up_proj
#      │                 │
#      ▼                 ▼
#   SiLU(.)          identity
#      │                 │
#      └──── elementwise × ────┐
#                               │
#                           down_proj
#                               │
#                               ▼
#                            output 

# So, x.shape = (batch_size, seq_len, d_model) . Mathematically, g = gate_proj(x) = x @ W_gate.T and u = up_proj(x) = x @ W_up.T . 
# Here, shape of g = (batch_size, seq_len, d_ff) and shape of u = (batch_size, seq_len, d_ff) . 
# After that, we do element-wise multiplication between g and sigmoid(g) to get g = g * sigmoid(g). 
# Shape of g is still (batch_size, seq_len, d_ff). 
# Now, we do down projection , y = down_proj(g * u) = (g * u) @ W_down.T . 
# Shape of y = (batch_size, seq_len, d_model)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # We go from (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) via up_proj.
        self.gate_proj = Linear(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )


        # # We go from (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) via up_proj.
        self.up_proj = Linear(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )


        # We go from (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) via down_proj.
        self.down_proj = Linear(
            d_ff,
            d_model,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 : Compute gate = sigmoid(gate_proj(x)) = sigmoid(x @ W_g.T)
        gate = self.gate_proj(x)
        gate = gate * torch.sigmoid(gate)

        # Step 2 : Compute up = up_proj(x) = x @ W_u.T 
        up = self.up_proj(x)

        # Step 3 : Compute hidden = hadamard product between gate and up = gate * up 
        hidden = gate * up

        # Step 4 : Compute output = down_proj(hidden) = hidden @ W_d.T
        return self.down_proj(hidden)