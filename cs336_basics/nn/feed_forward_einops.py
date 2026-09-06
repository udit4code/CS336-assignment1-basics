import torch
import torch.nn as nn
from einops import einsum

from .linear_einops import LinearEinops


class SwiGLUEinops(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.gate_proj = LinearEinops(d_model, d_ff)
        self.up_proj = LinearEinops(d_model, d_ff)
        self.down_proj = LinearEinops(d_ff, d_model)

    def forward(self, x):
        # In Gate Projection, we have :
        # x has Shape: (..., d_model) = (batch_size, seq_len, d_model) = (b, t, d_model)
        # and W1 has Shape (d_ff, d_model)
        # Einstein notation: g[b,t,i] = sum_j x[b,t,j] W_gate[i,j].
        # The repeated d_model axis j is reduced; b, t, and d_ff axis i remain.
        # Result Shape: (..., d_ff) = (batch_size, seq_len, d_ff)
        ####################################################################
        gate = einsum(
            x,
            self.gate_proj.weight,
            "... d_model, d_ff d_model -> ... d_ff",
        )

        gate = gate * torch.sigmoid(gate)

        # In Up Projection, we have :
        # x has Shape: (..., d_model) = (batch_size, seq_len, d_model) = (b, t, d_model)
        # and W3 has Shape (d_ff, d_model).
        # u[b,t,i] = sum_j x[b,t,j] W_up[i,j], again reducing d_model.
        # Output Shape: (..., d_ff)
        up = einsum(
            x,
            self.up_proj.weight,
            "... d_model, d_ff d_model -> ... d_ff",
        )

        hidden = gate * up

        # In Down Projection, we have :
        # hidden has Shape: (..., d_ff) = (batch_size, seq_len, d_ff) = (b, t, d_ff) and
        # W2 has Shape: (d_model, d_ff)
        # y[b,t,d] = sum_i hidden[b,t,i] W_down[d,i]. The d_ff axis i
        # is reduced and d_model axis d remains.
        # Output Shape: (..., d_model) = (batch_size, seq_len, d_model)

        return einsum(
            hidden,
            self.down_proj.weight,
            "... d_ff, d_model d_ff -> ... d_model",
        )
