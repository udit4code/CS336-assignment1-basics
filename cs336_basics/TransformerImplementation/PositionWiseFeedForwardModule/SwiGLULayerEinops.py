import torch
import torch.nn as nn
from einops import einsum

from cs336_basics.TransformerImplementation.LinearModule.LinearLayerWithoutBiasEinops import LinearEinops

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
        # Einstein notation: g_i = Summation over j (x_btj * W1_ij) = x_btj * W1_ij, where i is the free index and j is the repeated index , which maps to d_model.
        # Here, j appears once in x and once in W1, therefore it is summed over.
        # The remaining dimension is d_ff.
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
        # Einstein notation: u_i = Σ_d x_d W³_id = Summation over j over (x_btj * W3_ij), where i is the free index and j is the repeated index , which maps to d_model.
        # Again, j for d_model is the repeated index, and therefore it is summed away.
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
        # Einstein notation: y_d = Σ_i hidden_i W²_di = Summation over j over (hidden_btj * W2_dj), where d is the free index and j is the repeated index , which maps to d_ff.
        # Here, j for d_ff is repeated, therefore it is summed over.
        # Remaining dimension: d_model
        # Output Shape: (..., d_model) = (batch_size, seq_len, d_model)

        return einsum(
            hidden,
            self.down_proj.weight,
            "... d_ff, d_model d_ff -> ... d_model",
        )