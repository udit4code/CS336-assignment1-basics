import torch
import torch.nn as nn


# RMSNorm scales by root mean square without subtracting the mean. Unlike
# LayerNorm, it does not center activations and this implementation has no bias.
class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Learnable gain parameter g
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 : Save original dtype (float16/bfloat16/etc.)
        in_dtype = x.dtype

        # Upcast low-precision inputs before squaring to reduce overflow and rounding error.
        x = x.to(torch.float32)

        # Step 3 : Compute RMS over the last dimension
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

        # Step 4 : Normalize via broadcasting
        x = x / rms

        # Step 5 : Apply learnable gain
        x = x * self.weight

        # Step 6 : Downcast back to original dtype
        return x.to(in_dtype)


# Say, x.shape = (2, 3, 4), which can be interpreted as batch_size = 2 (we have a batch of 2 sentences), seq_len = 3 (each sequence has a maximum length of 3), d_model = 4 (each sequence is made of 4-dimensional vectors).
# So, we can have x as something like (after passing through Embedding Layer):
# Sentence 1 :
# [
#  [1 2 3 4]
#  [5 6 7 8]
#  [9 10 11 12]
# ]
# Sentence 2 :
# [
#  [13 14 15 16]
#  [17 18 19 20]
#  [21 22 23 24]
# ]
# Here, each row is a token-embedding generated via table lookup.
# In 1st step, we save in_dtype = x.dtype, so that we can restore the output back in the end.
# Upcasting makes overflow and rounding much less likely: for example, 300^2
# exceeds float16's finite range but is representable in float32.
# In 3rd step, we compute RMS over the last dimension (d_model) for each token embedding (by the flag dim=-1).
# So, for one token embedding [1, 2, 3, 4], we have : square = [1, 4, 9, 16] -> mean = (1 + 4 + 9 + 16)/4 = 7.5 -> rms = sqrt(7.5) = 2.7386127875258306.
# With keepdim=True, the RMS shape is (batch_size, seq_len, 1).
# In 4th step, we do x = x/rms via broadcasting. So, x (batch_size, seq_len, d_model) / rms (batch_size, seq_len, 1) = x (batch_size, seq_len, d_model).

# In 5th step, we do x = x * self.weight via broadcasting. Why ? So that every token embedding in every batch uses the same learnable gain vector.
# There is one learnable parameter per feature dimension, shared across all tokens and all examples.
# So, x (batch_size, seq_len, d_model) * self.weight (d_model,) = x (batch_size, seq_len, d_model).


# Finally, we convert back to the original dtype (float16/bfloat16/etc.) and return the output.
