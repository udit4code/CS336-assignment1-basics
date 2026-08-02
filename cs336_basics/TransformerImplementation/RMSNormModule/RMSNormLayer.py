import torch
import torch.nn as nn

# We could have implemented it using torch.nn.RMSNorm, but we will implement it from scratch instead. 
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

        # Step 2 : Upcast to a higher datatype for numerical stability
        x = x.to(torch.float32)

        # Step 3 : Compute RMS over the last dimension
        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True)
            + self.eps
        )

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
# In 2nd step, we upcast x to float32 for numerical stability. Why ? For example, 300^2 doesn't fit into float16, so it will overflow and give us garbage values. 
# So, by shifting to a bigger datatype float32, we make x * x safe. 
# In 3rd step, we compute RMS over the last dimension (d_model) for each token embedding (by the flag dim=-1). 
# So, for one token embedding [1, 2, 3, 4], we have : square = [1, 4, 9, 16] -> mean = (1 + 4 + 9 + 16)/4 = 7.5 -> rms = sqrt(7.5) = 2.7386127875258306.
# With this step, the shape becomes (batch_size, seq_len, 1) instead of (batch_size, seq_len, d_model). 
# In 4th step, we do x = x/rms via broadcasting. So, x (batch_size, seq_len, d_model) / rms (batch_size, seq_len, 1) = x (batch_size, seq_len, d_model).  

# In 5th step, we do x = x * self.weight via broadcasting. Why ? So that every token embedding in every batch uses the same learnable gain vector. 
# There is one learnable parameter per feature dimension, shared across all tokens and all examples.
# So, x (batch_size, seq_len, d_model) * self.weight (d_model,) = x (batch_size, seq_len, d_model). 


# Finally, we convert back to the original dtype (float16/bfloat16/etc.) and return the output. 