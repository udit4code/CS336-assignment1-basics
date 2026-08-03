import torch
import torch.nn as nn

from cs336_basics.TransformerImplementation.LinearModule.LinearLayerWithoutBias import Linear
from cs336_basics.TransformerImplementation.RoPEModule.RoPE import RotaryPositionalEmbedding
from cs336_basics.TransformerImplementation.ScaledDotProductAttentionModule.ScaledDotProductAttention import (
    scaled_dot_product_attention,
)


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = 10000.0,
        max_seq_len: int = 4096,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.use_rope = use_rope

        assert d_model % num_heads == 0, f"d_model {d_model} is not divisible by num_heads {num_heads}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q,K,V projections : All map from d_model to d_model. So, initially, Q, K, V are all d_model x d_model matrices. 
        # Later we'll reshape into num_heads × d_k
        
        self.q_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )

        self.k_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )

        self.v_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )

        # Output projection : The output projection (W_O) mixes information across different attention heads and maps the concatenated head outputs back into the model dimension (d_model), 
        # producing the representation used by the next Transformer block.
        self.out_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )

    
        # Rotary Position Embedding, which is applied independently to every head.
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        # Input x has Shape (batch_size, seq_len, d_model)
        # So, we extract batch_size and seq_len out of its shape. We already have self.d_model . 
        batch_size, seq_len, _ = x.shape

        
        # Step 1 : Compute Q,K,V
        # Shape of query, key and value : (batch_size, seq_len, d_model). Why ? 
        # Because, q_proj has shape (batch_size, d_model, d_model) and 
        # x (that has shape (batch_size, seq_len, d_model)) @ q_proj.T leads to query with shape (batch_size, seq_len, d_model) 
        # As, our Linear Module stores weight as weight.shape = (out_features, in_features), and so, the forward pass is y = x @ W.T under the hood. 
        # In our Linear Module, x has shape (..., in_features), weight has shape (out_features, in_features) and weight.T has shape (in_features, out_features)
        
        query = self.q_proj(x) # x @ W_q.T
        key = self.k_proj(x) # x @ W_k.T
        value = self.v_proj(x) # x @ W_v.T

        # Step 2 : Split embedding dimension into heads.
        # Before : (batch, seq_len, d_model)
        # After : (batch, seq_len, num_heads, d_k)
        
        # In this step, nothing is being computed. All we are doing is only changing how we interpret the memory. 
        # Say, query.shape = (2, 3, 8) and one of the query token vectors is [11, 12, 13, 14, 15, 16, 17, 18]
        # Here, since num_heads = 2, so, each head gets d_k = d_model / num_heads = 8 / 2 = 4 elements. 
        # So, instead of one long 1 x 8 vector, we get 2 smaller row vectors of shape 1 x 4, 
        # as head_0 = [11, 12, 13, 14] and head_1 = [15, 16, 17, 18]. Key idea : Same Numbers, but just grouped differently. 
        # Thus, we go from (batch_size, seq_len, d_k) to (batch_size, seq_len, num_heads, d_k). 
        # In our case, we went from (2, 3, 8) to (2, 3, 2, 4). Please note that 2 x 3 x 8 = 2 x 3 x 4 x 2 = 48 => Number of elements in the query tensor remained the same. 
        # query.view(...) is extremely cheap, because view() does not allocate any new memory (so, it doesn't involve any expensive malloc or cudamalloc). 
        # query.view(...) only changes the shape and strides (strides decide how indices map to memory). So, view() is essentially O(1). 
        # Same logic for both key.view(...) and value.view(...)
        query = query.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k,
        )

        key = key.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k,
        )

        value = value.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k,
        )

        
        # Step 3 : Move heads before sequence dimension.
        # Shape of query before : (batch, seq, heads, d_k)
        # query.transpose(1, 2) means swap the dimensions 1 and 2, which are seq and heads. 
        # After swapping, shape of query : (batch, heads, seq, d_k)
        # We do this because each attention head should be processed independently. 
        # The scaled dot-product attention function expects its input to be organized as (batch, heads, seq, d_k), so that for every (batch, head) pair it computes a separate attention matrix of shape (seq, seq). 
        # In other words, transpose(1, 2) moves the head dimension next to the batch dimension, effectively treating each head as an independent mini-batch while keeping all computations fully vectorized.
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # NOTE : transpose() is generally O(1) because it does not move or copy any data. It simply creates a new tensor view with different strides.
        # Under the hood, it allocates a tiny new tensor object containing updated metadata (shape, strides, storage pointer) but does not copy the underlying tensor data. 
        # The expensive operation comes later if we need a physically contiguous layout (e.g., before calling view()), at which point contiguous() performs an O(n) memory copy.


        # Step 4 : Apply RoPE on query and key tensors.
        # Incoming query and key tensors have shape : (batch_size, num_heads, seq_len, d_k) 
        # RoPE treats the head dimension as just another batch dimension.
        if self.use_rope:
            query = self.rope(
                query,
                token_positions,
            )

            key = self.rope(
                key,
                token_positions,
            )

        # Step 5 : Build causal mask, on the same device as x.
        # Token i can attend only to j <= i
        # Shape of causal mask : (seq_len, seq_len)
        # torch.ones(seq_len, seq_len) creates a square matrix filled entirely with 1s. 
        # torch.tril(...) ("triangular lower") keeps only the lower triangular part of a matrix (including the main diagonal) and sets everything above the diagonal to zero.
        # So, torch.tril(torch.ones(4, 4)) would lead to a mask tensor as : 
        #  [
        #   [1, 0, 0, 0],
        #   [1, 1, 0, 0],
        #   [1, 1, 1, 0],
        #   [1, 1, 1, 1]
        #  ]
        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )
        )

        
        # Step 6 : Perform scaled dot attention.
    
        # Shapes: Query has shape (batch_size, num_heads, seq_len, d_k)
        # Output of Attention will have shape (batch_size, num_heads, seq_len, d_v) 
        # In standard implementations, if d_k = d_v, then, output will have shape (batch_size, num_heads, seq_len, d_v).
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            mask,
        )

        
        # Step 7 : Restore back sequence_len dimension before num_heads.
        # Before : (batch_size, num_heads, seq_len, d_k)
        # After : (batch_size, seq_len, num_heads, d_k)
        output = output.transpose(1, 2)

        # Step 8 : Merge all heads.
        # Before : (batch_size, seq_len, num_heads, d_k)
        # After : (batch, seq, d_model)
        # Because, we had done transpose() calls earlier, the underlying tensor output became non-contigupus. 
        # But, view() assumes that the elements it need are laid out consecutively in memory. 
        # If we don't do contiguous(), we will get the Runtime error : "view size is not compatible with input tensor's size and stride". 
        # This is because, PyTorch cannot reinterpret the memory as one flat contiguous block.  
        # output.contiguous() creates a new tensor whose memory is physically rearranged into the current logical order. 
        output = output.contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )


        # Step 9 : Final output projection.
        output = self.out_proj(output) # output @ W_out.T 

        return output
    
    
# Doubt : If transpose() can change only the strides, why can't view() also just change the strides? 
# This is because : view() is much more restrictive than transpose(). 
# It can only create a new shape if that shape can be represented by the existing stride pattern. After a transpose, that's often impossible.

# Eg : x = torch.arange(12).view(3, 4) 
# So, memory is [0 1 2 3 4 5 6 7 8 9 10 11], whose shape is (3, 4) and stride is (4, 1), meaning row_jump = 4 and col_jump = 1
# So, x[i][j] = row_jump*i + col_jump*j = 4*i + j. 
# The logical x is : 
# [
# 0 1 2 3
# 4 5 6 7
# 8 9 10 11
# ]
# Now, if we do x.view(2, 6), then, it still works because the memory is still one continuous block. 
# 0 1 2 3 4 5
# 6 7 8 9 10 11
# In this case, after view(2, 6), shape of x is (2, 6) and stride is (6, 1). PyTorch recomputes a new stride that is compatible with the new shape.

# Now, let's check what happens in transpose. 
# Assume x has shape (3, 4)
# Say, y = x.transpose(0, 1). So, y has shape (4, 3) , while x had shape (3, 4).  
# In transpose, we swap the strides. So, original x had shape (3, 4) and stride (4, 1),
# while, transpose y had shape (4, 3) and stride (1, 4).
# Here, during transpose, we notice something interesting : The shape changed because we swapped dimensions, 
# and the stride changed because those dimensions now play different roles. We did not recompute a new contiguous stride—we simply swapped the existing stride values.

# After transpose, The logical tensor for y is : 
# [
# 0 4 8
# 1 5 9
# 2 6 10
# 3 7 11
# ]
# But, the memory is still : 0 1 2 3 4 5 6 7 8 9 10 11 
# Nothing moved. 

# Now, when we do y.view(2, 6), what should the first row be ? 
# Logically, it should be 0 4 8 1 5 9, but the memory is still : 0 1 2 3 4 5 6 7 8 9 10 11 . 
# To obtain [0 4 8 1 5 9], we have to jump from 0 -> by +4 -> 4 -> by +4 -> 8 -> by +4 -> 1 -> by +4 -> 5 -> by +4 -> 9, on the original memory.
# There is no single constant stride that can describe this traversal. 

# Why not make view() copy automatically ? 
# Because view() has a very specific contract: "Return another view of the same storage, without copying."
# If it silently copied data, it would no longer be a view. 
# PyTorch keeps that guarantee explicit. 
# If a copy is required, we either call contiguous() yourself or use reshape(), which will return a view when possible and make a copy only when necessary.
# view() can only succeed if the requested new shape is compatible with the current memory layout and stride pattern.
# After a transpose(), many reshapes (like merging the transposed dimensions back together) require a different physical ordering of elements, which cannot be represented by strides alone. 
# That's when a real memory copy (contiguous()) becomes necessary.

# A good mental model : 

# We can perceive a tensor as 3 pieces of metadata : 
# 1. pointer to memory 2. shape 3. strides 
# view(): Keeps the same memory, changes the shape, and computes a compatible set of strides for that new shape (only if possible).
# transpose(): Keeps the same memory, swaps the shape dimensions, and correspondingly swaps the stride values.
# contiguous(): Allocates new memory, copies the data into contiguous order, and assigns the standard contiguous strides.

# The important difference is that view() can only work if the existing memory layout can be described by the new shape and a valid set of strides. 
# After a transpose(), that's often no longer true, which is why view() may fail unless we first call contiguous().