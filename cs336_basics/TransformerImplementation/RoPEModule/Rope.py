import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even."

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # k = 0,1,...,d_k/2-1. Why ? 
        # Say, d_k = 8 and then, torch.arange(0, 8, 2) = [0, 2, 4, 6] . These are not indices into the tensor. Instead, they correspond to 
        # pair-indices. Eg : Say, one query vector looks like q = [q0, q1, q2, q3, q4, q5, q6, q7]. Then, we have 8/2 = 4 pairs, which are 
        # pair 0 with frequency_index = 0 : (q0, q1) , pair 1 with frequency_index = 2: (q2, q3), 
        # pair 2 with frequency_index = 4: (q4, q5) and pair 3 with frequency_index = 6: (q6, q7). Therefore, we have 4 pairs.
        # We need only 1 angle per pair, which is mapped via frequency_index : an even integer. Every value in freq_seq represents one 2D rotation block, not one individual embedding position.
        freq_seq = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )

        # Say, theta = 10000 and d_k = 8. Then, freq_seq/d_k = [0, 2, 4, 6]/8 = [0, 0.25, 0.5, 0.75]. 
        # inv_freq = 1 / (10000 ** [0, 0.25, 0.5, 0.75]) = [1, 1/10, 1/100, 1/1000] = [1, 0.1, 0.01, 0.001]. 
        # What do these frequencies mean ? The 1st pair (q0, q1) uses frequency 1, the 2nd pair (q2, q3) uses frequency 0.1, and so on.  
        # For a given position, different pairs rotate at different speeds. Every pair of dimensions for an embedding at a given position is viewed as a clock, with its unique revolutions_per_second. 
        # Lower dimensions rotate quickly, while Higher dimensions rotate slowly : Together, they encode both short-term and long-term positional information. 
        inv_freq = 1.0 / (theta ** (freq_seq / d_k))

        # positions = [0,1,...,max_seq_len-1]
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        # θ_{i,k} = i / theta^(2k/d_k) : The angle θ_{i,k} depends on token position i and embedding pair index k, both of which are independent of each other.
        # Why did we do an outer product ? To understand it, inv_freq = [1, 0.1, 0.01, 0.001] with theta = 10000 and d_k = 8 and positions = [0, 1, 2, 3] for a 4-token sequence. 
        # So, all we are doing is positions x inv_freq. positions is a (1 x 4) row vector and inv_freq is also a 1 x 4 row_vector. 
        # We want to capture every combination of i and k. 
        #      1     0.1    0.01   0.001
        # 0
        # 1
        # 2
        # 3
        # So, we need to multiply every row by every column. The result is : 
        # [ 
        #  [0 * 1,   0 * 0.1,   0.01 * 0,   0 * 0.001],
        #  [1 * 1,   0.1 * 1,   0.01 * 1,   0.001 * 1],
        #  [2 * 1,   0.1 * 2,   0.01 * 2,   0.001 * 2],
        #  [3 * 1,   0.1 * 3,   0.01 * 3,   0.001 * 3]
        # ]
        # For this, we have to do outer product. So, θ_{i,k} = position_{i} * inv_freq_{k}
        # Shape of angles = (max_seq_len x 1) @ (1 x d_k/2) = (max_seq_len x d_k/2) = (max_seq_len, d_k/2)
        # In tensor angles (a 2d matrix), each row corresponds to a token position, and each column corresponds to one pair of embedding dimensions that will be rotated together.
        angles = torch.outer(positions, inv_freq)
        
        
        # Why use register_buffer for cosine and sine values of angles ? 
        # Every tensor insider an nn.Module falls into one of 3 categories : 
        # 1. Parameters : learned during trainin. For them, we use nn.Parameter because we want gradient descent to update it.
        # 2. Buffers : part of the model's state, but not learned (heance, has no gradients).
        # 3. Ordinary attributes : just vanilla Python variables
        # Here, torch.cos(angles) and torch.sin(angles) should not be learned and optimized by gradient descent. 
        # Instead, we want them to be reused in every forward pass, so that we don't recompute it. That is why, we store it in some cache for reuse : Caching 101. 
        # But, why not do self.cos_cached = torch.cos(angles) ? Say, we decide to do model.to("cuda"), due to which we move the model to GPU. 
        # In this process, only nn.Parameter objects moves, while self.cos_cached stays on the CPU. As a result, we end up getting : "Expected all tensors to be on the same device".
        # Therefore, when we do register_buffer("cos_cached", torch.cos(angles)), PyTorch knows that this tensor cos_cached belongs to the module and unlike parameters, it need not be optimized, BUT, it needs to be moved to GPU too. 
        # Moreover, when we do torch.save(model.state_dict(), "model.pt") , what gets saved ? Only nn.Parameter objects and buffer objects, while ordinary attributes do not get saved.
        # Also, because cos_cached is registered as a buffer, cos_cached.requires_grad = False always, so that it does not get updated during backpropagation.
        self.register_buffer(
            "cos_cached",
            torch.cos(angles),
            # Why did we put persistent=False? Because we want to move this tensor to device and make it accessible via cos_cached, but at the same time, we do not want it to include in state_dict. 
            # Because cos_cached and sin_cached are deterministic: given theta, d_k, and max_seq_len, we can always recompute them in __init__. 
            # Saving large lookup tables would only make checkpoints bigger without adding information.
            persistent=False,
        )

        # So, for RoPE, buffers are used because the cosine and sine tables are model-owned tensors that should travel with the model during execution, but are not trainable parameters.
        self.register_buffer(
            "sin_cached",
            torch.sin(angles),
            persistent=False,
        )


    # KEY IDEA : The RoPE paper describes a giant block-diagonal rotation matrix, but nobody actually constructs/materializes that matrix. 
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        # x shape: (..., seq_len, d_k) = (batch_size, seq_len, d_k) = (2, 3, 8). 
        # So, x has 2 batches and each batch has 3 embedding. Each embedding is a 8-dimensional vector.  
        # Eg : one token embedding can be visualised as = [x0, x1, x2, x3, x4, x5, x6, x7], which gets grouped into pairs : (x0, x1) , (x2, x3), (x4, x5) and (x6, x7). 
        # Each of these pairs gets rotated independently.
        
        # Shape of cos_cached and sin_cached = seq_len x d_k/2 . 
        # Each row corresponds to a token position and each column corresponds to an embedding pair. 
        # Say, token_positions = [[5, 6, 7], [2, 3, 4]]. 
        # Because PyTorch performs advanced indexing, under the hood, we end up getting : [ cos_cached[5], cos_cached[6], cos_cached[7], cos_cached[2], cos_cached[3], cos_cached[4] ], stacked together. 
        # So, shape of seq_cos = (batch_size, seq_len, d_k/2) -> meaning : exactly one cosine value for every embedding pair in a given token vector within a chosen batch. 
        seq_cos = self.cos_cached[token_positions]
        seq_sin = self.sin_cached[token_positions]

        # (..., seq_len, d_k/2) -> This means slicing. 
        # Say, x = [10, 20, 30, 40, 50, 60, 70, 80]
        # Here, x[..., 0::2] = Within x, start at 0 index and update step by 2 = chosen indices are 0, 2, 4, 6 = [10, 30, 50, 70]
        # And, x[..., 1::2] = Within x, start at 1 index and update step by 2 = chosen indices are 1, 3, 5, 7 = [20, 40, 60, 80]
        # So, we have separated even_coordinates and odd_coordinates.
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply 2D rotation. How ? 
        # Say, our 2-d column vector is [x, y]. Then, R[x, y] = [x', y'] where, x' = x cos_theta - y sin_theta and y' = x sin_theta + y cos_theta  
        # So, in the code below, we are literally implementing the matrix multiplication formula, without explicitly creating the huge block diagonal rotation matrix.
        # Say, d_k = 8. Then, the full block diagonal rotation 8 x 8 matrix is as shown below : 
        # c0 -s0  0   0   0   0   0   0
        # s0  c0  0   0   0   0   0   0
        # 0   0   c1 -s1  0   0   0   0
        # 0   0   s1  c1  0   0   0   0
        # 0   0   0   0   c2 -s2  0   0
        # 0   0   0   0   s2  c2  0   0
        # 0   0   0   0   0   0   c3 -s3
        # 0   0   0   0   0   0   s3  c3
        # Here, we find that most entries are 0. So, this matrix is sparse, which leads to wastage of space in memory. It worsens further as we go for higher d_k. 
        # So, fo d_k = 128, the matrix is 128 x 128 , which leads to 16384 entries. Out of these, each block contributes 4 numbers and there are d_k/2 = 64 such blocks. 
        # So, non-zero entries = 4 x 64 = 256. So, zero-entries = 16384 - 256 = 16128 . More than 98% of the matrix is empty. Constructing it would waste memory and computation.
        # Hence, we compute rotated_even and rotated_odd, whose time complexity is O(d_k).
        rotated_even = (
            x_even * seq_cos
            - x_odd * seq_sin
        )
        rotated_odd = (
            x_even * seq_sin
            + x_odd * seq_cos
        )

        # We still need memory for the output tensor. What we avoid is allocating and multiplying by the enormous rotation matrix.
        # For output, we prefer to create a new tensor, because it is safer and simpler to debug, without any interference from PyTorch's autograd. 
        # Had we opted for explicit rotation matrix, then, for d_k = 128, the rotation matrix would be of shape (128 , 128), with each cell being float32. 
        # So, for a given token position, memory = d_k x d_k x size of float32 = 128 x 128 x 4 Bytes = 64 kB 
        # Every token position has a different rotation matrix. For max_seq_len = 4096, we would need = 4096 x 64 kB = 2^12 x 2^6 kB = 2^18 kB = 256 MB 
        # So, for just rotation matrices, we need 256 MB. 
        # With the current implementation, where we do not materialize the rotation matrix, we use cos_cached ans sin_cached, each of which is (max_seq_len, d_k/2) 
        # So, memory = max_seq_len x d_k/2 x 2 x size of float32 = 4096 x 128/2 x 2 x 4 B = 2^12 x 2^7 x 2^2 = 2^21 B = 2 MB 
        # So, we have reduced runtime memory for computation by : 256 / 2 = 128 -> 100 times ! 
        
        # But, torch.empty_like(x) creates an empty tensor , whose shape is (batch_size, max_seq_len, d_k). Its memory = 8 x 4096 x 128 x 4 B = 2^(3 + 12 + 7 + 2) = 2^24 = 16 MB 
        # This allocation is unavoidable because the output of RoPE has to be a new tensor. Every output in a neural network produces an output tensor. 
        # The expensive object is not the out tensor, it is the materialized rotation matrix. Had we opted for the materialized rotation matrix approach, how many FLOPs would we need ? 
        # d_k x d_k multiply-adds per token. So, 128 x 128 = 2^14 = 16384 multiply-adds per token = 16384 x 2 FLOPS = 32768 FLOPS per token. 
        # A multiply-add operation is a x b + c (also known as fused multiply addition), which modern GPU can perform as a single hardware instruction.
        # Why ? Because matrix multiplication, under the hood, does : a_1 x b_1 + a_2 x b_2 + a_3 x b_3 + ... . This can be viewed as : 
        # sum = 0
        # sum = a_1 x b_1 + sum 
        # sum = a_2 x b_2 + sum 
        # sum = a_3 x b_3 + sum 
        # ... and so on. So, every line is a single multiply + add , which can be done in 1 FLOP via Fused Multiply Add (FMA instruction). Each FMA consumes 2 FLOPs. 
        # When we opt for the current implementation, per token, for each pair, we would need 2 multiplications + 1 addition for x' and y' = 4 multiplications + 2 additions. Per token, We have d_k/2 = 128/2 = 64 pairs. 
        # So, we would need : 64 x (4 multiplications + 2 additions) = 256 multiplications + 128 additions. Each multiplication and addition individually consume 1 FLOP. 
        # Thus, per token, we have : 256 x 1 + 128 x 1 = 384 FLOPs
        out = torch.empty_like(x)

        out[..., 0::2] = rotated_even
        out[..., 1::2] = rotated_odd

        return out