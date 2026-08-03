import math
import torch

from cs336_basics.TransformerImplementation.SoftmaxModule.Softmax import softmax

# torch.matmul() is PyTorch's general-purpose matrix multiplication function, while the @ operator is simply Python syntax that internally calls torch.matmul(). 
# They produce the same result, use the same optimized backend (cuBLAS on GPUs, MKL/OpenBLAS on CPUs), and have identical performance. 
# Unlike elementwise multiplication (*), torch.matmul() automatically handles vector-vector (dot product), matrix-vector, matrix-matrix, and batched matrix multiplication, which makes it ideal for transformer operations like attention. 
# The choice between torch.matmul(A, B) and A @ B is purely one of readability: @ is more concise and closely matches mathematical notation (QKᵀ), 
# whereas torch.matmul() is often preferred in teaching or when function arguments are long and benefit from clearer formatting.

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Args:
        query: Shape = (..., seq_len, d_k. So, it can be (batch_size, seq_len, d_k)
        key: Shape = (..., seq_len, d_k). So, it can be (batch_size, seq_len, d_k)
        value: Shape = (..., seq_len, d_v). So, it can be (batch_size, seq_len, d_v)
        mask: Optional boolean mask with Shape: (seq_len, seq_len)
            True  -> allow attention
            False -> block attention

    Returns: Tensor of shape (..., seq_len, d_v). So, it can be (batch_size, seq_len, d_v)
    """

    # Step 1 : 
    # Compute QKᵀ.
    # Shape : Query = Key = (..., seq_len, d_k) 
    # To compute Q @ K.T, we have to transpose Key over its last 2 dimensions. So, we do : key.transpose(-2,-1) and end up with the shape (..., d_k, seq_len)
    # Result of Q @ K.T has shape = (..., seq_len, seq_len), as ignoring batch_size which is same for both, (seq_len, d_k) x (d_k, seq_len) = (seq_len, seq_len) = (batch_size, seq_len, seq_len)
    # Each element (i,j) in Q @ K.T is q_i * k_j
    scores = torch.matmul(
        query,
        key.transpose(-2, -1),
    )


    # Step 2 :Scale the attention scores.
    # Paper: QKᵀ / sqrt(d_k)
    # This prevents extremely large dot products, which would make Softmax saturate.
    # Shape of query is (batch_size, seq_len, d_k); so query.shape(-1) = d_k
    d_k = query.shape[-1]
    scores = scores / math.sqrt(d_k)

  
    # Step 3 : Apply attention mask.
    # Wherever mask == False, replace the score with -∞.
    # as, softmax(-∞) = 0, so those positions receive zero attention.
    if mask is not None:
        # For tensor y, y.masked_fill(mask, value) returns a new tensor where every value of y corresponding to a True value in the boolean mask is replaced with value. 
        # Elements where the mask is False are left unchanged. The input tensor y is not modified (unless you use the in-place version masked_fill_()).
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    # Step 4 : Convert scores into probabilities, where every row sums to one.
    # Shape of attention : (..., seq_len, seq_len) = (batch_size, seq_len, seq_len). 
    # Under the hood, all it does is apply exp on each cell and divide it by row-sum or along the chosen dim. 
    # So, it alters each cell, while keeping the shape intact.
    attention = softmax(
        scores,
        dim=-1,
    )


    # Step 5 :
    # Compute the weighted sum of Value vectors.
    # Attention has shape (..., seq_len, seq_len) = (batch_size, seq_len, seq_len)
    # Value has shape (..., seq_len, d_v) = (batch_size, seq_len, d_v)
    # Result has shape (..., seq_len, d_v) = (batch_size, seq_len, d_v); as keeping batch_size dimensions fixed, (seq_len, seq_len) @ (seq_len, d_v) = (seq_len, d_v) = (batch_size, seq_len, d_v)
    # Every output vector is basically : Summation over j for  attention(i,j) * value(j)
    output = torch.matmul(
        attention,
        value,
    )

    return output