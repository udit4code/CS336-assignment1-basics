import torch
import torch.nn as nn
from jaxtyping import Integer
from torch import Tensor 

# We do not use einops for Embedding layer, because at the end of the day, embedding layer is just a hash-table lookup, not a reshape or a matrix multiplication. 
# Yes, theoretically speaking, we can express embeddings as a one-hot matrix multiplied by the embedding matrix (using einsum), something like : 
# one_hot = F.one_hot(token_ids, vocab_size)
# embeddings = einsum(one_hot, self.weight,  "... v, v d -> ... d")

# But its time complexity is O(batch_size * sequence_length * vocab_size * embedding_dim), which is very inefficient for a large vocabulary size. 
# That is why, every deep learning framework implements embedding layers as a hash-table lookup, not as a matrix multiplication.
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # We could have used self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device, dtype=dtype) 
        # But, instead, we will implement embedding layer from scratch. 
        # Pytorch's own nn.Embedding expects exactly token_ids.dtype == torch.long and the output has shape (batch_size, sequence_length, embedding_dim).
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Here, we just tell PyTorch to reserve some memory for self.weight and it will have garbage values for now. 
        # Later, we will initialize it properly.
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings, 
                embedding_dim,
                device=device, 
                dtype=dtype,
            )
        )
        
        # Now, we will initialise the self.weights 
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )
        
        
    # Why LongTensor ? 
    # The input should be a LongTensor because token IDs are indices, and PyTorch only allows integer tensors (typically torch.int64, i.e., LongTensor) for indexing. 
    # We expect the dataloader or dataset to provide token_ids as torch.long. 
    def forward(
        self,
        token_ids: Integer[Tensor, "..."]
    ) -> torch.Tensor:
        # For every token ID in the input tensor, use it as a row index into the embedding matrix and return the corresponding embedding vector.
        return self.weight[token_ids] 
    
# Why does self.weight[token_ids] work ? 
# Say, our vocabulary is : 
# 0 -> <pad>
# 1 -> I
# 2 -> love
# 3 -> pizza
# 4 -> cats
# Suppose, the embedding dimension is 4. 
# Our Embedding Matrix would look like : 
#         d0    d1    d2    d3
# 0      [0.1, 0.2, 0.3, 0.4]
# 1      [1.0, 1.1, 1.2, 1.3]
# 2      [2.0, 2.1, 2.2, 2.3]
# 3      [3.0, 3.1, 3.2, 3.3]
# 4      [4.0, 4.1, 4.2, 4.3]

# Say, we have token_ids = torch.tensor([1, 2, 3]) which corresponds to the sentence "I love pizza". 
# Then, self.weight[token_ids] is equivalent to : 
# torch.stack([
#     self.weight[1],
#     self.weight[2],
#     self.weight[3],
# ]), which returns : 
# [
#  [1.0, 1.1, 1.2, 1.3],
#  [2.0, 2.1, 2.2, 2.3],
#  [3.0, 3.1, 3.2, 3.3],
# ]

# What happens when we are dealing with a batch of token_ids ? 
# Eg : token_ids = torch.tensor([
#     [123,57,981], # For Sentence 1
#     [ 44,12,100] # For Sentence 2
# ]), whose shape is (batch_size, sequence_length) = (2, 3). Then, self.weight[token_ids] is equivalent to : 
# torch.stack([
#    # For sentence 1
#     torch.stack([
#         self.weight[123], 
#         self.weight[57],
#         self.weight[981],
#      ]),
#    # For sentence 2    
#     torch.stack([
#         self.weight[44],
#         self.weight[12],
#         self.weight[100],
#     ]),
# )

# Why didn't we do a Matrix Multiplication for Embedding Layer ? 
# But an embedding layer is fundamentally a lookup table.
# Suppose our vocabulary has 50,000 tokens.
# If token 57 appears (which maps to the word "love" in vocabulary), we already know exactly which row we need: row 57. 
# Why compute 50000 x 768  multiplications when we can just look up the row? Indexing is much cheaper than matrix multiplication. 
# Initially, row 57 is a random [0.11, -0.42, ...]. During Training, backpropagation updates that row. Eventually, row 57 becomes a useful representation of the word for that position in the vocabulary : "love". 
# The embedding matrix is just another learnable parameter of the model.
