import torch
import torch.nn as nn
from jaxtyping import Integer
from torch import Tensor

# An embedding is indexed row selection, not a hash-table lookup. It can be
# written as dense one-hot matrix multiplication for illustration:
# one_hot = F.one_hot(token_ids, vocab_size)
# embeddings = einsum(one_hot, self.weight,  "... v, v d -> ... d")


# Dense one-hot multiplication does work proportional to vocabulary size and
# materializes a large one-hot tensor. Direct indexing touches only requested rows.
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # We could have used self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device, dtype=dtype)
        # But, instead, we will implement embedding layer from scratch.
        # nn.Embedding provides the same core lookup plus options such as
        # padding_idx, sparse gradients, max_norm, and frequency scaling.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # torch.empty allocates uninitialized storage; initialization follows immediately.
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        # Initialize every embedding row from the assignment's truncated normal.
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    # Why LongTensor ?
    # Token IDs must use an integer indexing dtype; this project supplies int64.
    def forward(self, token_ids: Integer[Tensor, "..."]) -> torch.Tensor:
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

# Why not matrix multiplication? An embedding layer is fundamentally row selection.
# Suppose our vocabulary has 50,000 tokens.
# If token 57 appears (which maps to the word "love" in vocabulary), we already know exactly which row we need: row 57.
# Dense one-hot multiplication would process all 50,000 rows; indexing selects
# row 57 directly. Backpropagation accumulates gradients into selected rows.
# A token ID represents a vocabulary token, which may be a byte or subword—not necessarily a word.
# The embedding matrix is just another learnable parameter of the model.
