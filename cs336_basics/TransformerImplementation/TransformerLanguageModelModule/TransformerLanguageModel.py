import torch
import torch.nn as nn

from cs336_basics.TransformerImplementation.EmbeddingModule.EmbeddingLayer import (
    Embedding,
)
from cs336_basics.TransformerImplementation.LinearModule.LinearLayerWithoutBias import (
    Linear,
)
from cs336_basics.TransformerImplementation.RMSNormModule.RMSNormLayer import (
    RMSNorm,
)
from cs336_basics.TransformerImplementation.TransformerBlockModule.TransformerBlock import (
    TransformerBlock,
)


# DOUBT 1 : Why did we use nn.ModuleList(...) for the layers of transformer blocks ? Why not use a python list instead ?

# To understand this question, let us build from first principles.  
# When we build a Linear Layer by inheriting the base class nn.Module, we specify self.weight as nn.Parameter(..). 
# Why nn.Parameter(...) ? Because PyTorch walks through the model and collects every parameter. Internally, model.parameters() does something like this : 

# for attribute in self.__dict__:
#     if isinstance(attribute, Parameter):
#         yield attribute
#     if isinstance(attribute, Module):
#         recurse(attribute)

# This is pure recursive code. So, PyTorch recursively explores paramters and modules.
# Now, let us come back to the architecture of the Transformer LM. It consists of multiple Transformer Blocks. 
# One block contains RMSNorm, Attention, SwiGLU, etc., each of them being a nn.Module object under the hood. 
# So, if we perceive the Transformer Model as a computation graph, we end up with something like this : 
# TransformerLM
# └── TransformerBlock
#         ├── Attention
#                 ├── q_proj.weight
#                 ├── k_proj.weight
#                 ├── ...
# Here, PyTorch recursively discovers everything. 
# Suppose, we have N blocks, with N or num_layers = 32. The obvious idea that comes to our mind is : 
# We can write the following for loop : 

# layers = []
# for i in range(num_layers):
#     layers.append(TransformerBlock(...))

# It looks fine on the surface-level. But, layers is only a Python list, and PyTorch knows nothing about Python lists. 
# So, during parameter discovery via recursive traversal, when PyTorch does for attribute in self.__dict__: for layers, 
# it finds that layers is a Python list, not a nn.Module and thus, it stops the recursion. So, it never enters the list. 

# TransformerLM
# └── layers  (ordinary Python list) -> END 

#         block0 -> not traversed further
#         block1 -> not traversed further
#         block2 -> not traversed further 
# .... and so on till block31.
# Hence, PyTorch never discovers the blocks block0, block1, block2, ... block31 as layers is a python list. 
# What is the consequence of this behaviour by PyTorch ? Say block0 has 100 million parameters. So, for 32 blocks,
# we have 3200 million parameters, which disappear because PyTorch doesn't recurse further ! PyTorch thinks that the model 
# has only Embedding, Final RMSNorm and LM head. 

# As a result, optimizer will never update these 3200 million parameters. Even worse, when we do model.to("cuda"), it only 
# moves the registered modules. Those inside the python list stay on CPU and we end up with runtime error : "Expected all tensors to be on same device". 
# Similarly, torch.save(model.state_dict()) will omit every transformer block.

# All these problems can be solved, if we use a nn.ModuleList instead of a python list. A nn.ModuleList is itself an extension of nn.Module.
# So, under the hood, it looks something like : 
# TransformerLM
#         │
#         ▼
# ModuleList
#         │
#         ├────────────┐
#         ▼            ▼
#  Transformer0   Transformer1
#         │            │
#         ▼            ▼
#  Attention     Attention
#         │            │
#         ▼            ▼
#      q_proj       q_proj
# So, now, recursion works and PyTorch is able to reach every parameter inside every Transformer Block. 
# Moreover, unlike Python list which can contain enything, nn.ModuleList ensures that everything inside it is an object of type nn.Module. 



# DOUBT 2 : Why not use nn.Sequential instead of nn.ModuleList ? 

# The reason is that Sequential assumes each module receives the previous module's output as its only input:
# x -> Layer1 -> Layer2 -> Layer3. 
# But, our transformer block needs 2 inputs : x and token_positions during forward pass.
# Since token_positions must be passed to every block, we need an explicit loop. 
# Thus, ModuleList is therefore the appropriate container: it registers the layers while leaving us in control of the forward-pass logic.

# KEY TAKEAWAY : We can think of nn.Module as a directory in a filesystem. 

# TransformerLM/
# │
# ├── token_embedding/
# ├── layers/
# │      ├── 0/
# │      ├── 1/
# │      ├── 2/
# │      └── ...
# ├── final_norm/
# └── lm_head/

# A regular Python list is like a cardboard box sitting on the floor:
# TransformerLM/
# │
# ├── token_embedding/
# ├── cardboard_box   ← PyTorch ignores its contents
# ├── final_norm/
# └── lm_head/

# A ModuleList is like a properly indexed folder in the directory tree. Because it's an nn.Module, PyTorch can descend into it, discover every TransformerBlock, 
# and consequently find every RMSNorm, attention projection, and feed-forward weight. 
# That's why ModuleList is the standard choice whenever we have a variable number of submodules, as in a stack of Transformer blocks.

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=theta,
                    max_seq_len=context_length,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        # We want token_ids to always be a LongTensor. 
        assert token_ids.dtype in [torch.long, torch.int64], f"Expected token_ids.dtype to be torch.long, but got {token_ids.dtype}."
        # Step 1 : Get the token embeddings for a sequence of token_ids
        x = self.token_embedding(token_ids)

        batch_size, seq_len = token_ids.shape

        # Step 2 : Construct position index tensor for the entire batch. 
        # Why ? RoPE doesn't learn positional embeddings; instead it needs to know which position is occupied by each token, so that it can rotate the Query and Key vectors accordingly. 
        # Say, s1 = "I love pizza" and s2 = "Cats chase mice". Now, after tokenizatiom, it becomes [[12, 45, 78], [91, 22, 17]]. 
        # So, token_ids is a tensor with shape (2, 3), where 2 is the batch_size and 3 is the seq_len. 
        # RoPE only cares that "I" is position 0, "love" is position 1, and so on.It doesn't care about token_id of tokens.
        # So, instead of [[12, 45, 78], [91, 22, 17]], what we need is [[0, 1, 2], [0, 1, 2]].
        # Now, torch.arange(seq_len) for seq_len=5 gives Tensor([0, 1, 2, 3, 4]), whose shape is (5,) -> 1 x 5 row vector. 
        # Then, we apply .unsqueeze(0) . Why ? Because currently, it doesn't have a batch-dimension. So, by .unsqueeze(0), we add a dimension at index 0. 
        # As a result, the shape becomes (1, 5) instead of (5,) . Now, torch.arange(seq_len).unsqueeze(0) = Tensor([[0, 1, 2, 3, 4]]). 
        # After that, we apply .expand(batch_size, seq_len). Why ? Because each sentence needs identical positions and there are 3 sentences in a given batch. 
        # So, torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len) gives a (3, 5) Tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]). 
        token_positions = (
            torch.arange(
                seq_len,
                device=token_ids.device,
            )
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        # Why did we use expand() instead of repeat() ? 
        # expand creates a view that behaves as if the single row were repeated, without allocating new memory. repeat actually copies the data.
        # Since every batch element shares the same position indices, expand is sufficient and more memory-efficient. 
        # It's a common PyTorch pattern whenever identical data needs to be broadcast across a batch.

        # Step 3 : Pass through all transformer blocks
        for layer in self.layers:
            x = layer(
                x,
                token_positions,
            )

        # Step 4 : Apply RMSNorm on x after passing through layers
        x = self.final_norm(x)
        
        # Step 5 : Apply a final linear projection that converts each token's hidden representation (B, S, d_model) into vocabulary scores (B, S, vocab_size). 
        # So, we can have one score for every possible next token.
        logits = self.lm_head(x)

        return logits