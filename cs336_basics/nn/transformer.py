import torch
import torch.nn as nn

from .embedding import Embedding
from .linear import Linear
from .normalization import RMSNorm
from .transformer_block import TransformerBlock


# Why ModuleList instead of a plain Python list?

# To understand this question, let us build from first principles.
# When we build a Linear Layer by inheriting the base class nn.Module, we specify self.weight as nn.Parameter(..).
# Why nn.Parameter(...) ? Because PyTorch walks through the model and collects every parameter. Internally, model.parameters() does something like this :

# for attribute in self.__dict__:
#     if isinstance(attribute, Parameter):
#         yield attribute
#     if isinstance(attribute, Module):
#         recurse(attribute)

# This pseudocode is only a mental model: nn.Module maintains dedicated
# registries rather than scanning arbitrary attributes recursively.
# Now, let us come back to the architecture of the Transformer LM. It consists of multiple Transformer Blocks.
# Each block contains registered RMSNorm, attention, and SwiGLU submodules.
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
# Hence blocks held only in that plain list are not registered as child modules.
# Their parameters still exist and autograd can compute gradients if forward uses
# them, but model.parameters() will omit them, so a normally constructed optimizer
# will not update them.

# As a result, a standard optimizer will miss those parameters. model.to("cuda") only
# moves the registered modules. Those inside the python list stay on CPU and we end up with runtime error : "Expected all tensors to be on same device".
# Similarly, torch.save(model.state_dict()) will omit every transformer block.

# ModuleList is a Module container that registers every appended child.
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
# Unlike a plain list, ModuleList accepts Module instances and exposes them to
# traversal, device moves, mode changes, and serialization.


# Why not nn.Sequential?

# The reason is that Sequential assumes each module receives the previous module's output as its only input:
# x -> Layer1 -> Layer2 -> Layer3.
# But, our transformer block needs 2 inputs : x and token_positions during forward pass.
# Since token_positions must be passed to every block, we need an explicit loop.
# Thus, ModuleList is therefore the appropriate container: it registers the layers while leaving us in control of the forward-pass logic.

# Revision model: think of registered modules as an indexed directory tree.

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

        # Retain the architectural contract on the model itself. Inference can
        # then validate token IDs and crop rolling context windows without
        # duplicating configuration that could drift from the loaded weights.
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.theta = theta

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
        # Embedding lookup requires integer token IDs; this API requires int64.
        assert token_ids.dtype in [torch.long, torch.int64], (
            f"Expected token_ids.dtype to be torch.long, but got {token_ids.dtype}."
        )
        # Step 1: Map each token ID to its embedding vector.
        x = self.token_embedding(token_ids)

        batch_size, seq_len = token_ids.shape

        # Step 2: Construct one position index per token. RoPE uses these
        # positions to choose rotation angles for query and key vectors.
        # For token_ids [[12, 45, 78], [91, 22, 17]], the position tensor is
        # [[0, 1, 2], [0, 1, 2]]. `unsqueeze(0)` adds the batch axis and
        # `expand` creates a non-copying broadcasted view for all batch items.
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
