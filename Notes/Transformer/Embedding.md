# Token embedding

Current source: `cs336_basics/nn/embedding.py`.

## Contract and first principles

An embedding table is `E ∈ R^(V×D)`. Given integer token IDs of shape `(...)`, the layer returns `E[token_ids]` with shape `(...,D)`. Each vocabulary item owns one learned row.

Conceptually, if token `i` were represented by one-hot row vector `e_i ∈ R^V`, then `e_i E = E[i]`. Indexing obtains the same answer without constructing a mostly-zero vector or performing `V×D` arithmetic.

## Code walkthrough

- The class subclasses `nn.Module` and calls `super().__init__()` to participate in PyTorch registration.
- `num_embeddings` is vocabulary size `V`; `embedding_dim` is model width `D`.
- `torch.empty((V,D), device=device, dtype=dtype)` allocates the dense table.
- Wrapping it in `nn.Parameter` makes all rows learnable, even though only selected rows receive nonzero gradients for a given batch.
- `nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3)` initializes each coordinate. This matches the assignment implementation; real architectures often choose a model-specific smaller standard deviation.
- `forward(token_ids)` returns `self.weight[token_ids]`. PyTorch advanced indexing appends the embedding axis.

The source describes this as a hash-table lookup. The useful intuition is “lookup,” but the implementation is more precisely a dense array row gather: token IDs are direct integer row indices, with no hashing or collision handling.

## Worked example

```text
E = [[0.1, 0.2],
     [1.0, 1.5],
     [2.0, 2.5]]
token_ids = [[2, 0], [1, 2]]
output = [[[2.0,2.5], [0.1,0.2]],
          [[1.0,1.5], [2.0,2.5]]]
```

Input shape `(2,2)` becomes `(2,2,2)`. Repeated ID `2` uses the same parameter row; gradients from both occurrences accumulate into `E[2]`.

## Complexity and edge cases

- Parameters: `V D`, often a large fraction of a small language model.
- Forward work and output memory are `O(BSD)`; it does not scan all `V` rows.
- IDs must be integer and satisfy `0 <= id < V`; negative or too-large IDs are invalid for this contract.
- The model later maps hidden vectors back to `V` logits. Weight tying can reuse `E` as the LM-head weight, reducing parameters and coupling input/output representations; this repository creates a separate head.

## Interview questions

**How does embedding lookup backpropagate?**  The gradient with respect to a selected table row is the sum of upstream gradients at every position containing that ID. Unselected rows receive zero gradient for that batch.

**Why not use one-hot vectors?**  They require `O(V)` storage per token and a large multiply, whereas gathering directly retrieves the identical row in `O(D)` output work.

**Does an embedding encode position?**  No. This table maps token identity to a vector. The repository supplies position information later by applying RoPE to queries and keys.

**What is weight tying?**  Use the input embedding matrix as the transpose-compatible output vocabulary projection. It saves `VD` parameters and can improve statistical efficiency, but constrains the two roles to share a representation.

**Why do repeated tokens start with identical vectors?**  Lookup is context-free. Attention and feed-forward layers turn those identical initial rows into context-dependent hidden states.

## Senior interview depth

The dense table has `VD` parameters and, with a dense optimizer, equally dense
optimizer state even though a minibatch touches relatively few rows. A gather
has irregular memory access and low arithmetic intensity; embedding throughput
is often bandwidth- or communication-bound rather than FLOP-bound. Large-vocab
systems may shard rows across devices and route token IDs to their owning shard.

Repeated IDs are important in backward: gradients are a scatter-add, so write
collisions must accumulate rather than overwrite. `nn.Embedding(sparse=True)`
can expose sparse gradients, but optimizer support is restricted. This custom
advanced-indexing implementation produces ordinary dense parameter gradients.

Weight tying makes logits `hEᵀ` and saves `VD` parameters. It also ties the
geometry used to encode tokens to the geometry used to score them; it is not
merely a memory trick. Padding support would additionally require deciding
whether a padding row is frozen and ensuring padded targets are excluded from
the loss and attention mask.
