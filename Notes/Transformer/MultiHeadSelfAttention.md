# Multi-head causal self-attention

Current source: `cs336_basics/nn/multihead_attention.py`.

## Purpose and shape story

Self-attention lets each token construct a content-dependent mixture of tokens in the same sequence. Multiple heads perform different learned similarity/aggregation operations in parallel.

```text
x                         (B,S,D)
Q,K,V projections         three × (B,S,D)
split D = H×K             (B,S,H,K)
transpose                 (B,H,S,K)
RoPE(Q), RoPE(K)          (B,H,S,K)
attention per head        (B,H,S,K)
transpose + merge         (B,S,D)
output projection         (B,S,D)
```

The constructor asserts `D % H == 0`, sets `K=D/H`, creates four `D->D` bias-free projections, and creates a RoPE object sized for one head. Four dense projection matrices give `4D²` parameters.

## Forward walkthrough

- Unpacking `batch_size, seq_len, _ = x.shape` enforces a rank-three input
  structurally; there is no explicit last-dimension validation.
- The three projections calculate `Q=XWqᵀ`, `K=XWkᵀ`, and `V=XWvᵀ`.
- `view(B,S,H,K)` partitions the final `D` coordinates into heads without
  changing values, and transpose produces `(B,H,S,K)`.
- RoPE rotates Q and K when enabled. The RoPE object is still constructed when
  disabled, which is semantically harmless but unnecessary setup.
- A lower-triangular boolean `(S,S)` mask allows the diagonal and past. It
  broadcasts across batch and heads.
- Attention returns `(B,H,S,K)`. Transpose, `contiguous`, and `view` restore
  `(B,S,D)`, after which `W_o` mixes concatenated head channels.

## Worked shape example

For `B=2,S=3,D=8,H=2`, `K=4`. Q/K/V each remain `(2,3,8)`, split to `(2,3,2,4)`, transpose to `(2,2,3,4)`, generate scores `(2,2,3,3)`, and return head outputs `(2,2,3,4)`. Merging restores `(2,3,8)`.

At query position one, the mask row is `[True,True,False]`; that token can use positions zero and one but not position two. This is what makes next-token training autoregressive rather than allowing target leakage.

## Why multiple heads?

A single head produces one attention distribution per query. Heads have distinct projection subspaces, so they can specialize in different relationships. Splitting keeps total projected width `D`; it does not multiply Q/K/V width by `H`.

The output projection is essential because concatenation alone leaves head channels isolated. `W_o` learns how their results interact before returning to the residual stream.

## Memory, performance, and design limitations

- Projection work is `O(BSD²)`; attention work is `O(BHS²K)=O(BS²D)`.
- Scores/weights occupy `O(BHS²)` memory here.
- The causal mask is rebuilt on every forward call. It could be cached as a buffer or avoided using a fused attention primitive with `is_causal=True`.
- Q/K/V can be one fused `D->3D` projection for fewer kernel launches, then split.
- Autoregressive generation should cache past K/V; this implementation recomputes the full prefix every call and has no KV-cache API.
- `.reshape` could replace `.contiguous().view`; it copies when required, but explicit contiguous conversion teaches the memory-layout issue clearly.

## Interview questions

**Why must `D` be divisible by `H`?**  This implementation partitions the feature dimension into equal-width heads, so `K=D/H` must be integral.

**Does adding heads necessarily add parameters?**  Not when total model width stays `D`: the four full projection matrices remain `D×D`. Head count changes their partitioning.

**What is the difference between self- and cross-attention?**  Self-attention derives Q/K/V from the same sequence. Cross-attention derives queries from one sequence and keys/values from another.

**Why call `contiguous()` after transpose?**  The desired merged axis order is usually incompatible with the transposed strides; `view` promises no copy and cannot invent the needed physical layout.

**What is a KV cache?**  During generation, store prior keys and values and compute only those for the new token. It eliminates repeated K/V projection of the prefix, though attention over cached positions remains.

## Senior interview depth

Splitting into heads changes the inductive structure, not the total width: each
head forms its own `S×S` probability matrix in a learned `K`-dimensional
subspace. Concatenation followed by `W_o` is equivalent to applying learned
linear combinations across all head outputs; without it, subsequent residual
features would remain tied to fixed head partitions.

For full-sequence training, projections cost roughly `8BSD²` FLOPs for Q/K/V/O
combined, while the two attention contractions cost roughly `4BS²D`. During
single-token cached decoding, new-token projection stays `O(D²)`, cache reads
and attention are `O(SD)`, and KV memory grows `O(LBSD)` across layers. At long
contexts, memory bandwidth for reading the cache is often central.

Multi-query attention shares K/V across all query heads; grouped-query attention
uses an intermediate number of K/V heads. Both reduce cache size and bandwidth
at a possible quality trade-off. This repository implements standard MHA, has
no padding mask or dropout, rebuilds its causal mask each call, and recomputes
the entire prefix during generation.

High-value tests check causality by perturbing future tokens, head split/merge
ordering, RoPE on/off behavior, reference equivalence, non-contiguous layouts,
and gradients for `B != H`—a shape choice that catches accidental broadcasting.
