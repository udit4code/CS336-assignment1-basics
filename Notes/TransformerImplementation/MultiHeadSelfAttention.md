# Multi-head causal self-attention

Source: `MultiHeadSelfAttentionModule/MultiHeadSelfAttention.py`.

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

## Forward walkthrough by source lines

- Lines 80–82 unpack the expected rank-three input. A malformed input fails naturally during unpacking.
- Lines 85–94 calculate `Q=XWqᵀ`, `K=XWkᵀ`, and `V=XWvᵀ`. These are separate learned views of the same residual representation.
- Lines 96–129 use `view(B,S,H,K)`. No values change; the final `D` coordinates are partitioned into heads.
- Lines 132–144 transpose to `(B,H,S,K)`. This makes `(B,H)` batch-like leading axes for the generic attention function. Transpose normally changes metadata/strides without copying.
- Lines 147–159 rotate Q and K when `use_rope` is true. The RoPE object is currently constructed even when disabled—small unnecessary setup, but no semantic error.
- Lines 161–180 build a lower-triangular boolean `(S,S)` mask on the input device. `True` includes the diagonal and past; `False` blocks future keys.
- Lines 183–193 call scaled dot-product attention. Broadcasting applies the same causal mask to every batch/head.
- Lines 196–213 transpose results to `(B,S,H,K)`, then call `contiguous().view(B,S,D)`. The copy restores logical head ordering in contiguous storage before flattening.
- Lines 216–219 apply `W_o`, which mixes information across the concatenated head coordinates.

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

