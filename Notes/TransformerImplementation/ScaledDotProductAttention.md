# Scaled dot-product attention

Source: `ScaledDotProductAttentionModule/ScaledDotProductAttention.py` and `ScaledDotProductAttentionEinops.py`.

## Core equation

`Attention(Q,K,V) = softmax(QKᵀ / sqrt(K) + mask) V`.

General shape contract permits any leading batch dimensions:

```text
Q (..., Q, K)   K (..., L, K)   V (..., L, Dv)
scores (..., Q, L)
output (..., Q, Dv)
```

Self-attention has `Q=L=S`. In multi-head attention, the leading dimensions are `(B,H)`.

## Code walkthrough

- `torch.matmul(query, key.transpose(-2,-1))`: computes every query-key dot product. Transposing only the last two axes preserves batch/head axes.
- `d_k = query.shape[-1]`: derives the dot-product width from data rather than duplicating configuration.
- `scores / sqrt(d_k)`: controls score variance.
- If a mask exists, `scores.masked_fill(~mask, -inf)` blocks locations. This repository's convention is `True=allowed`; `~mask` selects the forbidden cells.
- `softmax(scores, dim=-1)`: normalizes each query's row across keys.
- `attention @ value`: forms a weighted sum of value vectors for each query.
- The einops equations `... q d, ... k d -> ... q k` and `... q k, ... k v -> ... q v` encode the same contractions without explicit transpose.

## Why scale by `sqrt(K)`?

Assume independent zero-mean query/key coordinates with variance one. A dot product is a sum of `K` products, so its variance is roughly `K` and standard deviation roughly `sqrt(K)`. Dividing by `sqrt(K)` restores order-one scale. Otherwise, larger heads drive softmax toward near one-hot outputs and weak gradients.

## Worked causal example

Suppose one query has scaled scores `[2,1,5]`, but at its current position only the first two keys are legal. Masking gives `[2,1,-∞]`; softmax gives approximately `[0.731,0.269,0]`; output is `0.731 V₀ + 0.269 V₁`. The future value contributes exactly zero.

## Complexity and bottlenecks

- Score and value products cost `O(B H Q L K)` when `Dv≈K`.
- The score/probability matrix requires `O(B H Q L)` memory in this pedagogical implementation—quadratic in sequence length for self-attention.
- PyTorch's fused scaled-dot-product attention/FlashAttention can avoid materializing the full matrix in high-bandwidth memory while computing mathematically equivalent results within floating-point tolerances.
- Mask shape must be broadcastable to scores. A `(S,S)` causal mask broadcasts across batch and heads.

## Interview questions

**Why are Q and K different projections?**  They let the model learn asymmetric “what I seek” and “what I offer” representations. Values carry the content being aggregated.

**Why normalize over keys?**  For each query, the model chooses a distribution over candidate source positions.

**Does masking after softmax work?**  Zeroing probabilities afterward breaks row normalization unless renormalized. Adding `-∞` before softmax is the standard formulation.

**Why is attention quadratic?**  Every one of `S` queries compares with `S` keys, creating `S²` pair scores.

**What does FlashAttention change?**  Primarily the execution strategy and memory I/O through tiling and online softmax, not the conceptual attention equation.

