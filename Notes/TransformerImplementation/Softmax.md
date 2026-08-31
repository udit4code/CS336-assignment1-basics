# Numerically stable softmax

Source: `SoftmaxModule/Softmax.py`.

## Definition

Along a chosen axis, `softmax(x)_i = exp(x_i) / Σ_j exp(x_j)`. The output is positive and sums to one, so attention can treat each row as weights over keys.

## Code walkthrough

- `max_values = torch.max(x, dim=dim, keepdim=True).values`: finds one maximum per softmax slice. `keepdim=True` preserves a singleton axis for broadcasting.
- `shifted_x = x - max_values`: makes the largest input zero and all others non-positive.
- `exp_x = torch.exp(shifted_x)`: the largest exponential is now one, preventing overflow from large positive logits.
- `sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)`: calculates the normalizer with a broadcast-compatible shape.
- `return exp_x / sum_exp`: normalizes every slice.

Subtracting a constant does not change the result because the common `exp(-c)` factor cancels between numerator and denominator.

## Worked example

For `[1000,1001,1002]`, direct exponentiation overflows in limited precision. Subtract `1002` to get `[-2,-1,0]`; exponentiate to about `[0.1353,0.3679,1]`; divide by `1.5032` to obtain `[0.0900,0.2447,0.6652]`.

For causal attention, blocked scores are set to `-∞`; their exponentials become zero. If an entire row is blocked, however, subtracting a maximum of `-∞` produces undefined `-∞ - (-∞)` and NaNs. Valid attention masks should leave at least one key available per query.

## Mathematics and gradients

The Jacobian couples all outputs:

`∂p_i/∂x_j = p_i(δ_ij - p_j)`.

Increasing one logit raises its own probability and lowers the others. Softmax is invariant to adding the same scalar to every element of a slice, but not to multiplying all logits: scaling changes distribution sharpness.

## Complexity and practical notes

- Time is linear in the number of elements, but it requires reductions and multiple memory passes.
- In mixed precision, frameworks often perform reductions in higher precision.
- Cross-entropy should generally use a fused log-softmax formulation; separately materializing probabilities is less stable and wastes memory.
- The `dim` argument is semantic. Attention normalizes over keys (`-1`), never across batches or heads.

## Interview questions

**Why divide attention scores by `sqrt(d_k)` before softmax?**  Dot-product variance grows with `d_k`; scaling keeps logits in a range where softmax is not excessively sharp and gradients remain useful.

**Why does max subtraction work?**  Softmax is translation invariant, and selecting the maximum ensures every exponent's argument is at most zero.

**What is softmax temperature?**  `softmax(x/T)`: smaller `T` sharpens the distribution; larger `T` flattens it.

**Why use `-inf` for masks?**  Its exponential is exactly zero in the ideal arithmetic, assigning blocked locations zero probability.

