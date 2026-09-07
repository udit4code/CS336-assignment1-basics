# Numerically stable softmax

Current source: `cs336_basics/nn/softmax.py`.

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

**What happens if we compose softmax with itself many times—say 100 layers?**

Assume each layer applies an ordinary temperature-one softmax over the same
`n` coordinates with no learned transformation in between:

`p^(l+1) = softmax(p^l)`.

After the first layer, every coordinate lies in `(0,1)` and the coordinates sum
to one. Treating that probability vector as the next layer's logits greatly
reduces its range, so the next softmax is flatter. Repeating this process
converges to the uniform distribution

`p* = (1/n, ..., 1/n)`,

which is a fixed point because `softmax(p*) = p*`. For example,
`softmax([2,1,0]) ≈ [0.665,0.245,0.090]`; applying softmax again gives roughly
`[0.450,0.296,0.253]`, already closer to `[1/3,1/3,1/3]`. After 100 direct
applications, the result is numerically indistinguishable from uniform for
ordinary dimensions and inputs.

The backward signal also vanishes rapidly. Each layer contributes Jacobian
`J = diag(p)-ppᵀ`; composing 100 layers multiplies 100 such Jacobians. Softmax's
Euclidean operator norm is at most `1/2`, and near the uniform fixed point its
nonzero eigenvalues are `1/n`. Differences between coordinates and gradients
are therefore contracted exponentially with depth. The all-ones direction has
zero derivative because softmax is invariant to a common logit shift.

This is not what a 100-layer Transformer does. Its softmax produces attention
weights that multiply V; learned projections, residual connections,
normalization, and nonlinear FFNs intervene before the next block's softmax.
Those softmax operations are not composed directly, so the model is not forced
to converge to a uniform distribution. Temperature or other transformations
also change the repeated-map analysis; the uniform-limit statement above is
specifically for direct, standard softmax composition.

## Senior interview depth

Softmax is the gradient of `logsumexp`; its Jacobian is
`diag(p) - ppᵀ`. The Jacobian has a zero eigenvector in the all-ones direction,
matching translation invariance. When one probability approaches one, most
Jacobian entries approach zero, which is why very large attention logits cause
poor gradient flow.

Max subtraction prevents positive overflow but cannot rescue non-finite input
or an all-`-inf` slice. The current function also relies on PyTorch to validate
`dim`; it does not explicitly upcast reductions. A production implementation
would normally call a fused backend primitive and define behavior for empty
dimensions, mixed precision, and fully masked rows.

Test invariants include non-negativity, unit sums, translation invariance,
agreement with `torch.softmax`, finite gradients, and correct behavior on
non-contiguous tensors and every legal positive/negative dimension index.
