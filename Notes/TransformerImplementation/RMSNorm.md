# RMSNorm

Source: `RMSNormModule/RMSNormLayer.py` and `RMSNormLayerWithReduce.py`.

## Equation and contract

For each final-axis vector `x ∈ R^D`:

`rms(x) = sqrt((1/D) Σ_i x_i² + ε)`

`y_i = g_i x_i / rms(x)`

where learned scale `g` starts at ones. Shape remains `(...,D)` and the module has `D` parameters.

## Code walkthrough

- Constructor stores `eps` and creates `weight = nn.Parameter(torch.ones(D,...))`.
- `in_dtype = x.dtype`: remembers the external activation dtype.
- `x = x.to(torch.float32)`: squares and reductions are vulnerable to overflow/rounding, so the repository computes them in FP32.
- `torch.mean(x*x, dim=-1, keepdim=True)`: calculates mean square independently for every token vector.
- `torch.sqrt(... + eps)`: converts mean square to RMS; epsilon prevents division by zero.
- `x / rms`: normalizes the vector's magnitude.
- `self.weight * normalized_x`: broadcasts the learned coordinatewise scale across all leading axes.
- `.to(in_dtype)`: restores the caller-facing dtype.

The alternative file replaces `torch.mean` with `einops.reduce(x*x, "... d_model -> ... 1", "mean")`. It is mathematically equivalent and makes the axis contract readable; it is not automatically an optimization.

## Worked example

For `x=[3,4]`, mean square is `(9+16)/2=12.5`, RMS is about `3.536`, and normalized output before learned scale is `[0.849,1.131]`. Its mean square is approximately one. Unlike LayerNorm, its mean is not forced to zero.

## RMSNorm versus LayerNorm

| Property | RMSNorm | LayerNorm |
|---|---|---|
| Centers by subtracting mean | No | Yes |
| Scales by magnitude/variance | RMS | standard deviation |
| Learned affine values here | scale only | commonly scale and bias |
| Reductions per vector | mean square | mean and variance |

RMSNorm is approximately invariant to positive rescaling: `RMSNorm(cx)≈RMSNorm(x)` when epsilon is negligible. It is not invariant to adding a constant because it does not center.

## Numerical and implementation details

- Epsilon is inside the square root in this implementation. Moving it outside changes behavior near zero.
- Upcasting is particularly important because squaring can overflow FP16 even when original values are representable.
- Multiplication by the parameter may promote dtype; the final cast makes the API explicit.
- Norm placement is decided at block level. This repository is pre-norm: normalization occurs before attention and before the FFN.

## Interview questions

**Why can pre-norm Transformers train deeply?**  The residual path retains a direct identity route for activations and gradients, while each nonlinear branch receives normalized input.

**What information does RMSNorm preserve that LayerNorm removes?**  It does not remove the vector's mean component; it only controls root-mean-square magnitude.

**Why is `keepdim=True` needed?**  It leaves the reduced axis at length one, allowing direct broadcasting back across `D`.

**Is RMSNorm parameter-free?**  No. The normalization statistic is nonlearned, but `g ∈ R^D` is learned.

