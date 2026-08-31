# Linear layer without bias

Source: `LinearModule/LinearLayerWithoutBias.py` and `LinearLayerWithoutBiasEinops.py`.

## What it does

For every input vector `x ∈ R^D_in`, the layer computes

`y = x Wᵀ`, where `W ∈ R^(D_out × D_in)`.

The leading dimensions are preserved: `(..., D_in) -> (..., D_out)`. There is no bias. A Transformer uses this primitive for Q/K/V/output projections, the three SwiGLU projections, and the language-model head.

## Code walkthrough

- Imports: `math` supplies the initialization scale; `torch` supplies tensors; `nn.Module` gives registration, device transfer, state dictionaries, and call hooks.
- `class Linear(nn.Module)`: makes the object composable inside a model.
- `super().__init__()`: initializes PyTorch's module registries. Omitting it prevents correct parameter/submodule registration.
- `self.weight = nn.Parameter(torch.empty(out_features, in_features, ...))`: allocates `W` in PyTorch's conventional `(output, input)` order and marks it trainable. `empty` is uninitialized, so the following initialization is mandatory.
- `sigma = sqrt(2/(D_in + D_out))`: Xavier-style scale intended to keep signal variance controlled across the layer.
- `nn.init.trunc_normal_(..., std=sigma, a=-3*sigma, b=3*sigma)`: fills weights from a normal distribution with extreme values clipped by resampling outside three standard deviations.
- `return x @ self.weight.T`: transposes `(D_out,D_in)` to `(D_in,D_out)` and performs the projection across the last input axis.
- The einops version uses `einsum("... d_in, d_out d_in -> ... d_out")`. It expresses axis intent explicitly but computes the same mapping.

## Worked shape example

Let `x.shape = (2, 3, 4)` and `W.shape = (6, 4)`. Each of the `2×3` vectors of length four is multiplied by `Wᵀ.shape = (4,6)`, producing `(2,3,6)`. Parameter count is `6×4=24`.

For `x=[1,2]` and `W=[[3,4],[5,6],[7,8]]`, `y=[11,17,23]` because each output is a dot product with one row of `W`.

## Why transpose the weight?

Storing rows as output neurons makes `weight[o]` the complete receptive vector for output `o`, matches `torch.nn.Linear`, and makes checkpoint copying straightforward. The source comment suggesting `(4,3) @ (2,3)` is valid is inaccurate: the executable code correctly uses `weight.T`, so `(4,3) @ (3,2) -> (4,2)`.

## Complexity and numerical details

- Parameters: `D_in D_out`; add `D_out` only if a bias exists.
- Work for `N` input vectors: approximately `2 N D_in D_out` FLOPs when one multiply and one add are counted separately.
- Activation output memory: `N D_out`; backward also needs data sufficient to calculate `dX` and `dW`.
- Truncated initialization limits rare extreme weights, but its main goal is variance propagation—not regularization.
- `einsum` is not inherently faster. Benchmark compiled execution on the target device; both may lower to similar kernels.

## Interview questions

**Why is `nn.Parameter` necessary?**  Assignment of a `Parameter` to a `Module` registers it, so optimizers, `.to(device)`, and `state_dict()` discover it. A plain tensor is not a trainable registered parameter.

**Derive the gradients.**  If `Y=XWᵀ` and upstream gradient is `G`, then `dX=GW`, and after flattening all leading axes, `dW=GᵀX`.

**Why omit bias in many modern LLM projections?**  Bias adds little representational value when surrounding normalization and learned projections already provide offsets/scales, while removing it saves parameters and operations. It is an architectural choice, not a theorem.

**Why does initialization scale matter?**  If weights are too large, activation and gradient variance can grow through depth; if too small, signals shrink. Fan-aware initialization targets a controlled variance.

**Is a linear layer actually linear?**  With no bias, yes: `f(ax+by)=af(x)+bf(y)`. With bias it is affine, though ML convention still calls it linear.

## Revision checklist

You should be able to state the weight layout, perform the shape calculation, derive `dX/dW`, explain `Parameter`, and distinguish mathematical equivalence from kernel performance.

