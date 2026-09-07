# SiLU activation

Current source: `cs336_basics/nn/activation.py`.

## Definition

`SiLU(x) = x σ(x)`, with `σ(x)=1/(1+e^-x)`. It is also called Swish with fixed coefficient one. The implementation is exactly `x * torch.sigmoid(x)`, elementwise, so input shape and dtype-shaped output are unchanged.

## How it behaves

- For large positive `x`, `σ(x)≈1`, so SiLU behaves like `x`.
- At `x=0`, output is zero and slope is `1/2`.
- For large negative `x`, output approaches zero from below.
- It is smooth and slightly non-monotonic on the negative side, unlike ReLU's hard zero and kink.

Derivative:

`d SiLU/dx = σ(x) + x σ(x)(1-σ(x))`.

At `x=-1,0,1`, approximate outputs are `-0.269, 0, 0.731`. Negative values are attenuated rather than all discarded.

## Code walkthrough

- `import torch`: supplies the stable sigmoid tensor operation.
- `SiLU(nn.Module)` makes the stateless activation composable with other
  modules; it has no learned parameters or buffers.
- `sigmoid_x = torch.sigmoid(x)`: creates a gate in `(0,1)` for each coordinate.
- `return x * sigmoid_x`: gates the original value elementwise.

`torch.nn.functional.silu` may use a fused implementation and is normally preferable in performance-sensitive code. The expanded version is valuable for learning and testing the formula.

## Role in this repository

SwiGLU applies SiLU to one learned projection and multiplies it by a second learned projection. SiLU therefore controls a data-dependent gate; it is not merely placed between two ordinary linear layers.

## Interview questions

**Why can SiLU help compared with ReLU?**  It is smooth, retains small negative signals, and provides input-dependent soft gating. Whether it wins is empirical and architecture-dependent.

**Is SiLU zero-centered?**  Not exactly. It allows negative outputs, but its output distribution need not have zero mean.

**Does SiLU saturate?**  Its sigmoid gate saturates, but the positive branch remains approximately linear, so the full activation does not saturate for large positive inputs.

**What is the memory/performance concern?**  A literal sigmoid followed by multiplication can create an intermediate and two kernels. A fused primitive can reduce memory traffic and launch overhead.

## Senior interview depth

The derivative can be negative for sufficiently negative inputs, which explains
the small non-monotonic basin; SiLU is not simply a smooth ReLU. Near zero,
`SiLU(x) ≈ x/2 + x²/4`, so it initially transmits half the local linear signal.
For large negative `x`, both output and derivative approach zero; for large
positive `x`, output and derivative approach `x` and one respectively.

In the current model SiLU appears inside SwiGLU, where its output is multiplied
by an independently learned up projection. Consequently, initialization and
activation scale affect a product of two branches. A useful test compares both
values and gradients with `torch.nn.functional.silu`, including extreme values
and low-precision dtypes.
