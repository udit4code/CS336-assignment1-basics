# Position-wise feed-forward network: SwiGLU

Current source: `cs336_basics/nn/feed_forward.py` and `feed_forward_einops.py`.

## Equation

For each token vector independently:

`SwiGLU(x) = W_down [ SiLU(W_gate x) ⊙ (W_up x) ]`

With the repository's weight convention, each displayed mathematical map corresponds in code to `x @ W.T`. Shapes are:

```text
x                         (..., D)
gate = W_gate(x)          (..., F)
up   = W_up(x)            (..., F)
SiLU(gate) * up           (..., F)
down projection           (..., D)
```

“Position-wise” means the same weights operate independently at every `(batch, sequence)` location. Tokens mix in attention; features mix in the FFN.

## Code walkthrough

- Three bias-free custom `Linear` modules are constructed: gate `D->F`, up `D->F`, and down `F->D`.
- `gate = self.gate_proj(x)`: learns which intermediate features should open or close.
- `gate = gate * sigmoid(gate)`: applies SiLU explicitly.
- `up = self.up_proj(x)`: creates content to pass through the gate.
- `hidden = gate * up`: elementwise multiplicative interaction. This is the defining gated part.
- `return self.down_proj(hidden)`: mixes intermediate features back into model space so a residual addition with `x` is valid.

The einops version directly contracts `x` with each module's weights using named
`einsum` axes. That bypasses their `forward` methods but shares the same
registered parameters. The down projection is `d_ff -> d_model`; reversing
those dimensions would make the residual addition impossible.

## Worked example

Let `D=2`, `F=2`, identity gate/up weights, identity down weight, and `x=[1,-1]`. Then gate and up are both `[1,-1]`; `SiLU(gate)≈[0.731,-0.269]`; elementwise multiplication gives `[0.731,0.269]`; the down projection leaves it unchanged. Notice how multiplication lets the sign and magnitude of one learned path control another.

## Parameters and compute

- Parameter count without biases: `DF + DF + FD = 3DF`.
- Leading matrix-multiply work: about `6NDF` FLOPs for `N` tokens, plus elementwise activation/gating.
- Compared with a two-matrix FFN at the same `F`, SwiGLU has 50% more projection parameters. Architectures often reduce `F` to compare at similar parameter/compute budgets.
- Large intermediate tensors make fusion and memory bandwidth important in production.

## Interview questions

**Why is the FFN necessary if attention already mixes information?**  Attention mixes information across token positions; the FFN performs a learned nonlinear transformation across features at each position. They serve complementary axes.

**What makes GLU-style layers more expressive than a single activation?**  Two independent learned projections interact multiplicatively, allowing input-dependent feature selection.

**Why must the down projection return `D`?**  The block adds the FFN result to the residual stream, whose last dimension is `D`.

**Are token positions processed in a Python loop?**  No. Batched matrix multiplication applies identical position-wise logic to all tokens in parallel.

## Senior interview depth

SwiGLU introduces a multiplicative interaction between two affine subspaces.
At fixed `F`, it uses three matrices instead of the two in a conventional FFN;
for equal parameter budget against a `D -> 4D -> D` FFN, a common rough choice
is `F ≈ 8D/3`, often rounded for hardware alignment. That ratio is a budgeting
heuristic, not part of the SwiGLU definition.

For `N=BS` tokens, projection activation storage includes two `N×F` branches
and the gated intermediate unless kernels or autograd rematerialization reduce
it. Tensor parallel implementations commonly shard the two expanding
projections consistently, perform their elementwise product locally, then
reduce across the down projection.

The einops version reads registered weights directly instead of invoking each
`LinearEinops.forward`. This is mathematically valid here but would silently
bypass future behavior such as hooks, parametrizations, quantization wrappers,
or a bias added to those submodules. That is a useful code-review distinction
between equivalent equations and equivalent module semantics.
