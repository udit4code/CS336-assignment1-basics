# Pre-norm Transformer block

Current source: `cs336_basics/nn/transformer_block.py`.

## Computation

This decoder block uses two pre-normalized residual branches:

```text
h = x + MHSA(RMSNorm_attn(x), positions)
y = h + SwiGLU(RMSNorm_ffn(h))
```

Every visible tensor has shape `(B,S,D)`, which is required for both residual additions.

## Constructor walkthrough

- `attention_norm`: a distinct RMSNorm scale for the attention input.
- `attention`: causal multi-head self-attention with RoPE.
- `ffn_norm`: a second RMSNorm. It must not be shared accidentally; attention and FFN inputs need independently learned scales.
- `feed_forward`: SwiGLU with intermediate width `F`.
- Passing `device`/`dtype` through constructors creates parameters directly in their intended placement/precision.

## Forward walkthrough

- `normalized_x = attention_norm(x)`: conditions the branch input but leaves the identity residual `x` untouched.
- `attention_output = attention(normalized_x,token_positions)`: mixes information across allowed positions and returns model width.
- `residual_after_attention = x + attention_output`: creates the first identity shortcut.
- `normalized_residual = ffn_norm(residual_after_attention)`: normalizes the updated stream before the feature transformation.
- `ffn_output = feed_forward(normalized_residual)`: transforms each token independently.
- `output = residual_after_attention + ffn_output`: second identity shortcut.

## Why residual connections matter

If a branch computes `F(x)`, residual output is `x+F(x)`, whose Jacobian is `I+J_F`. The identity term gives gradients a direct path and lets a branch learn a small correction rather than rebuilding the entire representation. Shapes must match; hence attention output and FFN down projection both return `D`.

## Pre-norm versus post-norm

Pre-norm applies normalization before each branch and preserves a clean residual highway. Post-norm applies it after residual addition. Pre-norm is typically easier to optimize at depth, while exact stability and representation tradeoffs depend on architecture and initialization.

## Parameter estimate

Ignoring norm scales, attention has `4D²` weights and SwiGLU has `3DF`; norms add `2D`. Thus one block has `4D² + 3DF + 2D` parameters in this repository.

## Interview questions

**Where does token mixing happen?**  In self-attention. The FFN mixes feature channels but acts separately at every token position.

**Why are there two norms?**  They normalize different residual states before different learned branches and have separate learned scales.

**Does normalization alter the residual path?**  In pre-norm, only the branch input is normalized; the value added through the identity path is not.

**Where would dropout normally appear?**  Depending on the architecture, on attention probabilities and/or branch outputs before residual addition. This assignment block intentionally has none.

**What makes this a decoder block?**  Its self-attention is causal, preventing access to future tokens. It has no encoder-decoder cross-attention.

## Senior interview depth

For branch `x_{l+1}=x_l+F(Norm(x_l))`, the residual Jacobian includes an
identity path. Pre-norm therefore improves optimization at depth, but it does
not guarantee bounded residual-stream magnitude: branch outputs accumulate
across layers. Final normalization controls the scale seen by the LM head.

The block has no dropout, stochastic depth, learned residual scaling, or bias.
Its two sequential residual dependencies prevent attention and FFN from running
in parallel. Parallel-block variants evaluate both from a shared normalized
input, trading a different architecture for potential execution advantages.

Activation checkpointing can discard branch intermediates and recompute them in
backward, exchanging compute for memory. Tensor parallelism typically shards
attention heads/projections and FFN width, while pipeline parallelism partitions
blocks. Residual and normalization placement determine where communication and
precision-sensitive reductions occur.
