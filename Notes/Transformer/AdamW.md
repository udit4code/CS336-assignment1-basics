# AdamW optimizer

Current source: `cs336_basics/nn/adamw.py`.

## Update equations

For gradient `g_t`:

```text
m_t = β1 m_(t-1) + (1-β1) g_t
v_t = β2 v_(t-1) + (1-β2) g_t²
m̂_t = m_t / (1-β1^t)
v̂_t = v_t / (1-β2^t)
θ_t = (1 - lr·λ) θ_(t-1) - lr · m̂_t/(sqrt(v̂_t)+ε)
```

The repository algebra folds bias correction into `alpha_t = lr sqrt(1-β2^t)/(1-β1^t)` and applies `alpha_t m/(sqrt(v)+eps)`. This is nearly the displayed conventional form, but epsilon placement is not algebraically identical: conventional rewriting would scale epsilon with `sqrt(1-β2^t)`. Tests may intentionally target the assignment's stated formula.

## Code walkthrough

- Constructor packs learning rate, betas, epsilon, and weight decay into optimizer defaults, then lets the base class create parameter groups.
- Optional closure executes under enabled gradients and returns a recomputed loss.
- For every parameter with a gradient, `self.state[p]` lazily initializes scalar step plus first/second moment tensors `m` and `v` using `zeros_like(p)`.
- `step = previous+1` matters because bias-correction denominators would be zero at step zero.
- Inside `no_grad`, `p -= lr*weight_decay*p` applies decoupled shrinkage.
- Moment updates form exponential moving averages of gradients and squared gradients.
- `p -= alpha_t*m/(sqrt(v)+eps)` divides by a coordinatewise estimate of gradient scale, giving adaptive steps.
- Updated tensors and step are stored back in optimizer state and the optional loss is returned.

## Worked scalar example

Let `p=10`, `g=2`, `lr=.1`, `β1=.9`, `β2=.99`, `λ=.01`, first step. Decay changes `p` by `-.01` to `9.99`. Moments become `m=.2`, `v=.04`. Ignoring epsilon, `alpha=.1*sqrt(.01)/.1=.1`; adaptive update is `.1*.2/.2=.1`, giving approximately `9.89`.

## Adam versus AdamW

Adding `λθ` to Adam's gradient couples regularization to adaptive rescaling. AdamW applies parameter shrinkage separately, so decay strength is not divided coordinatewise by `sqrt(v)`. This is why it is called decoupled weight decay.

## Complexity and memory

- Work is linear in parameter count.
- Optimizer state stores two tensors per trained parameter, usually substantial memory in addition to parameters and gradients.
- Production optimizers fuse elementwise operations, use foreach kernels, shard state, or quantize state.
- Biases and normalization scales are often placed in a zero-weight-decay parameter group; this implementation leaves grouping to the caller.

## Interview questions

**What do first and second moments do?**  The first smooths gradient direction; the second estimates coordinatewise squared magnitude and normalizes step size.

**Why bias-correct?**  Moments start at zero and are biased low in early steps. Dividing by `1-β^t` corrects that initialization bias.

**Why epsilon?**  It prevents division by zero and influences update behavior where estimated variance is tiny.

**Why skip `grad is None`?**  The parameter did not participate in the graph (or gradient was explicitly set to none), so its moments and step should not necessarily advance.

**Is AdamW always better than SGD?**  No. AdamW is usually convenient and robust for Transformers; optimizer choice still interacts with data, scale, scheduling, and generalization.

## Senior interview depth

Adam is invariant to rescaling a coordinate's gradients only approximately:
epsilon, finite-history moments, clipping, and parameter decay break exact
invariance. `β1` controls directional smoothing; `β2` controls the time horizon
of squared-gradient scale. Large `β2` reduces noise but adapts slowly after a
distribution change. Bias correction matters most during those early horizons.

The repository uses `alpha_t = lr·sqrt(1-β2^t)/(1-β1^t)` with denominator
`sqrt(v_t)+eps`. This is not exactly the conventional
`lr·m_hat/(sqrt(v_hat)+eps)` unless epsilon is correspondingly rescaled. That
difference is tiny when `sqrt(v)` dominates epsilon but should be preserved when
matching assignment tests or checkpoints.

Production review: validate `lr`, betas, epsilon, and decay; reject or support
sparse gradients explicitly; use in-place/foreach/fused kernels; avoid replacing
moment tensors each step; and define capturable/differentiable behavior when
needed. Parameter groups commonly exclude biases and normalization scales from
decay. Distributed training may shard the two full-size moment tensors because
they add roughly two parameter-sized states before any FP32 master weights.

A robust test compares several steps—including `grad=None` and multiple groups—
against an equation-level reference, then round-trips `state_dict` and verifies
resume produces exactly the same next update.
