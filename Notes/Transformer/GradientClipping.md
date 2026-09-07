# Global L2 gradient clipping

Current source: `cs336_basics/nn/gradient_clipping.py`.

## Algorithm

Treat all parameter gradients as one conceptual flattened vector `g`. Compute

`||g||₂ = sqrt(Σ_p Σ_i g_(p,i)²)`.

If the norm exceeds threshold `M`, replace every gradient by

`g_p <- g_p · M/(||g||₂+ε)`.

All tensors receive the same scalar factor, so the global gradient direction is preserved while magnitude is capped.

## Code walkthrough

- Initialize the squared-norm accumulator.
- Iterate parameters and skip `grad is None`.
- Sum every `p.grad**2`, then take one square root after all parameters. Summing individual parameter norms would be mathematically wrong for a global norm.
- Return early when `total_norm <= max_l2_norm`; clipping must never amplify a small gradient.
- Calculate scale with epsilon and multiply each existing gradient by that shared scale.

## Worked example

If two parameter gradients flatten to `[3,4]` and `[0,12]`, global norm is `sqrt(9+16+144)=13`. With maximum five, scale is `5/13`; gradients become approximately `[1.154,1.538]` and `[0,4.615]`, whose combined norm is five.

## Training order

```text
optimizer.zero_grad()
loss.backward()
unscale gradients if using automatic mixed precision
gradient_clipping(model.parameters(), max_norm)
optimizer.step()
```

Clipping before backward has nothing to clip; clipping after step is too late. With gradient accumulation, normally clip after all microbatches have accumulated.

## Implementation caveats

- The function accepts any iterable and immediately materializes it as a tuple,
  so a one-shot `model.parameters()` generator is safe across both passes.
- Gradients are scaled in place with `parameter.grad.mul_(scale)` inside
  `no_grad`, avoiding replacement of the gradient buffers.
- Parameters on multiple devices cannot be added into one device-local tensor without coordination.
- Accumulation starts as Python `0.0` and becomes a tensor on the first gradient. Empty gradients leave a float and still work for the early return, though validation would be clearer.
- Non-finite norms deserve explicit handling in production.

## Interview questions

**Value clipping versus norm clipping?**  Value clipping clamps each coordinate and changes direction. Norm clipping applies one factor and preserves direction.

**Does clipping solve the cause of exploding gradients?**  It limits immediate damage but may hide bad initialization, unstable precision, excessive learning rates, or faulty data.

**Why global rather than per-parameter clipping?**  Global clipping constrains the true concatenated update direction uniformly; per-parameter clipping changes relative layer scales.

**Does clipping alter gradients below the threshold?**  No. The early return is essential.

## Senior interview depth

Global clipping implements `g' = g·min(1, M/(||g||+eps))`. It constrains the
first-order optimizer input, not necessarily the eventual parameter-update norm:
Adam subsequently rescales coordinates and weight decay adds another update.
Clipping also introduces bias into the stochastic gradient estimator because
large-norm samples are transformed nonlinearly.

With gradient accumulation, clip after all microbatches so the constraint
applies to the effective batch gradient. With AMP, call the scaler's unscale
operation first; clipping scaled gradients enforces the wrong threshold. Under
data parallelism, gradients are normally reduced before clipping if the goal is
the norm of the global averaged gradient. Fully sharded systems require a
distributed norm reduction rather than local clipping per shard.

The current accumulator lives on the first gradient's device and copies every
other squared norm there. That is functionally limited for heterogeneous-device
parameters and can synchronize devices. It also lacks a non-finite policy.
Production code should define whether NaN/Inf raises, skips the step, or interacts
with loss scaling, and should accumulate norms in a sufficiently wide dtype.
