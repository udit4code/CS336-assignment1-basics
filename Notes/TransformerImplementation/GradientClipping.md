# Global L2 gradient clipping

Source: `GradientClippingModule/GradientClipping.py`.

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

- The annotation says `list`, but callers commonly pass the one-shot generator `model.parameters()`. This function iterates `parameters` twice, so a generator would be exhausted and the scaling pass would do nothing. Materialize once (`parameters=list(parameters)`) or accept/reuse a concrete collection.
- Reassigning `p.grad = p.grad*scale` allocates tensors; in-place `p.grad.mul_(scale)` is leaner.
- Parameters on multiple devices cannot be added into one device-local tensor without coordination.
- Accumulation starts as Python `0.0` and becomes a tensor on the first gradient. Empty gradients leave a float and still work for the early return, though validation would be clearer.
- Non-finite norms deserve explicit handling in production.

## Interview questions

**Value clipping versus norm clipping?**  Value clipping clamps each coordinate and changes direction. Norm clipping applies one factor and preserves direction.

**Does clipping solve the cause of exploding gradients?**  It limits immediate damage but may hide bad initialization, unstable precision, excessive learning rates, or faulty data.

**Why global rather than per-parameter clipping?**  Global clipping constrains the true concatenated update direction uniformly; per-parameter clipping changes relative layer scales.

**Does clipping alter gradients below the threshold?**  No. The early return is essential.

