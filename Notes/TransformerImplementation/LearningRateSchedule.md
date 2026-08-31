# Linear warmup and cosine learning-rate decay

Source: `LearningRateScheduleModule/LearningRateSchedule.py`.

## Piecewise schedule

With maximum rate `α_max`, floor `α_min`, warmup endpoint `T_w`, and cosine endpoint `T_c`:

```text
t < T_w:
    α(t) = (t/T_w) α_max

T_w <= t <= T_c:
    α(t) = α_min + 1/2 [1 + cos(π(t-T_w)/(T_c-T_w))]
                         (α_max-α_min)

t > T_c:
    α(t) = α_min
```

At `t=T_w`, cosine is one and rate is `α_max`; at `t=T_c`, cosine is minus one and rate is `α_min`. The pieces are value-continuous under nondegenerate boundaries.

## Code walkthrough

- Assertions enforce nonnegative warmup and a decay endpoint no earlier than warmup.
- The first branch linearly increases from zero, reducing violent early updates while optimizer moments and network statistics are immature.
- The second maps normalized decay progress from zero to one, then half-cosine maps it smoothly from maximum to minimum.
- The last branch holds the floor after decay.

## Worked example

Let `α_max=10^-3`, `α_min=10^-4`, `T_w=100`, `T_c=1000`.

- `t=0`: `0`
- `t=50`: `5×10^-4`
- `t=100`: `10^-3`
- `t=550`: midpoint cosine is zero, so `5.5×10^-4`
- `t=1000`: `10^-4`
- later: `10^-4`

## Edge cases in the current function

- `T_w=0` is safe only because `t<0` is false for ordinary nonnegative steps; negative `t` would enter a division by zero.
- `T_c=T_w` causes division by zero at `t=T_w`. Validation should require `T_c>T_w` when the cosine branch can execute, or special-case a zero-length decay.
- Negative time, negative learning rates, and `alpha_min>alpha_max` are not rejected.
- Be explicit whether `t` counts optimizer updates or microbatches. With gradient accumulation, schedules usually advance per optimizer step.

## Interview questions

**Why warm up?**  Early gradients and adaptive moment estimates can be poorly calibrated; gradually raising step size improves stability, especially for large batches/models.

**Why cosine decay?**  It provides a smooth fall with zero slope at endpoints and few hyperparameters. Its superiority is empirical, not universal.

**Why keep a nonzero floor?**  It permits continued adaptation late in training; zero may be appropriate when training ends exactly at `T_c`.

**Does AdamW's adaptivity eliminate scheduling?**  No. AdamW rescales coordinates, while the global learning rate still controls overall update magnitude over training.

