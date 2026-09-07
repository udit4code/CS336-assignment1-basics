# Cross-entropy from logits

Current source: `cs336_basics/nn/cross_entropy.py`.

## Contract and equation

`logits` has shape `(...,V)`; integer `targets` has shape `(...)`. For each item with correct class `y`:

`L = log Σ_i exp(z_i) - z_y`.

The function returns the mean over every leading location. In language modeling those locations are batch-token pairs.

## Code walkthrough

- Assertions require at least one sample axis plus class axis, exact target/leading-shape agreement, and 64-bit integer targets.
- `max_logits = logits.max(-1,keepdim=True).values`: one stabilizing maximum per item.
- `stabilized_logits = logits-max_logits`: softmax/log-sum-exp are invariant to this common shift.
- `exp_logits`, `sum(dim=-1)`, then `log`: computes `log Σ exp(z_i-m)` without positive exponential overflow.
- `targets.unsqueeze(-1)`: makes indices match gather's rank.
- `.gather(-1,...).squeeze(-1)`: selects exactly `z_y-m` for every item without one-hot expansion.
- `loss = log_sum_exp-target_logits`: equals the negative log-probability of the correct class.
- `.mean()`: produces a scalar suitable for backward.

## Worked example

For logits `[2,1,0]` and target zero, subtract maximum to get `[0,-1,-2]`. `log_sum_exp≈log(1+0.368+0.135)=0.408`. Correct stabilized logit is zero, so loss is `0.408`, equal to `-log(0.665)`.

## Gradient

For one example, `∂L/∂z_i = softmax(z)_i - 1[i=y]`. The model is pushed to increase the correct logit and decrease competing logits, with force proportional to its current confidence.

## Practical caveats

- This function has no `ignore_index`; padding tokens would incorrectly contribute unless filtered elsewhere.
- There is no label smoothing or per-token weighting.
- `torch.logsumexp` or fused framework cross-entropy is typically faster and can be more robust in mixed precision.
- Target values must lie in `[0,V)` or `gather` fails.
- Perplexity is `exp(mean cross-entropy)` when the loss uses natural logarithms and the token averaging convention is appropriate.

## Interview questions

**Why not softmax first and then take log?**  It materializes probabilities and can underflow correct-class probabilities to zero. Log-sum-exp computes the same quantity stably.

**Why does subtracting the maximum not alter loss?**  Both the normalizer's log and the correct logit shift by the same constant, which cancels.

**What does cross-entropy measure here?**  The negative log-likelihood assigned to observed next tokens, averaged over positions.

**Why are targets integers rather than one-hot?**  The correct logit can be gathered directly, saving `O(V)` target storage per position.

## Senior interview depth

For a batch of `N` tokens, the gradient is `(P-Y)/N`, where `P` is softmax and
`Y` is the one-hot target matrix. The division by `N` comes from this function's
mean reduction. Changing reduction, masking padding, or distributing the batch
changes the effective gradient scale and therefore interacts with learning rate.

The stable loss can be written directly as `logsumexp(z)-z_y`; subtracting the
maximum explicitly is sound, but a fused cross-entropy avoids materializing all
intermediates and can combine reduction efficiently. Its unavoidable logits
work is `O(NV)`, which is expensive for large vocabularies even though target
selection itself is `O(N)`.

Production gaps are explicit range checking, `ignore_index`, label smoothing,
per-token weights, selectable reduction, and defined behavior for zero valid
tokens. Tests should include huge-magnitude logits, arbitrary leading shapes,
invalid targets, gradient comparison with `torch.nn.functional.cross_entropy`,
and the identity `perplexity = exp(natural-log NLL)` under identical masking.
