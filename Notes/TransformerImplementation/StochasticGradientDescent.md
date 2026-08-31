# Stochastic gradient descent and the training loop

Source: `StochasticGradientDescentModule/SGD.py` and `SGD_training_loop.py`.

## What this implementation actually is

The class is called SGD, but its update includes per-parameter step decay:

`p_t = p_(t-1) - lr / sqrt(t+1) * g_t`.

Plain SGD would use constant `lr` unless an external scheduler changes it. This distinction is important in an interview and when comparing with `torch.optim.SGD`.

## Optimizer walkthrough

- Subclassing `torch.optim.Optimizer` provides parameter groups, state dictionaries, zeroing utilities, and standard integration.
- Constructor rejects negative learning rate, builds `defaults={"lr":lr}`, and delegates parameter-group creation to the base class.
- Optional closure recomputes and returns a loss. Robust implementations normally invoke it under `torch.enable_grad()`.
- The outer loop visits parameter groups, permitting different hyperparameters for different parameter sets.
- Parameters with `grad is None` are skipped; this differs from a numeric zero gradient because no update/state advance occurs.
- `self.state[p]` stores step `t` independently per parameter.
- Gradient is read, effective learning rate is `lr/sqrt(t+1)`, and the parameter is changed under `torch.no_grad()` so optimizer arithmetic does not enter autograd's graph.
- `state["t"] = t+1` records the next step.

The code uses `p.data` inside `no_grad`. Modern style is simply `p.add_(grad, alpha=-effective_lr)` within `no_grad`; `.data` can bypass safety checks and is unnecessary here.

## Training-loop walkthrough

1. Create an `nn.Parameter` and optimizer.
2. `optimizer.zero_grad()` clears gradients because PyTorch accumulates gradients by default.
3. Compute scalar loss `(weights**2).mean()`.
4. `loss.backward()` populates `weights.grad` with `2*weights/numel`.
5. `optimizer.step()` mutates weights using that gradient.

Zeroing before backward and after step are both valid conventions as long as it occurs before the next unwanted accumulation.

## Interview questions

**Why does PyTorch accumulate gradients?**  Accumulation enables microbatching and objectives composed from multiple backward calls. The training loop must clear them deliberately.

**What makes SGD stochastic?**  The gradient is estimated from a sampled minibatch rather than the full dataset; the update equation itself is ordinary gradient descent.

**Why use `no_grad` for updates?**  Parameter updates are an optimization procedure, not differentiable model computation; recording them would grow the graph and corrupt training semantics.

**What is a parameter group?**  A collection of parameters sharing optimizer hyperparameters, enabling different learning rates or decay policies.

**How does this differ from momentum SGD?**  Momentum stores a velocity that accumulates past gradients. This implementation stores only a step counter and decays learning rate.

