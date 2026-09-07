# Decoder-only Transformer language model

Current source: `cs336_basics/nn/transformer.py`.

## End-to-end contract

Input `token_ids` has shape `(B,S)` and integer dtype. Output `logits` has shape `(B,S,V)`. `logits[b,s,v]` is an unnormalized score for vocabulary item `v` after observing the legal prefix through position `s`.

For next-token training, inputs and targets are shifted:

```text
tokens:  [BOS, I, like, tea]
input:   [BOS, I, like]
target:  [I,   like, tea]
```

The model returns all positions in parallel; causality prevents leakage.

## Constructor walkthrough

- `token_embedding`: maps `(B,S)` IDs to `(B,S,D)` learned vectors.
- `nn.ModuleList([...])`: creates and registers `num_layers` blocks. A plain Python list would not recursively register its contained modules for `parameters()`, device movement, train/eval traversal, or state dictionaries.
- `ModuleList` rather than `Sequential`: forward must explicitly pass both `x` and `token_positions` to each block. `Sequential`'s simple one-output-to-next-input pipeline is inconvenient here.
- `final_norm`: normalizes the completed residual stream.
- `lm_head`: maps each `D` vector to `V` scores. It is separate from the embedding table; there is no weight tying.

## Forward walkthrough

- The dtype assertion requires long integer IDs. Since `torch.long` is an alias of `torch.int64`, the two-list check is redundant but clear.
- `x = token_embedding(token_ids)`: shape becomes `(B,S,D)`.
- `batch_size,seq_len = token_ids.shape`: establishes rank-two token input.
- `arange(seq_len,device=token_ids.device)`: positions `0...S-1` on the correct device.
- `.unsqueeze(0).expand(B,S)`: creates a zero-copy broadcasted `(B,S)` view because every ordinary batch item uses the same positions. `repeat` would allocate copies.
- The explicit loop replaces `x` with each block's output while reusing positions.
- `final_norm(x)`: prepares the residual stream for decoding.
- `lm_head(x)`: projects every position independently to `(B,S,V)` logits. Softmax is deliberately not applied here because cross-entropy can work stably from logits.

## Worked shape example

With `B=2,S=4,V=10_000,D=512`, IDs `(2,4)` become embeddings `(2,4,512)`. Every block preserves that shape. The head produces `(2,4,10_000)`. Targets remain `(2,4)`; cross-entropy selects one correct logit from each length-10,000 vector and averages eight losses.

## Parameter accounting

Approximate total, without tied embeddings:

`VD` embedding + `N(4D²+3DF+2D)` blocks + `D` final norm + `VD` head.

The two `VD` matrices can dominate small models with large vocabularies. Weight tying would replace the separate head matrix with the embedding matrix.

## Missing production features worth recognizing

- No padding mask or variable-length sequence handling; only causal masking.
- No dropout, KV cache, sampling/generation method, or cache position offset.
- No fused QKV, fused attention, tensor parallelism, or activation checkpointing.
- Position cache fixes maximum context at construction.
- Initialization is module-local rather than a coordinated architecture-wide scheme.

These omissions make the educational forward pass easier to inspect; they do not make the mathematical model invalid.

## Interview questions

**Why output logits rather than probabilities?**  Logits preserve numerical information and allow fused stable cross-entropy; softmax probabilities are only needed for sampling or interpretation.

**Why final normalization?**  It controls the scale of the accumulated residual stream before vocabulary projection, consistent with pre-norm decoder designs.

**What does `ModuleList` do that a list does not?**  It registers child modules in PyTorch's module tree while still allowing an explicit custom loop.

**How is inference different from training?**  Training scores many shifted positions in parallel with a causal mask. Autoregressive inference chooses one token, appends it, and repeats—ideally using a KV cache.

**Where are positional embeddings added?**  Nowhere in the residual stream. RoPE rotates Q/K inside each attention layer.

## Senior interview depth

The causal LM objective factorizes sequence probability as
`p(x_1:T)=Π_t p(x_t | x_<t)`. Teacher forcing computes all conditional logits
in parallel because the triangular mask prevents information leakage. Loss must
still shift targets correctly; feeding identical input and target indices would
train token reconstruction rather than next-token prediction.

The current model accepts only rank-two token IDs in practice because it unpacks
`token_ids.shape` into `(B,S)`. It validates dtype with an assertion, has no
explicit vocabulary-range or context-length check, and generates positions from
zero on every call. Consequently it is a training/full-prefix interface, not a
correct incremental-decoding API with cache offsets.

Training memory includes parameters, gradients, optimizer states, saved
activations, and temporary kernels. For Adam-like training in mixed precision,
optimizer/master-weight state can exceed parameter storage; for long sequences,
attention and residual activations can dominate. Production levers include
weight tying, fused kernels, activation checkpointing, sequence/tensor/pipeline
parallelism, sharded optimizer state, and reduced-precision formats.

An end-to-end test should verify output shape, finite loss/gradients, causal
invariance to future-token perturbations, save/load identity, deterministic
initialization, device/dtype transfer, maximum-context behavior, and parity
between full-prefix and cached decoding once a KV-cache path exists.
