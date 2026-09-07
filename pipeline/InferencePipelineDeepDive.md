# Inference pipeline deep dive

This document explains how `pipeline inference` turns a trained
`TransformerLM` artifact and a user prompt into a sampled text completion. It
is a revision guide as well as a design document: every phase states its input,
output, invariants, and systems trade-offs.

The implementation is intentionally split into two layers:

- `cs336_basics/generation.py` owns tokenizer-agnostic token-ID generation.
- `pipeline/inference.py` owns artifact loading, tiktoken, devices, and text
  output.

This separation means the model-side generator can later work with the custom
BPE tokenizer without duplicating autoregressive sampling logic.

## 1. The problem from first principles

A causal language model estimates a probability distribution for the next token
given the tokens seen so far:

~~~text
P(x_(t+1) | x_1, x_2, ..., x_t)
~~~

The model does not produce an entire paragraph in one call. One forward pass
returns a logit vector for every position in the supplied sequence. The vector
at the final position is used to sample exactly one next token. That token is
appended to the context, and the process repeats:

~~~text
prompt text
    |
    | tokenizer.encode
    v
token IDs [x_1, ..., x_t]
    |
    | TransformerLM([x_1, ..., x_t])
    v
final-position logits z in R^V
    |
    | temperature scaling -> top-p filtering -> random sample
    v
x_(t+1)
    |
    +--> append and repeat until EOT or the safety limit
~~~

Here `V` is the vocabulary size. “Generation” is the repeated prediction and
sampling loop; “decoding” in the tokenizer sense is the final conversion from
generated token IDs back into text. Keeping these terms separate prevents a
common mistake: tokenization is not the same operation as model generation.

## 2. Phase 0 — parse and validate the request

### Input

The CLI accepts:

~~~text
artifact, prompt, max_new_tokens, min_words,
temperature, top_p, seed, device, dtype
~~~

`InferenceConfig` in `pipeline/config.py` validates the path, non-empty prompt,
non-negative limits, positive finite temperature, `0 < top_p <= 1`, optional
non-negative seed, and supported dtype.

`max_new_tokens` counts only tokens generated after the prompt. It is a hard
safety bound even when a minimum word count has been requested. The minimum
word count is zero by default, preserving ordinary EOT behavior.

### Output

The validated request is passed to `run_inference`. Invalid input fails before
model construction or sampling, which makes errors cheap and deterministic.

## 3. Phase 1 — resolve and validate the model artifact

### Input

The `--artifact` argument may name either:

~~~text
pipeline/artifacts/RUN/artifact.pt
pipeline/artifacts/RUN/             # resolves to artifact.pt
~~~

The final artifact contains the model state dictionary, model configuration,
tokenizer metadata, training step, data metadata, and metrics.

### Processing

`pipeline.artifacts.load_final_artifact`:

1. Resolves a run directory to `artifact.pt`.
2. Loads tensors onto CPU with `weights_only=True`.
3. Verifies the schema version and required fields.
4. Verifies that model configuration and tokenizer metadata are present.

Loading onto CPU first is a portability boundary. An artifact saved on a Modal
CUDA machine can be consumed by an M4 Mac, and an artifact trained on a Mac can
be moved to a CUDA device later. The loader uses a restricted weights-only
format, but artifacts should still come from a trusted source.

### Output

The output is a validated Python payload. No model has been sampled yet; this
phase only establishes that the artifact is structurally usable.

## 4. Phase 2 — reconstruct the exact model

### Input

The loader reads `payload["config"]["model"]`:

~~~python
ModelConfig(
    context_length=S,
    vocab_size=V,
    d_model=D,
    num_layers=L,
    num_heads=H,
    d_ff=F,
    rope_theta=theta,
)
~~~

### Processing

`run_inference` constructs the existing `TransformerLM` with these dimensions
and loads `model_state_dict` using `strict=True`. Strict loading matters: a
missing or unexpected parameter must not silently produce a partially random
model.

The model is placed on the requested backend:

~~~text
--device auto: CUDA -> MPS -> CPU
--device cuda: NVIDIA GPU
--device mps: Apple Silicon GPU
--device cpu: host CPU
~~~

Inference uses `model.eval()` and `torch.inference_mode()`. Evaluation mode
disables training-only behavior, while inference mode avoids autograd graphs and
their memory overhead. The model is not updated during generation.

### Why are both `model.eval()` and `torch.inference_mode()` necessary?

They solve two different problems. `model.eval()` selects the module's
**behavioral mode**; `torch.inference_mode()` selects the tensor/autograd
**execution mode**. Neither one is a substitute for the other.

#### `model.eval()` changes module behavior

Calling `model.eval()` is equivalent to recursively calling `train(False)` on
the module and all registered child modules. It changes how stateful or
stochastic layers behave, for example:

~~~text
Dropout       training: randomly zero activations and rescale survivors
              evaluation: identity (no random mask)

BatchNorm     training: batch statistics and running-stat updates
              evaluation: frozen running statistics
~~~

The current `TransformerLM` does not contain Dropout or BatchNorm, so its
current numerical output is not changed by this flag. Setting it is still the
correct contract: a future block may add dropout, and a third-party module may
already have train/eval-dependent behavior. Without `eval()`, inference could
remain stochastic or mutate running statistics, making repeated prompts
produce different results even with the same random sampling seed.

`eval()` does **not**:

- disable gradient tracking;
- set every parameter's `requires_grad` to `False`;
- freeze weights in the optimizer; or
- change the model's numerical dtype or device.

For example, this is still an autograd-enabled forward pass:

~~~python
model.eval()
logits = model(input_ids)
print(logits.requires_grad)  # usually True because model parameters require grad
print(logits.grad_fn)        # a backward Function, such as an add/matmul node
~~~

The `eval` flag answers “should layers behave as they do at evaluation time?”
It does not answer “should PyTorch remember this computation for backward?”

#### What an autograd graph actually is

PyTorch autograd builds a dynamic directed acyclic graph while executing a
forward pass. The graph is not the model architecture itself; it is a record of
the particular operations and tensors used by this particular input. A useful
mental model is:

~~~text
leaf parameters W_q, W_k, W_v, ...  (requires_grad=True)
                 |
                 v
       embedding -> projections -> attention -> MLP -> logits
                 |
                 v
       output.grad_fn points to the last recorded operation
~~~

When an operation receives at least one tensor requiring gradients, autograd
creates a node for that operation. The node stores references to whatever
intermediate values are needed by the chain rule. For a linear operation, the
backward pass needs the relevant input and weight; for attention, it may need
queries, keys, scaled scores, masks, softmax outputs, and value tensors. The
exact saved set is operator-dependent, but the principle is fixed: forward
activations are retained so `backward()` can later compute parameter gradients.

For a scalar loss `L`, backward applies the chain rule through this graph:

~~~text
logits -> loss
   ^       |
   |       | loss.backward()
   +-------+----> dL/dW for every parameter W
~~~

The graph is therefore useful for training, Jacobians, higher-order
derivatives, and any other operation that asks for gradients. It is not needed
to evaluate the already-defined function `logits = f_θ(input_ids)`.

#### What happens if we build the graph during inference?

If inference runs without a gradient-disabled context, the model still computes
the correct logits, but it also performs bookkeeping that no consumer needs:

1. Each parameter-dependent operation creates an autograd node.
2. Intermediate activations are retained for a possible backward pass.
3. Those allocations increase peak memory and memory-bandwidth traffic.
4. Graph construction and tensor version/view bookkeeping add CPU overhead.
5. Lower memory headroom can reduce batch size or trigger out-of-memory errors.

In this generator, one token is sampled per forward pass. If the graph is not
disabled, this unnecessary work is repeated for every generated token. A graph
may be reclaimed after a forward result and its `grad_fn` become unreachable,
so a short loop does not necessarily leak forever. That does **not** make the
work free: every step still allocated and populated saved activations before
discarding them.

The more dangerous case is retaining graph-connected tensors:

~~~python
logits_history = []
for _ in range(num_steps):
    logits = model(input_ids)       # graph is built
    logits_history.append(logits)   # keeps each graph alive through grad_fn
~~~

Here each list element keeps its computation history and saved activations.
GPU memory can grow roughly with the number of retained steps. Appending
`logits.detach()` would sever the history, but for pure inference the stronger
and clearer solution is to prevent graph construction at the source.

#### Why a forward-only pass does not need a graph

“Forward pass” describes the direction of numerical computation; it does not
imply that gradients must be recorded. Inference needs the values produced by
the forward equations:

~~~text
input IDs -> embeddings -> Transformer blocks -> logits -> sampling
~~~

It does not need:

~~~text
logits -> loss -> backward -> parameter gradients
~~~

The graph exists only to make that second path possible later. Since inference
never calls `loss.backward()`, never updates parameters, and never requests a
Jacobian, recording the graph is dead bookkeeping. `torch.inference_mode()`
executes the first path while disabling the second.

#### `torch.no_grad()` versus `torch.inference_mode()`

Both contexts stop autograd from recording operations:

~~~python
with torch.no_grad():
    logits = model(input_ids)
    # logits.requires_grad is False; logits.grad_fn is None
~~~

`torch.inference_mode()` provides the same no-backward guarantee and additionally
disables autograd's view tracking and tensor version-counter updates for tensors
created inside the context. Version counters let autograd detect in-place
mutations; view tracking records aliasing relationships. Those safeguards are
valuable in training, but they are unnecessary when tensors are only being
read for prediction. Avoiding that metadata work generally reduces overhead and
is why inference mode is preferred for a serving/generation path:

~~~python
with torch.inference_mode():
    logits = model(input_ids)
    # logits.requires_grad is False; logits.grad_fn is None
~~~

Inference-mode tensors have a deliberate restriction: they cannot later be
turned into gradient-tracked tensors with `requires_grad_(True)` or safely fed
into a gradient-enabled computation that expects normal autograd metadata. If
an output must be reused for training or differentiation, use `no_grad()` or
leave inference mode and explicitly `clone()` into a normal tensor. That is not
needed here because the logits are immediately reduced to a sampled integer ID.

#### How the current code applies the contract

`cs336_basics/generation.py` uses the following sequence:

~~~python
was_training = model.training
model.eval()                         # evaluation behavior
try:
    with torch.inference_mode():     # no graph, no backward metadata
        logits = model(input_ids)
        next_token = sample_next_token(logits[0, -1], ...)
finally:
    model.train(was_training)        # do not surprise a caller's mode
~~~

`pipeline/inference.py` also sets the freshly reconstructed model to evaluation
mode before invoking the generator. The generator restores the mode that was
active before its call, so using it does not permanently mutate a caller's
module state. The sampled token is converted to a Python integer; no
graph-connected tensor is stored between decoding iterations.

#### Memory and performance intuition for this Transformer

Let `L` be the number of layers, `S_active` the active context length, `D` the
model width, `H` the head count, and `V` the vocabulary size. A training-style
forward can retain activation tensors whose total size grows with layer count
and includes attention intermediates proportional to `H * S_active^2`, in
addition to hidden states proportional to `S_active * D`. The exact allocator
footprint depends on kernels and dtype, but inference mode removes the need to
retain these backward intermediates. It does not remove the memory needed for
the current forward computation or the final `(1, S_active, V)` logits.

This is why `eval()` plus `inference_mode()` is not ceremonial boilerplate:
`eval()` makes the function deterministic with respect to module semantics, and
`inference_mode()` makes execution match the actual requirement—evaluate the
function once, sample from its result, and never differentiate it.

### Output contract

The model accepts integer IDs of shape `(1, S_active)` and returns logits of
shape `(1, S_active, V)`. The final slice, `logits[0, -1]`, has shape `(V,)` and
represents scores for the next token after the current context.

## 5. Phase 3 — load the tokenizer and encode the prompt

### Tokenizer compatibility

The artifact records the tiktoken encoding name and vocabulary size. The
pipeline loads that exact named encoding and checks:

~~~text
artifact vocabulary == installed tokenizer vocabulary == model vocabulary
~~~

It also verifies that `<|endoftext|>` exists and that a newly recorded EOT ID
agrees with `encoding.eot_token`. Older schema-v1 artifacts that predate the
explicit EOT field remain compatible because the ID can be derived from the
encoding name.

### Prompt encoding

The prompt uses the same special-token policy as training:

~~~python
prompt_token_ids = encoding.encode(
    prompt,
    allowed_special={"<|endoftext|>"},
)
~~~

The result is a list `[x_1, ..., x_t]` of integer IDs. A literal
`<|endoftext|>` is one ID; other unexpected special tokens remain rejected.
The prompt must produce at least one token because the model needs a final
position from which to predict.

An EOT in the prompt is context, not an automatic stop signal. The stopping
rule applies to an EOT sampled after the prompt.

## 6. Phase 4 — construct the active context

The prompt may be longer than the model's trained context length, and a long
completion may eventually make the complete sequence longer as well. Before
each forward pass, generation selects:

~~~python
active_ids = all_token_ids[-model.context_length:]
~~~

Thus the model always receives at most `S` IDs. Older IDs remain in the result
for reporting but are no longer visible to attention. With RoPE, resetting the
retained window's positions to `0..S_active-1` preserves relative positions
inside that window; the trade-off is lost long-range history.

The current Transformer has no KV cache. Every step therefore recomputes the
embedding, projections, attention, and feed-forward layers for the active
window. This is straightforward and reuses the existing implementation, but
it is slower than a serving implementation that caches prior keys and values.

## 7. Phase 5 — obtain final-position logits

For the active sequence `[x_1, ..., x_t]`, the model computes:

~~~text
(1, t) -> token embeddings (1, t, D)
        -> L causal Transformer blocks (1, t, D)
        -> RMSNorm and lm_head (1, t, V)
~~~

Only the final row is used:

~~~python
next_token_logits = model(input_ids)[0, -1]
~~~

These values are logits, not probabilities. They may be shifted by any common
constant without changing the eventual softmax distribution. The generator
checks that the shape is `(V,)` and rejects non-finite values before sampling.

## 8. Phase 6 — apply temperature scaling

Given logits `z_i` and temperature `T > 0`, sampling probabilities are:

~~~text
q_i(T) = exp(z_i / T) / sum_j exp(z_j / T)
~~~

Subtracting the maximum logit before exponentiation is numerically equivalent
and prevents overflow. The implementation performs sampling arithmetic in
float32 even when model weights use float16 or bfloat16.

Temperature changes the shape of the entire distribution:

~~~text
T = 1       unchanged distribution
0 < T < 1   sharper; high-probability tokens dominate
T > 1       flatter; more tokens receive meaningful probability
~~~

Temperature zero is rejected. Greedy argmax decoding is a separate decoding
policy and should not be hidden behind an invalid division by zero.

## 9. Phase 7 — apply nucleus (top-p) filtering

Temperature produces a distribution over all `V` tokens. Nucleus sampling
removes its low-probability tail dynamically:

1. Sort tokens by descending probability.
2. Find the smallest prefix `1..k` whose cumulative mass satisfies
   `sum(i=1..k) q_i >= p`.
3. Set logits outside that prefix to negative infinity.
4. Renormalize the retained probabilities.
5. Sample one ID with `torch.multinomial`.

`top_p=1` keeps the full vocabulary. A very small `p` may keep only the most
likely token. The first token crossing the threshold is retained, so the
candidate set can never be empty. Temperature is applied first because it
changes the ranking and cumulative masses used to define the nucleus.

The sampler can also suppress selected IDs on a particular step. This is how
minimum-word generation temporarily masks only the EOT ID while leaving all
ordinary tokens available.

## 10. Phase 8 — enforce minimum words and stop conditions

The normal loop stops when either condition is true:

~~~text
sampled token == encoding.eot_token
or
number of sampled tokens == max_new_tokens
~~~

For `--min-words N`, the pipeline supplies an `endoftext_allowed` callback to
the tokenizer-agnostic generator. Before each sample, it decodes the generated
IDs so far and counts Unicode word spans. Until the count reaches `N`, EOT is
masked from the distribution. Once the threshold is reached, EOT is eligible
again and can terminate the paragraph naturally.

This policy belongs in the pipeline because “word” is a text-level concept;
token IDs alone cannot reliably identify words. One BPE token may be a complete
word, a word fragment, whitespace, or punctuation. Apostrophes inside a word
are retained by the current counter, while hyphenated terms count as two words.

If `max_new_tokens` is reached before the minimum word count, the pipeline
raises an error instead of returning a result that violates the request. A
useful starting point for 50 English words is `--min-words 50
--max-new-tokens 128`; difficult prompts or weak models may need 256 or more.

The raw `GenerationResult` retains a sampled terminal EOT ID for auditability.
The human-readable completion removes only that final sentinel.

## 11. Phase 9 — decode IDs and persist output

After generation, the pipeline has:

~~~text
prompt IDs       [x_1, ..., x_t]
generated IDs    [x_(t+1), ..., x_(t+n)]
stop reason      endoftext | max_new_tokens
~~~

It removes a terminal EOT from the display sequence, then calls:

~~~python
completion = encoding.decode(completion_token_ids)
~~~

The CLI writes only the completion text to stdout. Diagnostics go to stderr so
the completion can be redirected into another command. `--show-token-ids`
prints the exact IDs, and `--output-json PATH` stores the prompt, completion,
token IDs, stop reason, device, dtype, and token/word counts.

The structured result is useful for debugging because text alone hides whether
the model stopped at EOT, hit the token cap, or produced a different number of
tokens than expected.

## 12. Complete command

~~~bash
uv run python -m pipeline inference \
  --artifact pipeline/artifacts/tinystories-500/tinystories-500/artifact.pt \
  --prompt "Write a paragraph about a little dog:" \
  --min-words 50 \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42 \
  --device mps \
  --output-json pipeline/artifacts/inference/generation.json
~~~

On Modal or another NVIDIA host, replace `mps` with `cuda`; `auto` selects
CUDA, then MPS, then CPU.

## 13. Failure modes and engineering boundaries

| Failure | Meaning | Resolution |
| --- | --- | --- |
| Artifact schema mismatch | The payload is from an incompatible pipeline version | Use a compatible artifact or migrate its schema |
| Model/tokenizer vocabulary mismatch | IDs would index the wrong embedding/output row | Use the tokenizer recorded by the artifact |
| Empty prompt | No final position exists for next-token prediction | Supply non-empty text |
| Invalid temperature/top-p | The sampling distribution is undefined | Use `temperature > 0` and `0 < top_p <= 1` |
| EOT before minimum words | EOT is intentionally masked | Increase `max_new_tokens` if the cap is reached |
| Requested backend unavailable | PyTorch cannot access CUDA or MPS | Use `--device auto` or an available backend |
| Non-finite logits | Model weights or numerical execution are invalid | Inspect the artifact, dtype, and device |

The current implementation is a single-prompt, correctness-first generator. It
does not yet provide KV caching, batching, streaming output, beam search, or
greedy decoding. Those are separate serving features and should be added only
after preserving the token-level contracts described above.

The quality of generated language depends on training. A 500-step smoke-test
artifact can exercise the complete inference path while still producing weak
or repetitive text; that is a model-quality limitation, not necessarily an
inference bug.
