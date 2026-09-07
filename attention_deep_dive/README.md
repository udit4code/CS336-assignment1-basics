# Token-level attention deep dive

This package visualizes the causal attention probabilities produced by the
repository's `cs336_basics.nn.MultiHeadSelfAttention` implementation.

## Important interpretation

The default mode intentionally uses a deterministic random embedding table and
random MHA weights. It is useful for studying shapes, masking, head-specific
patterns, RoPE, and softmax normalization. It is **not** evidence that a model
understands the sentence. Meaningful linguistic patterns require a trained
checkpoint whose tokenizer, embedding, normalization, projections, RoPE
configuration, and architecture are compatible.

Checkpoint mode is reserved in the API but is not implemented yet. Passing
`--mode checkpoint` fails explicitly instead of silently visualizing random
weights.

## Command-line interface

Run from the repository root:

```bash
.venv/bin/python -m attention_deep_dive \
  --sentence "While waiting for the bus, Sam got frustrated, as it was too late"
```

The CLI prints the token sequence and writes the HTML/JSON artifacts. See all
options with:

```bash
.venv/bin/python -m attention_deep_dive --help
```

Important options:

| Option | Default | Purpose |
|---|---:|---|
| `--sentence TEXT` | the demo sentence | Text to tokenize and inspect |
| `--encoding-name NAME` | `gpt2` | Any encoding supported by tiktoken |
| `--output-dir PATH` | `attention_deep_dive/outputs` | Artifact directory |
| `--d-model INT` | `128` | Embedding/MHA width |
| `--num-heads INT` | `4` | Equal-width attention heads |
| `--theta FLOAT` | `10000` | RoPE base theta |
| `--context-length INT` | `128` | Maximum RoPE position |
| `--seed INT` | `0` | Reproducible random initialization |
| `--device {auto,cpu,cuda,mps}` | `auto` | Execution device |
| `--dtype {float32,float64,float16,bfloat16}` | `float32` | Model/activation dtype |
| `--no-rope` | disabled | Disable RoPE inside MHA |
| `--mode {random,checkpoint}` | `random` | Model source; checkpoint is reserved |
| `--checkpoint-path PATH` | none | Reserved for future checkpoint support |

Examples:

```bash
# Inspect a short custom sentence and put artifacts in /tmp.
.venv/bin/python -m attention_deep_dive \
  --sentence "The cat sat on the mat." \
  --output-dir /tmp/cat-attention \
  --seed 42

# Use a different tiktoken encoding and CPU explicitly.
.venv/bin/python -m attention_deep_dive \
  --sentence "A small test" \
  --encoding-name cl100k_base \
  --device cpu \
  --d-model 64 \
  --num-heads 4
```

`d_model` must be divisible by `num_heads`, and the resulting head width must be
even because the current MHA constructs its RoPE cache even when rotation is
disabled. `float16`/`bfloat16` support depends on the selected device. The
default `float32`/`auto` combination is the most portable.

The command writes:

```text
attention_deep_dive/outputs/token_attention.html
attention_deep_dive/outputs/token_attention.json
```

Open the HTML file in a browser. It contains one matrix averaged across heads
and one matrix per head. Rows are query tokens (“who attends?”), columns are
key tokens (“what is attended to?”), and gray upper-triangular cells are blocked
by the causal mask. Values are post-softmax probabilities, not raw QK logits.

## Why tiktoken and the custom embedding are both needed

`tiktoken` supplies integer IDs. The custom `Embedding` maps those IDs to
vectors of width `d_model`; it is not correct to pass token IDs directly to
MHA. The demo uses:

```text
GPT-2 tiktoken IDs (B,S)
  -> cs336_basics.nn.Embedding (B,S,D)
  -> cs336_basics.nn.RMSNorm (B,S,D)
  -> cs336_basics.nn.MultiHeadSelfAttention
```

RoPE is applied internally to Q and K after their projections and head split;
it is not applied to the embedding vectors.

GPT-2 tokenization is subword tokenization. A label such as `␠waiting` may be a
single token, while another word may be split into several tokens. This demo
deliberately stays at token level and preserves exact UTF-8 token bytes.

## What the probe verifies

The probe calls `MultiHeadSelfAttention.forward_with_attention()`, which is the
same projection/head-layout/RoPE/mask path used by ordinary `forward()`. That
method delegates score computation to the shared scaled-dot-product attention
implementation and returns the diagnostic tensors without duplicating the
score or softmax logic. It reconstructs

```text
softmax(QKᵀ / sqrt(d_k) + causal mask) V
  -> concatenate heads
  -> output projection
```

and asserts that the result matches the public MHA forward pass. This makes the
visualization an inspection path rather than a second, unverified attention
implementation.

## Programmatic use

```python
import torch
import tiktoken

from attention_deep_dive import build_random_model, probe_attention, tokenize_sentence

sentence = tokenize_sentence("While waiting for the bus")
model = build_random_model(tiktoken.get_encoding("gpt2").n_vocab, seed=7)
ids = torch.tensor([sentence.token_ids])
positions = torch.arange(ids.shape[1]).unsqueeze(0)
_, hidden = model.hidden_states(ids)
trace = probe_attention(model.attention, hidden, positions)
print(trace.probabilities.shape)  # (1, heads, sequence, sequence)
```

For a real model, the embedding vocabulary size must equal the tokenizer's
vocabulary and the checkpoint must have been trained with the same token-ID
mapping. Loading only an unrelated pretrained embedding matrix is not enough:
Q/K projections and normalization were co-trained with it. The future
checkpoint adapter should validate every such compatibility condition before
loading.
