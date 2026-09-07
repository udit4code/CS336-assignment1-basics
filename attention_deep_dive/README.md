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

## Run

From the repository root:

```bash
.venv/bin/python -m attention_deep_dive \
  --sentence "While waiting for the bus, Sam got frustrated, as it was too late"
```

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

The probe reuses the module's Q/K/V projections, RoPE object, custom softmax,
head layout, and causal-mask convention. It reconstructs

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
