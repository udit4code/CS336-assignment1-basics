# Transformer implementation: senior MLE revision map

These notes are a code-guided companion to the current implementation in
`cs336_basics/nn`. They are designed for two levels of recall: first explain the
mathematics cleanly, then connect it to tensor layouts, numerical behavior,
training dynamics, and production trade-offs. Read them in this order:

1. [Linear](Linear.md) and [Embedding](Embedding.md)
2. [SiLU](SiLU.md), [SwiGLU](PositionWiseFeedForward.md), and [RMSNorm](RMSNorm.md)
3. [Softmax](Softmax.md), [Scaled dot-product attention](ScaledDotProductAttention.md), [RoPE](RoPE.md), and [Multi-head self-attention](MultiHeadSelfAttention.md)
4. [Transformer block](TransformerBlock.md) and [Transformer language model](TransformerLanguageModel.md)
5. [Cross-entropy](CrossEntropyLoss.md), [SGD](StochasticGradientDescent.md), [AdamW](AdamW.md), [gradient clipping](GradientClipping.md), and [learning-rate scheduling](LearningRateSchedule.md)

## Current source-to-note map

| Current source | Corresponding note |
|---|---|
| `nn/adamw.py` | [AdamW.md](AdamW.md) |
| `nn/cross_entropy.py` | [CrossEntropyLoss.md](CrossEntropyLoss.md) |
| `nn/embedding.py` | [Embedding.md](Embedding.md) |
| `nn/gradient_clipping.py` | [GradientClipping.md](GradientClipping.md) |
| `nn/schedules.py` | [LearningRateSchedule.md](LearningRateSchedule.md) |
| `nn/linear.py`, `nn/linear_einops.py` | [Linear.md](Linear.md) |
| `nn/multihead_attention.py` | [MultiHeadSelfAttention.md](MultiHeadSelfAttention.md) |
| `nn/feed_forward.py`, `nn/feed_forward_einops.py` | [PositionWiseFeedForward.md](PositionWiseFeedForward.md) |
| `nn/normalization.py`, `nn/normalization_einops.py` | [RMSNorm.md](RMSNorm.md) |
| `nn/rotary_embedding.py`, `nn/rotary_embedding_einops.py` | [RoPE.md](RoPE.md) |
| `nn/attention.py`, `nn/attention_einops.py` | [ScaledDotProductAttention.md](ScaledDotProductAttention.md) |
| `nn/activation.py` | [SiLU.md](SiLU.md) |
| `nn/softmax.py` | [Softmax.md](Softmax.md) |
| `nn/sgd.py`, `nn/sgd_example.py` | [StochasticGradientDescent.md](StochasticGradientDescent.md) |
| `nn/transformer_block.py` | [TransformerBlock.md](TransformerBlock.md) |
| `nn/transformer.py` | [TransformerLanguageModel.md](TransformerLanguageModel.md) |

## Shared notation

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `S` | sequence length |
| `D` | model width, `d_model` |
| `H` | number of attention heads |
| `K` | per-head width, `d_k = D/H` |
| `F` | feed-forward width, `d_ff` |
| `V` | vocabulary size |

Here, `N = B S` is the number of token positions and `L` is the number of
Transformer blocks. Unless stated otherwise, attention is self-attention, so
query and key length are both `S`.

## Whole-model data flow

```text
token ids (B,S)
  -> embedding (B,S,D)
  -> N × [RMSNorm -> causal MHSA -> residual
          RMSNorm -> SwiGLU -> residual]
  -> final RMSNorm
  -> vocabulary projection
  -> logits (B,S,V)
  -> cross-entropy against next-token targets (B,S)
  -> scalar loss -> backward -> clip -> AdamW/SGD step
```

## Source-specific cautions

- These implementations prioritize learning and transparency. Production systems normally fuse projections, normalization, attention, and optimizer operations where possible.
- Assertions are useful teaching checks, but public library code should generally raise explicit exceptions; Python assertions can be disabled.
- The notes distinguish mathematical complexity from actual speed. Equal big-O does not imply equal runtime because memory traffic, kernel launches, fusion, dtype, and device matter.

## The 90-second architecture answer

This is a decoder-only, pre-norm Transformer. Integer token IDs are gathered
from a learned embedding table. Each block applies RMSNorm, causal multi-head
self-attention with RoPE, and a residual addition; it then applies a second
RMSNorm, a SwiGLU feed-forward network, and another residual addition. A final
RMSNorm and bias-free linear head produce next-token logits. Training minimizes
mean token cross-entropy, optionally clips the global gradient norm, and updates
parameters with AdamW or the educational SGD variant.

## Complexity summary

For one block, projections and the FFN cost `O(B S D² + B S D F)`, while dense
self-attention costs `O(B S² D)` compute and `O(B H S²)` score/probability
memory. At short contexts the dense layers may dominate; at long contexts the
quadratic attention term dominates. Parameter count per block is
`4D² + 3DF + 2D`; the untied embedding and output matrices contribute `2VD`.

## Interview method

For any component, answer in this order:

1. State its contract and equation.
2. Walk the shape transformation without hand-waving.
3. Give parameter count, compute, and activation-memory cost.
4. Name the numerical-stability mechanism and its failure mode.
5. Contrast this educational implementation with a production implementation.
6. Explain how you would test it: reference equivalence, invariants, edge cases,
   gradient checks, mixed precision, and device coverage.
