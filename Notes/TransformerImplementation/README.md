# Transformer Implementation: revision map

These notes are a code-guided companion to `cs336_basics/TransformerImplementation`. There is one note for every substantive module folder. Read them in this order:

1. [Linear](Linear.md) and [Embedding](Embedding.md)
2. [SiLU](SiLU.md), [SwiGLU](PositionWiseFeedForward.md), and [RMSNorm](RMSNorm.md)
3. [Softmax](Softmax.md), [Scaled dot-product attention](ScaledDotProductAttention.md), [RoPE](RoPE.md), and [Multi-head self-attention](MultiHeadSelfAttention.md)
4. [Transformer block](TransformerBlock.md) and [Transformer language model](TransformerLanguageModel.md)
5. [Cross-entropy](CrossEntropyLoss.md), [SGD](StochasticGradientDescent.md), [AdamW](AdamW.md), [gradient clipping](GradientClipping.md), and [learning-rate scheduling](LearningRateSchedule.md)

## Complete folder-to-note map

| Source subfolder | Corresponding note |
|---|---|
| `AdamWOptimizerModule` | [AdamW.md](AdamW.md) |
| `CrossEntropyLossModule` | [CrossEntropyLoss.md](CrossEntropyLoss.md) |
| `EmbeddingModule` | [Embedding.md](Embedding.md) |
| `GradientClippingModule` | [GradientClipping.md](GradientClipping.md) |
| `LearningRateScheduleModule` | [LearningRateSchedule.md](LearningRateSchedule.md) |
| `LinearModule` | [Linear.md](Linear.md) |
| `MultiHeadSelfAttentionModule` | [MultiHeadSelfAttention.md](MultiHeadSelfAttention.md) |
| `PositionWiseFeedForwardModule` | [PositionWiseFeedForward.md](PositionWiseFeedForward.md) |
| `RMSNormModule` | [RMSNorm.md](RMSNorm.md) |
| `RoPEModule` | [RoPE.md](RoPE.md) |
| `ScaledDotProductAttentionModule` | [ScaledDotProductAttention.md](ScaledDotProductAttention.md) |
| `SiLUModule` | [SiLU.md](SiLU.md) |
| `SoftmaxModule` | [Softmax.md](Softmax.md) |
| `StochasticGradientDescentModule` | [StochasticGradientDescent.md](StochasticGradientDescent.md) |
| `TransformerBlockModule` | [TransformerBlock.md](TransformerBlock.md) |
| `TransformerLanguageModelModule` | [TransformerLanguageModel.md](TransformerLanguageModel.md) |

All 16 substantive subfolders are covered. Generated `__pycache__` directories are intentionally excluded because they contain Python bytecode caches rather than implementation concepts.

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

The walkthroughs use line ranges from the current source. Blank lines and comments are grouped with the operation they explain: “line by line” means every executable statement is accounted for, without repeating import boilerplate word for word.

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
