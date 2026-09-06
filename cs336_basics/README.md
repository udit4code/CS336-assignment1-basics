# Package layout

The package uses short, domain-oriented, PEP 8 module paths. Import public neural-network components from
`cs336_basics.nn`, and tokenizers or BPE trainers from `cs336_basics.tokenization`.

```python
from cs336_basics.nn import AdamW, TransformerLM, gradient_clipping
from cs336_basics.tokenization import Tokenizer
from cs336_basics.tokenization.trainers import train_bpe
```

`Tokenizer` selects the portable linked-list/heap implementation. `train_bpe` selects the compacting integer-heap
trainer. Explicit implementation classes remain available for benchmarking and education.

## Tokenizer naming

| Previous name | Descriptive name | Strategy |
| --- | --- | --- |
| `v1` / `TokenizerV1` | `naive` / `NaiveTokenizer` | Replays every learned merge |
| `v2` / `TokenizerV2` | `rank_scan` / `RankScanTokenizer` | Repeated merge-rank scans |
| `v3` / `TokenizerV3` | `rebuilding_heap` / `RebuildingHeapTokenizer` | Rebuilds a candidate heap after each merge |
| `v4` / `TokenizerV4` | `linked_heap` / `LinkedHeapTokenizer` | Linked nodes with local heap updates |
| `v5` / `TokenizerV5` | `native_batch` / `NativeBatchTokenizer` | Batched native merge loop |
| `v6` / `TokenizerV6` | `cached_native` / `CachedNativeTokenizer` | Native persistent heap with bounded LRU cache |

## Trainer naming

The corresponding trainer modules are `naive`, `word_frequency`, `incremental`, `heap`, `integer`, and
`compacting`. Names describe the algorithm and remain meaningful when implementations evolve.
