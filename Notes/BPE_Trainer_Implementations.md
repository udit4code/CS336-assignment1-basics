# BPE trainer implementations and optimization trade-offs

This note documents the current BPE training code in
`cs336_basics/tokenization/trainers`. Implementations are named by their data
structure and update strategy:

- `NaiveBPETrainer`
- `WordFrequencyBPETrainer`
- `IncrementalBPETrainer`
- `HeapBPETrainer`
- `IntegerBPETrainer`
- `CompactingHeapBPETrainer`

The public default is:

```python
from cs336_basics.tokenization.trainers import train_bpe

# train_bpe delegates to train_bpe_compacting.
vocab, merges = train_bpe(corpus_path, vocab_size, special_tokens)
```

## 1. Keep the three stages separate

```text
BPE training
  raw corpus -> vocabulary and ranked merge rules

BPE encoding
  text + frozen vocabulary/merges -> token IDs

Language-model pretraining
  token IDs -> gradient-based learning of neural-network parameters
```

BPE training is a discrete greedy counting algorithm. It does not use loss,
backpropagation, embeddings, or an optimizer. The vocabulary and merge ranks are
normally frozen before language-model training so each token ID keeps a stable
meaning.

## 2. Exact trainer contract

Input:

- A UTF-8 corpus path.
- A desired total `vocab_size`.
- Zero or more special-token strings.

Output:

```python
vocab: dict[int, bytes]                  # token ID -> bytes
merges: list[tuple[bytes, bytes]]        # chronological learned pairs
```

`initialize_vocab` creates all singleton bytes at IDs `0..255`, then appends
each special token in caller order. Learned tokens use `len(vocab)` as the next
ID and contain the concatenation of the winning pair.

The requested size includes base bytes and special tokens. The current code
does not validate that `vocab_size` is at least the initialized size. If it is
smaller, training performs no merge and still returns the larger initialized
vocabulary. Duplicate special-token strings are also not rejected and receive
distinct IDs in trainer output. Production validation should make both policies
explicit.

Training can stop below the requested size if no adjacent pairs remain.

## 3. Canonical BPE training algorithm

Start each ordinary regex pretoken as singleton UTF-8 byte symbols. At every
iteration:

1. Count all adjacent symbol-pair occurrences across the corpus.
2. Weight each occurrence by the pretoken's corpus frequency.
3. Select the pair with the greatest global count.
4. If counts tie, select the lexicographically greatest pair of byte strings.
5. Merge all non-overlapping occurrences of that pair left-to-right in every
   pretoken.
6. Append the pair to `merges` and its concatenation to `vocab`.

Selection is precisely:

```python
best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
```

Python compares a pair lexicographically: left bytes first, then right bytes.
This deterministic tie-break is part of the repository's observable contract;
changing it changes learned token IDs and merge ranks.

### Overlapping counts versus non-overlapping replacement

Pair counting includes overlapping adjacent windows. In `[a,a,a]`, `(a,a)` has
two occurrences. If that pretoken frequency is two, its global contribution is
four.

Replacement is non-overlapping and left-to-right:

```text
[a, a, a] --merge (a,a)--> [aa, a]
```

Counting and replacement therefore intentionally have different overlap
behavior.

### Byte conservation

For every segmented pretoken `w`:

```text
concat(w before merge) == concat(w after merge)
```

Training changes segmentation and vocabulary, never corpus byte order or
content.

## 4. Pretokenization and multiprocessing

Sources: `trainers/base.py`, `patterns.py`, `naive.py`, and
`word_frequency.py`.

The GPT-2-style regex separates contractions, letters, numbers, punctuation,
and whitespace. Training pairs never cross regex-pretoken boundaries. Special
tokens are inserted directly into vocabulary and excluded from ordinary pair
statistics.

### Safe file boundaries

`find_chunk_boundaries` starts from approximately equal byte offsets and scans
forward until it finds `split_special_token`. Boundaries are deduplicated and
may yield fewer chunks than requested. Ending one chunk and starting the next at
a complete special-token boundary avoids splitting ordinary training content at
an arbitrary byte offset.

Only the first configured special token is used to find chunk boundaries. All
configured special tokens are still removed inside a chunk. With no configured
special token, the code searches for `<|endoftext|>`; if it is absent after a
candidate boundary, that boundary advances to EOF. A corpus with no delimiter
may therefore collapse to one effective chunk and lose multiprocessing speedup.

The naive worker decodes chunk bytes with `errors="ignore"`, while the
word-frequency worker uses strict UTF-8 decoding. Valid UTF-8 corpora split at
valid delimiters behave consistently; malformed input has different failure
semantics between these implementations.

### Process costs

Each worker opens and reads its byte range, decodes, regex-pretokenizes, and
returns either expanded words or a frequency `Counter`. The parent combines
results in submission order. Multiprocessing bypasses the GIL for CPU-bound
regex work, but startup, file reads, pickling, duplicated process memory, and
result transfer can dominate small corpora.

## 5. Complexity notation

Because corpus state shrinks and affected sets vary, a single `O(...)` label is
often misleading. Use:

- `M`: number of learned merges actually performed.
- `C_t`: total symbol occurrences across all expanded pretokens at iteration
  `t`.
- `U_t`: number of distinct segmented pretoken records at iteration `t`.
- `S_t`: total symbols across those distinct records.
- `P_t`: number of distinct active pair types.
- `A_t`: records containing the selected pair.
- `S(A_t)`: total symbols across affected records.
- `Q_t`: number of pair types whose counts change in iteration `t`.

All counts refer to the current segmentation, not raw characters.

## 6. `NaiveBPETrainer`: expanded corpus and global recomputation

Source: `cs336_basics/tokenization/trainers/naive.py`.

The corpus representation is a list containing every regex-pretoken occurrence,
each as a mutable-style list of byte strings. Repeated pretokens are repeated in
memory.

Every learned merge:

1. Scans every current symbol occurrence to rebuild global pair counts.
2. Scans the pair dictionary to select the winner.
3. Scans and reconstructs every pretoken occurrence.

Approximate total merge-loop work is:

```text
O(sum_t (C_t + P_t))
```

with `O(C_t)` corpus-state memory, excluding multiprocessing copies. A coarse
upper bound is `O(M C_0 + sum_t P_t)`, though `C_t` decreases as merges occur.

Strength: the state directly mirrors the definition and is easy to inspect.
Weakness: common pretokens are processed once per occurrence on every merge.

## 7. `WordFrequencyBPETrainer`: compress repeated pretokens

Source: `cs336_basics/tokenization/trainers/word_frequency.py`.

The representation becomes:

```python
Counter({tuple_of_byte_symbols: corpus_frequency})
```

Pair counts multiply within-word occurrence count by word frequency. If a
pretoken occurs a million times, its segmentation is scanned once per merge,
not a million times.

After each winning merge, the trainer still reconstructs every distinct word
and recounts every pair globally:

```text
O(sum_t (S_t + P_t)) time
O(S_t + P_t) principal state
```

This class is the readable optimized reference used by tests. It often provides
the largest real improvement because natural-language pretoken frequencies are
highly skewed.

When two old distinct records become the same tuple after a merge,
`merged_counter[...] += frequency` correctly coalesces them.

## 8. `IncrementalBPETrainer`: inverted index and affected-word updates

Sources: `trainers/incremental.py` and `trainers/_state.py`.

`IncrementalMergeState` assigns stable integer record IDs and stores:

```text
words[word_id]             -> current tuple of symbols
frequencies[word_id]       -> corpus multiplicity
pair_counts[pair]          -> weighted global occurrence count
pair_to_words[pair]        -> record IDs currently containing the pair
```

Initialization counts overlapping pairs inside each word and adds
`occurrences × frequency` globally.

For a winning pair, only IDs in `pair_to_words[pair]` can change. For each such
record, the implementation:

1. Removes all of its old pair contributions and inverted-index memberships.
2. Merges the selected pair left-to-right.
3. Adds all pair contributions from the new segmentation.

This is incremental across records but not maximally local inside a record: it
rescans the complete affected word before and after mutation. Its update cost is
roughly `O(S(A_t))`, plus hash/set operations.

Winner selection still calls `max` over all active pairs:

```text
O(sum_t (S(A_t) + P_t)) after initialization
```

The inverted index adds memory but helps when selected pairs occur in a small
fraction of unique pretokens. If almost every record is affected, the advantage
shrinks. Records that converge to identical tuples remain separate stable IDs;
counts stay correct, but future work may exceed a freshly coalesced Counter.

## 9. `HeapBPETrainer`: incremental state plus lazy max-heap

Source: `cs336_basics/tokenization/trainers/heap.py`.

This class keeps the same incremental corpus state but replaces the `O(P_t)`
winner scan with `LazyMaxPairHeap`.

Python's `heapq` is a min-heap. `_MaxHeapEntry.__lt__` reverses comparison so
the root is ordered by:

1. Greatest count.
2. Greatest lexicographic byte-pair key.

Initial construction creates one entry per positive count and uses
`heapq.heapify`, which is `O(P_0)`.

### Lazy invalidation

Heap entries snapshot a pair and count. When a count changes, `refresh` pushes a
new entry instead of locating/removing the old entry. `pop_best` discards entries
until the stored count equals authoritative `state.pair_counts[pair]`.

Correctness depends on this separation:

- `pair_counts` is the source of truth.
- Heap entries are disposable selection hints.

Each iteration costs roughly:

```text
O(S(A_t)) state maintenance
+ O(Q_t log H_t) refreshed heap entries
+ stale-pop cost amortized over entries previously inserted
```

where `H_t` is physical heap size. Lazy invalidation avoids arbitrary heap
deletion but can retain many stale entries and increase both memory and pop time.

## 10. `IntegerBPETrainer`: compact symbols in the hot loop

Source: `cs336_basics/tokenization/trainers/integer.py`.

The initial byte-tuples are converted once from singleton byte objects to IDs
`0..255`. Each learned token receives ID `len(vocab)`. Incremental words, pairs,
sets, counters, and heap entries then contain integers rather than repeated
variable-length `bytes` objects.

The public output remains byte-based. When pair `(left_id, right_id)` wins:

```python
left_bytes = vocab[left_id]
right_bytes = vocab[right_id]
vocab[new_id] = left_bytes + right_bytes
merges.append((left_bytes, right_bytes))
```

Integer ID order is creation order, not lexicographic byte order. To preserve
the required tie-break, the heap order key is:

```python
(vocab[left_id], vocab[right_id])
```

Comparing `(left_id, right_id)` directly would silently learn different merges.

This optimization reduces hashing/comparison payload and repeated byte-string
storage in the mutable state. It adds vocabulary indirection and a one-time
conversion, so tiny training jobs may not benefit.

## 11. `CompactingHeapBPETrainer`: bound stale-heap amplification

Source: `cs336_basics/tokenization/trainers/compacting.py`.

This is the public default. It is the integer trainer with
`heap_compact_factor = 4`. After refreshing touched pairs, the selector rebuilds
from authoritative positive counts when both conditions hold:

```text
physical heap size >= 1024
physical heap size > 4 × max(1, active pair count)
```

Rebuilding drops every stale snapshot and uses linear-time `heapify`. The 1,024
floor avoids paying rebuild overhead on small heaps; the factor prevents stale
entries from growing without relation to live pair diversity once the heap is
large.

This is a relative stale-entry bound after threshold checks, not a fixed memory
cap. Active pairs, inverted-index sets, word state, and the heap can all grow
with corpus diversity. Compaction does not change BPE semantics because the heap
is reconstructed from the source-of-truth counts.

## 12. Strategy comparison

| Class | Corpus representation | Winner selection | Update scope | Principal trade-off |
|---|---|---|---|---|
| `NaiveBPETrainer` | Every occurrence as byte lists | Full pair scan | Entire expanded corpus | Clearest, highest repetition |
| `WordFrequencyBPETrainer` | Counter of unique byte tuples | Full pair scan | Every unique record | Strong frequency compression |
| `IncrementalBPETrainer` | Stable records + inverted index | Full pair scan | Affected records | More state, less recomputation |
| `HeapBPETrainer` | Same incremental byte state | Lazy max-heap | Affected records | Avoids max scan; stale growth |
| `IntegerBPETrainer` | Incremental integer state | Lazy max-heap | Affected records | Compact hot state; indirection |
| `CompactingHeapBPETrainer` | Incremental integer state | Compacting lazy heap | Affected records | Bounds stale amplification |

These are not six quality levels. Each isolates a systems idea and can win for a
different corpus size, repetition distribution, merge count, and pair density.
The reference implementation is often preferable for debugging; the compacting
trainer is the current general-purpose default.

## 13. Correctness proof sketch for incremental training

Assume before a merge that `pair_counts` equals the weighted sum of adjacent
pairs in all current records and `pair_to_words` contains exactly the records
where each pair occurs.

For the selected pair:

- Records outside its inverted-index set cannot contain it and therefore do not
  change.
- For every affected record, removing all old contributions subtracts exactly
  its previous weighted pairs.
- Deterministic `merge_pair` produces the same left-to-right segmentation as a
  full-corpus implementation.
- Adding all new contributions restores exactly that record's new weighted
  pairs and memberships.

Thus both invariants hold after the update. By induction, incremental and full
recomputation produce identical counts, winners, vocabulary, and merge order,
provided they use the same tie-break.

The heap does not enter this state proof. It is correct because `pop_best`
accepts only entries whose count still matches the authoritative map and its
ordering matches the full `max` key.

## 14. Failure modes and production gaps

- Invalid `vocab_size`, duplicate special tokens, unreadable paths, and
  `desired_num_chunks <= 0` lack complete public validation.
- Special-token alternation is escaped but not sorted longest-first in trainer
  preprocessing; overlapping configured tokens therefore depend on caller
  order, unlike `BaseTokenizer` encoding.
- Only the first special token determines process boundaries.
- Malformed UTF-8 behavior differs between expanded and frequency workers.
- `assert` guards inside `_train` are internal assumptions, not robust public
  validation, because assertions can be disabled.
- Process-based pretokenization can oversubscribe CPU/memory and has no explicit
  backpressure or streaming aggregation.
- The whole unique pretoken state, inverted index, and pair table reside in
  memory; this is not a distributed or external-memory trainer.
- A worker failure is surfaced when its future is consumed; partial work is not
  checkpointed.
- Output artifacts need an explicit serialization/versioning format binding
  vocabulary, ordered merges, regex, normalization policy, and special tokens.

## 15. Testing and benchmarking rigor

Current tests verify:

- Optimized trainers equal `WordFrequencyBPETrainer` on the reference corpus.
- Runtime stays below the assignment budget.
- Overlapping pair counts and frequency weighting.
- Lexicographically greatest tie-breaking.
- Lazy removal of stale counts.
- Heap compaction when stale entries dominate.
- Exact expected vocabulary/merge artifacts and special-token exclusion.

Run:

```bash
.venv/bin/python -m pytest tests/test_train_bpe.py \
  tests/test_train_bpe_naive.py \
  tests/test_train_bpe_versions.py -q
```

Additional production tests should cover empty/tiny corpora, no remaining
pairs, repeated/overlapping special tokens, malformed UTF-8, requested sizes
below initialization, many processes on a tiny file, deterministic output
across process counts, resume/checkpoint behavior, memory limits, and adversarial
pair distributions that create many stale heap entries.

Benchmark end-to-end pretokenization separately from the merge loop. Report
corpus bytes, unique pretokens, total occurrences, average/max pretoken length,
requested merges, process count, wall time, peak RSS, and artifact equality.
Microbenchmarks that exclude process startup or use only repeated words can
reverse the apparent ranking.

## 16. Senior MLE interview questions

**What is the most important optimization?**

Compress repeated pretokens into a frequency counter. It changes work from
being proportional to all corpus occurrences to all unique segmented records,
while preserving exact weighted pair counts.

**Why is an inverted index useful?**

After choosing pair `p`, only records in `pair_to_words[p]` can change. It
converts a global corpus update into a sparse affected-set update at the cost of
sets and consistency bookkeeping.

**Why remove and re-add all pairs for an affected word?**

It is simpler and easier to prove correct than updating only immediate
boundaries. It remains incremental across words, though not optimally local
within long words.

**Why not trust the heap as state?**

Lazy heaps contain stale historical entries. Authoritative counts must live in
a separate map; heap entries are validated snapshots used only to accelerate
selection.

**Why can integer IDs not determine a tied winner?**

IDs encode creation order. The required tie-break compares token byte strings,
so the integer implementation maps IDs back to byte values for its ordering key.

**What does compaction guarantee?**

Once the heap is at least 1,024 entries, it is rebuilt when physical entries
exceed four times active pair count. This controls stale amplification, not
absolute memory and not the size of other trainer state.

**Why is training harder to distribute than initial counting?**

Initial pretoken frequencies combine associatively across workers. Every later
merge is sequentially dependent: the next global winner depends on counts after
the previous global mutation. Distributed implementations need synchronized
winner selection and efficient sparse/global count updates.

**How would you scale beyond one machine?**

Stream and aggregate initial pretoken counts; shard word state and inverted
indices; reduce global pair-count deltas; deterministically select and broadcast
one winner per iteration; checkpoint corpus state and merge history; and measure
communication versus computation. Approximate or batched merges may scale
better but alter the exact greedy algorithm and require quality evaluation.

## 17. Revision checklist

You should be able to derive and explain:

- The trainer's input/output contract and why merge order matters later.
- Greatest-frequency plus greatest-byte-pair tie-breaking.
- Overlapping counting versus non-overlapping merging.
- How pretoken frequency weights pair counts.
- Why safe file boundaries use a special-token delimiter.
- Expanded corpus versus unique-word Counter complexity.
- The four invariants in `IncrementalMergeState`.
- Why affected-word updates are correct but not fully local.
- Lazy heap invalidation and its source of truth.
- Why integer state must preserve byte-based ordering.
- Exactly when compaction triggers and what it does not bound.
- How to test deterministic artifact equality, not merely runtime.
