# BPE Trainer Optimization: From V1 to V6

## Purpose of this note

This document explains how the repository's BPE **training** implementation evolves from v1 through v6.

Before discussing optimization, we must separate three processes whose similar names cause a great deal of confusion:

```text
1. BPE tokenizer training
2. Tokenization/encoding
3. LLM pretraining
```

Only the first process learns tokenization rules. Only the third process learns neural-network weights.

---

## Start here: BPE Trainer versus BPE Tokenizer versus LLM training

### The shortest possible distinction

```text
BPE Trainer
    Learns how text should be divided into tokens.

BPE Tokenizer
    Uses those learned rules to turn text into token IDs
    and token IDs back into text.

Transformer / LLM
    Uses token IDs to learn neural-network weights that predict
    the next token.
```

The BPE Trainer does not train the Transformer. The BPE Tokenizer does not learn embeddings. The Transformer does not ordinarily decide how raw text should be split into tokens.

They form a pipeline, but they have separate responsibilities.

### Why the word “training” is confusing

There are two completely different training procedures:

```text
Tokenizer training
    Learns discrete vocabulary entries and merge rules.
    Uses pair-frequency counting.
    Does not use gradient descent.
    Does not produce neural-network weights.

LLM pretraining
    Learns floating-point model parameters.
    Uses forward passes, cross-entropy, backpropagation, and an optimizer.
    Does not normally change the tokenizer's merge rules.
```

When someone says “train BPE,” they mean the first procedure. When someone says “pretrain the language model,” they mean the second.

### Input and output of the BPE Trainer

The BPE Trainer receives:

```text
Input 1: raw text corpus path
Input 2: desired total vocabulary size
Input 3: list of special tokens
```

For example:

```python
vocab, merges = train_bpe_v6(
    input_path="tinystories.txt",
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
)
```

It returns two artifacts:

```text
Output 1: vocabulary
          dict[int, bytes]
          token ID → token bytes

Output 2: ordered merge list
          list[tuple[bytes, bytes]]
          pairs ordered by when they were learned
```

A tiny conceptual result might look like:

```python
vocab = {
    0: b"\x00",
    ...,
    97: b"a",
    98: b"b",
    ...,
    256: b"<|endoftext|>",
    257: b"ab",
    258: b"abc",
}

merges = [
    (b"a", b"b"),   # creates b"ab"
    (b"ab", b"c"),  # creates b"abc"
]
```

These are discrete data structures. They are not tensors of learned model parameters.

### What does the BPE Trainer actually learn?

It learns a compression-oriented segmentation scheme from corpus statistics.

Initially, every UTF-8 byte is a separate token. The trainer repeatedly asks:

```text
Which adjacent token pair occurs most frequently?
```

It merges that pair and records the decision. Frequent byte sequences gradually become tokens representing characters, word fragments, spaces plus fragments, or sometimes complete words.

After the requested vocabulary size is reached, the trainer stops. The resulting vocabulary and ordered merge list are normally frozen.

### Input and output of the BPE Tokenizer

The BPE Tokenizer is constructed from the Trainer's outputs:

```python
tokenizer = TokenizerV6(
    vocab=vocab,
    merges=merges,
    special_tokens=["<|endoftext|>"],
)
```

Its encoding operation receives:

```text
Input:  ordinary text string
Output: list of integer token IDs
```

For example:

```python
token_ids = tokenizer.encode("abc")
```

Using the conceptual vocabulary above:

```text
"abc"
  → UTF-8 bytes [a, b, c]
  → apply learned merge (a,b)
  → [ab, c]
  → apply learned merge (ab,c)
  → [abc]
  → vocabulary lookup
  → [258]
```

The Tokenizer's decoding operation reverses the ID-to-byte lookup:

```text
Input:  list of token IDs
Output: decoded text string
```

```python
text = tokenizer.decode([258])
```

Conceptually:

```text
[258]
  → vocab[258]
  → b"abc"
  → UTF-8 decode
  → "abc"
```

Decoding does not replay the merge algorithm backward. It simply maps IDs to stored bytes, concatenates those bytes, and UTF-8 decodes them.

### Relationship between merge frequency and merge rank

During BPE training, the Trainer repeatedly selects the pair with the greatest current corpus frequency.

During later tokenization, the corpus is no longer available and frequencies are no longer recalculated. The Tokenizer uses the order in which merges were learned:

```text
earlier merge in the list → smaller merge rank → higher priority
```

Therefore:

```text
Trainer asks:   Which pair is most frequent in the training corpus now?
Tokenizer asks: Which currently adjacent pair has the earliest learned rank?
```

The Trainer produces the priority rules. The Tokenizer applies them.

### Input and output of the Transformer language model

The language model does not receive strings or byte sequences. It receives the integer IDs produced by the Tokenizer:

```text
Input:  token ID tensor, shape (batch_size, sequence_length)
Output: logits, shape (batch_size, sequence_length, vocabulary_size)
```

Inside this repository:

```text
token IDs
    → Embedding table lookup
    → Transformer blocks
    → Language-model head
    → one score for every possible next token
```

The embedding table has shape:

```text
(vocabulary_size, d_model)
```

If token ID `258` occurs, the model retrieves:

```python
embedding.weight[258]
```

This embedding vector is not supplied by the tokenizer. When pretraining from scratch, it is randomly initialized as part of the model and learned through gradient descent.

### What is frozen and what is learned during LLM pretraining?

Normally:

```text
Frozen during LLM pretraining:
    vocabulary
    token-to-ID mapping
    merge list/ranks
    special-token IDs

Learned during LLM pretraining:
    token embedding matrix
    attention projection weights
    feed-forward weights
    normalization scales
    language-model output head
```

The token ID assigned to a byte sequence must not change halfway through model training. If ID `258` meant `b"abc"` yesterday and something else today, the learned embedding row would no longer have a stable meaning.

### The complete LLM pretraining pipeline

```text
PHASE A — Train the tokenizer once

Tokenizer-training text corpus
        │
        ▼
     BPE Trainer
        │
        ├── vocabulary: ID → bytes
        └── ordered merge list
        │
        ▼
Freeze and save tokenizer artifacts


PHASE B — Tokenize the LLM datasets

Full LLM training text ──┐
                         ├── same frozen BPE Tokenizer ──► train token IDs
Validation text ─────────┘                              └─► validation token IDs


PHASE C — Pretrain the language model

train token IDs
        │
        ├── create shifted input/target windows
        ▼
randomly initialized Transformer
including random token embeddings
        │
        ├── forward pass
        ├── next-token cross-entropy
        ├── backpropagation
        └── optimizer updates model weights
        │
        ▼
trained model checkpoint


PHASE D — Inference

prompt string
        │
        ▼
same frozen BPE Tokenizer
        │
        ▼
prompt token IDs
        │
        ▼
trained Transformer checkpoint
        │
        ▼
next-token ID
        │
        ├── append and repeat generation
        ▼
same frozen BPE Tokenizer.decode(...)
        │
        ▼
generated text
```

### Input/output table for the whole pipeline

| Component | Input | Output | What is learned? |
|---|---|---|---|
| BPE Trainer | Raw text corpus, target vocab size, special tokens | Vocabulary and ordered merges | Discrete merge rules from frequency counts |
| BPE Tokenizer encode | Text plus frozen vocabulary/merges | Integer token IDs | Nothing |
| Dataset/batcher | Long token-ID stream | Shifted input and target tensors | Nothing |
| Transformer | Input token IDs | Vocabulary logits | Floating-point neural-network weights |
| Cross-entropy | Logits and target IDs | Scalar loss | Nothing directly; supplies gradients |
| Optimizer | Model parameters and gradients | Updated model parameters | Updates embeddings and other weights |
| BPE Tokenizer decode | Generated IDs plus frozen vocabulary | Text | Nothing |

### Where `tiktoken` fits

If you use:

```python
encoding = tiktoken.get_encoding("gpt2")
```

you are selecting an already-defined tokenizer vocabulary and merge-ranking table. You do not need to run this repository's BPE Trainer unless you want to learn your own tokenizer.

However, `tiktoken` still does not provide a trained language model or token embeddings. It only supplies tokenization artifacts and efficient encode/decode logic.

Two valid pipeline choices are therefore:

```text
Choice 1: custom tokenizer
    text → this repo's BPE Trainer → this repo's BPE Tokenizer → token IDs

Choice 2: existing tokenizer
    text → tiktoken GPT-2 encoding → token IDs
```

In either case, the Transformer embedding table is learned separately during LLM pretraining.

### Why vocabulary compatibility is non-negotiable

The following values must agree:

```text
tokenizer vocabulary size
model embedding row count
model LM-head output size
meaning of every token ID
special-token IDs
```

If the tokenizer can output IDs from `0` through `9,999`, the model needs at least 10,000 embedding rows and 10,000 output logits.

The same tokenizer artifacts must be used for:

- Training tokenization
- Validation tokenization
- Test tokenization
- Inference prompt encoding
- Generated-token decoding

### A practical lifecycle for this repository

If using the custom BPE implementation:

```text
1. Train BPE on a representative text corpus.
2. Save vocab and merges.
3. Construct one tokenizer from those saved artifacts.
4. Encode training and validation text into integer arrays.
5. Set TransformerLM.vocab_size to the tokenizer vocabulary size.
6. Randomly initialize TransformerLM, including its embedding table.
7. Train model weights on shifted token-ID batches.
8. Save the model checkpoint and tokenizer artifacts together.
9. At inference, restore both and generate token IDs.
10. Decode generated IDs with the same tokenizer.
```

If using `tiktoken`, replace steps 1–3 with selection of a fixed encoding such as GPT-2, and record that encoding name with the model configuration.

### One-sentence memory aid

> The BPE Trainer learns the dictionary, the BPE Tokenizer uses the dictionary to produce IDs, and the Transformer learns what those IDs mean in context.

---

## Optimization roadmap after the conceptual distinction

BPE training and BPE tokenization are related, but they solve different computational problems:

```text
BPE training
    Input:  a text corpus, target vocabulary size, special tokens
    Output: a vocabulary and an ordered merge list

BPE tokenization
    Input:  new text, the learned vocabulary, the learned merge list
    Output: token IDs
```

The trainer discovers merge rules from corpus-wide frequencies. The tokenizer later applies those fixed rules to new text. Consequently, an optimization useful for encoding—such as caching repeated input strings—may not be the right optimization for training.

The six trainer versions are:

| Version | Main representation | Pair selection | Pair-count update |
|---|---|---|---|
| v1 | Every pretoken occurrence as byte lists | Full maximum scan | Recount entire expanded corpus |
| v2 | Counter of unique byte tuples | Full maximum scan | Recount every unique word |
| v3 | Unique byte tuples plus inverted index | Full maximum scan | Update only affected words |
| v4 | Same byte representation | Lazy max-heap | Update only affected words |
| v5 | Integer token IDs | Lazy max-heap | Update only affected words |
| v6 | Integer token IDs | Compacting lazy max-heap | Update only affected words |

---

## 1. BPE training from first principles

### Initial vocabulary

Byte-level BPE begins with all 256 possible byte values:

```text
vocab[0]   = b"\x00"
vocab[1]   = b"\x01"
...
vocab[255] = b"\xff"
```

Special tokens are then appended as indivisible vocabulary entries:

```text
vocab[256] = b"<|endoftext|>"
```

Special tokens are excluded from ordinary pair learning. They must never accidentally merge with surrounding text.

### Pretokenization

The GPT-2-style regular expression divides text into pretokens. BPE merges never cross a pretoken boundary.

For example, the text:

```text
" a cat"
```

might produce pretokens similar to:

```text
" a"
" cat"
```

Each pretoken is UTF-8 encoded and initially represented as individual bytes.

### Training iteration

Every BPE iteration performs four conceptual operations:

1. Count every adjacent token pair across the corpus.
2. Select the most frequent pair.
3. If frequencies tie, choose the lexicographically greatest byte pair.
4. Merge every non-overlapping occurrence of that pair inside each pretoken.

The merged byte string is appended to the vocabulary, and the chosen pair is appended to the ordered merge list.

Training stops when:

- The requested vocabulary size is reached, or
- No adjacent pairs remain.

---

## 2. A running example

Suppose pretokenization produces:

```text
"aaab" with frequency 2
"ab"   with frequency 3
```

Initially:

```text
"aaab" → [a, a, a, b]
"ab"   → [a, b]
```

### Initial pair counts

For `[a,a,a,b]`:

```text
(a,a) occurs twice per word × frequency 2 = 4
(a,b) occurs once  per word × frequency 2 = 2
```

For `[a,b]`:

```text
(a,b) occurs once per word × frequency 3 = 3
```

Global counts are:

```text
(a,a) → 4
(a,b) → 5
```

The trainer selects `(a,b)` and creates token `ab`:

```text
[a,a,a,b] → [a,a,ab]
[a,b]     → [ab]
```

Only words containing `(a,b)` changed. That observation becomes the key optimization in v3.

### Overlapping counts versus non-overlapping replacement

In `[a,a,a]`, pair `(a,a)` appears at positions `(0,1)` and `(1,2)`, so its frequency contribution is two.

But if `(a,a)` is selected, replacement proceeds left to right without overlap:

```text
[a,a,a] → [aa,a]
```

The middle `a` cannot participate in two replacements during the same merge iteration.

Counting adjacent candidates and applying non-overlapping replacement are therefore related but distinct operations.

---

## 3. Notation for complexity

Let:

| Symbol | Meaning |
|---|---|
| `C` | Total number of pretoken occurrences before frequency compression |
| `U` | Number of unique pretokens |
| `L` | Total number of current token pieces across unique pretokens |
| `P` | Number of distinct adjacent pairs currently present |
| `M` | Number of merges learned |
| `A_m` | Pieces contained in words affected by merge `m` |

These quantities change during training because every merge shortens affected segmentations.

Big-O expressions describe scaling pressure, not exact runtime. Python object allocation, hashing byte strings, process startup, cache locality, and corpus distribution can dominate small benchmarks.

---

## 4. V1: expanded occurrences and complete recomputation

Source: `BPE_Trainer/v1.py`

### Representation

V1 stores every pretoken occurrence separately:

```python
words = [
    [b"a", b"a", b"a", b"b"],
    [b"a", b"a", b"a", b"b"],
    [b"a", b"b"],
    [b"a", b"b"],
    [b"a", b"b"],
]
```

If a word appears one million times, its byte-token list is represented one million times.

### What v1 already does well

- Splits the input file at safe special-token boundaries.
- Pretokenizes chunks in worker processes.
- Implements correct non-overlapping replacement.
- Uses the required deterministic selection key:

```python
max(pair_counts.items(), key=lambda item: (item[1], item[0]))
```

The word “naive” describes the merge loop, not necessarily every part of the ingestion pipeline.

### Per-merge behavior

For every learned merge, v1:

1. Traverses every occurrence to recount all pairs.
2. Scans the pair table to select the best pair.
3. Traverses every occurrence again to apply the merge.
4. Allocates a new list for every word occurrence.

### Approximate complexity

If the expanded corpus contains `L_expanded` current pieces, each merge costs roughly:

```text
O(L_expanded + P)
```

Across `M` merges:

```text
O(M × L_expanded)
```

as a useful worst-case mental model.

### Tradeoff

V1 is the easiest version to reason about and is an excellent correctness oracle on small inputs. Its repeated data and repeated whole-corpus scans make it unsuitable for larger training corpora.

### Optimization pressure

Natural language repeats pretokens heavily. We should store each distinct pretoken once and attach a frequency.

---

## 5. V2: compress repeated pretokens

Source: `BPE_Trainer/v2.py`

### Representation change

V2 replaces the expanded list with a `Counter`:

```python
word_counter = {
    (b"a", b"a", b"a", b"b"): 2,
    (b"a", b"b"): 3,
}
```

Pair counts are weighted by each word's corpus frequency:

```python
for word, frequency in word_counter.items():
    for pair in zip(word, word[1:]):
        pair_counter[pair] += frequency
```

### Why this is a major improvement

If one pretoken occurs a million times, v2 processes its segmentation once per merge rather than one million times.

The amount of merge-loop work moves from the number of occurrences `C` toward the number of unique pretokens `U`.

### Additional improvements

- A table pre-creates the 256 one-byte `bytes` objects.
- Worker processes return local `Counter` objects.
- The parent combines counters rather than enormous lists of occurrences.

### Remaining bottleneck

After selecting one pair, v2 still:

1. Traverses every unique word.
2. Rebuilds a complete `Counter` of merged words.
3. Recalculates every pair frequency from scratch.

Most words usually do not contain the selected pair, so most of this work is redundant.

### Approximate complexity

If unique current segmentations contain `L` pieces:

```text
per merge: O(L + P)
total:     O(M × (L + P))
```

This is frequently far better than v1 because `L` excludes duplicate word occurrences, but it still performs global work for every local change.

### Tradeoff

V2 is compact, readable, and fast enough for the assignment reference corpus. Its full recomputation is robust and simple, which can beat more elaborate structures on tiny inputs.

### Optimization pressure

When pair `(x,y)` is selected, only words containing `(x,y)` can change. We need a reverse mapping from a pair to those words.

---

## 6. V3: incremental counts through an inverted index

Sources: `BPE_Trainer/v3.py` and `_incremental.py`

### New data structures

V3 assigns each unique word a stable integer ID and maintains:

```text
words[word_id]             → current segmented word
frequencies[word_id]       → corpus frequency
pair_counts[pair]          → weighted global occurrence count
pair_to_words[pair]        → IDs of words containing that pair
```

The last structure is an inverted index.

For the running example:

```text
pair_to_words[(a,b)] = {word_0, word_1}
pair_to_words[(a,a)] = {word_0}
```

### Local update algorithm

When `(a,b)` wins:

1. Read `pair_to_words[(a,b)]`.
2. Visit only those affected word IDs.
3. Subtract each old word's weighted pair contributions.
4. Merge `(a,b)` within that word.
5. Add the new word's weighted pair contributions.
6. Update the inverted index.

Unrelated words are never traversed.

### Why subtract and re-add the affected word?

A merge changes only local boundaries, so an even more specialized algorithm could update just neighboring pairs. V3 instead removes and rebuilds all pair contributions for each affected unique word.

This is a deliberate middle ground:

- Much less work than scanning the whole corpus.
- Considerably simpler and safer than tracking every pair occurrence node.
- Easy to verify against v2.

### Pair selection is still global

V3 still finds the winner using:

```text
max over all P active pair counts
```

Therefore:

```text
per merge ≈ O(P + A_m)
```

and across training:

```text
O(Σ_m (P_m + A_m))
```

### Tradeoff

V3 introduces more memory and bookkeeping:

- Stable word IDs
- Global counts
- A set of word IDs for every pair

In exchange, it avoids revisiting unaffected words. The global maximum scan becomes the next visible bottleneck.

---

## 7. V4: lazy max-heap pair selection

Source: `BPE_Trainer/v4.py`

### Why a heap?

V3 scans all `P` active pairs to find the maximum after every merge. A heap can return its highest-priority entry in logarithmic time after updates.

The desired ordering is:

```text
larger frequency wins
if tied, lexicographically greater byte pair wins
```

Python's `heapq` is a min-heap, so the custom heap entry reverses this comparison. The root behaves like a maximum.

### Why entries become stale

Suppose the heap contains:

```text
count 20 for pair (a,b)
```

After a different merge changes affected words, `(a,b)` may now have count 13. Removing an arbitrary old heap entry is awkward and costly.

V4 uses lazy invalidation:

1. Leave the old count-20 entry in the heap.
2. Push a new count-13 entry.
3. When count-20 reaches the top, compare it with the current authoritative `pair_counts` value.
4. Discard it because it is stale.

### Authoritative state

The heap is only a candidate index. `pair_counts` remains the source of truth.

An entry is valid only when:

```text
entry.count == pair_counts[entry.pair]
```

### Approximate complexity

If a merge changes `K_m` distinct pair counts:

```text
pair updates: O(K_m log H)
word work:    O(A_m)
selection:    amortized heap pops
```

where `H` is heap size, including stale entries.

### Tradeoff

V4 avoids the `O(P)` maximum scan, but:

- Heap pushes cost `O(log H)`.
- Stale entries consume memory.
- Small corpora may not benefit.
- Correct tie-breaking is more subtle.

This version optimizes selection while retaining readable byte-string tokens.

### Optimization pressure

Byte strings are convenient at the API boundary but expensive in the hot loop. Pair keys contain Python objects, and every merge concatenates bytes. Internally, token IDs can be much smaller and cheaper.

---

## 8. V5: integer symbols in the hot loop

Source: `BPE_Trainer/v5.py`

### Representation change

V5 represents initial byte tokens using IDs `0...255`:

```text
b"a" → 97
b"b" → 98
```

When `(97,98)` is selected, the next vocabulary ID becomes the merged symbol:

```text
new ID 257 → b"ab"
```

The corpus state stores:

```text
(97, 97, 97, 98)
```

instead of:

```text
(b"a", b"a", b"a", b"b")
```

### Why integers help

- Integer pairs are cheaper to hash than variable-length byte strings.
- Merged words store compact IDs rather than repeated byte content.
- Merging creates one new integer ID instead of concatenating bytes at every corpus occurrence.
- The vocabulary table stores the byte value once per learned token.

### Critical correctness trap: tie-breaking

Token IDs express creation order, not lexicographic byte order.

It would be wrong to break frequency ties using:

```python
max(integer_pair)
```

V5 converts the pair to its vocabulary byte values for priority:

```python
(vocab[left_id], vocab[right_id])
```

Thus v5 preserves exactly the same result as v2–v4.

### Boundary conversion

Pretokenization still produces byte tuples. V5 converts them to integer tuples once before the merge loop. At the end, learned integer pairs are emitted as the public byte-pair merge list expected by tokenizers.

### Tradeoff

V5 has more indirection: debugging a sequence of IDs requires consulting `vocab`. It substantially reduces hot-loop object weight while preserving the external API.

This repository's tokenizer v5 uses C++, but trainer v5 intentionally focuses first on representation. Trainer and tokenizer version numbers describe their own optimization journeys; they need not use identical mechanisms.

---

## 9. V6: bound lazy-heap growth through compaction

Source: `BPE_Trainer/v6.py`

### The remaining v5 problem

Lazy invalidation avoids expensive arbitrary heap deletion, but stale entries accumulate.

Conceptually, the heap may contain:

```text
(a,b), old count 100
(a,b), old count 80
(a,b), old count 60
(a,b), current count 42
```

Only the last entry is useful. The others still consume memory and will eventually need to be popped.

### Adaptive compaction

V6 monitors:

```text
heap entries / active pair counts
```

When the heap is sufficiently large and exceeds four times the number of active pairs, it rebuilds the heap from authoritative current counts.

In the repository:

```text
compact_factor = 4
compact_min_size = 1,024
```

The minimum prevents constant rebuilding on tiny heaps.

### Why rebuilding can be cheaper

Heap rebuilding is `O(P)`. That looks expensive, but occasionally paying `O(P)` can be cheaper than:

- Retaining many stale Python objects
- Performing repeated stale heap pops
- Increasing `log H` as `H` grows far beyond active pair count

This is an amortization tradeoff: tolerate cheap stale entries temporarily, then compact in bulk.

### Tradeoff

The compaction threshold is heuristic and workload-dependent.

- Too small: rebuild too frequently and lose the advantage of lazy deletion.
- Too large: waste memory and time processing stale entries.

V6 does not change BPE semantics. It bounds a performance side effect of v4/v5's selection data structure.

---

## 10. Cumulative optimization map

```text
V1
Expanded pretoken occurrences
Full recount + full rewrite
        │
        │ compress identical pretokens
        ▼
V2
Unique words with corpus frequencies
Still full recount + full rewrite
        │
        │ index pair → affected words
        ▼
V3
Incremental corpus updates
Still scans every active pair for maximum
        │
        │ lazy maximum-priority heap
        ▼
V4
Incremental updates + faster pair selection
Byte objects remain in hot loop
        │
        │ replace internal byte tokens with IDs
        ▼
V5
Compact integer corpus state
Lazy heap can accumulate stale entries
        │
        │ adaptive heap rebuilding
        ▼
V6
Integer state + bounded lazy-heap growth
```

---

## 11. Correctness invariants shared by every version

An optimization is valid only if it preserves all of these rules.

### Same initial vocabulary

IDs `0...255` represent individual bytes. Special tokens follow in caller-supplied order.

### Same pretoken boundaries

Merges cannot cross GPT-2 regex pretoken boundaries.

### Same special-token behavior

Special-token text is excluded from ordinary merge learning and remains indivisible.

### Same weighted pair counts

If a unique word occurs `f` times and a pair appears `k` times in it, the global contribution is `f × k`.

### Same deterministic winner

Choose by:

```text
(frequency, byte_pair)
```

with the maximum tuple.

### Same non-overlapping merge semantics

Replacement proceeds from left to right, consuming two pieces when a match is merged.

### Same output API

Every version returns:

```python
(
    dict[int, bytes],
    list[tuple[bytes, bytes]],
)
```

The merge list is ordered by learning iteration.

---

## 12. Tests that protect the optimization sequence

`tests/test_train_bpe_versions.py` verifies that v3–v6 exactly match v2 on the complete 500-token reference training case.

It also tests:

- Weighted pair counting
- Overlapping adjacent-pair counts
- Non-overlapping replacement results
- Lexicographic tie-breaking
- Stale heap-entry rejection
- Adaptive heap compaction
- The 1.5-second assignment speed budget for every new version

The speed test and correctness comparison are intentionally separate:

```text
fast but wrong  → fails correctness
correct but slow → fails speed budget
```

Observed local timings on the small reference corpus were approximately:

```text
v3: 0.22 seconds
v4: 0.20 seconds
v5: 0.21 seconds
v6: 0.18 seconds
```

These are smoke measurements, not rigorous benchmarks. Process startup, machine load, filesystem cache, CPU model, corpus size, and vocabulary target affect results.

---

## 13. Why a theoretically later version may not always benchmark faster

### Fixed overhead dominates small corpora

Multiprocessing startup and regex pretokenization can dominate a 130 KB test file. Improving the merge loop cannot remove that fixed cost.

### A heap has constants

For small `P`, scanning a Python dictionary can beat heap allocation, comparison, and stale-entry management.

### Integer conversion has an upfront cost

V5 converts the byte corpus to IDs before training. A short merge run may not perform enough hot-loop work to repay that conversion.

### Compaction is insurance

V6 may behave almost exactly like v5 when the heap never becomes stale enough to cross its threshold. Its benefit is bounded degradation on longer runs, not a guaranteed speedup on every tiny input.

### Corpus distribution matters

If a winning pair occurs in nearly every unique word, v3's affected set is almost the entire corpus and local updates save little. If winning pairs are sparse, the inverted index is extremely valuable.

---

## 14. Memory tradeoffs

### V1

Largest corpus representation because repeated pretokens are duplicated.

### V2

Compresses repeated pretokens, usually providing the biggest immediate memory reduction.

### V3

Adds `pair_to_words` sets and stable word metadata. Uses more auxiliary memory than v2 to save repeated computation.

### V4

Adds a heap whose stale entries can grow beyond active pair count.

### V5

Reduces corpus-token and pair-key weight by replacing byte objects with integers. Retains heap and inverted-index overhead.

### V6

Periodically discards stale heap entries, trading occasional rebuild work for a more controlled memory footprint.

---

## 15. When should each version be used?

### Use v1 when

- Learning the algorithm from scratch
- Debugging a tiny corpus
- Needing the simplest correctness oracle

### Use v2 when

- You want a compact, readable baseline
- The unique pretoken set is modest
- Simplicity matters more than maximum scalability

### Use v3 when

- Selected pairs affect a small fraction of unique words
- Global pair selection is not yet the dominant cost
- You want incremental behavior without heap complexity

### Use v4 when

- The number of active distinct pairs is large
- Repeated `max()` scans are costly
- Heap memory growth remains acceptable

### Use v5 when

- The merge loop dominates runtime
- Python byte-string hashing and storage are material costs
- You can tolerate a less human-readable internal representation

### Use v6 when

- Training learns many merges
- Lazy heap history becomes large
- You want the most bounded Python implementation in this sequence

---

## 16. Interview questions and answers

### What is the biggest conceptual optimization from v1 to v2?

Replace repeated pretoken occurrences with one unique segmentation plus a corpus frequency. Pair contributions are multiplied by frequency.

### What is the biggest conceptual optimization from v2 to v3?

Exploit locality. A selected pair can change only words containing that pair, so maintain a pair-to-word inverted index and update only those words.

### Why does v3 still scan all pairs?

Its global `pair_counts` table changes incrementally, but it still uses `max()` to locate the best active pair. V4 targets that remaining global selection cost.

### Why use lazy heap invalidation?

Python's binary heap does not efficiently update or delete an arbitrary existing entry. Pushing a replacement and validating candidates when popped is simpler and often faster.

### What is the source of truth in v4–v6?

`pair_counts`, not the heap. Heap entries are cached candidates and may be stale.

### Why can v5 not use integer IDs for lexicographic tie-breaking?

IDs encode vocabulary creation order. The assignment requires comparison of the actual byte strings in the pair, so v5 obtains the ordering key from `vocab`.

### Why is there no LRU cache in trainer v6 like tokenizer v6?

Training already compresses repeated pretokens into a frequency counter. Their repetition is represented once before the merge loop, so an encoding-style cache would duplicate an optimization already achieved by v2.

### Does counting `(a,a)` twice in `[a,a,a]` conflict with merging only once?

No. Pair frequency counts adjacent candidate positions, which may overlap. Once the pair is selected, replacement follows deterministic non-overlapping left-to-right semantics.

### Why not always jump straight to v6?

Each intermediate version isolates one idea and is easier to validate. Complex data structures have memory costs and can lose on small workloads. Optimization should follow measured bottlenecks.

### What must be benchmarked besides elapsed time?

Peak memory, preprocessing time, merge-loop time, number of unique words, active pair count, affected words per merge, heap size, and stale-entry rate.

---

## 17. Final revision checklist

You should be able to explain:

1. Why BPE training outputs a vocabulary and ordered merge list.
2. Why repeated pretokens can be compressed into a frequency counter.
3. Why a selected pair changes only words containing it.
4. How an inverted index enables incremental count maintenance.
5. Why a heap entry can become stale.
6. Why `pair_counts` remains authoritative.
7. Why integer IDs reduce hot-loop cost.
8. Why tie-breaking must still compare bytes in v5/v6.
9. Why adaptive heap rebuilding bounds lazy-deletion overhead.
10. Why later asymptotic designs do not guarantee faster tiny-corpus benchmarks.

The overall optimization pattern is reusable beyond BPE:

```text
remove duplicate work
    → update only affected state
    → index expensive global selection
    → use compact internal representations
    → bound the auxiliary data structure's degradation
```
