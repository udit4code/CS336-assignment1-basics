# BPE tokenizer implementations and optimization trade-offs

This note documents the current encoding implementations in
`cs336_basics/tokenization`. The implementations are named by strategy, not by
version:

- `NaiveTokenizer`
- `RankScanTokenizer`
- `RebuildingHeapTokenizer`
- `LinkedHeapTokenizer`
- `NativeBatchTokenizer`
- `CachedNativeTokenizer`

The portable public default is:

```python
from cs336_basics.tokenization import Tokenizer

# Tokenizer is LinkedHeapTokenizer.
```

The native implementations are optional and require `make native`.

## 1. Training and encoding are different algorithms

BPE training learns two artifacts from a corpus:

```python
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

Encoding treats those artifacts as immutable. It does not count corpus pairs or
learn new tokens. If

```python
merges = [(b"l", b"o"), (b"lo", b"w")]
```

then the list index is the merge rank:

```text
(l, o)   rank 0   higher priority
(lo, w)  rank 1   lower priority
```

At each encoding step, the legal adjacent pair with the smallest rank wins.
Training uses the opposite kind of priority: greatest current corpus frequency,
then a deterministic tie-break.

## 2. Shared tokenizer contract

`BaseTokenizer` owns vocabulary loading, special-token handling,
pretokenization, streaming, and decoding. Subclasses implement only the BPE
merge loop for an ordinary regex pretoken.

### Construction

The constructor copies `vocab` into `id_to_token` and builds its reverse map
`token_to_id`. Missing special-token byte strings are appended at consecutive
IDs beginning after the greatest existing ID—not necessarily at `len(vocab)`.

The implementation assumes ordinary vocabulary values are unique. If two IDs
map to the same bytes, the reverse dictionary keeps only the later insertion,
so the format is not truly bijective.

`from_files` reads GPT-2-formatted assets:

- JSON vocabulary: printable GPT-2 byte representation → integer ID.
- Merge text: two encoded symbols per non-comment line.
- `decode_gpt2_token` reverses GPT-2's byte-to-printable-Unicode mapping.

The native constructors additionally verify that every merge operand and its
concatenated result exist in the vocabulary. Pure-Python constructors defer
most malformed-artifact failures until lookup or encoding.

### Encoding pipeline

```text
input str
  -> split around registered special tokens
  -> emit special-token IDs directly
  -> GPT-2 regex over each ordinary chunk
  -> UTF-8 bytes for each regex pretoken
  -> BPE merge loop, independently per pretoken
  -> vocabulary IDs
```

The GPT-2-style regex separates contractions, letter runs, number runs,
punctuation, and whitespace. BPE never crosses a regex-pretoken boundary.

Special-token alternatives are escaped and sorted longest-first. This ensures
that when registered special tokens overlap, a longer token is tested before
its prefix. Special tokens bypass both regex pretokenization and BPE.

`encode_iterable` calls `encode` on each string and yields IDs lazily across
items. Each item is an independent text boundary; concatenating the input
strings first can tokenize differently because regex pretokens may cross that
join.

### Why byte-level BPE is open-vocabulary

Every valid Python string has a UTF-8 byte representation. If the base
vocabulary contains all 256 singleton bytes, any input can be represented
before merges:

```text
"A"  -> 41
"é"  -> c3 a9
"🙃" -> f0 9f 99 83
```

This eliminates an unknown-token requirement for valid text. It does not mean
that one Unicode character maps to one token.

### Decoding

```python
data = b"".join(id_to_token[token_id] for token_id in ids)
text = data.decode("utf-8", errors="replace")
```

Decoding does not reverse the merge history; it concatenates stored bytes.
`decode(encode(text)) == text` should hold for valid strings and valid tokenizer
artifacts. Arbitrary token-ID sequences can concatenate to invalid UTF-8, in
which case the implementation emits replacement characters. Therefore decoding
is intentionally not bijective over every possible ID sequence.

## 3. Correct BPE semantics

For one pretoken with initial UTF-8 byte length `L`, maintain a segmentation

```text
s = [s_0, s_1, ..., s_(m-1)]
```

whose concatenation always equals the original byte string. A legal operation
replaces adjacent `(s_i, s_(i+1))` by `s_i + s_(i+1)` if that pair has a learned
rank.

The invariants are:

1. Bytes are never reordered, inserted, or removed.
2. Only adjacent symbols merge.
3. The smallest available merge rank wins.
4. Equal-rank occurrences resolve left-to-right.
5. Overlapping occurrences cannot both consume the same symbol.
6. Final pieces must map to vocabulary IDs.
7. Merges remain inside one regex pretoken.

Example:

```text
merges: (l,o)=0, (lo,w)=1, (e,r)=2, (low,er)=3

[l, o, w, e, r]
[lo,   w, e, r]
[low,     e, r]
[low,     er]
[lower]
```

For `(a,a)` in `[a,a,a]`, left-to-right non-overlap produces `[aa,a]`, not
`[a,aa]` and not an attempt to merge both overlapping occurrences.

## 4. Complexity notation

- `R`: total learned merge rules.
- `L`: initial UTF-8 byte length of one pretoken.
- `K`: successful merges for that pretoken, `0 <= K <= L-1`.
- `P`: number of currently mergeable adjacent candidates, `P <= L-1`.

The merge loop runs per regex pretoken. Whole-text cost is the sum across
pretokens plus Python regex and special-token processing.

Big-O below describes structural operations. Python object allocation, byte
concatenation, cache locality, native-boundary conversion, and typical short
pretokens can dominate observed latency.

## 5. `NaiveTokenizer`: replay learned rules

Source: `cs336_basics/tokenization/naive.py`.

Algorithm:

```python
pieces = singleton_utf8_bytes(pretoken)
for pair in merges:                 # rank order
    pieces = merge_all_left_to_right(pieces, pair)
return ids(pieces)
```

It scans the current segmentation once for every learned rule, including rules
irrelevant to the input. Applying all non-overlapping occurrences of one rule in
one pass is consistent with ranked BPE: a later learned token cannot be needed
to create an operand for an earlier rule.

- Worst-case time: `O(RL)`.
- Peak segmentation memory: `O(L)`.
- Hidden cost: a fresh Python list on every rule pass.

This is simple and useful as a conceptual oracle, but runtime depends on the
entire merge table even for a one-byte pretoken.

## 6. `RankScanTokenizer`: query only pairs that exist

Source: `cs336_basics/tokenization/rank_scan.py`.

The constructor builds `pair -> rank`. Each iteration scans the current adjacent
pairs, retains the strictly smallest rank, merges one occurrence by list
slicing, and repeats.

Strict rank comparison preserves the first occurrence encountered, giving the
leftmost tie-break.

- One selection scan and list reconstruction: `O(m)` at current length `m`.
- At most `L-1` successful merges.
- Worst-case time: `O(L²)`.
- Peak live segmentation memory: `O(L)`.

This removes `R` from the hot loop. Despite a worse-looking dependence on `L`,
it is usually much faster than replaying tens of thousands of rules because
natural-language pretokens are generally short.

## 7. `RebuildingHeapTokenizer`: priority queue with the wrong lifetime

Source: `cs336_basics/tokenization/rebuilding_heap.py`.

For every iteration it:

1. Scans all current adjacent pairs.
2. Pushes each legal `(rank, index)` into a min-heap.
3. Pops one winner.
4. Reconstructs the Python list after that merge.
5. Discards the heap.

The tuple ordering implements smallest rank, then leftmost index. However, the
heap accelerates selection only after its construction, and this implementation
rebuilds it before every single pop.

- Current repeated-`heappush` build: `O(m log m)`.
- Worst-case time: `O(L² log L)`.
- `heapify` could reduce each build to `O(m)`, but would still rediscover every
  unaffected candidate after each merge.

This class is deliberately educational: choosing an asymptotically useful data
structure does not help if its reusable state is thrown away.

## 8. `LinkedHeapTokenizer`: persistent candidates and local mutation

Source: `cs336_basics/tokenization/linked_heap.py`.

This is the portable default. It combines:

- A doubly linked list of `Node(value, position, prev, next, alive)` objects.
- A min-heap of `HeapEntry(rank, position, left_node)` candidates.
- Stable original byte positions for deterministic left-to-right tie-breaking.
- Lazy invalidation instead of arbitrary heap deletion.
- At most two newly created neighbor candidates after a merge.

When two nodes merge, the implementation creates a new node, splices it between
the surviving neighbors, and marks both old nodes dead. Old heap entries can
still reference those nodes, so every popped entry is revalidated:

- Is the left node alive?
- Does it still have a live right neighbor?
- Does the current pair still exist in `merge_rank`?
- Does its current rank equal the stored rank?

Only `(previous, merged)` and `(merged, next)` can be newly created. Distant
candidates remain valid and remain in the heap.

There are `O(L)` initial candidates and at most two insertions per successful
merge, so total heap entries inserted are `O(L)`. With repeated `heappush`:

- Structural time: `O(L log L)`.
- Structural auxiliary memory: `O(L)`.
- Linked-list splice: `O(1)` per merge.

Important qualification: node values are Python `bytes`. Constructing
`left.value + right.value` copies the merged bytes. A chain of progressively
larger merges can therefore add `O(L²)` byte-copy work even though candidate
maintenance is `O(L log L)`. Python nodes, dataclasses, heap entries, pointer
chasing, and branches also give this design high constants on tiny pretokens.

## 9. `NativeBatchTokenizer`: simple quadratic algorithm, compact execution

Sources: `cs336_basics/tokenization/native.py` and
`tokenization/_native/bpe_native.cpp` (`BPEEngine`).

Construction translates byte-string vocabulary entries to integer IDs and
builds:

```text
byte value -> base token ID
(left ID, right ID) -> (rank, merged result ID)
```

The Python override batches all regex pretokens into one
`encode_pretokens(list(pretokens))` call. C++ reserves an output capacity equal
to total input bytes, a safe upper bound because merges only reduce token count.

For each pretoken, `BPEEngine` stores token IDs in a contiguous vector. It scans
every adjacent ID pair for the lowest rank, overwrites the left slot with the
result ID, erases the right slot, and repeats.

- Worst-case time: `O(L²)` from repeated scans and vector erases.
- Per-pretoken working memory: `O(L)` compact integers.
- Native call processes a batch and releases the GIL.

Releasing the GIL permits other Python threads to run; it does not make this
single call internally parallel. `list(pretokens)` and pybind conversion still
materialize and copy boundary data. The algorithm can nevertheless outperform
more sophisticated Python structures because contiguous integers, compiled
loops, fewer allocations, and predictable branches have much smaller constants.

## 10. `CachedNativeTokenizer`: native local updates plus bounded reuse

Sources: `cs336_basics/tokenization/cached.py` and
`tokenization/_native/bpe_native.cpp` (`CachedBPEEngine`).

The cold path uses an array-backed linked structure:

```text
CachedNode = value ID, prev index, next index, generation, alive
```

Candidate entries contain rank, original position, both node indices, and both
captured generations. A min-priority queue orders by rank and then position.
After a valid merge, the left node is updated in place, its generation is
incremented, the right node is killed, links are rewired, and only the two local
candidate pairs are pushed. Generation and adjacency checks make stale entries
safe.

- Structural cold-path time: `O(L log L)`.
- Cold-path working memory: `O(L)` contiguous nodes/candidates.
- Hot cache-hit work: `O(L + number of returned token IDs)` because hashing the
  pretoken key reads its bytes and the cached ID vector is copied to output.

The LRU cache maps the raw UTF-8 pretoken bytes to its final ID vector. A
`std::list` records recency and an unordered map provides lookup. Default
capacity is 65,536 entries; capacity zero disables lookup and storage. Capacity
is a count of entries, not a byte budget, so long keys/results can still consume
substantial memory.

Cache metadata is protected by a mutex. Miss computation happens outside the
critical section, so concurrent misses for the same pretoken may duplicate work
but do not corrupt the cache. `clear_cache()` removes entries and resets hit/miss
counters. Hits and misses count ordinary regex pretokens, not complete calls to
`encode`.

The cache is most valuable because natural language repeats common pretokens.
It can hurt cold, high-cardinality workloads through hashing, copying, locking,
and eviction overhead. Always benchmark with an explicit cold/warm policy.

## 11. Strategy comparison

| Class | Candidate selection | Sequence update | Structural worst case per pretoken | Main trade-off |
|---|---|---|---:|---|
| `NaiveTokenizer` | Replay all `R` rules | Rebuild list | `O(RL)` | Simplest; rule-set-sized work |
| `RankScanTokenizer` | Rescan current pairs | Rebuild list | `O(L²)` | Excellent constants for short inputs |
| `RebuildingHeapTokenizer` | Rebuild heap every merge | Rebuild list | `O(L² log L)` | Heap work is not reused |
| `LinkedHeapTokenizer` | Persistent Python heap | Linked nodes | `O(L log L)` structural | Portable; Python/byte-copy overhead |
| `NativeBatchTokenizer` | Native rescan | Vector erase | `O(L²)` | Compact and simple compiled hot loop |
| `CachedNativeTokenizer` | Persistent native heap | Array-backed links | `O(L log L)` structural | Better long/repeated inputs; cache memory |

There is no universal winner:

- Tiny cold pretokens often favor the native scan.
- Long adversarial pretokens favor persistent local updates.
- Repeated pretokens favor the LRU cache.
- Environments without a compiler/native wheel require pure Python.
- The repository chooses `LinkedHeapTokenizer` as a portable default, not as a
  claim that it wins every benchmark.

## 12. Benchmarking rigor

Run:

```bash
make native
make benchmark-tokenizers
```

The benchmark checks token IDs against a local tiktoken reference and verifies
round-trip decoding before reporting median latency, throughput, and peak Python
allocation data.

Interpret it carefully:

- Warmup means the cached implementation is warm unless the cache is cleared.
- `tracemalloc` tracks Python allocations, not all native C++ allocations or
  process RSS; native memory comparisons are incomplete.
- A single validation file is not a workload distribution.
- Character limits are not byte limits, and UTF-8 byte length drives BPE work.
- Median latency hides tails, cache-miss spikes, and concurrency effects.
- End-to-end time includes Python regex and special-token handling.

A defensible benchmark matrix varies pretoken byte length, repetition rate,
Unicode mix, cache state/capacity, merge density, thread count, corpus size, and
cold-start construction. Report environment, compiler flags, CPU, samples,
percentiles, token/byte throughput, RSS, and output equivalence.

## 13. Correctness and regression tests

High-value tests include:

1. Empty input, one byte, multibyte Unicode, whitespace, and punctuation.
2. Overlapping merge occurrences such as `aaa` with `(a,a)`.
3. Multiple equal-rank occurrences and leftmost resolution.
4. Newly created neighbor pairs with higher priority than distant candidates.
5. Special tokens at boundaries, adjacent occurrences, regex metacharacters,
   and overlapping registered tokens.
6. Missing byte tokens, malformed merges, invalid IDs, and invalid UTF-8 ID
   sequences.
7. Equivalence of every implementation to the same reference IDs.
8. Cache hits, misses, eviction, capacity zero, clearing, and bounded entry
   count.
9. Long pretokens designed to trigger many stale candidates.
10. Concurrent native calls if thread safety is part of the supported contract.

Current targeted commands:

```bash
make test-native-batch
make test-cached-native
.venv/bin/python -m pytest tests/test_tokenizer.py -q
```

## 14. Senior MLE interview questions

**Why can `O(L²)` beat `O(L log L)`?**

`L` is usually small, while contiguous compiled scans have excellent locality.
The asymptotically better Python implementation pays object allocation, pointer
chasing, hashing, branches, heap operations, and byte-copy costs.

**Why is the heap entry ordered by rank and position?**

Rank implements learned BPE priority; original position implements deterministic
left-to-right resolution among occurrences of the same ranked pair.

**Why is lazy invalidation safe?**

The authoritative state is the current linked segmentation plus the immutable
merge table. Heap entries are only hints. Revalidating liveness, adjacency,
generation/value, and rank before mutation prevents stale hints from changing
semantics.

**Why batch the native boundary?**

Per-pretoken binding dispatch and Python/C++ conversion can dominate short hot
loops. Batching amortizes fixed overhead, though it is not zero-copy and it
materializes the pretoken list.

**What does an LRU cache change semantically?**

Nothing if vocabulary, merges, regex, and special-token configuration are
immutable for the tokenizer lifetime. The cache changes latency and memory, not
token IDs. Mutable tokenizer state would require invalidation or versioned keys.

**Why does decoding not need merges?**

Each token ID already maps to its complete byte string. Merge history is needed
to choose segmentation during encoding, not to reconstruct bytes from IDs.

**How would you productionize this?**

Validate tokenizer artifacts eagerly; define special-token policy explicitly;
ship native wheels rather than requiring local compilation; use an end-to-end
native regex/BPE pipeline; expose byte-budgeted cache controls and metrics;
benchmark representative cold/warm workloads; fuzz against a reference; and
version vocabulary, merges, regex, and special-token configuration together.

## 15. Revision checklist

You should be able to explain without notes:

- Frequency priority during training versus rank priority during encoding.
- Why byte-level BPE can encode unseen Unicode text.
- Why merges cannot cross regex-pretoken boundaries.
- The leftmost/non-overlap rule.
- Why rule replay depends on `R` and rank scanning does not.
- Why a rebuilt heap is not useful state reuse.
- How linked nodes, local updates, and lazy invalidation work.
- Why structural `O(L log L)` does not include Python byte-copy cost.
- Why compiled quadratic code can beat object-heavy Python code.
- What the native GIL release does and does not imply.
- What the LRU caches, how it is synchronized, and why capacity is not a byte
  bound.
- Why benchmark equivalence and workload design matter as much as timing.
