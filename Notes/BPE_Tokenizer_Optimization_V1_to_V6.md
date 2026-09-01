# BPE Tokenizer Optimization: From V1 to V6

> A living, first-principles revision guide to the BPE tokenizer implementations in this repository.
>
> Despite this file’s historical name, V1–V6 optimize **encoding/tokenization**, not BPE training. Training learns a vocabulary and ordered merge list; these implementations apply that fixed state to new text.

---

## Contents

1. The mental model
2. Training versus tokenization
3. Tokenizer state and the shared pipeline
4. Correctness invariants and running example
5. V1: replay every merge rule
6. V2: search current adjacent pairs
7. V3: build a heap, then throw it away
8. V4: persistent heap and linked list
9. V5: batched integer BPE in C++
10. Complexity and benchmark comparison
11. Why Big-O did not predict the benchmark
12. Correctness traps
13. Where time and memory go
14. V6: compact native heap plus bounded LRU
15. Commands, revision questions, and final model
16. The complete lifecycle: training, loading, tokenizing, and tiktoken

---

## 1. The mental model

Byte Pair Encoding has two separate phases:

```text
Training corpus
    │ count frequent adjacent pairs and learn merges
    ▼
Vocabulary + ranked merge list
    │ apply those fixed merges to new text
    ▼
Token IDs
```

This repository implements a **byte-level BPE tokenizer**. For each regex pretoken, it:

1. Converts Unicode text to UTF-8 bytes.
2. Starts with one symbol per byte.
3. Applies valid learned merges in priority order.
4. Converts the final symbols to vocabulary IDs.

Every version computes the same logical result. They answer two implementation questions differently:

- How do we find the adjacent pair with the best merge rank?
- How do we update the sequence after merging it?

The evolution is:

```text
V1  Scan every learned rule, relevant or not
 ↓
V2  Scan only adjacent pairs that currently exist
 ↓
V3  Put candidates in a heap, but rebuild it after every merge
 ↓
V4  Preserve the heap and update only the local neighborhood
 ↓
V5  Represent symbols as integers and run the hot loop in C++
 ↓
V6  Combine a compact native heap with bounded pretoken reuse
```

The surprising result is that V4 has the strongest Python-level structural complexity, but V5 deliberately returns to a simpler V2-style scan. V6 then reintroduces persistent local updates only after moving them into compact native arrays, and adds a bounded cache for repeated pretokens. The best version depends on whether inputs are tiny, long, cold, or repeated.

---

## 2. Training versus tokenization

### Training learns the rules

A simplified BPE training loop is:

```python
while len(vocab) < target_vocab_size:
    pair_counts = count_adjacent_pairs(corpus)
    best_pair = choose_most_frequent_pair(pair_counts)
    merge_pair_everywhere(best_pair)
    merges.append(best_pair)
```

During training, **corpus frequency** chooses the next merge. Training produces:

```python
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

### Tokenization applies the rules

Tokenization neither counts global frequencies nor learns new tokens. The position of a pair in `merges` is its priority:

```python
merge_rank = {pair: rank for rank, pair in enumerate(merges)}
```

Smaller rank means earlier learned and higher priority.

| Phase | Pair selection | State being changed |
|---|---|---|
| Training | Highest corpus frequency, with a defined tie-break | Corpus representation and learned rules |
| Tokenization | Smallest existing merge rank | One input pretoken’s segmentation |

Both phases merge adjacent symbols, which causes the naming confusion. V1–V6 optimize the second row.

---

## 3. Tokenizer state and shared pipeline

### Vocabulary

The vocabulary maps IDs to byte strings:

```python
id_to_token = {
    75: b"l",
    78: b"o",
    86: b"w",
    256: b"lo",
    257: b"low",
    258: b"er",
    259: b"lower",
}
```

The reverse map supports encoding:

```python
token_to_id = {token: token_id for token_id, token in id_to_token.items()}
```

### Ordered merges

Suppose training learned:

```python
merges = [
    (b"l", b"o"),       # rank 0; result b"lo"
    (b"lo", b"w"),      # rank 1; result b"low"
    (b"e", b"r"),       # rank 2; result b"er"
    (b"low", b"er"),    # rank 3; result b"lower"
]
```

The result of `(left, right)` is `left + right`, and that result must exist in the vocabulary.

### The shared pipeline

`BaseTokenizer` owns behavior common to all versions:

```text
Input string
    │
    ├── split around registered special tokens
    ├── emit special-token IDs directly
    └── GPT-2 regex pretokenize normal chunks
            │
            ├── "Hello"
            ├── ","
            ├── " how"
            └── " are"
                    │
                    └── BPE each pretoken independently
```

The regex separates contractions, letters, numbers, punctuation, and whitespace. BPE merges never cross a regex pretoken boundary.

Special tokens such as `<|endoftext|>` bypass ordinary BPE. Overlapping special tokens are sorted longest-first, so a registered longer token wins over its prefix.

### Why bytes?

Unicode characters have variable-length UTF-8 encodings:

```text
"A"  → 41
"é"  → c3 a9
"🙃" → f0 9f 99 83
```

Starting from all 256 possible bytes guarantees that any valid UTF-8 text is representable, even if a character was absent during training.

Decoding concatenates token bytes and performs one UTF-8 decode:

```python
data = b"".join(id_to_token[token_id] for token_id in ids)
text = data.decode("utf-8", errors="replace")
```

---

## 4. Correctness invariants and running example

At every stage:

```text
b"".join(current_pieces) == pretoken.encode("utf-8")
```

A merge changes segmentation, never content:

```text
[b"l", b"o", b"w"] → [b"lo", b"w"]
```

Both sides concatenate to `b"low"`.

The full invariants are:

1. Every current piece exists in the vocabulary.
2. Only adjacent pieces are merged.
3. The applicable pair with the smallest rank has priority.
4. Equal-rank occurrences are resolved left-to-right.
5. A byte occurrence cannot participate in two overlapping merges.
6. Special tokens never enter ordinary BPE.
7. `decode(encode(text)) == text` under the tokenizer contract.

### Running example: `lower`

```text
[l,   o, w,  e, r]
 └─r0─┘

[lo,     w,  e, r]
 └──r1───┘

[low,        e, r]
             └r2┘

[low,          er]
 └─────r3──────┘

[lower]
```

Notation used below:

- `R`: number of learned rules—about 50,000 for GPT-2.
- `L`: UTF-8 byte length of one pretoken.
- `K`: successful merges for the pretoken, where `K ≤ L - 1`.
- `P`: currently mergeable adjacent candidates.

---

## 5. V1: replay every merge rule

### Algorithm

V1 walks the entire merge list in rank order. For each rule, it scans the current pieces and merges every non-overlapping occurrence.

```python
pieces = one_byte_pieces(pretoken)

for pair in merges:
    pieces = merge_pair_everywhere(pieces, pair)

return map_pieces_to_ids(pieces)
```

### Worked example

```text
Start                   [l, o, w, e, r]
Apply rank 0 (l, o)     [lo, w, e, r]
Apply rank 1 (lo, w)    [low, e, r]
Apply rank 2 (e, r)     [low, er]
Apply rank 3 (low, er)  [lower]
```

With the real GPT-2 list, V1 keeps considering tens of thousands of irrelevant rules. Even after the pretoken becomes one piece, it still loops over all remaining merge rules.

### Non-overlap example

For `[a, a, a]` and rule `(a, a)`, the left-to-right scan returns:

```text
[a, a, a] → [aa, a]
```

The middle `a` cannot be consumed twice.

### Complexity

Each of `R` rules scans up to `L` pieces:

```text
Time: O(RL)
Live sequence space: O(L)
Upper-bound transient element writes: O(RL)
```

Every pass creates a new Python list. V1 is easy to verify but ties hot-loop work to the complete vocabulary rather than the tiny input.

### Lesson and optimization pressure

V1 asks the wrong directional question:

> For every learned rule, does this rule occur here?

V2 instead inspects only pairs that actually exist.

---

## 6. V2: search current adjacent pairs

### Algorithm

V2 inverts the lookup:

```text
V1: learned rule → search current pieces
V2: current adjacent pair → look up learned rank
```

It builds an `O(1)` average-time rank table once. On every iteration it scans the current adjacencies, selects the smallest rank, and merges one occurrence.

### Worked example

For `[l, o, w, e, r]`, the first scan examines four pairs rather than all `R` rules:

| Pair | Rank |
|---|---:|
| `(l, o)` | 0 |
| `(o, w)` | absent |
| `(w, e)` | absent |
| `(e, r)` | 2 |

Rank 0 wins:

```text
[l, o, w, e, r] → [lo, w, e, r]
```

The next scan sees `(lo, w)` at rank 1 and `(e, r)` at rank 2, so rank 1 wins. Scanning repeats until no adjacent pair is in `merge_rank`.

For an eight-byte pretoken and 50,000 rules:

```text
V1 first-pass possibilities: 50,000 rules
V2 first-pass possibilities:      7 pairs
```

### Tie behavior

V2 updates the winner only for a **strictly smaller** rank. Multiple occurrences of the same pair share a rank, so the leftmost remains selected.

```text
[a, a, a], rank(a,a)=0
index 0: rank 0 ← winner
index 1: rank 0 ← not strictly smaller
result: [aa, a]
```

### Complexity

Finding the winner scans up to `O(L)` pairs. `_merge_at()` slices and reconstructs up to `O(L)` elements. Across at most `K` merges:

```text
Time: O(KL), worst case O(L²)
Per-pretoken live space: O(L)
Persistent rank table: O(R)
```

### Lesson and optimization pressure

The large gain is an indexing decision, not a micro-optimization: V2 removes `R` from the hot loop.

But after each merge it rescans all surviving adjacencies, even though only the merged pair’s neighborhood changed. That motivates a priority queue.

---

## 7. V3: build a heap, then throw it away

### Algorithm

V3 builds a min-heap of `(rank, index)` entries for currently mergeable pairs. The heap top identifies the next merge.

For `lower`:

```text
State 1: [l, o, w, e, r]
Heap 1:  [(0, 0), (2, 3)]
Pop:     (0, 0)
Merge:   [lo, w, e, r]

State 2: [lo, w, e, r]
Heap 2:  [(1, 0), (2, 2)]  ← rebuilt from scratch
Pop:     (1, 0)
Merge:   [low, e, r]
```

The critical implementation shape is:

```python
while True:
    heap = build_heap(pieces)
    if not heap:
        break
    _, index = heapq.heappop(heap)
    pieces = merge_at(pieces, index)
```

### Why the heap does not yet help

A heap makes minimum selection cheap **after the heap has been constructed**. V3 discards that construction after one pop.

The implementation uses repeated `heappush`, so building up to `L` candidates costs up to `O(L log L)`. List reconstruction still costs `O(L)`.

```text
Time: O(KL log L), worst case O(L² log L)
```

Using `heapify()` could make each rebuild `O(L)`, but V3 would still rediscover every candidate after every merge.

### Working “failed optimization” example

In the first `lower` heap, `(e, r)` at rank 2 is already known. Merging `(l, o)` does not affect it. Nevertheless, V3 destroys the heap and rediscovers `(e, r)` during the next build.

This is the main lesson:

> Improving one operation—heap pop—does not improve the algorithm when constructing its supporting data structure dominates and no state is reused.

### Optimization pressure

To reuse the heap, entries need stable references after merges. Python list indices shift when elements disappear, so V4 changes both the candidate structure and sequence representation.

---

## 8. V4: persistent heap and linked list

### Algorithm

V4 combines:

1. A doubly linked list of current pieces.
2. A persistent min-heap of candidates.
3. Local updates around each merge.
4. Lazy invalidation of stale heap entries.

### Constant-time structural merging

```text
prev <-> left <-> right <-> next
```

becomes:

```text
prev <-> merged <-> next
```

with a constant number of pointer changes. Old nodes are marked `alive = False`.

### Locality of change

For:

```text
a <-> b <-> c <-> d
```

merging `b + c → bc` changes only:

```text
Removed: (a,b), (b,c), (c,d)
Added:   (a,bc), (bc,d)
```

All distant candidates remain valid. V4 pushes at most the two new neighbor pairs.

### Worked `lower` example

```text
l(0) <-> o(1) <-> w(2) <-> e(3) <-> r(4)
```

Initial heap:

```text
(rank=0, position=0, left=l)
(rank=2, position=3, left=e)
```

After merging `l + o`:

```text
lo(0) <-> w(2) <-> e(3) <-> r(4)
```

Only `(lo, w)` is newly pushed at rank 1. The existing `(e, r)` entry remains useful:

```text
(rank=1, position=0, left=lo)
(rank=2, position=3, left=e)
```

Unlike V3, V4 preserves unaffected work.

### Lazy invalidation

Deleting arbitrary entries from a binary heap is inconvenient. V4 leaves old entries in place and checks validity when they are popped.

An entry is stale if:

- Its left node is dead.
- Its right neighbor is missing or dead.
- The adjacent values no longer form a mergeable pair.
- The current pair’s rank differs from the recorded rank.

For `[a, a, a]`, the heap initially contains occurrences at positions 0 and 1. Merging position 0 kills the node at position 1. Its old heap entry stays present but is rejected later.

This pattern is broadly useful:

> Lazy validation can be cheaper and simpler than eagerly removing every invalidated reference.

### Stable tie-breaking

Entries are ordered by `(rank, position)`. A memory address such as `id(node)` is not textual order and would make overlapping equal-rank behavior unstable. The merged node inherits its left node’s position.

### Complexity—with an important caveat

- Build linked list: `O(L)`.
- Initial heap via repeated pushes: up to `O(L log L)`.
- Structural merge: `O(1)`.
- At most two neighbor pushes per merge: `O(log L)` each.
- Every inserted entry, valid or stale, is eventually popped at most once.

```text
Structural time: O(L log L)
Structural space: O(L)
```

However, `left.value + right.value` copies bytes. A pathological chain of increasingly large strings can add up to `O(L²)` byte-copy work. The `O(L log L)` statement describes heap and sequence structure, not every byte allocation.

### Why V4 loses on ordinary text

V4 creates substantial Python machinery:

- One `Node` object per byte.
- `prev`, `next`, `position`, and `alive` state.
- `HeapEntry` objects and comparisons.
- Pointer chasing through non-contiguous memory.
- Stale-entry branches.

Most natural-language pretokens are short. For small `L`, V2’s quadratic contiguous scan can execute fewer actual CPU instructions.

> Better asymptotic complexity wins only beyond its constant-factor break-even point.

---

## 9. V5: batched integer BPE in C++

### Design choice

V5 does not port V4 line by line. It chooses the simpler V2-style scan, but executes it over compact integers in optimized C++.

```text
Python
  ├── special-token splitting
  ├── GPT-2 regex pretokenization
  └── batch pretokens
          │ one pybind11 call
          ▼
C++
  ├── bytes → integer symbol IDs
  ├── scan adjacent ID pairs
  ├── replace winner with its result ID
  └── return flattened IDs
```

### Integer symbols

Python versions represent a merge key as byte strings:

```python
(b"lo", b"w")
```

V5 stores something like:

```text
(256, 86) → {rank: 1, result_id: 257}
```

One native hash lookup answers:

- Is this pair mergeable?
- What is its priority?
- Which token ID replaces it?

No merged byte string is created in the hot loop.

### Direct byte table

```cpp
TokenId byte_to_id_[256];
bool has_byte_[256];
```

The initial symbol for a byte is found by array indexing, not a Python dictionary lookup.

### Native merge table

```cpp
std::unordered_map<std::pair<TokenId, TokenId>, Merge, PairHash>
```

where:

```cpp
struct Merge {
    std::size_t rank;
    TokenId result;
};
```

The constructor validates that the left token, right token, and concatenated result all exist in the vocabulary.

### Batch the Python/C++ boundary

A separate native call per pair—or even per short pretoken—would waste time on dispatch and conversion. V5 overrides `_encode_pretokens()`:

```python
self._engine.encode_pretokens(list(pretokens))
```

The input is not zero-copy: pybind11 converts Python strings to `std::string`, then converts the output vector to Python integers. The gain comes from **amortizing** this fixed cost across a batch.

### Release the GIL

```cpp
py::call_guard<py::gil_scoped_release>()
```

After argument conversion, the CPU-bound native loop runs without holding Python’s Global Interpreter Lock. This does not make one encode call parallel, but it permits unrelated Python threads to run. The engine tables are read-only during encoding.

### Worked ID example

Suppose:

```text
l=75, o=78, w=86, e=68, r=81
lo=256, low=257, er=258, lower=259
```

Input vector:

```text
[75, 78, 86, 68, 81]
```

First native scan:

```text
(75,78) → rank 0, result 256
(78,86) → absent
(86,68) → absent
(68,81) → rank 2, result 258
```

Then:

```text
[75, 78, 86, 68, 81]
 → [256, 86, 68, 81]
 → [257, 68, 81]
 → [257, 258]
 → [259]
```

### Complexity

V5 scans the vector once per successful merge. `vector::erase` shifts the suffix:

```text
Time: O(KL), worst case O(L²)
Per-pretoken space: O(L)
Persistent native table: O(R)
```

This resembles V2 asymptotically, but is far faster because it uses:

- Compiled loops rather than Python bytecode.
- Contiguous integer vectors.
- Fixed-size byte arrays.
- Native hash lookups.
- No Python allocation per merge.
- No byte-string concatenation in the hot loop.
- One native call for a batch.

### What is still Python?

- Special-token splitting.
- GPT-2 regex pretokenization.
- Materializing the list of pretokens.
- Converting results back to `list[int]`.

This is part of the remaining gap to tiktoken.

---

## 10. Complexity and benchmark comparison

| Version | Candidate selection | Sequence update | Approximate time | Principal weakness |
|---|---|---|---:|---|
| V1 | Replay all `R` rules | Rebuild list | `O(RL)` | Work depends on entire rule set |
| V2 | Scan current pairs | Slice/rebuild list | `O(L²)` worst case | Repeated global scan |
| V3 | Rebuild heap, pop once | Slice/rebuild list | `O(L² log L)` worst case | Heap state discarded |
| V4 | Persistent heap | Linked-list rewiring | `O(L log L)` structurally | Python object overhead; byte copying remains |
| V5 | Native scan of ID pairs | Native vector erase | `O(L²)` worst case | Python pretokenization and quadratic core remain |
| V6 | Persistent native heap | Array-backed links | `O(L log L)` structurally | Higher constants on tiny cold pretokens |

### Measured smoke benchmark

On the repository’s 482-byte TinyStories sample with ten repetitions:

| Implementation | Encode time | Tokens/second | MB/second |
|---|---:|---:|---:|
| V1 | 427.79 ms | 281 | ~0.00 |
| V2 | 0.26 ms | 464,666 | 1.87 |
| V3 | 0.27 ms | 440,333 | 1.77 |
| V4 | 0.65 ms | 184,918 | 0.74 |
| V5 C++ | 0.05 ms | 2,618,172 | 10.52 |
| V6 heap, cache disabled | 0.08 ms | 1,569,058 | 6.30 |
| V6 heap + warm LRU | 0.03 ms | 3,573,181 | 14.35 |
| tiktoken | 0.03 ms | 3,931,718 | 15.79 |

Treat this as a smoke test, not a universal ranking:

- The input is tiny and sub-millisecond measurements are noisy.
- Initialization is excluded.
- Pretoken length distributions differ by corpus.
- `tracemalloc` does not observe all native allocations.
- Adversarial long pretokens may change the V2/V4 crossover.

Robust conclusions:

1. Removing `R` from the hot loop makes V2 enormously faster than V1.
2. Rebuilding a heap does not create a reliable win in V3.
3. V4’s Python object overhead dominates for ordinary short pretokens.
4. A batched native integer loop gives V5 a large constant-factor gain.
5. V6 trades higher tiny-input heap cost for better long-input scaling and cached repetition.
6. A mature end-to-end native implementation still wins on the measured sample.

---

## 11. Why Big-O did not predict the benchmark

Big-O describes growth and hides constants.

```text
V2: potentially quadratic, but a simple contiguous list
V4: structurally near L log L, but many Python objects and branches
```

When `L` is small, these V4 costs can exceed a complete V2 rescan:

- Dataclass allocation.
- Reference updates.
- Heap comparisons.
- Pointer chasing.
- Stale-entry checks.

Three regimes help explain the results:

### Huge rule set, short input

V1 loses because `R ≫ L`.

### Short input in Python

V2 is competitive because `L²` remains small and its representation is simple.

### Hot loop in native code

V5 makes V2’s algorithmic shape cheap enough that compact representation matters more than V4’s structural advantage.

The engineering loop is:

```text
Establish correctness
 → measure a real workload
 → identify dominant work
 → change algorithm or representation
 → measure again
```

---

## 12. Correctness traps

### Rank is not frequency during encoding

Training frequencies are no longer consulted. Only merge-list order matters.

### Smaller rank wins

Rank 0 outranks rank 100; hence V3/V4 use a min-heap.

### Occurrences cannot overlap

`[a,a,a]` with `(a,a)` becomes `[aa,a]`, not a result that consumes the middle byte twice.

### Tie-breaking must use textual order

Object addresses and hash iteration order are not textual position. V4 stores `position` explicitly.

### Heap entries become stale

A heap entry records what was true at insertion time. Mutable neighbors may invalidate it, so V4 rechecks on pop.

### UTF-8 code points span byte tokens

An emoji begins as four pieces. Decoding an isolated partial token may show replacement characters, while decoding the complete sequence reconstructs the original text.

### GPT-2 file encoding is special

GPT-2 vocabulary files use a reversible bytes-to-printable-Unicode mapping. Every character in a serialized token must be mapped back to its byte. Calling `.encode("utf-8")` on the serialized key is not equivalent.

### Pretoken boundaries are semantic

BPE cannot merge across GPT-2 regex pretokens.

### Iterable chunks are independent

`encode_iterable()` independently encodes every yielded string. Arbitrary chunking is not guaranteed to equal encoding the concatenation because a regex pretoken could span the boundary.

---

## 13. Where time and memory go

### V1

- Tens of thousands of rule checks per pretoken.
- A new list for every rule.
- Repeated byte-pair tuple construction.

### V2

- Pair lookups on each scan.
- Byte-string hashing and concatenation.
- List slicing and reconstruction.

### V3

- V2-like list work.
- Heap allocation and pushes on every iteration.
- No reuse of unaffected candidates.

### V4

- Node and heap-entry allocation.
- Pointer-rich layout.
- Heap comparisons and validation.
- Byte concatenation remains.

### V5

- Python regex work.
- Python/C++ argument conversion.
- Native table probes and vector shifts.
- Native/Python result conversion.

V5 reserves output capacity equal to total input bytes. This is a safe upper bound because merging can only reduce the initial one-symbol-per-byte count.

### V6

- Compact `V6Node` arrays rather than Python node objects.
- Persistent native heap entries.
- Generation checks for lazy invalidation.
- Mutex-protected LRU lookup and result copying.
- Python regex and Python/native conversion still remain.

---

## 14. V6: compact native heap plus bounded LRU

V6 implements the two most important recommendations from the earlier design analysis:

1. Preserve V4’s local-update algorithm inside compact C++ data structures.
2. Avoid recomputing BPE for repeated pretokens with a bounded cache.

### Array-backed linked structure

Each input byte receives one entry in a contiguous `std::vector<V6Node>`:

```cpp
struct V6Node {
    TokenId value;
    std::int32_t prev;
    std::int32_t next;
    std::uint32_t generation;
    bool alive;
};
```

`prev` and `next` are integer indices, not pointers to separately allocated objects. Merging keeps the left slot, replaces its token ID with the result ID, links it around the right slot, and marks the right slot dead.

This retains V4’s constant-time structural update while improving locality and eliminating Python node allocation.

### Persistent native candidate heap

Candidate entries store:

```text
rank
original left position
left and right node indices
captured left and right generation numbers
```

The heap is ordered by `(rank, position)`, preserving merge priority and left-to-right ties.

After a merge, only `prev + merged` and `merged + next` are pushed. Unaffected candidates remain in the heap.

### Generation-based lazy invalidation

When a node’s value changes or it is removed, its generation increments. A heap entry is valid only if:

- Both captured nodes are alive.
- They are still adjacent in both directions.
- Their current generations match the captured generations.
- Their current pair still has the recorded rank.

Generation counters catch a subtle case that `alive` alone cannot: a left node may remain alive but now contain a different merged token.

### Bounded, thread-safe pretoken LRU

V6 caches:

```text
pretoken bytes → final token IDs
```

The default capacity is 65,536 distinct pretokens. Least-recently-used eviction prevents unbounded growth. A mutex protects the cache because encoding releases the GIL and the same tokenizer may be called concurrently.

Caching can be disabled for algorithmic measurement:

```python
tokenizer = TokenizerV6(vocab, merges, cache_capacity=0)
```

Cache state is observable and resettable:

```python
tokenizer.cache_info()   # {"hits": ..., "misses": ..., "size": ...}
tokenizer.clear_cache()
```

### Complexity

Ignoring cache hits:

```text
Build node array: O(L)
Build candidate heap: up to O(L log L) in the current implementation
Each local merge/update: O(log L)
Total structural time: O(L log L)
Per-pretoken working memory: O(L)
Bounded cache memory: proportional to configured capacity and cached output sizes
```

On a cache hit, BPE work is replaced by hash lookup, LRU maintenance, and copying cached IDs to the result.

### Measured crossover

With the cache disabled and one long regex pretoken formed by repeating `antidisestablishmentarianism`:

| Input bytes | V5 scan | V6 heap | V6 speedup |
|---:|---:|---:|---:|
| 280 | 0.080 ms | 0.015 ms | 5.20× |
| 1,400 | 1.804 ms | 0.099 ms | 18.15× |
| 2,800 | 7.237 ms | 0.215 ms | 33.62× |
| 7,000 | 44.115 ms | 0.699 ms | 63.13× |

The widening speedup is the expected `O(L²)` versus near-`O(L log L)` crossover.

### Honest interpretation

- **Tiny, cold pretokens:** V5’s simpler scan can be faster.
- **Long, cold pretokens:** V6’s persistent heap wins increasingly strongly.
- **Repeated pretokens:** V6’s LRU can bypass both algorithms.

V6 is therefore optimized for a broader workload, not guaranteed to beat V5 on every individual pretoken.

Remaining opportunities are native pretokenization, heap construction via linear-time heapify, parallel native batches, and typed-buffer output.

---

## 15. Commands, revision questions, and final model

### Build and test

V5 and V6 use the direct pybind11 Makefile workflow:

```bash
uv sync
make native
make test-v5
make test-v6
```

Force a rebuild after compiler-option or header changes:

```bash
make clean-native
make native
```

Run all tokenizer tests:

```bash
uv run pytest tests/test_tokenizer.py -v
```

Run the offline-reference benchmark:

```bash
uv run python \
  -m cs336_basics.SubWordEncodingImplementation.BenchmarkTokenizer.benchmark_script \
  --dataset data/TinyStoriesV2-GPT4-valid-100w.txt \
  --repeat 10
```

The benchmark verifies every implementation’s IDs against a local tiktoken encoding before reporting speed.

### Rapid revision questions

**Why is V1 slow?** Its hot loop depends on all `R` learned rules, even for a tiny input.

**What changes in V2?** It looks up ranks for current adjacent pairs, removing `R` from the hot loop.

**Why is V2 quadratic?** It rescans and reconstructs an `O(L)` list after up to `O(L)` merges.

**Why does V3’s heap not help much?** It rebuilds and discards the heap after one pop.

**Why does V4 need stable nodes?** A persistent heap cannot safely retain ordinary list indices that shift after deletion.

**Why lazy invalidation?** Arbitrary heap deletion is awkward; checking stale entries at pop time is simpler and bounded by total insertions.

**Why can V4 lose to V2?** Typical `L` is small, while Python nodes, heap entries, pointer chasing, and branches have large constants.

**Why does V5 resemble V2 instead of V4?** Simple contiguous integer scans are extremely cheap in compiled code for the observed input distribution.

**What does V5 precompute?** `(left_id, right_id) → (rank, result_id)` plus a 256-entry byte-to-ID table.

**Is V5 zero-copy?** No. It amortizes conversions by batching pretokens.

**Does releasing the GIL make V5 parallel?** No. It permits concurrency; the current native loop itself is single-threaded.

**Why can tiktoken remain faster?** V5/V6 retain Python pretokenization and boundary conversion; tiktoken has a more mature end-to-end native implementation.

**What does V6 add?** A compact array-backed linked structure, persistent native heap, generation-based stale-entry checks, and bounded LRU caching.

**Does V6 always beat V5?** No. V5 can win on tiny cold pretokens; V6 wins on sufficiently long inputs or repeated pretokens.

### Final mental model

Remember each version by its unit of work:

```text
V1: learned rules are the unit of search
V2: current adjacent pairs are the unit of search
V3: candidates become heap entries, but only temporarily
V4: local mutations are the unit of maintenance
V5: compact integer operations are the unit of execution
V6: local native updates and reusable pretoken results are the unit of work
```

Or remember the performance story:

```text
V1 removes no irrelevant work.
V2 removes irrelevant rules.
V3 introduces the right structure with the wrong lifetime.
V4 preserves state but pays Python object costs.
V5 simplifies the algorithm and removes Python from the hot loop.
V6 combines native locality with bounded reuse.
```

The deepest lesson is broader than tokenization:

> Performance comes from choosing the algorithm, representation, update scope, and execution boundary together.

The recurring systems principles are:

1. **Index in the direction of the query.** V2 queries ranks for pairs that exist.
2. **Preserve reusable state.** V4 keeps unaffected candidates alive.
3. **Optimize for the real distribution.** Short pretokens make constants decisive.
4. **Cross expensive boundaries in batches.** V5 does enough work per pybind11 call to justify native execution.

To revise this chapter, reproduce the `lower` trace, rebuild the complexity table from memory, and explain why V4 loses to V5 despite its better structural Big-O. If that explanation is clear, the optimization journey is clear.

---

## 16. The complete lifecycle: training, loading, tokenizing, and tiktoken

This section resolves the most common source of confusion: **the BPE trainer and the BPE tokenizer are different programs used at different times**.

### The entire lifecycle in one diagram

```text
PHASE A — TRAIN THE TOKENIZER, usually once

Raw text corpus
    + target vocabulary size
    + pretokenization rule
    + special-token policy
              │
              ▼
        BPE training algorithm
              │
              ├── vocabulary: token ID ↔ token bytes
              └── ordered merge rules / equivalent rank table


PHASE B — CONSTRUCT OR LOAD A TOKENIZER

Vocabulary
    + merge priorities
    + regex pattern
    + special-token mapping
              │
              ▼
       Tokenizer/Encoding object


PHASE C — TOKENIZE TEXT, repeated many times

New text
    + already-constructed tokenizer
              │
              ▼
           token IDs
              │
              ▼
     language-model embedding table


PHASE D — DETOKENIZE OUTPUT

Generated token IDs
    + same tokenizer vocabulary
              │
              ▼
          output text
```

Training may take minutes or hours and is performed once for a tokenizer design. Encoding is performed continually—on every training document, prompt, and generated response.

### Phase A: what exactly goes into BPE training?

A BPE trainer normally receives:

| Input | Meaning |
|---|---|
| Raw training corpus | Representative text from the desired language/domain distribution |
| Target vocabulary size | How many byte tokens, learned tokens, and special tokens may exist |
| Pretokenization rule | Defines boundaries that learned merges cannot cross |
| Special tokens | Reserved symbols such as `<|endoftext|>` |
| Tie-breaking policy | Makes equal-frequency merge choices deterministic |

For byte-level BPE, the initial ordinary vocabulary contains all 256 one-byte tokens. This is the base alphabet.

The trainer then:

1. Pretokenizes the corpus.
2. Represents every pretoken as byte symbols.
3. Counts adjacent symbol-pair occurrences, weighted by pretoken frequency.
4. Selects the most frequent pair, using the specified tie-breaker.
5. Creates a new symbol representing the concatenation.
6. Replaces non-overlapping occurrences of that pair.
7. Records the pair in the ordered merge list.
8. Repeats until the target vocabulary size is reached.

### Small training example

Suppose the pretokenized training corpus is:

```text
low low lower
```

Ignoring spaces for the moment, the initial representations are:

```text
low   → [l, o, w]       frequency 2
lower → [l, o, w, e, r] frequency 1
```

Initial weighted pair counts include:

```text
(l, o): 3
(o, w): 3
(w, e): 1
(e, r): 1
```

Assume the tie-breaking rule selects `(l, o)` first:

```text
Learn rank 0: (l, o) → lo

low   → [lo, w]       frequency 2
lower → [lo, w, e, r] frequency 1
```

Recount or incrementally update affected pairs:

```text
(lo, w): 3
(w, e):  1
(e, r):  1
```

The next learned merge is:

```text
Learn rank 1: (lo, w) → low
```

Later iterations might learn:

```text
rank 2: (e, r)    → er
rank 3: (low, er) → lower
```

The important output is not merely the strings `low` and `lower`. It is the **ordered history of how those symbols were constructed**:

```python
merges = [
    (b"l", b"o"),
    (b"lo", b"w"),
    (b"e", b"r"),
    (b"low", b"er"),
]
```

This order is what V1 replays and what V2–V5 convert into rank lookups.

### Phase A output

The trainer’s conceptual outputs are:

```python
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

In a complete tokenizer definition, also preserve:

```python
pretokenizer_pattern: str
special_tokens: dict[str, int]
```

The merge list alone is not sufficient to reproduce the entire tokenizer. Two implementations using identical merges but different pretokenization or special-token rules can produce different IDs.

### Phase B: tokenizer construction is not training

Construction loads the artifacts and creates efficient lookup structures:

```python
tokenizer = TokenizerV5(
    vocab=trained_vocab,
    merges=trained_merges,
    special_tokens=["<|endoftext|>"],
)
```

The constructor may build:

- `id_to_token`
- `token_to_id`
- `pair → rank`
- V5’s `(left_id, right_id) → (rank, result_id)` native table
- A compiled pretokenization regex

This is **index construction over already learned data**, not learning. No corpus frequencies are counted and no new merge is selected.

The distinction resembles a database:

```text
Training             ≈ create the persistent dataset/schema
Tokenizer loading    ≈ load it and build in-memory indexes
Encoding             ≈ execute queries against those indexes
```

### Phase C: tokenizer input and output

At runtime:

```python
ids = tokenizer.encode("lower")
```

Inputs:

- New text
- Fixed tokenizer configuration learned earlier

Output:

- A sequence of integer token IDs

The tokenizer does not modify its learned vocabulary or merge priorities. Encoding `lower` one million times does not make `lower` a newly learned token if it was not already in the vocabulary.

Those IDs are meaningful only relative to that exact tokenizer. ID `259` is not universally “lower”; it means whatever token bytes vocabulary entry 259 contains.

### Why the language model and tokenizer must agree

A language model has an embedding matrix approximately shaped:

```text
[vocabulary_size, model_dimension]
```

Token ID 259 selects row 259. The model learned the meaning of that row from sequences produced by its training tokenizer.

If you change the merge list or token-to-ID assignments after model training, the same text may produce different IDs, and those IDs select unrelated embedding rows. Therefore:

> A pretrained model must be used with the tokenizer configuration for which it was trained.

Training your own BPE tokenizer does not automatically make it compatible with GPT-2 or another pretrained model.

### Phase D: decoding does not run BPE backward

Decoding does not reverse the merge algorithm step by step. It performs a vocabulary lookup for each ID, concatenates the bytes, and UTF-8 decodes the result:

```text
[257, 258]
   │     │
   │     └── b"er"
   └──────── b"low"

b"low" + b"er" = b"lower" → "lower"
```

The merge list is needed to decide segmentation during encoding. The vocabulary alone is sufficient for ordinary decoding.

---

### How tiktoken fits into this lifecycle

#### `import tiktoken` does not choose or train an encoding

```python
import tiktoken
```

loads the Python API and its compiled native implementation. It does **not** by itself:

- Train BPE.
- Select GPT-2, `cl100k_base`, or `o200k_base`.
- Tokenize any text.
- Necessarily load a merge table.

It mainly makes APIs such as `get_encoding`, `encoding_for_model`, and `Encoding` available.

#### `get_encoding()` selects a pretrained tokenizer definition

```python
enc = tiktoken.get_encoding("gpt2")
```

asks tiktoken’s registry for the constructor named `gpt2`. In the installed tiktoken version, that constructor supplies:

```text
name
regex pattern
mergeable ranks
special-token mapping
expected vocabulary size
```

It then constructs an `Encoding`, which creates the native `CoreBPE` object used by `encode()` and `decode()`.

No BPE training occurs here. The encoding parameters were trained previously by the tokenizer’s authors.

#### Does tiktoken already have the merge list?

The precise answer is:

> tiktoken knows **how to locate and load pretrained encoding data**, but merely importing the package does not mean every encoding’s complete merge data is already resident in memory.

For the installed GPT-2 constructor, tiktoken references two pretrained GPT-2 assets:

```text
vocab.bpe
encoder.json
```

On the first `get_encoding("gpt2")` call, its loader may download these files if they are not already cached. It verifies expected hashes and converts the GPT-2 files into tiktoken’s internal `mergeable_ranks` representation. Later loads can use the on-disk cache.

This explains why code can behave as follows in an offline environment:

```python
import tiktoken                    # succeeds
tiktoken.get_encoding("gpt2")     # may fail if assets are not cached
```

Other named encodings, such as `cl100k_base`, are loaded from pretrained `.tiktoken` rank files. Again, these are learned artifacts being loaded—not rules being trained at import time.

Within one Python process, `get_encoding()` also caches the constructed `Encoding` object in a registry, so requesting the same name again normally returns the already constructed object.

#### Why tiktoken says `mergeable_ranks`, not always `merges`

Our educational tokenizer accepts explicit pair rules:

```python
(left_bytes, right_bytes) → rank
```

tiktoken’s `Encoding` constructor receives:

```python
mergeable_ranks: dict[bytes, int]
```

This maps every mergeable token’s byte sequence to its rank/token value. During BPE, adjacent pieces can be concatenated and the resulting byte sequence looked up in this table. The rank determines priority.

These representations package related learned information differently:

```text
Educational representation:
    ordered pair merges + vocabulary

tiktoken representation:
    mergeable token bytes → rank
```

For legacy GPT-2 files, tiktoken’s loader reads the explicit BPE merge data and encoder vocabulary, validates that they agree, and converts them into `mergeable_ranks` before constructing `CoreBPE`.

#### `encoding_for_model()` adds one mapping step

```python
enc = tiktoken.encoding_for_model("some-model-name")
```

conceptually performs:

```text
model name → encoding name → get_encoding(encoding name)
```

It does not derive a tokenizer from model weights and does not train one. It uses a maintained model-to-encoding-name mapping.

#### What happens when tiktoken encodes?

Conceptually:

```text
Python text
    │ validate allowed/disallowed special tokens
    ▼
Native CoreBPE
    │ regex pretokenization
    │ byte-level BPE using pretrained mergeable ranks
    ▼
Python list of token IDs
```

The computational core is native, which avoids most per-pair Python overhead. Batch APIs can encode multiple strings concurrently, and `encode_to_numpy()` can avoid copying the output into a Python list.

### How this repository’s benchmark avoids a tiktoken download

The repository already contains local GPT-2 vocabulary and merge files. Its benchmark constructs a tiktoken `Encoding` explicitly from those local artifacts rather than relying on:

```python
tiktoken.get_encoding("gpt2")
```

That makes the benchmark usable without downloading GPT-2 assets and ensures that V1–V6 and the tiktoken reference use the same learned tokenizer definition.

### Final input/output table

| Step | Input | Output | Happens how often? |
|---|---|---|---:|
| BPE training | Corpus, target vocab size, regex policy, special tokens | Vocabulary and ranked merge information | Usually once per tokenizer design |
| Tokenizer loading | Learned artifacts and configuration | In-memory tokenizer/`Encoding` object | Once per process or configuration |
| Encoding | New text and fixed tokenizer | Token IDs | Constantly |
| Model forward pass | Token IDs and model weights | Logits/hidden states | Constantly |
| Decoding | Token IDs and tokenizer vocabulary | Bytes/text | Constantly during output |

The shortest correct summary is:

> BPE training decides **which byte sequences deserve tokens and their priority**. The tokenizer later uses that frozen decision to decide **which token IDs represent a new piece of text**. tiktoken normally loads an already trained encoding; it does not train BPE when imported or when `get_encoding()` is called.
