# Optimizing BPE Training: From Naive to Production

This note explains how to optimize BPE training from first principles.

## 1. Naive algorithm

-   Pretokenize corpus into byte words.
-   Count every adjacent pair.
-   Pick the most frequent pair.
-   Merge everywhere.
-   Repeat.

Time per merge: scan entire corpus twice (count + merge).

If N is corpus size and M merges are learned, complexity is roughly
O(M×N).

## 2. Observation: language repeats

Instead of storing every word occurrence, store:

``` python
Counter({(b'h',b'e',b'l',b'l',b'o'):1000})
```

Now work scales with unique words rather than all occurrences.

## 3. Parallel pretokenization

Split the corpus into chunks aligned on special-token boundaries. Each
worker tokenizes independently and returns a Counter. Merge Counters in
the parent.

## 4. Maintain pair counts

Initialize pair frequencies once from the word Counter.

## 5. Incremental updates

When merging (b,c) in `a b c d -> a bc d`, only the neighbouring pairs
change. Update only those counts instead of rescanning the corpus.

## 6. Tradeoffs

-   Counter: less memory, slightly more bookkeeping.
-   Incremental updates: much faster, more complex.
-   Pair-location index: fastest, highest memory usage.

## Napkin math

100M pre-token occurrences, 50k merges:

-   Naive: \~5e12 token visits.
-   Counter representation: work proportional to unique words.
-   Incremental updates: work proportional to affected words per merge.

The central systems idea is replacing repeated global recomputation with
local incremental maintenance.
