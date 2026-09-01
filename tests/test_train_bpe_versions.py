from collections import Counter
import time

import pytest

from cs336_basics.SubWordEncodingImplementation.BPE_Trainer._incremental import (
    IncrementalMergeState,
    LazyMaxPairHeap,
    choose_best_pair,
)
from cs336_basics.SubWordEncodingImplementation.BPE_Trainer.v2 import OptimisedBPETrainer
from cs336_basics.SubWordEncodingImplementation.BPE_Trainer.v3 import IncrementalBPETrainer
from cs336_basics.SubWordEncodingImplementation.BPE_Trainer.v4 import HeapBPETrainer
from cs336_basics.SubWordEncodingImplementation.BPE_Trainer.v5 import IntegerBPETrainer
from cs336_basics.SubWordEncodingImplementation.BPE_Trainer.v6 import CompactingHeapBPETrainer

from .common import FIXTURES_PATH


TRAINER_VERSIONS = (
    IncrementalBPETrainer,
    HeapBPETrainer,
    IntegerBPETrainer,
    CompactingHeapBPETrainer,
)


@pytest.mark.parametrize(
    "trainer_cls",
    TRAINER_VERSIONS,
    ids=("v3", "v4", "v5", "v6"),
)
def test_train_bpe_speed_optimized_versions(trainer_cls):
    """Keep every post-v2 trainer below the assignment's 1.5-second budget."""
    start_time = time.perf_counter()
    trainer_cls().train(
        FIXTURES_PATH / "corpus.en",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    elapsed = time.perf_counter() - start_time

    assert elapsed < 1.5, (
        f"{trainer_cls.__name__} took {elapsed:.2f} seconds; "
        "expected less than 1.50 seconds"
    )


@pytest.fixture(scope="module")
def reference_training_result():
    return OptimisedBPETrainer().train(
        FIXTURES_PATH / "corpus.en",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )


@pytest.mark.parametrize(
    "trainer_cls",
    TRAINER_VERSIONS,
    ids=("v3", "v4", "v5", "v6"),
)
def test_optimized_trainer_versions_match_v2(trainer_cls, reference_training_result):
    actual = trainer_cls().train(
        FIXTURES_PATH / "corpus.en",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    assert actual == reference_training_result


def test_incremental_state_counts_overlapping_pairs_and_weighted_words():
    state = IncrementalMergeState(
        Counter(
            {
                (b"a", b"a", b"a"): 2,
                (b"a", b"b"): 3,
            }
        )
    )

    assert state.pair_counts[(b"a", b"a")] == 4
    assert state.pair_counts[(b"a", b"b")] == 3

    state.apply_merge((b"a", b"a"), b"aa")

    assert (b"a", b"a") not in state.pair_counts
    assert state.pair_counts[(b"aa", b"a")] == 2
    assert state.pair_counts[(b"a", b"b")] == 3


def test_pair_selectors_use_lexicographically_greatest_pair_to_break_ties():
    pair_counts = {
        (b"a", b"z"): 5,
        (b"b", b"a"): 5,
        (b"z", b"z"): 4,
    }
    expected = (b"b", b"a")

    assert choose_best_pair(pair_counts, pair_order=lambda pair: pair) == expected

    heap = LazyMaxPairHeap(pair_counts, pair_order=lambda pair: pair)
    assert heap.pop_best(pair_counts) == expected


def test_lazy_heap_discards_stale_counts():
    pair = (b"a", b"b")
    pair_counts = {pair: 10}
    heap = LazyMaxPairHeap(pair_counts, pair_order=lambda candidate: candidate)

    pair_counts[pair] = 3
    heap.refresh({pair}, pair_counts)

    assert heap.pop_best(pair_counts) == pair


def test_compacting_heap_rebuilds_when_stale_entries_dominate():
    pair = (b"a", b"b")
    pair_counts = {pair: 10}
    heap = LazyMaxPairHeap(
        pair_counts,
        pair_order=lambda candidate: candidate,
        compact_factor=2,
        compact_min_size=1,
    )

    for count in range(9, 0, -1):
        pair_counts[pair] = count
        heap.refresh({pair}, pair_counts)

    assert len(heap.heap) <= 2 * len(pair_counts)
    assert heap.pop_best(pair_counts) == pair
