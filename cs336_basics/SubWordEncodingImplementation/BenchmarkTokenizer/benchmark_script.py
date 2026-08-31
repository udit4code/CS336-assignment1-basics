from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import tiktoken

from ..BPE_Tokenizer import TokenizerV1, TokenizerV2, TokenizerV3, TokenizerV4, TokenizerV5
from ..BPE_Tokenizer.base import GPT2_PRETOKENIZER
from ..BPE_Tokenizer.byte_mapping import decode_gpt2_token

REPO_ROOT = Path(__file__).resolve().parents[3]
VOCAB_PATH = REPO_ROOT / "data" / "gpt2_vocab.json"
MERGES_PATH = REPO_ROOT / "data" / "gpt2_merges.txt"

DATASET = REPO_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"

REPEAT = 10

SPECIAL_TOKENS = ["<|endoftext|>"]


TOKENIZERS = {
    "V1": TokenizerV1,
    "V2": TokenizerV2,
    "V3": TokenizerV3,
    "V4": TokenizerV4,
    "V5 (C++)": TokenizerV5,
}


def load_dataset(dataset_path: Path | None = None, max_chars: int | None = None) -> str:
    path = dataset_path or DATASET
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if max_chars is not None:
        return text[:max_chars]

    return text


def benchmark_encode(tokenizer, text):
    """
    Benchmark encode().
    Returns a dict of metrics.
    """

    tokenizer.encode(text)  # warmup

    times = []
    ids = None
    for _ in range(REPEAT):
        gc.collect()
        start = time.perf_counter()
        ids = tokenizer.encode(text)
        times.append(time.perf_counter() - start)

    tracemalloc.start()
    tokenizer.encode(text)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert ids is not None

    return {
        "time": statistics.median(times),
        "tokens": len(ids),
        "bytes": len(text.encode("utf-8")),
        "peak_memory": peak,
        "ids": ids,
    }


def benchmark_decode(tokenizer, ids):

    tokenizer.decode(ids)

    times = []
    for _ in range(REPEAT):
        gc.collect()
        start = time.perf_counter()
        tokenizer.decode(ids)
        times.append(time.perf_counter() - start)

    tracemalloc.start()
    tokenizer.decode(ids)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "time": statistics.median(times),
        "peak_memory": peak,
    }


def run_one(name, tokenizer_cls, text, reference_ids):

    tokenizer = tokenizer_cls.from_files(
        VOCAB_PATH,
        MERGES_PATH,
        SPECIAL_TOKENS,
    )

    encode_result = benchmark_encode(tokenizer, text)
    ids = encode_result["ids"]
    if ids != reference_ids:
        raise ValueError(f"{name} produced token IDs different from tiktoken")
    if tokenizer.decode(ids) != text:
        raise ValueError(f"{name} failed to decode its result back to the input")

    decode_result = benchmark_decode(tokenizer, ids)
    encode_time = encode_result["time"]

    return {
        "Implementation": name,
        "Encode(ms)": encode_time * 1000,
        "Decode(ms)": decode_result["time"] * 1000,
        "Tokens/sec": len(ids) / encode_time,
        "MB/sec": encode_result["bytes"] / encode_time / 1_000_000,
        "Peak MB": encode_result["peak_memory"] / (1024 * 1024),
    }


def benchmark_tiktoken(text):
    with open(VOCAB_PATH, encoding="utf-8") as vocab_file:
        raw_vocab = json.load(vocab_file)
    mergeable_ranks = {
        decode_gpt2_token(token): token_id
        for token, token_id in raw_vocab.items()
        if token != "<|endoftext|>"
    }
    enc = tiktoken.Encoding(
        name="local-gpt2",
        pat_str=GPT2_PRETOKENIZER.pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens={"<|endoftext|>": 50256},
    )

    enc.encode(text, allowed_special={"<|endoftext|>"})

    encode_times = []
    decode_times = []

    peak_memories = []

    ids = None

    for _ in range(REPEAT):

        gc.collect()
        start = time.perf_counter()

        ids = enc.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )

        encode_times.append(
            time.perf_counter() - start
        )

        gc.collect()
        start = time.perf_counter()

        enc.decode(ids)

        decode_times.append(
            time.perf_counter() - start
        )

    tracemalloc.start()
    enc.encode(text, allowed_special={"<|endoftext|>"})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memories.append(peak)

    encode_time = statistics.median(encode_times)

    return {
        "Implementation": "tiktoken",
        "Encode(ms)": encode_time * 1000,
        "Decode(ms)": statistics.median(decode_times) * 1000,
        "Tokens/sec": len(ids) / encode_time,
        "MB/sec": len(text.encode("utf-8")) / encode_time / 1_000_000,
        "Peak MB": max(peak_memories) / (1024 * 1024),
        "ids": ids,
    }


def print_table(results):

    print()

    print(
        f"{'Implementation':<15}"
        f"{'Encode(ms)':>15}"
        f"{'Decode(ms)':>15}"
        f"{'Tokens/sec':>18}"
        f"{'MB/sec':>12}"
        f"{'Peak(MB)':>12}"
    )

    print("-" * 87)

    for r in results:

        print(
            f"{r['Implementation']:<15}"
            f"{r['Encode(ms)']:>15.2f}"
            f"{r['Decode(ms)']:>15.2f}"
            f"{r['Tokens/sec']:>18,.0f}"
            f"{r['MB/sec']:>12.2f}"
            f"{r['Peak MB']:>12.2f}"
        )


def main():
    global REPEAT

    parser = argparse.ArgumentParser(description="Benchmark BPE tokenizers")
    parser.add_argument("--dataset", type=Path, default=DATASET, help="Path to a text file to tokenize")
    parser.add_argument("--max-chars", type=int, default=None, help="Optional character limit for the benchmark input")
    parser.add_argument("--repeat", type=int, default=REPEAT, help="Number of benchmark repetitions")
    args = parser.parse_args()

    REPEAT = args.repeat

    text = load_dataset(args.dataset, args.max_chars)

    print("Running tiktoken reference...")
    reference_result = benchmark_tiktoken(text)
    reference_ids = reference_result.pop("ids")

    results = []

    for name, tokenizer_cls in TOKENIZERS.items():
        print(f"Running {name}...")
        results.append(
            run_one(
                name,
                tokenizer_cls,
                text,
                reference_ids,
            )
        )

    results.append(reference_result)

    print_table(results)


if __name__ == "__main__":
    main()
