from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from pathlib import Path

import tiktoken

from ..Tokenizer import TokenizerV1, TokenizerV2, TokenizerV3, TokenizerV4

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
}


def load_dataset(dataset_path: Path | None = None, max_chars: int | None = None) -> str:
    path = dataset_path or DATASET
    with open(path, "r", encoding="utf-8") as f:
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

    gc.collect()

    tracemalloc.start()

    start = time.perf_counter()

    ids = tokenizer.encode(text)

    elapsed = time.perf_counter() - start

    current, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    return {
        "time": elapsed,
        "tokens": len(ids),
        "bytes": len(text.encode("utf-8")),
        "peak_memory": peak,
    }


def benchmark_decode(tokenizer, ids):

    tokenizer.decode(ids)

    gc.collect()

    tracemalloc.start()

    start = time.perf_counter()

    tokenizer.decode(ids)

    elapsed = time.perf_counter() - start

    current, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    return {
        "time": elapsed,
        "peak_memory": peak,
    }


def run_one(name, tokenizer_cls, text):

    tokenizer = tokenizer_cls.from_files(
        VOCAB_PATH,
        MERGES_PATH,
        SPECIAL_TOKENS,
    )

    encode_times = []
    decode_times = []

    peak_memories = []

    tokens = None
    ids = None

    for _ in range(REPEAT):

        encode_result = benchmark_encode(
            tokenizer,
            text,
        )

        encode_times.append(
            encode_result["time"]
        )

        peak_memories.append(
            encode_result["peak_memory"]
        )

        ids = tokenizer.encode(text)

        tokens = len(ids)

        decode_result = benchmark_decode(
            tokenizer,
            ids,
        )

        decode_times.append(
            decode_result["time"]
        )

    mean_encode = statistics.mean(encode_times)
    mean_decode = statistics.mean(decode_times)

    throughput = tokens / mean_encode

    return {
        "Implementation": name,
        "Encode(ms)": mean_encode * 1000,
        "Decode(ms)": mean_decode * 1000,
        "Tokens/sec": throughput,
        "Peak MB": max(peak_memories) / (1024 * 1024),
    }


def benchmark_tiktoken(text):

    enc = tiktoken.get_encoding("gpt2")

    enc.encode(text, allowed_special={"<|endoftext|>"})

    gc.collect()

    encode_times = []
    decode_times = []

    peak_memories = []

    ids = None

    for _ in range(REPEAT):

        tracemalloc.start()

        start = time.perf_counter()

        ids = enc.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )

        encode_times.append(
            time.perf_counter() - start
        )

        _, peak = tracemalloc.get_traced_memory()

        peak_memories.append(peak)

        tracemalloc.stop()

        tracemalloc.start()

        start = time.perf_counter()

        enc.decode(ids)

        decode_times.append(
            time.perf_counter() - start
        )

        tracemalloc.stop()

    return {
        "Implementation": "tiktoken",
        "Encode(ms)": statistics.mean(encode_times) * 1000,
        "Decode(ms)": statistics.mean(decode_times) * 1000,
        "Tokens/sec": len(ids) / statistics.mean(encode_times),
        "Peak MB": max(peak_memories) / (1024 * 1024),
    }


def print_table(results):

    print()

    print(
        f"{'Implementation':<15}"
        f"{'Encode(ms)':>15}"
        f"{'Decode(ms)':>15}"
        f"{'Tokens/sec':>18}"
        f"{'Peak(MB)':>12}"
    )

    print("-" * 75)

    for r in results:

        print(
            f"{r['Implementation']:<15}"
            f"{r['Encode(ms)']:>15.2f}"
            f"{r['Decode(ms)']:>15.2f}"
            f"{r['Tokens/sec']:>18,.0f}"
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

    results = []

    for name, tokenizer_cls in TOKENIZERS.items():
        print(f"Running {name}...")
        results.append(
            run_one(
                name,
                tokenizer_cls,
                text,
            )
        )

    print("Running tiktoken...")
    results.append(benchmark_tiktoken(text))

    print_table(results)


if __name__ == "__main__":
    main()