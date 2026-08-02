from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
import re
import regex
from typing import BinaryIO

GPT2_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d
    |[ ]?\p{L}+
    |[ ]?\p{N}+
    |[ ]?[^\s\p{L}\p{N}]+
    |\s+(?!\S)
    |\s+
    """,
    regex.VERBOSE,
)

# Pre-create all byte objects once.
BYTE_TABLE = tuple(bytes([i]) for i in range(256)) 

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk_counter(
    input_path: str,
    start: int,
    end: int,
    special_pattern: str | None,
    special_tokens: set[str],
) -> Counter[tuple[bytes, ...]]:

    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")

    if special_pattern:
        pieces = re.split(special_pattern, text)
    else:
        pieces = (text,)

    counter = Counter()

    update = counter.update

    for piece in pieces:

        if piece in special_tokens:
            continue

        for token in GPT2_PATTERN.findall(piece):

            encoded = token.encode("utf-8")

            update([
                tuple(BYTE_TABLE[b] for b in encoded)
            ])

    return counter


def load_and_pretokenize_counter(
    input_path: str,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter[tuple[bytes, ...]]:

    if num_processes is None:
        num_processes = os.cpu_count()

    split_token = (
        special_tokens[0].encode("utf-8")
        if special_tokens
        else b"<|endoftext|>"
    )

    with open(input_path, "rb") as f:

        boundaries = find_chunk_boundaries(
            f,
            num_processes,
            split_token,
        )

    work = list(zip(boundaries[:-1], boundaries[1:]))

    special_pattern = (
        "(" + "|".join(map(re.escape, special_tokens)) + ")"
        if special_tokens
        else None
    )

    special_token_set = set(special_tokens)

    global_counter = Counter()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:

        futures = [
            executor.submit(
                process_chunk_counter,
                input_path,
                start,
                end,
                special_pattern,
                special_token_set,
            )
            for start, end in work
        ]

        for future in futures:
            global_counter.update(future.result())

    return global_counter 



def initialize_pair_counts(
    word_counter: Counter[tuple[bytes, ...]]
) -> Counter[tuple[bytes, bytes]]:
    """
    Build the initial pair-frequency table.

    Args:
        word_counter:
            Counter mapping
                (token1, token2, ...) -> frequency

    Returns:
        Counter mapping
            (token_i, token_{i+1}) -> total frequency
    """

    pair_counter = Counter()

    for word, freq in word_counter.items():

        # words of length 1 contain no pairs
        if len(word) < 2:
            continue

        for pair in zip(word, word[1:]):
            pair_counter[pair] += freq

    return pair_counter 



def choose_best_pair(
    pair_counter: Counter[tuple[bytes, bytes]]
) -> tuple[bytes, bytes] | None:
    """
    Return the pair with

        1. highest frequency
        2. lexicographically greatest pair if tied

    Returns None if no pairs remain.
    """

    if not pair_counter:
        return None

    return max(
        pair_counter.items(),
        key=lambda item: (item[1], item[0]),
    )[0] 
    
    


Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]

def merge_words(
    word_counter: Counter[Word],
    pair: Pair,
) -> Counter[Word]:
    """
    Merge every occurrence of `pair` in every unique word.

    Returns a new Counter containing the merged words.
    """

    merged_counter = Counter()

    left, right = pair

    for word, freq in word_counter.items():

        merged = []

        i = 0

        while i < len(word):

            if (
                i < len(word) - 1
                and word[i] == left
                and word[i + 1] == right
            ):
                merged.append(left + right)
                i += 2
            else:
                merged.append(word[i])
                i += 1

        merged_counter[tuple(merged)] += freq

    return merged_counter

# This will not contain any implementation details. It will simply orchestrate the flow of BPE training. 
# We can think of it as a "main" function that will call other functions to perform the actual work.
def train_bpe_v2(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    # Step 1 : Initialize vocabulary
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    merges = []

    # Step 2 : Pre-tokenize corpus
    word_counter = load_and_pretokenize_counter(
        input_path,
        special_tokens,
    )

    pair_counter = initialize_pair_counts(word_counter)

    # Step 3 : Main BPE loop
    while len(vocab) < vocab_size:

        best_pair = choose_best_pair(pair_counter)

        if best_pair is None:
            break

        # Add merged token to vocab
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        merges.append(best_pair)

        # Merge every word
        word_counter = merge_words(
            word_counter,
            best_pair,
        )

        # Recompute pair counts
        pair_counter = initialize_pair_counts(word_counter)

    return vocab, merges