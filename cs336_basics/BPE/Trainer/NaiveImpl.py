import os
import re
import regex
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from typing import List
from typing import BinaryIO
from typing import List, Tuple

# GPT-2 pre-tokenization regex
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




def process_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: List[str],
) -> list[list[bytes]]:
    """
    Read one chunk from disk, split around special tokens,
    GPT-2 pre-tokenize it, and convert each pre-token into
    a list of byte tokens.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    text = raw.decode("utf-8", errors="ignore")

    # Split on special tokens while keeping them in the output.
    if special_tokens:
        special_pattern = re.compile(
            "(" + "|".join(map(re.escape, special_tokens)) + ")"
        )
        pieces = special_pattern.split(text)
    else:
        pieces = [text]

    words = []

    for piece in pieces:

        # Ignore special tokens completely.
        if piece in special_tokens:
            continue

        pretokens = GPT2_PATTERN.findall(piece)

        for token in pretokens:
            words.append(
                [bytes([b]) for b in token.encode("utf-8")]
            )

    return words


def load_and_pretokenize(
    input_path: str,
    special_tokens: List[str],
    num_processes: int = os.cpu_count(),
) -> list[list[bytes]]:
    """
    Reads a corpus, splits it into independent chunks,
    pre-tokenizes using the GPT-2 regex,
    and returns a list of words where every word is a
    list of byte tokens.

    Example output:

    [
        [b'H', b'e', b'l', b'l', b'o'],
        [b' ', b'w', b'o', b'r', b'l', b'd'],
        ...
    ]
    """

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

    work = [
        (start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    words = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:

        futures = [
            executor.submit(
                process_chunk,
                input_path,
                start,
                end,
                special_tokens,
            )
            for start, end in work
        ]

        for future in futures:
            words.extend(future.result())

    return words

from collections import Counter

def count_pairs(words: list[list[bytes]]) -> Counter[tuple[bytes, bytes]]:
    """
    Count the frequency of every adjacent token pair.

    Args:
        words: List of words, where each word is represented as a list of
               byte tokens.

    Returns:
        Counter mapping (token1, token2) -> frequency.
    """
    pair_counts = Counter()

    for word in words:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += 1

    return pair_counts




def merge_word(
    word: List[bytes],
    pair: Tuple[bytes, bytes],
) -> List[bytes]:
    """
    Merge every non-overlapping occurrence of `pair` in `word`.

    Example:
        word = [b'a', b'b', b'c', b'b', b'c']
        pair = (b'b', b'c')

        returns

        [b'a', b'bc', b'bc']
    """

    merged = []
    i = 0

    while i < len(word):

        # Found the pair -> merge it
        if (
            i < len(word) - 1
            and word[i] == pair[0]
            and word[i + 1] == pair[1]
        ):
            merged.append(pair[0] + pair[1])
            i += 2

        else:
            merged.append(word[i])
            i += 1

    return merged

# This will not contain any implementation details. It will simply orchestrate the flow of BPE training. 
# We can think of it as a "main" function that will call other functions to perform the actual work.
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    vocab = {}

    # byte vocab
    for i in range(256):
        vocab[i] = bytes([i])

    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode()

    words = load_and_pretokenize(input_path, special_tokens)

    merges = []

    while len(vocab) < vocab_size:

        pair_counts = count_pairs(words)

        if not pair_counts:
            break
        
        # max_freq = max(pair_counts.values())

        # candidates = [
        #     pair for pair, freq in pair_counts.items()
        #     if freq == max_freq
        # ]

        # print(f"Merge #{len(merges)}")
        # print("Max frequency:", max_freq)
        # print("Number of candidates:", len(candidates))
        # print(candidates[:10])


        best_pair = max(pair_counts.items(),
                        key=lambda x: (x[1], x[0]),
                    )[0]

        merges.append(best_pair)

        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        words = [merge_word(w, best_pair) for w in words]

    return vocab, merges 