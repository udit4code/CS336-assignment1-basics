"""Convert raw UTF-8 text into the memory-mapped format used by training."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import tiktoken


END_OF_TEXT = "<|endoftext|>"


def sha256_file(path: Path) -> str:
    """Hash a file in chunks so hashing does not require loading it into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_token_array(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as stream:
            np.save(stream, tokens, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def encode_text_file(source: Path, destination: Path, encoding_name: str) -> dict[str, Any]:
    """Encode one text file with tiktoken and persist a compact uint32 array.

    The end-of-text marker is explicitly allowed as a special token.  Therefore
    every literal ``<|endoftext|>`` in the source becomes exactly one ID
    (``encoding.eot_token``), rather than being split into ordinary text IDs.
    Other special tokens remain disallowed so a misspelled or unexpected marker
    cannot silently change the training corpus.
    """
    if not source.is_file():
        raise FileNotFoundError(f"text file does not exist: {source}")
    text = source.read_text(encoding="utf-8")
    encoding = tiktoken.get_encoding(encoding_name)
    if END_OF_TEXT not in encoding.special_tokens_set or encoding.eot_token is None:
        raise ValueError(f"encoding {encoding_name!r} does not define {END_OF_TEXT}")
    token_ids = np.asarray(
        encoding.encode(text, allowed_special={END_OF_TEXT}),
        dtype=np.uint32,
    )
    if token_ids.size == 0:
        raise ValueError(f"text file produced zero tokens: {source}")
    if int(token_ids.max()) >= encoding.n_vocab:
        raise ValueError("token ID exceeds tokenizer vocabulary")

    save_token_array(destination, token_ids)
    metadata = {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "encoding": encoding_name,
        "vocab_size": encoding.n_vocab,
        "endoftext_token": END_OF_TEXT,
        "endoftext_id": encoding.eot_token,
        "special_tokens_preserved": [END_OF_TEXT],
        "token_count": int(token_ids.size),
        "dtype": str(token_ids.dtype),
        "token_file": str(destination),
    }
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def split_tokens(tokens: np.ndarray, validation_fraction: float, context_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Make a deterministic suffix validation split without overlapping windows."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1) when splitting")
    split_at = int(tokens.size * (1.0 - validation_fraction))
    if split_at <= context_length or tokens.size - split_at <= context_length:
        raise ValueError("both train and validation splits must exceed context_length")
    return tokens[:split_at], tokens[split_at:]
