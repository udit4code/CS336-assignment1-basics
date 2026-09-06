import numpy as np
import torch
from torch.utils.data import Dataset


from pathlib import Path
from typing import Any
# A causal language model learns next-token prediction from one token stream.
# For tokens [10, 20, 30, 40, 50] and context_length=4, item 0 is:
#   input  = [10, 20, 30, 40]
#   target = [20, 30, 40, 50]
# Item i advances both windows by one token. The context length is the number
# of input tokens in a sample (and the model's maximum attention span here),
# not necessarily the number of preceding tokens available to every position.


class LanguageModelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        # A path is memory-mapped to avoid loading a large corpus into RAM;
        # callers may also inject an existing ndarray or memmap.
        tokens: str | Path | np.ndarray | np.memmap,
        context_length: int,
    ) -> None:

        if context_length <= 0:
            raise ValueError("context_length must be positive")

        if isinstance(tokens, (str, Path)):
            self.tokens = np.load(
                tokens,
                mmap_mode="r",
            )
        else:
            self.tokens = tokens
        if len(self.tokens) <= context_length:
            raise ValueError("tokens must contain more entries than context_length")
        self.context_length = context_length

    def __len__(self) -> int:
        # A stream of N tokens has N-context_length valid shifted windows.
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: Any) -> tuple[torch.Tensor, torch.Tensor]:
        # Slicing creates each overlapping window lazily. NumPy basic slices are
        # views, although conversion to a torch.long tensor may copy or cast.
        input_tokens = self.tokens[index : index + self.context_length]
        target = self.tokens[index + 1 : index + self.context_length + 1]

        # Token IDs must be integer tensors for embedding-table indexing.
        return (
            torch.as_tensor(input_tokens, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )
