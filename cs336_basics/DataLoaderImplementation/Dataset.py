import numpy as np
import torch
from torch.utils.data import Dataset


from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):

    def __init__(
        self,
        # We are using dependency injection here.
        tokens: Union[str, Path, np.ndarray, np.memmap],
        context_length: int
    ):

        if isinstance(tokens, (str, Path)):
            self.tokens = np.load(
                tokens,
                mmap_mode="r",
            )
        else:
            self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        inputs = self.tokens[idx : idx + self.context_length]
        targets = self.tokens[idx + 1 : idx + self.context_length + 1]

        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )