import numpy as np
import torch
from torch.utils.data import Dataset


from pathlib import Path
from typing import Union




# LanguageModelDataset is really just a function that converts one long stream of tokens into many training examples.
# As seen from the init method, we find that there are 2 inputs : tokens and context_length. 
# Say, we have a sentence : "I love deep learning".
# Now, the tokenizer converts words/subwords into integers.
# "I"         → 15
# "love"      → 421
# "deep"      → 98
# "learning"  → 617
# "."         → 9
# So, effectively, our dataset becomes : tokens = [15, 421, 98, 617, 9]
# The model never sees words (strings). Instead, it sees only token_ids. 

# What is Context-length ? It is the number of previous tokens we allow the model to look at. 
# Say, context_length = 4. Then, every training example will contain 4 input tokens and 4 target tokens.

# Say, tokens = [10, 20, 30, 40, 50, 60, 70, 80] and context-length = 4
# Here, the dataset does not copy the data. Instead, all it does is produce one training example whenever a downstream consumer asks it. 
# In this case, conceptually, the dataset knows how to generate the following : 
# ---------------------
# Example 0 :
# Input : [10 20 30 40]
# Target : [20 30 40 50]
# ---------------------
# Example 1
# Input : [20 30 40 50]
# Target : [30 40 50 60]
# ---------------------
# Example 2
# Input : [30 40 50 60]
# Target : [40 50 60 70]
# ---------------------
# Example 3
# Input : [40 50 60 70]
# Target : [50 60 70 80]

# So, we find that it is like sliding a window of size 4 across list of token-ids and generating (input, target) on the fly.
# We can think of LanguageModelDataset as a window generator.  
# Long Token Stream : 10 20 30 40 50 60 70 80 90 100 ...
#  ┌──────────────┐
#  │10 20 30 40   │ → Input
#  │20 30 40 50   │ → Target
#  └──────────────┘

#     slide by 1
#     ┌──────────────┐
#     │20 30 40 50   │ → Input
#     │30 40 50 60   │ → Target
#     └──────────────┘

#         slide by 1
#         ┌──────────────┐
#         │30 40 50 60   │ → Input
#         │40 50 60 70   │ → Target
#         └──────────────┘

# Each call to dataset[idx] asks for one of these sliding windows. 
# The dataset computes the appropriate slices from the original token stream and returns them as (input, target) tensors. 
# The training loop (or a DataLoader) can then request many such examples, batch them together, and feed them to the model.

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
        # How many (input, target) pairs ? 
        # In the above example, len(token_ids) = 8 and context_length = 4. So, len of dataset = 8 - 4 = 4
        # As evident from above, we have 4 (input, target) training examples.
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        # Starting from idx, we can generate (input, target) training example using python slicing.
        # So, the dataset doesn't precompute or store these windows. It generates them on demand.
        input = self.tokens[idx : idx + self.context_length]
        target = self.tokens[idx + 1 : idx + self.context_length + 1]

        # We return LongTensors for (input, target)
        return (
            torch.tensor(input, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )