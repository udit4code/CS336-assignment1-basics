import numpy as np
import torch

from cs336_basics.data import LanguageModelDataset

def get_batch(
    dataset: LanguageModelDataset,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    start_indices = np.random.randint(
        low=0,
        high=len(dataset),
        size=batch_size,
    )

    inputs = []
    targets = []

    for idx in start_indices:

        input_tokens, target_tokens = dataset[idx]

        inputs.append(input_tokens)
        targets.append(target_tokens)

    inputs = torch.stack(inputs).to(device)
    targets = torch.stack(targets).to(device)

    return inputs, targets
