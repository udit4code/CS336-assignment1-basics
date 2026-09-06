import os
from typing import BinaryIO, IO

import torch
import torch.nn as nn
import torch.optim as optim

# A resumable training checkpoint needs model state, optimizer state, and the
# training position. A module state_dict contains parameters and persistent
# buffers; an optimizer state_dict contains parameter-group settings and any
# per-parameter state, such as Adam's moments. torch.save serializes this
# object graph to a path or binary file-like object.


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(
        checkpoint,
        out,
    )


# Loading reverses that process: deserialize the mappings, copy tensors into
# the existing model, restore optimizer bookkeeping, and return the saved step.
# The caller must construct a compatible model and optimizer first.


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:

    checkpoint = torch.load(
        src,
    )
    # load_state_dict does not discover or return the saved mapping. It matches
    # keys in the supplied mapping to already-registered model state, for example:
    # {
    #     "embedding.weight": ...,
    #     "layers.0.attention.q_proj.weight": ...,
    #     "layers.0.attention.k_proj.weight": ...,
    #     ...
    #     "lm_head.weight": ...
    # }
    model.load_state_dict(checkpoint["model_state_dict"])
    # Optimizer state is keyed internally by parameter identifiers and also
    # records each parameter group's hyperparameters. For AdamW it includes:
    # {
    #     "state": {
    #         parameter_0: {
    #             "step": 100,
    #             "m": ...,
    #             "v": ...
    #         },
    #         ...
    #     },

    #     "param_groups": ...
    # }
    # Restoring moments and the step counter is necessary to continue with the
    # same updates; restoring model weights alone does not reproduce a resume.
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]
