import os
from typing import BinaryIO, IO

import torch
import torch.nn as nn
import torch.optim as optim

# How does Saving checkpoint look under the hood ? 
#  Model
#  │
#  │ state_dict()
#  ▼
#  Dictionary of Parameters
#  │
#  │
#  Optimizer
#  │
#  │ state_dict()
#  ▼
#  Dictionary of Moments
#  │
#  ▼
#  {
#     model_state_dict,
#     optimizer_state_dict,
#     iteration
#  }
#  │
#  ▼
#  torch.save(...)
#  │
#  ▼
#  checkpoint.pt

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
    
    
# How does loading checkpoint work under the hood ? 
# checkpoint.pt
#       │
#       ▼
# torch.load()
#       │
#       ▼
# Dictionary
#       │
#       ├──────────────┐
#       │              │
#       ▼              ▼
# model.load_      optimizer.load_
# state_dict()     state_dict()
#       │              │
#       └──────┬───────┘
#              ▼
#       Training resumes

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:

    checkpoint = torch.load(
        src,
    )
    # Key Idea : Calling model.load_state_dict() and optimizer.load_state_dict() returns their respective dictionary objects. 
    # Then, PyTorch can recursively traverse them and collect every learnable parameter. 
    # For Example, if model = TransformerLM(...), then, internally, it would have : embeddingLayer, layer0, ... layer31, LM head. 
    # So, the dictionary being loaded would be something like : 
    # {
    #     "embedding.weight": ...,
    #     "layers.0.attn.q_proj.weight": ...,
    #     "layers.0.attn.k_proj.weight": ...,
    #     ...
    #     "lm_head.weight": ...
    # }
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    # Similarly, for AdamW optimizer, the dictionary would look like : 
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
    # For AdamW, this includes the first moment (m), second moment (v), and step counter, which are essential for resuming optimization correctly.
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return checkpoint["iteration"]
    