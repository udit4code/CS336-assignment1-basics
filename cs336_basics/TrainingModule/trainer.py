import argparse

import torch
from torch.utils.data import DataLoader

from cs336_basics.DataLoaderImplementation.Dataset import LanguageModelDataset
from cs336_basics.TransformerImplementation.TransformerLanguageModelModule.TransformerLanguageModel import TransformerLM
from cs336_basics.TransformerImplementation.CrossEntropyLossModule.CrossEntropy import cross_entropy
from cs336_basics.TransformerImplementation.AdamWOptimizerModule.AdamW import AdamW
from cs336_basics.TransformerImplementation.LearningRateScheduleModule.LearningRateSchedule import (
    get_lr_cosine_schedule,
)
from cs336_basics.TransformerImplementation.GradientClippingModule.GradientClipping import (
    gradient_clipping,
)
from cs336_basics.CheckpointingImplementation.checkpoint_utils import (
    save_checkpoint,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--context_length", type=int, default=128)

    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--max_iters", type=int, default=10000)

    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    dataset = LanguageModelDataset(
        tokens=args.train_data,
        context_length=args.context_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
    ).to(device)

    criterion = cross_entropy()

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    step = 0

    while step < args.max_iters:

        for inputs, targets in dataloader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            loss = criterion(
                logits,
                targets,
            )

            optimizer.zero_grad()

            loss.backward()

            gradient_clipping(
                model.parameters(),
                args.max_grad_norm,
            )

            lr = get_lr_cosine_schedule(
                t=step,
                alpha_max=args.learning_rate,
                alpha_min=args.min_learning_rate,
                T_w=args.warmup_iters,
                T_c=args.max_iters,
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.step()

            if step % 10 == 0:

                print(
                    f"[Step {step:6d}] "
                    f"Loss={loss.item():.4f} "
                    f"LR={lr:.6e}"
                )

            if step > 0 and step % args.save_every == 0:

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=step,
                    out=args.checkpoint,
                )

                print(
                    f"Checkpoint saved at step {step}"
                )

            step += 1

            if step >= args.max_iters:
                break


if __name__ == "__main__":
    main()