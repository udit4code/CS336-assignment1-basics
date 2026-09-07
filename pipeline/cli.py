"""Command-line interface for the end-to-end language-model pipeline."""

import argparse
from pathlib import Path

import tiktoken

from .config import DataConfig, ModelConfig, OptimizerConfig, RuntimeConfig
from .train import run_training


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="prepare text and train a Transformer language model")
    parser.add_argument("--data", type=Path, required=True, help="UTF-8 training text file")
    parser.add_argument("--valid-data", type=Path, help="optional separate UTF-8 validation text file")
    parser.add_argument("--encoding", default="gpt2", help="tiktoken encoding name")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, help="defaults to the selected tiktoken vocabulary size")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--dtype", choices=("float32", "float64", "float16", "bfloat16"), default="float32")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("pipeline/artifacts"))
    parser.add_argument("--run-name", help="stable output directory name; defaults to UTC timestamp")
    parser.add_argument("--resume", type=Path, help="checkpoint produced by an earlier training run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline", description="Train and run the CS336 Transformer language model")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    _add_train_parser(subparsers)
    inference = subparsers.add_parser("inference", help="generate text from a trained artifact (planned)")
    inference.add_argument("--artifact", type=Path, required=True)
    inference.add_argument("--prompt", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "inference":
        parser.error("inference mode is reserved but not implemented yet")

    try:
        encoding = tiktoken.get_encoding(args.encoding)
        vocab_size = args.vocab_size or encoding.n_vocab
        data = DataConfig(args.data, args.valid_data, args.encoding, args.context_length, args.validation_fraction)
        model = ModelConfig(args.context_length, vocab_size, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta)
        optimizer = OptimizerConfig(args.learning_rate, args.min_learning_rate, args.warmup_steps, args.max_steps, args.weight_decay, args.max_grad_norm)
        runtime = RuntimeConfig(args.batch_size, args.seed, args.device, args.dtype, args.eval_interval, args.checkpoint_interval, args.eval_batches, args.log_interval)
        artifact = run_training(data, model, optimizer, runtime, args.artifacts_dir, args.run_name, args.resume)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Final artifact: {artifact}")
    return 0
