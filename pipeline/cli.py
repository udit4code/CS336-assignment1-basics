"""Command-line interface for the end-to-end language-model pipeline."""

import argparse
from pathlib import Path
import sys

import tiktoken

from .artifacts import json_dumps
from .config import DataConfig, InferenceConfig, ModelConfig, OptimizerConfig, RuntimeConfig
from .inference import count_words, run_inference
from .runtime import SUPPORTED_DTYPES
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
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="float32")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("pipeline/artifacts"))
    parser.add_argument("--run-name", help="stable output directory name; defaults to UTC timestamp")
    parser.add_argument("--resume", type=Path, help="checkpoint produced by an earlier training run")


def _add_inference_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("inference", help="generate text from a trained final artifact")
    parser.add_argument("--artifact", type=Path, required=True, help="artifact.pt or the run directory containing it")
    parser.add_argument("--prompt", required=True, help="non-empty text used as the generation prefix")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="maximum number of tokens to sample after the prompt (default: 100)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=0,
        help="minimum decoded completion words before EOT is allowed (default: 0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="positive sampling temperature; lower is sharper (default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="nucleus-sampling probability threshold in (0, 1] (default: 1.0)",
    )
    parser.add_argument("--seed", type=int, help="optional non-negative sampling seed")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="float32")
    parser.add_argument("--show-token-ids", action="store_true", help="write prompt and generated token IDs to stderr")
    parser.add_argument("--output-json", type=Path, help="also write a structured generation result to this path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline", description="Train and run the CS336 Transformer language model")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    _add_train_parser(subparsers)
    _add_inference_parser(subparsers)
    return parser


def _run_train(args: argparse.Namespace) -> int:
    encoding = tiktoken.get_encoding(args.encoding)
    vocab_size = args.vocab_size or encoding.n_vocab
    data = DataConfig(args.data, args.valid_data, args.encoding, args.context_length, args.validation_fraction)
    model = ModelConfig(
        args.context_length,
        vocab_size,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
    )
    optimizer = OptimizerConfig(
        args.learning_rate,
        args.min_learning_rate,
        args.warmup_steps,
        args.max_steps,
        args.weight_decay,
        args.max_grad_norm,
    )
    runtime = RuntimeConfig(
        args.batch_size,
        args.seed,
        args.device,
        args.dtype,
        args.eval_interval,
        args.checkpoint_interval,
        args.eval_batches,
        args.log_interval,
    )
    artifact = run_training(data, model, optimizer, runtime, args.artifacts_dir, args.run_name, args.resume)
    print(f"Final artifact: {artifact}")
    return 0


def _run_inference(args: argparse.Namespace) -> int:
    config = InferenceConfig(
        artifact_path=args.artifact,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        min_words=args.min_words,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )
    result = run_inference(config)

    # Keep stdout composable: it contains only completion text. Diagnostics go
    # to stderr so shell users can redirect the completion into another tool.
    print(result.completion)
    print(
        f"[generation: {len(result.generated_token_ids)} tokens, {count_words(result.completion)} words, "
        f"stop={result.stop_reason}, device={result.device}]",
        file=sys.stderr,
    )
    if args.show_token_ids:
        print(f"prompt_token_ids={result.prompt_token_ids}", file=sys.stderr)
        print(f"generated_token_ids={result.generated_token_ids}", file=sys.stderr)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_dumps(result.to_dict()), encoding="utf-8")
        print(f"[structured output: {args.output_json}]", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.mode == "train":
            return _run_train(args)
        return _run_inference(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))
