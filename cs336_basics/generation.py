"""Autoregressive token generation for the project's causal language model.

This module deliberately operates on integer token IDs rather than text.  A
tokenizer is an orchestration concern: the same generation code can therefore
be used with tiktoken today and the project's custom BPE tokenizer later.
"""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch

from .nn.softmax import softmax
from .nn.transformer import TransformerLM


@dataclass(frozen=True)
class GenerationResult:
    """Raw token-level result returned by :func:`generate`.

    ``generated_token_ids`` includes the end-of-text token when that token
    caused generation to stop.  Keeping the stop token makes the result
    auditable; the caller can omit it when decoding user-facing text.
    """

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    stopped_on_endoftext: bool

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        """Return the prompt followed by every sampled token."""
        return self.prompt_token_ids + self.generated_token_ids

    @property
    def stop_reason(self) -> str:
        """Return a stable, machine-readable termination reason."""
        return "endoftext" if self.stopped_on_endoftext else "max_new_tokens"


def _validate_sampling_parameters(temperature: float, top_p: float) -> None:
    if not isinstance(temperature, Real) or isinstance(temperature, bool):
        raise TypeError("temperature must be a real number")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    if not isinstance(top_p, Real) or isinstance(top_p, bool):
        raise TypeError("top_p must be a real number")
    if not math.isfinite(float(top_p)) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in the interval (0, 1]")


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    suppress_token_ids: Collection[int] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token ID from a next-token logit vector.

    Temperature scaling is applied before nucleus (top-p) filtering.  Top-p
    keeps the smallest highest-probability prefix whose cumulative probability
    reaches ``top_p``.  The retained distribution is then renormalized before
    sampling.

    Args:
        logits: Unnormalized next-token scores with shape ``(vocab_size,)``.
        temperature: Positive scale. Values below one sharpen the distribution;
            values above one flatten it.
        top_p: Nucleus threshold in ``(0, 1]``. One disables filtering.
        suppress_token_ids: Optional token IDs that must not be sampled on this
            step. At least one other finite logit must remain.
        generator: Optional device-compatible generator for reproducibility.

    Returns:
        A scalar ``torch.long`` tensor containing the sampled token ID.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError(f"logits must have shape (vocab_size,), got {tuple(logits.shape)}")
    if not logits.is_floating_point():
        raise TypeError(f"logits must be floating point, got {logits.dtype}")
    _validate_sampling_parameters(temperature, top_p)

    # Sampling in float32 avoids avoidable overflow/underflow when model
    # parameters use a lower-precision inference dtype.
    scaled_logits = logits.to(dtype=torch.float32) / float(temperature)
    if not bool(torch.isfinite(scaled_logits).all().item()):
        raise RuntimeError("model produced non-finite next-token logits")

    if suppress_token_ids:
        suppressed = tuple(suppress_token_ids)
        if any(not isinstance(token_id, Integral) or isinstance(token_id, bool) for token_id in suppressed):
            raise TypeError("suppress_token_ids must contain only integer token IDs")
        invalid = next((token_id for token_id in suppressed if not 0 <= token_id < logits.numel()), None)
        if invalid is not None:
            raise ValueError(f"suppressed token ID {invalid} is outside [0, {logits.numel()})")
        scaled_logits = scaled_logits.clone()
        scaled_logits[list(suppressed)] = float("-inf")
        if not bool(torch.isfinite(scaled_logits).any().item()):
            raise ValueError("cannot suppress every candidate token")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        sorted_probabilities = softmax(sorted_logits, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        # Remove token k only if tokens before k already reached the threshold.
        # This retains the first crossing token and always keeps index zero,
        # even when p is smaller than the most likely token's probability.
        remove = torch.zeros_like(cumulative_probabilities, dtype=torch.bool)
        remove[1:] = cumulative_probabilities[:-1] >= float(top_p)
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        probabilities = softmax(sorted_logits, dim=-1)
        sampled_sorted_index = torch.multinomial(probabilities, 1, generator=generator)
        return sorted_indices[sampled_sorted_index].squeeze(0).to(dtype=torch.long)

    probabilities = softmax(scaled_logits, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(0).to(dtype=torch.long)


def _normalize_prompt_token_ids(
    prompt_token_ids: Sequence[int] | torch.Tensor,
    *,
    vocab_size: int,
) -> tuple[int, ...]:
    if isinstance(prompt_token_ids, torch.Tensor):
        if prompt_token_ids.ndim != 1:
            raise ValueError(f"prompt_token_ids must be one-dimensional, got shape {tuple(prompt_token_ids.shape)}")
        if prompt_token_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise TypeError(f"prompt_token_ids must contain integers, got {prompt_token_ids.dtype}")
        normalized = tuple(int(token_id) for token_id in prompt_token_ids.detach().cpu().tolist())
    else:
        if isinstance(prompt_token_ids, (str, bytes)):
            raise TypeError("prompt_token_ids must contain integers, not text")
        normalized_list: list[int] = []
        for token_id in prompt_token_ids:
            if not isinstance(token_id, Integral) or isinstance(token_id, bool):
                raise TypeError("every prompt token ID must be an integer")
            normalized_list.append(int(token_id))
        normalized = tuple(normalized_list)

    if not normalized:
        raise ValueError("prompt_token_ids must contain at least one token")
    invalid = next((token_id for token_id in normalized if not 0 <= token_id < vocab_size), None)
    if invalid is not None:
        raise ValueError(f"prompt token ID {invalid} is outside [0, {vocab_size})")
    return normalized


def generate(
    model: TransformerLM,
    prompt_token_ids: Sequence[int] | torch.Tensor,
    *,
    endoftext_token_id: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
    endoftext_allowed: Callable[[tuple[int, ...]], bool] | None = None,
) -> GenerationResult:
    """Autoregressively sample a completion from ``model``.

    At each iteration the model predicts a distribution for the token after
    the current sequence. Generation stops when it samples ``endoftext_token_id``
    or reaches ``max_new_tokens``. If the sequence becomes longer than the
    trained context window, only its most recent ``model.context_length`` IDs
    are passed to the next forward call.

    ``endoftext_allowed`` can delay the stop token based on tokenizer-aware
    policy. It receives the generated IDs *before* each sample and should
    return ``False`` to temporarily suppress EOT. This keeps word-count policy
    outside this tokenizer-agnostic module.

    This correctness-first implementation recomputes the active context on
    every step because the current Transformer does not expose a KV cache.
    """
    if not isinstance(model, TransformerLM):
        raise TypeError(f"model must be a TransformerLM, got {type(model).__name__}")
    if not isinstance(max_new_tokens, Integral) or isinstance(max_new_tokens, bool):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not isinstance(endoftext_token_id, Integral) or isinstance(endoftext_token_id, bool):
        raise TypeError("endoftext_token_id must be an integer")
    _validate_sampling_parameters(temperature, top_p)

    vocab_size = model.vocab_size
    if not 0 <= endoftext_token_id < vocab_size:
        raise ValueError(f"endoftext_token_id must be in [0, {vocab_size})")
    prompt = _normalize_prompt_token_ids(prompt_token_ids, vocab_size=vocab_size)

    try:
        device = next(model.parameters()).device
    except StopIteration as error:  # Defensive: TransformerLM normally always has parameters.
        raise ValueError("model has no parameters") from error

    generated: list[int] = []
    all_token_ids = list(prompt)
    stopped_on_endoftext = False
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(int(max_new_tokens)):
                # The full prompt/result stays in CPU-native tuples/lists; only
                # the active attention window is materialized on the accelerator.
                active_ids = all_token_ids[-model.context_length :]
                input_ids = torch.tensor(active_ids, dtype=torch.long, device=device).unsqueeze(0)
                logits = model(input_ids)
                if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != vocab_size:
                    raise RuntimeError(
                        "TransformerLM returned unexpected logits shape: "
                        f"expected (1, sequence_length, {vocab_size}), got {tuple(logits.shape)}"
                    )
                next_token = int(
                    sample_next_token(
                        logits[0, -1],
                        temperature=temperature,
                        top_p=top_p,
                        suppress_token_ids=()
                        if endoftext_allowed is None or endoftext_allowed(tuple(generated))
                        else (int(endoftext_token_id),),
                        generator=generator,
                    ).item()
                )
                generated.append(next_token)
                all_token_ids.append(next_token)
                if next_token == int(endoftext_token_id):
                    stopped_on_endoftext = True
                    break
    finally:
        model.train(was_training)

    return GenerationResult(
        prompt_token_ids=prompt,
        generated_token_ids=tuple(generated),
        stopped_on_endoftext=stopped_on_endoftext,
    )


__all__ = ["GenerationResult", "generate", "sample_next_token"]
