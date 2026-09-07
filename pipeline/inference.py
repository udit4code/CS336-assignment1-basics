"""Load a trained artifact and orchestrate tokenizer-aware text generation."""

from dataclasses import asdict, dataclass
import re
from typing import Any

import tiktoken
import torch

from cs336_basics.generation import generate
from cs336_basics.nn import TransformerLM

from .artifacts import load_final_artifact
from .config import InferenceConfig, ModelConfig
from .prepare_data import END_OF_TEXT
from .runtime import resolve_device, resolve_dtype


_WORD_PATTERN = re.compile(r"\b[\w]+(?:['’][\w]+)*\b", flags=re.UNICODE)


def count_words(text: str) -> int:
    """Count human-readable words in generated text.

    This intentionally counts decoded text rather than token IDs: one BPE
    token can be part of a word, a whole word, or punctuation. Apostrophes
    inside a word remain part of that word; hyphenated terms count as two.
    """
    return len(_WORD_PATTERN.findall(text))


@dataclass(frozen=True)
class InferenceResult:
    """Text plus token-level metadata for one completed inference request."""

    artifact_path: str
    tokenizer: str
    device: str
    dtype: str
    prompt: str
    completion: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    stop_reason: str

    @property
    def text(self) -> str:
        """Return the original prompt followed by its generated completion."""
        return self.prompt + self.completion

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["text"] = self.text
        payload["num_prompt_tokens"] = len(self.prompt_token_ids)
        payload["num_generated_tokens"] = len(self.generated_token_ids)
        payload["num_generated_words"] = count_words(self.completion)
        return payload


def _model_config_from_artifact(payload: dict[str, Any]) -> ModelConfig:
    try:
        model_config = ModelConfig(**dict(payload["config"]["model"]))
    except (KeyError, TypeError) as error:
        raise ValueError("artifact model configuration does not match the current ModelConfig schema") from error
    model_config.validate()
    return model_config


def _load_tokenizer(payload: dict[str, Any], model_config: ModelConfig) -> tiktoken.Encoding:
    tokenizer_metadata = payload["tokenizer"]
    encoding_name = tokenizer_metadata["name"]
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError as error:
        raise ValueError(f"artifact requires unavailable tiktoken encoding {encoding_name!r}") from error

    recorded_vocab_size = int(tokenizer_metadata["vocab_size"])
    if recorded_vocab_size != encoding.n_vocab:
        raise ValueError(
            "artifact tokenizer vocabulary does not match the installed tiktoken encoding: "
            f"artifact={recorded_vocab_size}, installed={encoding.n_vocab}"
        )
    if model_config.vocab_size != encoding.n_vocab:
        raise ValueError(
            "artifact model and tokenizer vocabulary sizes differ: "
            f"model={model_config.vocab_size}, tokenizer={encoding.n_vocab}"
        )
    if END_OF_TEXT not in encoding.special_tokens_set or encoding.eot_token is None:
        raise ValueError(f"encoding {encoding_name!r} does not define {END_OF_TEXT}")

    # New artifacts record the exact ID. Older schema-v1 artifacts did not, so
    # absence remains backwards compatible while a conflicting value fails.
    recorded_endoftext_id = tokenizer_metadata.get("endoftext_id")
    if recorded_endoftext_id is not None and int(recorded_endoftext_id) != encoding.eot_token:
        raise ValueError(
            "artifact end-of-text ID does not match the installed tokenizer: "
            f"artifact={recorded_endoftext_id}, installed={encoding.eot_token}"
        )
    return encoding


def run_inference(config: InferenceConfig) -> InferenceResult:
    """Generate one completion from a trusted final training artifact."""
    config.validate()
    artifact_path = config.artifact_path / "artifact.pt" if config.artifact_path.is_dir() else config.artifact_path
    payload = load_final_artifact(artifact_path, map_location="cpu")
    model_config = _model_config_from_artifact(payload)
    encoding = _load_tokenizer(payload, model_config)
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)

    model = TransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        num_layers=model_config.num_layers,
        theta=model_config.rope_theta,
        device=device,
        dtype=dtype,
    )
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError("artifact weights are incompatible with its recorded model configuration") from error
    model.eval()

    # Match training's special-token contract: a literal <|endoftext|> in the
    # prompt is one token, while unexpected special markers remain disallowed.
    prompt_token_ids = encoding.encode(config.prompt, allowed_special={END_OF_TEXT})
    if not prompt_token_ids:
        raise ValueError("prompt produced zero tokens")

    sampling_generator = None
    if config.seed is not None:
        sampling_generator = torch.Generator(device=device)
        sampling_generator.manual_seed(config.seed)

    generation = generate(
        model,
        prompt_token_ids,
        endoftext_token_id=encoding.eot_token,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        generator=sampling_generator,
        endoftext_allowed=(
            None
            if config.min_words == 0
            else lambda generated_ids: count_words(encoding.decode(generated_ids)) >= config.min_words
        ),
    )

    # Keep EOT in generated_token_ids for diagnostics, but do not print the
    # sentinel as natural-language output.
    completion_token_ids = list(generation.generated_token_ids)
    if generation.stopped_on_endoftext:
        completion_token_ids = completion_token_ids[:-1]
    completion = encoding.decode(completion_token_ids)
    generated_word_count = count_words(completion)
    if generated_word_count < config.min_words:
        raise RuntimeError(
            f"generation reached max_new_tokens={config.max_new_tokens} with only "
            f"{generated_word_count} words; increase --max-new-tokens to produce at least {config.min_words} words"
        )

    return InferenceResult(
        artifact_path=str(artifact_path),
        tokenizer=encoding.name,
        device=str(device),
        dtype=config.dtype,
        prompt=config.prompt,
        completion=completion,
        prompt_token_ids=list(generation.prompt_token_ids),
        generated_token_ids=list(generation.generated_token_ids),
        stop_reason=generation.stop_reason,
    )


__all__ = ["InferenceResult", "run_inference"]
