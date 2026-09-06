import pytest
import torch

from cs336_basics.nn.rotary_embedding import (
    RotaryPositionalEmbedding,
)
from cs336_basics.nn.rotary_embedding_einops import (
    RotaryPositionalEmbeddingWithReduce,
)
from cs336_basics.nn.transformer import (
    TransformerLM,
)


ROPE_IMPLEMENTATIONS = (
    RotaryPositionalEmbedding,
    RotaryPositionalEmbeddingWithReduce,
)


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_broadcasts_batched_positions_across_heads_when_batch_differs_from_heads(rope_cls):
    """Regression: (B, S) positions must broadcast over H when B != H."""
    batch_size = 3
    num_heads = 5
    seq_len = 7
    d_k = 8

    torch.manual_seed(0)
    x = torch.randn(batch_size, num_heads, seq_len, d_k)
    token_positions = torch.arange(seq_len).expand(batch_size, seq_len)
    rope = rope_cls(theta=10_000.0, d_k=d_k, max_seq_len=seq_len)

    output = rope(x, token_positions)

    assert output.shape == x.shape


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_shared_and_batched_positions_are_equivalent(rope_cls):
    """The same positions represented as (S,) or expanded (B, S) must agree."""
    batch_size = 3
    num_heads = 5
    seq_len = 7
    d_k = 8

    torch.manual_seed(1)
    x = torch.randn(batch_size, num_heads, seq_len, d_k)
    shared_positions = torch.arange(seq_len)
    batched_positions = shared_positions.expand(batch_size, seq_len)
    rope = rope_cls(theta=10_000.0, d_k=d_k, max_seq_len=seq_len)

    output_with_shared_positions = rope(x, shared_positions)
    output_with_batched_positions = rope(x, batched_positions)

    torch.testing.assert_close(
        output_with_shared_positions,
        output_with_batched_positions,
    )


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_supports_different_position_offsets_per_batch(rope_cls):
    """A batched call must equal applying RoPE to every batch item separately."""
    batch_size = 3
    num_heads = 5
    seq_len = 4
    d_k = 8

    torch.manual_seed(2)
    x = torch.randn(batch_size, num_heads, seq_len, d_k)
    token_positions = torch.stack(
        (
            torch.arange(0, 4),
            torch.arange(3, 7),
            torch.arange(6, 10),
        )
    )
    rope = rope_cls(theta=10_000.0, d_k=d_k, max_seq_len=10)

    batched_output = rope(x, token_positions)
    per_batch_output = torch.stack(
        [rope(x[index], token_positions[index]) for index in range(batch_size)]
    )

    torch.testing.assert_close(batched_output, per_batch_output)


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_at_position_zero_is_identity(rope_cls):
    torch.manual_seed(3)
    x = torch.randn(2, 5, 1, 8)
    token_positions = torch.zeros(2, 1, dtype=torch.long)
    rope = rope_cls(theta=10_000.0, d_k=8, max_seq_len=1)

    output = rope(x, token_positions)

    torch.testing.assert_close(output, x)


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_preserves_each_vector_l2_norm(rope_cls):
    """Every coordinate pair is rotated, so each full vector keeps its norm."""
    torch.manual_seed(4)
    x = torch.randn(3, 5, 7, 8)
    token_positions = torch.arange(7).expand(3, 7)
    rope = rope_cls(theta=10_000.0, d_k=8, max_seq_len=7)

    output = rope(x, token_positions)

    torch.testing.assert_close(
        torch.linalg.vector_norm(output, dim=-1),
        torch.linalg.vector_norm(x, dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_backward_with_batched_positions_and_multiple_heads(rope_cls):
    torch.manual_seed(5)
    x = torch.randn(3, 5, 7, 8, requires_grad=True)
    token_positions = torch.arange(7).expand(3, 7)
    rope = rope_cls(theta=10_000.0, d_k=8, max_seq_len=7)

    loss = rope(x, token_positions).square().mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_rope_implementations_agree_for_batched_multihead_input():
    torch.manual_seed(6)
    x = torch.randn(3, 5, 7, 8)
    token_positions = torch.arange(7).expand(3, 7)
    slicing_rope = RotaryPositionalEmbedding(
        theta=10_000.0,
        d_k=8,
        max_seq_len=7,
    )
    einops_rope = RotaryPositionalEmbeddingWithReduce(
        theta=10_000.0,
        d_k=8,
        max_seq_len=7,
    )

    torch.testing.assert_close(
        slicing_rope(x, token_positions),
        einops_rope(x, token_positions),
    )


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_rejects_odd_head_dimension(rope_cls):
    with pytest.raises(ValueError, match="positive even integer"):
        rope_cls(theta=10_000.0, d_k=7, max_seq_len=8)


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_rejects_position_outside_cache(rope_cls):
    rope = rope_cls(theta=10_000.0, d_k=8, max_seq_len=4)
    x = torch.randn(1, 1, 8)

    with pytest.raises(ValueError, match=r"must lie in \[0, 4\)"):
        rope(x, torch.tensor([4]))


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
@pytest.mark.parametrize("theta", (0.0, -1.0, float("inf"), float("nan")))
def test_rope_rejects_invalid_theta(rope_cls, theta):
    with pytest.raises(ValueError, match="finite and positive"):
        rope_cls(theta=theta, d_k=8, max_seq_len=8)


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
@pytest.mark.parametrize("d_k", (0, -2, 3))
def test_rope_rejects_non_positive_or_odd_head_dimension(rope_cls, d_k):
    with pytest.raises(ValueError, match="positive even integer"):
        rope_cls(theta=10_000.0, d_k=d_k, max_seq_len=8)


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
@pytest.mark.parametrize("max_seq_len", (0, -1))
def test_rope_rejects_non_positive_max_sequence_length(rope_cls, max_seq_len):
    with pytest.raises(ValueError, match="must be positive"):
        rope_cls(theta=10_000.0, d_k=8, max_seq_len=max_seq_len)


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_rejects_negative_positions(rope_cls):
    rope = rope_cls(theta=10_000.0, d_k=8, max_seq_len=4)
    x = torch.randn(1, 2, 8)

    with pytest.raises(ValueError, match=r"must lie in \[0, 4\)"):
        rope(x, torch.tensor([-1, 0]))


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_rejects_non_integer_positions(rope_cls):
    rope = rope_cls(theta=10_000.0, d_k=8, max_seq_len=4)
    x = torch.randn(1, 2, 8)

    with pytest.raises(TypeError, match="must have dtype"):
        rope(x, torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_rejects_non_floating_point_input(rope_cls):
    rope = rope_cls(theta=10_000.0, d_k=8, max_seq_len=4)
    x = torch.ones(1, 2, 8, dtype=torch.long)

    with pytest.raises(TypeError, match="must be floating point"):
        rope(x, torch.tensor([0, 1]))


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_rejects_input_with_wrong_head_dimension(rope_cls):
    rope = rope_cls(theta=10_000.0, d_k=8, max_seq_len=4)
    x = torch.randn(1, 2, 6)

    with pytest.raises(ValueError, match=r"x\.shape\[-1\]"):
        rope(x, torch.tensor([0, 1]))


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_rejects_mismatched_sequence_lengths(rope_cls):
    rope = rope_cls(theta=10_000.0, d_k=8, max_seq_len=4)
    x = torch.randn(2, 3, 8)

    with pytest.raises(ValueError, match="same sequence length"):
        rope(x, torch.tensor([0, 1]))


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_rejects_non_broadcastable_batch_dimensions(rope_cls):
    rope = rope_cls(theta=10_000.0, d_k=8, max_seq_len=4)
    x = torch.randn(3, 5, 4, 8)
    positions_for_wrong_batch_size = torch.arange(4).expand(2, 4)

    with pytest.raises(ValueError, match="not broadcastable"):
        rope(x, positions_for_wrong_batch_size)


@pytest.mark.parametrize(
    "rope_cls",
    ROPE_IMPLEMENTATIONS,
    ids=("slicing", "einops"),
)
def test_rope_accepts_an_empty_sequence(rope_cls):
    rope = rope_cls(theta=10_000.0, d_k=8, max_seq_len=4)
    x = torch.empty(2, 3, 0, 8)
    token_positions = torch.empty(2, 0, dtype=torch.long)

    output = rope(x, token_positions)

    assert output.shape == x.shape
    assert output.numel() == 0


def test_transformer_lm_forward_and_backward_when_batch_differs_from_heads():
    """Integration regression for the failure originally exposed inside RoPE."""
    batch_size = 3
    num_heads = 4
    seq_len = 6
    vocab_size = 32

    torch.manual_seed(7)
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=seq_len,
        d_model=32,
        num_heads=num_heads,
        d_ff=64,
        num_layers=1,
    )
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(token_ids)
    logits.square().mean().backward()

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
