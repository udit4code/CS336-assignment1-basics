"""Neural-network layers, losses, optimizers, and model components."""

from .activation import SiLU
from .adamw import AdamW
from .attention import scaled_dot_product_attention
from .attention_einops import scaled_dot_product_attention_with_einops
from .cross_entropy import cross_entropy
from .embedding import Embedding
from .feed_forward import SwiGLU
from .feed_forward_einops import SwiGLUEinops
from .gradient_clipping import gradient_clipping
from .linear import Linear
from .linear_einops import LinearEinops
from .multihead_attention import MultiHeadSelfAttention
from .normalization import RMSNorm
from .normalization_einops import RMSNormReduce
from .rotary_embedding import RotaryPositionalEmbedding
from .rotary_embedding_einops import RotaryPositionalEmbeddingWithReduce
from .schedules import get_lr_cosine_schedule
from .sgd import StochasticGradientDescentOptimizer
from .softmax import softmax
from .transformer import TransformerLM
from .transformer_block import TransformerBlock

__all__ = [
    "AdamW",
    "Embedding",
    "Linear",
    "LinearEinops",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RMSNormReduce",
    "RotaryPositionalEmbedding",
    "RotaryPositionalEmbeddingWithReduce",
    "SiLU",
    "StochasticGradientDescentOptimizer",
    "SwiGLU",
    "SwiGLUEinops",
    "TransformerBlock",
    "TransformerLM",
    "cross_entropy",
    "get_lr_cosine_schedule",
    "gradient_clipping",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_with_einops",
    "softmax",
]
