from typing import Any

from transformers import PretrainedConfig


class Config(PretrainedConfig):
    model_type = "nfm"

    def __init__(
        self,
        dim: int = 384,
        hidden_dim: int = 384 * 4,
        num_heads: int = 12,
        num_cross_layers: int = 24,
        num_self_layers: int = 24,
        rope_theta: float = 10000,
        **kwargs: Any,
    ) -> None:
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_cross_layers = num_cross_layers
        self.num_self_layers = num_self_layers
        self.rope_theta = rope_theta
        super().__init__(**kwargs)
