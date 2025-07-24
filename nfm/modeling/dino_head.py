import torch
from torch import Tensor, nn


class DINOHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        num_layers: int,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        assert num_layers > 1

        # build the MLP
        layers = []
        h = [hidden_dim] * (num_layers - 1)
        for n, k in zip([input_dim, *h], h, strict=False):
            layers.append(nn.Linear(n, k))
            layers.append(act_layer())

        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        self.apply(self.init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, output_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)

    def init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2, eps=torch.finfo(x.dtype).eps)
        return self.last_layer(x)
