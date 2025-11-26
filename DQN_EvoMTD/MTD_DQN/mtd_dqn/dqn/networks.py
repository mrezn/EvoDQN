from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


def _orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DuelingQNetwork(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            input_dim = hidden
        self.feature_extractor = nn.Sequential(*layers)

        value_layers = [nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, 1)]
        advantage_layers = [nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, action_dim)]
        self.value_stream = nn.Sequential(*value_layers)
        self.advantage_stream = nn.Sequential(*advantage_layers)
        self.apply(_orthogonal_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
 
        features = self.feature_extractor(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values
