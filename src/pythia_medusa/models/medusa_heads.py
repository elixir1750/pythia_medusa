from __future__ import annotations

import torch
from torch import nn

from pythia_medusa.models.medusa_config import MedusaConfig


class ResBlock(nn.Module):
    """A simple residual MLP block used inside each Medusa head."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states + residual


class MedusaHead(nn.Module):
    """One future-offset prediction head producing hidden-state refinements."""

    def __init__(self, config: MedusaConfig) -> None:
        super().__init__()
        medusa_hidden_size = config.resolved_medusa_hidden_size
        self.input_proj = (
            nn.Linear(config.hidden_size, medusa_hidden_size)
            if medusa_hidden_size != config.hidden_size
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            ResBlock(medusa_hidden_size) for _ in range(config.medusa_num_layers)
        )
        self.output_proj = (
            nn.Linear(medusa_hidden_size, config.hidden_size)
            if medusa_hidden_size != config.hidden_size
            else nn.Identity()
        )
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_proj(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.output_proj(hidden_states)


class MedusaHeadStack(nn.Module):
    """A stack of independent Medusa heads returning hidden states per future offset."""

    def __init__(self, config: MedusaConfig) -> None:
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList(MedusaHead(config) for _ in range(config.medusa_num_heads))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        refined_states = [head(hidden_states) for head in self.heads]
        return torch.stack(refined_states, dim=0)
