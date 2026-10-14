"""Shared layers and loss helpers for the backbone model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class Decoder(nn.Module):
    def __init__(
        self, input_dimension: int, hidden_dimension: int, output_dimension: int, **factory: Any
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension, **factory),
            nn.GELU(),
            nn.Linear(hidden_dimension, output_dimension, **factory),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

class ResidualMLPBlock(nn.Module):
    """Pointwise capacity block that cannot introduce crop-boundary dependence."""

    def __init__(
        self,
        hidden_dimension: int,
        feedforward_multiplier: int,
        dropout: float,
        **factory: Any,
    ) -> None:
        super().__init__()
        inner_dimension = hidden_dimension * feedforward_multiplier
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dimension, **factory),
            nn.Linear(hidden_dimension, inner_dimension, **factory),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dimension, hidden_dimension, **factory),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)

def _distortion_terms(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    prefix: str,
    weights: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validity = batch["validity"].bool()
    action_mask = validity.unsqueeze(-1) & batch["action_dimension_mask"].bool()
    effect_mask = validity.unsqueeze(-1) & batch["effect_dimension_mask"].bool()
    contact_mask = validity.unsqueeze(-1) & batch["contact_observation_mask"].bool()
    key = lambda name: f"{prefix}{name}" if prefix else name
    action = masked_mean(
        F.smooth_l1_loss(prediction[key("action")], batch["action"], reduction="none"),
        action_mask,
    )
    effect_continuous = masked_mean(
        F.smooth_l1_loss(prediction[key("effect")], batch["effect"], reduction="none"),
        effect_mask,
    )
    effect_contact = masked_mean(
        F.binary_cross_entropy_with_logits(
            prediction[key("contact_logits")], batch["contact_active"].float(), reduction="none"
        ),
        contact_mask,
    )
    effect = effect_continuous + weights.contact_within_effect * effect_contact
    transition = masked_mean(
        F.smooth_l1_loss(
            prediction[key("state_delta")], batch["state_delta"], reduction="none"
        ),
        validity,
    )
    return action, effect, transition, effect_continuous, effect_contact
