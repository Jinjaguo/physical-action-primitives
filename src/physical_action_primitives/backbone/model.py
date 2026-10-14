"""Physical backbone: continuous mixture over learned prototype units with an explicit action decoder."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from physical_action_primitives.backbone.layers import Decoder, ResidualMLPBlock, _distortion_terms, masked_mean


@dataclass(frozen=True)
class UnitLearnerConfig:
    scale_name: str
    state_dimension: int = 108
    action_dimension: int = 8
    effect_dimension: int = 39
    contact_dimension: int = 7
    window_length: int = 16
    hidden_dimension: int = 768
    layers: int = 10
    feedforward_multiplier: int = 4
    unit_dimension: int = 128
    prototype_count: int = 64
    parameter_dimension: int = 4
    mixture_temperature: float = 0.5
    residual_bound: float = 0.5
    dropout: float = 0.0

    def validate(self) -> None:
        if self.scale_name not in {"50M", "200M", "800M", "test"}:
            raise ValueError("unknown capacity scale")
        if min(
            self.state_dimension,
            self.action_dimension,
            self.effect_dimension,
            self.contact_dimension,
            self.window_length,
            self.hidden_dimension,
            self.layers,
            self.unit_dimension,
            self.prototype_count,
            self.parameter_dimension,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.prototype_count < 4:
            raise ValueError("model requires at least four prototype units")
        if self.feedforward_multiplier < 2:
            raise ValueError("feedforward multiplier is too small")
        if not 0.0 < self.mixture_temperature <= 2.0:
            raise ValueError("mixture temperature must be in (0, 2]")
        if not 0.0 < self.residual_bound <= 1.0:
            raise ValueError("residual bound must be in (0, 1]")


MODEL_SCALES: dict[str, UnitLearnerConfig] = {
    "50M": UnitLearnerConfig(scale_name="50M", hidden_dimension=768, layers=10),
    "200M": UnitLearnerConfig(scale_name="200M", hidden_dimension=1184, layers=18),
    "800M": UnitLearnerConfig(scale_name="800M", hidden_dimension=2032, layers=24),
}


@dataclass(frozen=True)
class LossWeights:
    effect: float
    transition: float
    coarse: float
    boundary: float
    continuous_parameter: float
    parameter_smoothness: float
    residual_energy: float
    assignment_entropy: float
    marginal_balance: float
    prototype_diversity: float
    packing_consistency: float
    precision_consistency: float
    contact_within_effect: float = 1.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"invalid loss weight: {name}")
        if self.coarse <= 0:
            raise ValueError("loss requires a positive prototype-only coarse loss")
        if self.continuous_parameter <= 0:
            raise ValueError("loss requires an explicit continuous parameter rate")
        if self.assignment_entropy <= 0 or self.marginal_balance <= 0:
            raise ValueError("loss requires sparse, noncollapsed prototype activation")
        if self.prototype_diversity <= 0:
            raise ValueError("loss requires prototype diversity")
        if self.packing_consistency <= 0 or self.precision_consistency <= 0:
            raise ValueError("loss requires continuous packing/precision consistency")


class EmergentActionUnitLearner(nn.Module):
    """Continuous mixture over learned prototype units with bounded refinement."""

    def __init__(
        self,
        config: UnitLearnerConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        factory: dict[str, Any] = {"device": device, "dtype": dtype}
        input_dimension = (
            2 * config.state_dimension
            + config.action_dimension
            + config.effect_dimension
            + config.contact_dimension
        )
        self.input_norm = nn.LayerNorm(input_dimension, **factory)
        self.input_projection = nn.Linear(input_dimension, config.hidden_dimension, **factory)
        self.encoder = nn.ModuleList(
            [
                ResidualMLPBlock(
                    config.hidden_dimension,
                    config.feedforward_multiplier,
                    config.dropout,
                    **factory,
                )
                for _ in range(config.layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(config.hidden_dimension, **factory)
        self.prototype_logits = nn.Linear(
            config.hidden_dimension, config.prototype_count, **factory
        )
        self.prototype_embeddings = nn.Parameter(
            torch.empty(config.prototype_count, config.unit_dimension, **factory)
        )
        nn.init.normal_(self.prototype_embeddings, mean=0.0, std=0.02)
        self.parameter_projection = nn.Linear(
            config.hidden_dimension, 2 * config.parameter_dimension, **factory
        )

        coarse_action_input = config.unit_dimension
        residual_action_input = config.unit_dimension + config.parameter_dimension
        coarse_physical_input = config.state_dimension + config.unit_dimension
        residual_physical_input = coarse_physical_input + config.parameter_dimension
        effect_contact_dimension = config.effect_dimension + config.contact_dimension
        self.coarse_action_decoder = Decoder(
            coarse_action_input, config.hidden_dimension, config.action_dimension, **factory
        )
        self.residual_action_decoder = Decoder(
            residual_action_input, config.hidden_dimension, config.action_dimension, **factory
        )
        self.coarse_effect_decoder = Decoder(
            coarse_physical_input, config.hidden_dimension, effect_contact_dimension, **factory
        )
        self.residual_effect_decoder = Decoder(
            residual_physical_input,
            config.hidden_dimension,
            effect_contact_dimension,
            **factory,
        )
        self.coarse_transition_decoder = Decoder(
            coarse_physical_input, config.hidden_dimension, config.state_dimension, **factory
        )
        self.residual_transition_decoder = Decoder(
            residual_physical_input, config.hidden_dimension, config.state_dimension, **factory
        )

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototype_embeddings, dim=-1, eps=1e-6)

    def encode(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        effect: torch.Tensor,
        contact_active: torch.Tensor,
        state_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        config = self.config
        if state.ndim != 3 or state.shape[-1] != config.state_dimension:
            raise ValueError("state tensor violates the model contract")
        expected = {
            "action": (config.action_dimension, action),
            "effect": (config.effect_dimension, effect),
            "contact_active": (config.contact_dimension, contact_active),
            "state_delta": (config.state_dimension, state_delta),
        }
        for name, (dimension, tensor) in expected.items():
            if tensor.shape != (*state.shape[:2], dimension):
                raise ValueError(f"{name} tensor violates the model contract")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} tensor contains nonfinite values")
        if not bool(torch.isfinite(state).all()):
            raise ValueError("state tensor contains nonfinite values")
        if state.shape[1] > config.window_length:
            raise ValueError("sequence exceeds the model window length")
        values = torch.cat(
            (state, action, state_delta, effect, contact_active.to(state.dtype)), dim=-1
        )
        hidden = self.input_projection(self.input_norm(values))
        for block in self.encoder:
            hidden = block(hidden)
        hidden = self.encoder_norm(hidden)
        logits = self.prototype_logits(hidden)
        weights = torch.softmax(logits / config.mixture_temperature, dim=-1)
        unit_features = weights @ self.normalized_prototypes().to(weights.dtype)
        parameter_statistics = self.parameter_projection(hidden)
        parameter_mean, parameter_log_variance = parameter_statistics.chunk(2, dim=-1)
        parameter_log_variance = parameter_log_variance.clamp(-10.0, 10.0)
        parameters = torch.tanh(parameter_mean)
        boundary_probability = torch.zeros_like(weights[..., 0])
        if weights.shape[1] > 1:
            boundary_probability[:, 1:] = (
                0.5 * (weights[:, 1:] - weights[:, :-1]).abs().sum(dim=-1)
            ).clamp(0.0, 1.0)
        return {
            "hidden": hidden,
            "prototype_logits": logits,
            "projected_unit": weights,
            "unit_weights": weights,
            "unit_features": unit_features,
            "normalized_prototypes": self.normalized_prototypes(),
            "unit_ids": weights.argmax(dim=-1),
            "parameters": parameters,
            "parameter_mean": parameter_mean,
            "parameter_log_variance": parameter_log_variance,
            "boundary_probability": boundary_probability,
        }

    def _decode_features(
        self,
        state: torch.Tensor,
        unit_features: torch.Tensor,
        parameters: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        config = self.config
        if state.shape[:2] != unit_features.shape[:2] or state.shape[:2] != parameters.shape[:2]:
            raise ValueError("decoder inputs do not align")
        coarse_physical_input = torch.cat((state, unit_features), dim=-1)
        residual_physical_input = torch.cat((state, unit_features, parameters), dim=-1)
        residual_action_input = torch.cat((unit_features, parameters), dim=-1)
        coarse_action = self.coarse_action_decoder(unit_features)
        residual_action = config.residual_bound * torch.tanh(
            self.residual_action_decoder(residual_action_input)
        )
        coarse_effect_contact = self.coarse_effect_decoder(coarse_physical_input)
        residual_effect_contact = config.residual_bound * torch.tanh(
            self.residual_effect_decoder(residual_physical_input)
        )
        coarse_state_delta = self.coarse_transition_decoder(coarse_physical_input)
        residual_state_delta = config.residual_bound * torch.tanh(
            self.residual_transition_decoder(residual_physical_input)
        )
        effect_end = config.effect_dimension
        return {
            "action": coarse_action + residual_action,
            "effect": coarse_effect_contact[..., :effect_end]
            + residual_effect_contact[..., :effect_end],
            "contact_logits": coarse_effect_contact[..., effect_end:]
            + residual_effect_contact[..., effect_end:],
            "state_delta": coarse_state_delta + residual_state_delta,
            "coarse_action": coarse_action,
            "coarse_effect": coarse_effect_contact[..., :effect_end],
            "coarse_contact_logits": coarse_effect_contact[..., effect_end:],
            "coarse_state_delta": coarse_state_delta,
            "residual_action": residual_action,
            "residual_effect_contact": residual_effect_contact,
            "residual_state_delta": residual_state_delta,
        }

    def decode(
        self,
        state: torch.Tensor,
        unit_weights: torch.Tensor,
        parameters: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if unit_weights.shape != (*state.shape[:2], self.config.prototype_count):
            raise ValueError("decoder unit weights changed shape")
        if parameters.shape != (*state.shape[:2], self.config.parameter_dimension):
            raise ValueError("decoder parameters changed shape")
        if not bool(torch.isfinite(unit_weights).all()) or not bool(
            torch.isfinite(parameters).all()
        ):
            raise ValueError("decoder inputs contain nonfinite values")
        if bool((unit_weights < 0).any()):
            raise ValueError("prototype weights must be nonnegative")
        if bool((unit_weights.sum(dim=-1) <= 0).any()):
            raise ValueError("prototype weights must have positive mass")
        if bool((parameters.abs() > 1.0001).any()):
            raise ValueError("decoder parameters exceed the allowed bound")
        normalized = unit_weights / unit_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        features = normalized @ self.normalized_prototypes().to(normalized.dtype)
        return self._decode_features(state, features, parameters)

    def decode_from_posterior(
        self, state: torch.Tensor, posterior: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return self._decode_features(state, posterior["unit_features"], posterior["parameters"])

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        effect: torch.Tensor,
        contact_active: torch.Tensor,
        state_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        posterior = self.encode(state, action, effect, contact_active, state_delta)
        decoded = self.decode_from_posterior(state, posterior)
        return {**posterior, **decoded}


def reconstruction_distortion(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    action, effect, transition, effect_continuous, effect_contact = _distortion_terms(
        prediction, batch, prefix="", weights=weights
    )
    return {
        "total": action + weights.effect * effect + weights.transition * transition,
        "action": action,
        "effect": effect,
        "effect_continuous": effect_continuous,
        "effect_contact": effect_contact,
        "transition": transition,
    }


def unit_learner_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: LossWeights,
    *,
    packing_target: dict[str, torch.Tensor] | None = None,
    precision_target: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Soft-prototype rate-distortion objective with continuous stability."""
    weights.validate()
    validity = batch["validity"].bool()
    action, effect, transition, effect_continuous, effect_contact = _distortion_terms(
        prediction, batch, prefix="", weights=weights
    )
    coarse_action, coarse_effect, coarse_transition, coarse_effect_continuous, coarse_effect_contact = _distortion_terms(
        prediction, batch, prefix="coarse_", weights=weights
    )
    boundary = masked_mean(prediction["boundary_probability"], validity)
    continuous_parameter = masked_mean(
        0.5
        * (
            prediction["parameter_mean"].square()
            + prediction["parameter_log_variance"].exp()
            - 1.0
            - prediction["parameter_log_variance"]
        ),
        validity,
    )
    if prediction["parameters"].shape[1] > 1:
        pair_mask = validity[:, 1:] & validity[:, :-1]
        continuation = 1.0 - prediction["boundary_probability"][:, 1:]
        parameter_change = (
            prediction["parameters"][:, 1:] - prediction["parameters"][:, :-1]
        ).square().mean(dim=-1)
        parameter_smoothness = masked_mean(parameter_change * continuation, pair_mask)
    else:
        parameter_smoothness = boundary.new_zeros(())
    residual_energy = (
        masked_mean(prediction["residual_action"].square(), validity)
        + masked_mean(prediction["residual_effect_contact"].square(), validity)
        + masked_mean(prediction["residual_state_delta"].square(), validity)
    )
    probabilities = prediction["unit_weights"].clamp_min(1e-8)
    log_count = math.log(prediction["unit_weights"].shape[-1])
    sample_entropy = -(probabilities * probabilities.log()).sum(dim=-1) / log_count
    assignment_entropy = masked_mean(sample_entropy, validity)
    valid_weights = probabilities * validity.unsqueeze(-1).to(probabilities.dtype)
    marginal = valid_weights.sum(dim=(0, 1)) / validity.sum().clamp_min(1).to(probabilities.dtype)
    marginal = marginal.clamp_min(1e-8)
    marginal_balance = (log_count + (marginal * marginal.log()).sum()) / log_count
    model_prototypes = prediction.get("normalized_prototypes")
    if model_prototypes is None:
        prototype_diversity = assignment_entropy.new_zeros(())
    else:
        gram = model_prototypes.float() @ model_prototypes.float().transpose(0, 1)
        off_diagonal = gram - torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        prototype_diversity = off_diagonal.square().mean()

    def consistency(target: dict[str, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        if target is None:
            zero = assignment_entropy.new_zeros(())
            return zero, zero
        target_weights = target["projected_unit"].detach().float()
        reference = prediction["unit_weights"][
            : target_weights.shape[0], : target_weights.shape[1]
        ].float()
        selected_validity = validity[: target_weights.shape[0], : target_weights.shape[1]]
        distance = masked_mean((reference - target_weights).square().sum(dim=-1), selected_validity)
        cosine = masked_mean(
            F.cosine_similarity(reference, target_weights, dim=-1, eps=1e-8),
            selected_validity,
        )
        return distance, cosine

    packing_consistency, packing_cosine = consistency(packing_target)
    precision_consistency, precision_cosine = consistency(precision_target)
    coarse_total = coarse_action + weights.effect * coarse_effect + weights.transition * coarse_transition
    total = (
        action
        + weights.effect * effect
        + weights.transition * transition
        + weights.coarse * coarse_total
        + weights.boundary * boundary
        + weights.continuous_parameter * continuous_parameter
        + weights.parameter_smoothness * parameter_smoothness
        + weights.residual_energy * residual_energy
        + weights.assignment_entropy * assignment_entropy
        + weights.marginal_balance * marginal_balance
        + weights.prototype_diversity * prototype_diversity
        + weights.packing_consistency * packing_consistency
        + weights.precision_consistency * precision_consistency
    )
    return {
        "total": total,
        "action": action,
        "effect": effect,
        "effect_continuous": effect_continuous,
        "effect_contact": effect_contact,
        "transition": transition,
        "coarse_total": coarse_total,
        "coarse_action": coarse_action,
        "coarse_effect": coarse_effect,
        "coarse_effect_continuous": coarse_effect_continuous,
        "coarse_effect_contact": coarse_effect_contact,
        "coarse_transition": coarse_transition,
        "rate_boundary": boundary,
        "rate_continuous_parameter": continuous_parameter,
        "rate_parameter_smoothness": parameter_smoothness,
        "residual_energy": residual_energy,
        "assignment_entropy": assignment_entropy,
        "marginal_balance": marginal_balance,
        "prototype_diversity": prototype_diversity,
        "packing_weight_consistency": packing_consistency,
        "packing_weight_cosine": packing_cosine,
        "precision_weight_consistency": precision_consistency,
        "precision_weight_cosine": precision_cosine,
        "stability_probe_executed": total.new_tensor(
            float(packing_target is not None and precision_target is not None)
        ),
    }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
