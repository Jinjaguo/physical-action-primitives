from __future__ import annotations

import pytest
import torch

from physical_action_primitives.backbone.model import (
    MODEL_SCALES,
    EmergentActionUnitLearner,
    LossWeights,
    UnitLearnerConfig,
    trainable_parameter_count,
    unit_learner_loss,
)
from physical_action_primitives.backbone.training_core import should_run_consistency_probe


def tiny_config() -> UnitLearnerConfig:
    return UnitLearnerConfig(
        scale_name="test",
        state_dimension=9,
        action_dimension=3,
        effect_dimension=4,
        contact_dimension=2,
        window_length=5,
        hidden_dimension=24,
        layers=2,
        feedforward_multiplier=2,
        unit_dimension=8,
        prototype_count=6,
        parameter_dimension=2,
    )


def weights() -> LossWeights:
    return LossWeights(
        effect=1.0,
        transition=1.0,
        coarse=1.0,
        boundary=0.01,
        continuous_parameter=0.01,
        parameter_smoothness=0.01,
        residual_energy=0.001,
        assignment_entropy=0.01,
        marginal_balance=0.01,
        prototype_diversity=0.001,
        packing_consistency=0.02,
        precision_consistency=0.02,
    )


def batch(config: UnitLearnerConfig, batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "state": torch.randn(batch_size, config.window_length, config.state_dimension),
        "action": torch.randn(batch_size, config.window_length, config.action_dimension),
        "effect": torch.randn(batch_size, config.window_length, config.effect_dimension),
        "state_delta": torch.randn(batch_size, config.window_length, config.state_dimension),
        "contact_active": torch.randint(
            0, 2, (batch_size, config.window_length, config.contact_dimension)
        ).bool(),
        "validity": torch.ones(batch_size, config.window_length, dtype=torch.bool),
        "action_dimension_mask": torch.ones(
            batch_size, config.window_length, config.action_dimension, dtype=torch.bool
        ),
        "effect_dimension_mask": torch.ones(
            batch_size, config.window_length, config.effect_dimension, dtype=torch.bool
        ),
        "contact_observation_mask": torch.ones(
            batch_size, config.window_length, config.contact_dimension, dtype=torch.bool
        ),
    }


def posterior_inputs(values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (
        values["state"],
        values["action"],
        values["effect"],
        values["contact_active"],
        values["state_delta"],
    )


def test_forward_loss_and_simplex_contract() -> None:
    config = tiny_config()
    model = EmergentActionUnitLearner(config)
    values = batch(config)
    prediction = model(*posterior_inputs(values))
    terms = unit_learner_loss(prediction, values, weights())
    assert prediction["unit_weights"].shape == (
        2,
        config.window_length,
        config.prototype_count,
    )
    assert torch.allclose(
        prediction["unit_weights"].sum(dim=-1),
        torch.ones(2, config.window_length),
    )
    assert bool((prediction["unit_weights"] >= 0).all())
    assert bool((prediction["parameters"].abs() <= 1.0).all())
    assert set(terms) >= {
        "total",
        "assignment_entropy",
        "marginal_balance",
        "prototype_diversity",
        "packing_weight_consistency",
        "precision_weight_consistency",
    }
    assert all(bool(torch.isfinite(value)) for value in terms.values())
    terms["total"].backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_identical_continuous_targets_have_zero_penalty() -> None:
    config = tiny_config()
    model = EmergentActionUnitLearner(config)
    values = batch(config)
    prediction = model(*posterior_inputs(values))
    target = {"projected_unit": prediction["unit_weights"].detach().clone()}
    terms = unit_learner_loss(
        prediction, values, weights(), packing_target=target, precision_target=target
    )
    assert terms["packing_weight_consistency"].item() == 0.0
    assert terms["precision_weight_consistency"].item() == 0.0
    assert terms["packing_weight_cosine"].item() == pytest.approx(1.0)
    assert terms["precision_weight_cosine"].item() == pytest.approx(1.0)


def test_transition_local_weights_are_crop_invariant() -> None:
    config = tiny_config()
    model = EmergentActionUnitLearner(config).eval()
    values = batch(config)
    with torch.no_grad():
        full = model.encode(*posterior_inputs(values))
        suffix = model.encode(*(value[:, 2:] for value in posterior_inputs(values)))
    assert torch.allclose(
        full["unit_weights"][:, 2:], suffix["unit_weights"], rtol=1e-5, atol=1e-6
    )
    assert torch.allclose(
        full["parameters"][:, 2:], suffix["parameters"], rtol=1e-5, atol=1e-6
    )


def test_action_decoder_is_state_blind_but_physical_decoder_is_not() -> None:
    config = tiny_config()
    model = EmergentActionUnitLearner(config).eval()
    unit_weights = torch.softmax(torch.randn(2, 3, config.prototype_count), dim=-1)
    parameters = torch.tanh(torch.randn(2, 3, config.parameter_dimension))
    first = model.decode(
        torch.zeros(2, 3, config.state_dimension), unit_weights, parameters
    )
    second = model.decode(
        torch.ones(2, 3, config.state_dimension), unit_weights, parameters
    )
    assert torch.equal(first["action"], second["action"])
    assert not torch.equal(first["effect"], second["effect"])
    assert not torch.equal(first["state_delta"], second["state_delta"])


def test_prototype_only_coarse_path_cannot_read_parameters() -> None:
    config = tiny_config()
    model = EmergentActionUnitLearner(config).eval()
    state = torch.randn(2, 3, config.state_dimension)
    unit_weights = torch.softmax(torch.randn(2, 3, config.prototype_count), dim=-1)
    zeros = torch.zeros(2, 3, config.parameter_dimension)
    ones = torch.ones_like(zeros)
    first = model.decode(state, unit_weights, zeros)
    second = model.decode(state, unit_weights, ones)
    assert torch.equal(first["coarse_action"], second["coarse_action"])
    assert torch.equal(first["coarse_effect"], second["coarse_effect"])
    assert torch.equal(first["coarse_state_delta"], second["coarse_state_delta"])
    assert not torch.equal(first["action"], second["action"])


def test_decode_rejects_invalid_mixture_or_parameter() -> None:
    config = tiny_config()
    model = EmergentActionUnitLearner(config)
    state = torch.zeros(1, 2, config.state_dimension)
    parameters = torch.zeros(1, 2, config.parameter_dimension)
    invalid = torch.zeros(1, 2, config.prototype_count)
    with pytest.raises(ValueError, match="positive mass"):
        model.decode(state, invalid, parameters)
    invalid[..., 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        model.decode(state, invalid, parameters)
    valid = torch.ones(1, 2, config.prototype_count)
    with pytest.raises(ValueError, match="bound"):
        model.decode(state, valid, parameters + 1.01)


def test_scale_parameter_counts_are_in_declared_capacity_bands() -> None:
    observed = {
        name: trainable_parameter_count(
            EmergentActionUnitLearner(config, device=torch.device("meta"))
        )
        for name, config in MODEL_SCALES.items()
    }
    assert 40_000_000 <= observed["50M"] <= 65_000_000
    assert 160_000_000 <= observed["200M"] <= 260_000_000
    assert 650_000_000 <= observed["800M"] <= 950_000_000
    assert observed["50M"] < observed["200M"] < observed["800M"]


def test_loss_requires_all_soft_prototype_rates() -> None:
    bad = weights().__dict__.copy()
    bad["coarse"] = 0.0
    with pytest.raises(ValueError, match="coarse"):
        LossWeights(**bad).validate()
    bad = weights().__dict__.copy()
    bad["assignment_entropy"] = 0.0
    with pytest.raises(ValueError, match="sparse"):
        LossWeights(**bad).validate()
    bad = weights().__dict__.copy()
    bad["packing_consistency"] = 0.0
    with pytest.raises(ValueError, match="consistency"):
        LossWeights(**bad).validate()


def test_consistency_probe_runs_once_on_selected_optimizer_boundary() -> None:
    observed = [
        should_run_consistency_probe(
            optimizer_steps=7,
            micro_steps=micro_step,
            gradient_accumulation_steps=3,
            consistency_every_steps=8,
        )
        for micro_step in (21, 22, 23)
    ]
    assert observed == [False, False, True]
