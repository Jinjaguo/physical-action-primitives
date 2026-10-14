"""Training-schedule helpers and config validation for backbone training."""

from __future__ import annotations

import math
from typing import Any

from physical_action_primitives.backbone.model import MODEL_SCALES, LossWeights


def should_run_consistency_probe(
    *,
    optimizer_steps: int,
    micro_steps: int,
    gradient_accumulation_steps: int,
    consistency_every_steps: int,
) -> bool:
    """Run one stability probe on the microbatch that closes a selected step."""

    if optimizer_steps < 0 or micro_steps < 0:
        raise ValueError("step counters must be nonnegative")
    if gradient_accumulation_steps <= 0 or consistency_every_steps <= 0:
        raise ValueError("consistency scheduling intervals must be positive")
    return (
        (optimizer_steps + 1) % consistency_every_steps == 0
        and (micro_steps + 1) % gradient_accumulation_steps == 0
    )


def validate_training_config(config: dict[str, Any], architecture: dict[str, Any]) -> None:
    """Check the training config for consistency with the model and architecture files."""

    shared = architecture.get("shared_representation", {})
    reference = MODEL_SCALES["50M"]
    if int(shared.get("prototype_count", 0)) != reference.prototype_count:
        raise ValueError("architecture prototype count differs from the model")
    if float(shared.get("mixture_temperature", 0.0)) != reference.mixture_temperature:
        raise ValueError("architecture mixture temperature differs from the model")
    if int(shared.get("parameter_dimension", 0)) != reference.parameter_dimension:
        raise ValueError("architecture continuous parameter dimension differs from the model")
    if float(shared.get("residual_bound", 0.0)) != reference.residual_bound:
        raise ValueError("architecture residual bound differs from the model")
    if shared.get("action_decoder_state_conditioned") is not False:
        raise ValueError("the action decoder must not be conditioned on state")
    if shared.get("transition_local_posterior") is not True:
        raise ValueError("the posterior must be transition-local")
    if shared.get("crop_relative_inputs") is not False:
        raise ValueError("crop-relative inputs are not supported")
    if shared.get("identity_inference_precision") != "fp32_tf32_disabled":
        raise ValueError("unit inference must run in FP32 with TF32 disabled")
    scale = str(config.get("model_scale"))
    if scale not in MODEL_SCALES:
        raise ValueError("unknown model scale")
    model = MODEL_SCALES[scale]
    recorded = architecture["scales"][scale]
    if int(recorded["layers"]) != model.layers:
        raise ValueError("training model layer count differs from architecture")
    if int(recorded["hidden_dimension"]) != model.hidden_dimension:
        raise ValueError("training hidden dimension differs from architecture")
    if config.get("partition") != "source_training":
        raise ValueError("the backbone may train only on source-training tasks")
    if int(config.get("window_length", 0)) != model.window_length:
        raise ValueError("training window length differs from the model")
    if int(config.get("window_stride", 0)) < 1:
        raise ValueError("training window stride is invalid")
    if config.get("window_sampling") not in {
        "all_lazy_windows",
        "one_random_window_per_root_per_epoch",
    }:
        raise ValueError("unknown window-sampling mode")
    if config.get("precision") != "bfloat16":
        raise ValueError("training precision must be bfloat16")
    if int(config.get("per_gpu_batch_size", 0)) < 1:
        raise ValueError("training batch size is invalid")
    if int(config.get("gradient_accumulation_steps", 0)) < 1:
        raise ValueError("gradient accumulation is invalid")
    if int(config.get("epochs", 0)) < 1:
        raise ValueError("training epoch count is invalid")
    if int(config.get("max_optimizer_steps", 0)) < 1:
        raise ValueError("optimizer-step count is invalid")
    consistency_every = int(config.get("consistency_every_steps", 0))
    consistency_windows = int(config.get("consistency_windows", 0))
    if consistency_every < 1:
        raise ValueError("stability-probe interval is invalid")
    if consistency_windows < 2 or consistency_windows > int(config["per_gpu_batch_size"]):
        raise ValueError("stability-probe window count is invalid")
    if int(config.get("validation_windows", 0)) < 1:
        raise ValueError("validation window count is invalid")
    optimizer = config["optimizer"]
    if optimizer.get("name") != "AdamW":
        raise ValueError("only the AdamW optimizer is supported")
    if float(optimizer["learning_rate"]) <= 0:
        raise ValueError("learning rate is invalid")
    if int(optimizer["warmup_steps"]) < 0:
        raise ValueError("warmup is invalid")
    if int(optimizer["warmup_steps"]) >= int(config["max_optimizer_steps"]):
        raise ValueError("warmup consumes the entire run")
    LossWeights(**config["loss_weights"]).validate()
    if int(config["resource_envelope"]["gpus"]) < 1:
        raise ValueError("GPU count is invalid")


def cosine_learning_rate(
    step: int, total_steps: int, warmup_steps: int, peak: float, minimum_ratio: float
) -> float:
    if step < 0 or total_steps < 1 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("invalid learning-rate schedule")
    if not 0 <= minimum_ratio <= 1:
        raise ValueError("invalid minimum learning-rate ratio")
    if step < warmup_steps:
        return peak * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (minimum_ratio + (1.0 - minimum_ratio) * cosine)
