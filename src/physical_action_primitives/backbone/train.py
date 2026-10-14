"""Train the physical backbone (continuous prototype-mixture model with an explicit action decoder).

Usage: ``python -m physical_action_primitives.backbone.train --config configs/backbone/train_800m.json``
(launch with ``torchrun`` for multi-GPU runs).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from physical_action_primitives.manifest import atomic_write_json
from physical_action_primitives.paths import resolve
from physical_action_primitives.backbone.split_manifest import validate_split_manifest
from physical_action_primitives.backbone.model import (
    MODEL_SCALES,
    EmergentActionUnitLearner,
    LossWeights,
    reconstruction_distortion,
    trainable_parameter_count,
    unit_learner_loss,
)
from physical_action_primitives.backbone.training_core import (
    cosine_learning_rate,
    should_run_consistency_probe,
    validate_training_config,
)
from physical_action_primitives.backbone.training_data import (
    TrainingShard,
    iter_root_window_batches,
    iter_shard_batches,
    load_training_shard,
    normalization_from_manifest,
    partition_shards,
    root_path,
    training_shards,
)
from physical_action_primitives.backbone.data import select_unit_array_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: checkpoints/backbone/<scale> under PAP_ROOT",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def distributed_context(expected_world_size: int) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != expected_world_size:
        raise RuntimeError(
            f"training world size {world_size} differs from configured GPU count {expected_world_size}"
        )
    if world_size > 1:
        distributed.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def limit_shards_per_task(
    shards: list[TrainingShard], limit: int | None
) -> list[TrainingShard]:
    if limit is None:
        return shards
    if limit < 1:
        raise ValueError("max shards per task must be positive")
    by_task: dict[tuple[str, str], list[TrainingShard]] = defaultdict(list)
    for shard in shards:
        by_task[(shard.source, shard.environment_id)].append(shard)
    result: list[TrainingShard] = []
    for source_task in sorted(by_task):
        result.extend(
            sorted(by_task[source_task], key=lambda item: str(item.path))[:limit]
        )
    return result


def torch_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(np.ascontiguousarray(value)).to(device, non_blocking=True)
        for name, value in batch.items()
    }


def posterior_forward(
    model: EmergentActionUnitLearner | DistributedDataParallel,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Evaluate the transition-local posterior without crop-relative inputs."""

    return model(
        batch["state"],
        batch["action"],
        batch["effect"],
        batch["contact_active"],
        batch["state_delta"],
    )


def posterior_encode(
    model: EmergentActionUnitLearner | DistributedDataParallel,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    return unwrapped.encode(
        batch["state"],
        batch["action"],
        batch["effect"],
        batch["contact_active"],
        batch["state_delta"],
    )


def batch_slice(
    batch: dict[str, torch.Tensor], start: int, stop: int
) -> dict[str, torch.Tensor]:
    return {name: value[start:stop] for name, value in batch.items()}


def stability_targets(
    model: EmergentActionUnitLearner | DistributedDataParallel,
    batch: dict[str, torch.Tensor],
    window_count: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Create detached alternate-packing BF16 and high-precision teacher views."""
    selected_count = min(window_count, int(batch["state"].shape[0]))
    if selected_count < 2:
        raise RuntimeError("stability probe requires at least two windows")
    selected = batch_slice(batch, 0, selected_count)
    prior_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prior_float32_precision = torch.get_float32_matmul_precision()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            parts = [
                posterior_encode(model, batch_slice(selected, index, index + 1))
                for index in range(selected_count)
            ]
            packing = {
                name: torch.cat([part[name] for part in parts], dim=0)
                for name in ("projected_unit",)
            }
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        with torch.autocast(device_type="cuda", enabled=False):
            high_precision = posterior_encode(model, selected)
            precision = {
                name: high_precision[name]
                for name in ("projected_unit",)
            }
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32
        torch.set_float32_matmul_precision(prior_float32_precision)
    return packing, precision


def reduced_scalars(
    terms: dict[str, torch.Tensor], world_size: int
) -> dict[str, float]:
    names = sorted(terms)
    values = torch.stack([terms[name].detach().float() for name in names])
    if world_size > 1:
        distributed.all_reduce(values, op=distributed.ReduceOp.SUM)
        values /= world_size
    return {name: float(value) for name, value in zip(names, values.cpu())}


def optimizer_learning_rate(config: dict[str, Any], step: int) -> float:
    optimizer = config["optimizer"]
    return cosine_learning_rate(
        step,
        int(config["max_optimizer_steps"]),
        int(optimizer["warmup_steps"]),
        float(optimizer["learning_rate"]),
        float(optimizer["minimum_learning_rate_ratio"]),
    )


def validation(
    model: EmergentActionUnitLearner,
    shards: list[TrainingShard],
    normalization: Any,
    config: dict[str, Any],
    weights: LossWeights,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    limit = int(config["validation_windows"])
    observed = 0
    sums: dict[str, float] = defaultdict(float)
    prototype_mass = np.zeros(model.config.prototype_count, dtype=np.float64)
    wrong_mixture_sums: dict[str, float] = defaultdict(float)
    deleted_prototype_sums: dict[str, float] = defaultdict(float)
    zero_parameter_sums: dict[str, float] = defaultdict(float)
    deterministic_action_max_abs = 0.0
    deterministic_weight_max_abs = 0.0
    first_batch = True
    started = time.perf_counter()
    prior_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prior_float32_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    with torch.no_grad():
        for shard_index, shard in enumerate(
            sorted(shards, key=lambda item: str(item.path))
        ):
            arrays = load_training_shard(shard, normalization)
            for numpy_batch in iter_shard_batches(
                arrays,
                batch_size=min(int(config["per_gpu_batch_size"]), limit - observed),
                length=int(config["window_length"]),
                stride=int(config["window_stride"]),
                seed=int(config["seed"]) + shard_index,
            ):
                if len(numpy_batch["state"]) > limit - observed:
                    numpy_batch = {
                        name: value[: limit - observed]
                        for name, value in numpy_batch.items()
                    }
                batch = torch_batch(numpy_batch, device)
                prediction = posterior_forward(model, batch)
                terms = unit_learner_loss(
                    prediction,
                    batch,
                    weights,
                )
                if first_batch:
                    repeated = posterior_forward(model, batch)
                wrong_mixture = torch.roll(
                    prediction["unit_weights"], shifts=1, dims=0
                )
                wrong_prediction = model.decode(
                    batch["state"], wrong_mixture, prediction["parameters"]
                )
                deleted_mixture = prediction["unit_weights"].clone()
                dominant = deleted_mixture.argmax(dim=-1, keepdim=True)
                deleted_mixture.scatter_(-1, dominant, 0.0)
                deleted_mixture = deleted_mixture / deleted_mixture.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                deleted_prediction = model.decode(
                    batch["state"], deleted_mixture, prediction["parameters"]
                )
                zero_parameter_prediction = model.decode(
                    batch["state"],
                    prediction["unit_weights"],
                    torch.zeros_like(prediction["parameters"]),
                )
                wrong_terms = reconstruction_distortion(
                    wrong_prediction, batch, weights
                )
                deleted_terms = reconstruction_distortion(
                    deleted_prediction, batch, weights
                )
                zero_terms = reconstruction_distortion(
                    zero_parameter_prediction, batch, weights
                )
                if first_batch:
                    deterministic_action_max_abs = float(
                        (prediction["action"] - repeated["action"])
                        .abs()
                        .max()
                        .float()
                        .cpu()
                    )
                    deterministic_weight_max_abs = float(
                        (
                            prediction["unit_weights"]
                            - repeated["unit_weights"]
                        )
                        .abs()
                        .max()
                        .float()
                        .cpu()
                    )
                    first_batch = False
                count = len(numpy_batch["state"])
                observed += count
                for name, value in terms.items():
                    sums[name] += float(value.float().cpu()) * count
                for name, value in wrong_terms.items():
                    wrong_mixture_sums[name] += float(value.float().cpu()) * count
                for name, value in deleted_terms.items():
                    deleted_prototype_sums[name] += float(value.float().cpu()) * count
                for name, value in zero_terms.items():
                    zero_parameter_sums[name] += float(value.float().cpu()) * count
                valid_weights = prediction["unit_weights"][batch["validity"].bool()]
                prototype_mass += valid_weights.double().sum(dim=0).cpu().numpy()
                if observed >= limit:
                    break
            if observed >= limit:
                break
    if observed != limit:
        raise RuntimeError(f"validation supplied {observed} windows, expected {limit}")
    torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
    torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32
    torch.set_float32_matmul_precision(prior_float32_precision)
    probabilities = prototype_mass / prototype_mass.sum()
    positive = probabilities > 0
    entropy = float(-(probabilities[positive] * np.log(probabilities[positive])).sum())
    effective_complexity = float(np.exp(entropy))
    dominant_prototype = int(np.argmax(prototype_mass))
    material_threshold = float(config["material_prototype_minimum_mass_fraction"])
    material_prototypes = np.flatnonzero(probabilities >= material_threshold).astype(int)
    losses = {name: value / observed for name, value in sorted(sums.items())}
    wrong_losses = {
        name: value / observed for name, value in sorted(wrong_mixture_sums.items())
    }
    deleted_losses = {
        name: value / observed
        for name, value in sorted(deleted_prototype_sums.items())
    }
    zero_losses = {
        name: value / observed for name, value in sorted(zero_parameter_sums.items())
    }
    baseline_distortion = losses["action"] + weights.effect * losses["effect"] + weights.transition * losses["transition"]
    return {
        "window_count": observed,
        "losses": losses,
        "prototype_count": model.config.prototype_count,
        "prototype_activation_mass": prototype_mass.tolist(),
        "material_prototype_count": int(len(material_prototypes)),
        "material_prototype_ids": material_prototypes.tolist(),
        "prototype_entropy_nats": entropy,
        "effective_prototype_complexity": effective_complexity,
        "dominant_prototype": dominant_prototype,
        "dominant_prototype_mass_fraction": float(probabilities[dominant_prototype]),
        "decoder_interventions": {
            "baseline_distortion": baseline_distortion,
            "wrong_mixture_distortion": wrong_losses["total"],
            "wrong_mixture_delta_fraction": (wrong_losses["total"] - baseline_distortion) / max(baseline_distortion, 1e-12),
            "dominant_prototype_deletion_distortion": deleted_losses["total"],
            "dominant_prototype_deletion_delta_fraction": (deleted_losses["total"] - baseline_distortion) / max(baseline_distortion, 1e-12),
            "zero_parameter_distortion": zero_losses["total"],
            "zero_parameter_delta_fraction": (zero_losses["total"] - baseline_distortion) / max(baseline_distortion, 1e-12),
        },
        "deterministic_action_max_abs": deterministic_action_max_abs,
        "deterministic_unit_weight_max_abs": deterministic_weight_max_abs,
        "unit_inference_precision": "fp32_tf32_disabled",
        "elapsed_seconds": time.perf_counter() - started,
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    started = time.perf_counter()
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return time.perf_counter() - started


def verified_checkpoint_read(path: Path, scale: str, parameter_count: int) -> float:
    started = time.perf_counter()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only.
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["model_scale"] != scale:
        raise RuntimeError("checkpoint model scale failed round trip")
    if int(checkpoint["exact_trainable_parameters"]) != parameter_count:
        raise RuntimeError("checkpoint parameter count failed round trip")
    if not checkpoint["model_state"]:
        raise RuntimeError("checkpoint model state is empty")
    return time.perf_counter() - started


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("backbone training requires a CUDA device")
    config_path = args.config.resolve()
    config = load_json(config_path)
    architecture_path = config_path.parent / config.get("architecture", "architecture.json")
    architecture = load_json(architecture_path)
    validate_training_config(config, architecture)
    output_dir = (
        root_path(args.output_dir)
        if args.output_dir is not None
        else resolve("checkpoints/backbone", str(config["model_scale"]).lower())
    )
    split_path = root_path(config["split_manifest"])
    split = load_json(split_path)
    validate_split_manifest(split)
    rank, local_rank, world_size = distributed_context(
        int(config["resource_envelope"]["gpus"])
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed = int(config["seed"])
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)

    fresh = load_json(root_path(config["fresh_release_index"]))
    historical = load_json(root_path(config["historical_release_index"]))
    branched = load_json(root_path(config["branched_release_index"]))
    normalization = normalization_from_manifest(root_path(config["normalization"]))
    shards = training_shards(fresh, historical, branched)
    task_partitions = split["task_partitions"]
    final_tasks = set(task_partitions["composition_final_tasks"])
    if final_tasks & {shard.environment_id for shard in shards}:
        raise RuntimeError("held-out final task leaked into the training shard registry")
    source_shards = [shard for shard in shards if shard.partition == config["partition"]]
    development_shards = [
        shard for shard in shards if shard.partition == "composition_development"
    ]
    if not development_shards:
        raise RuntimeError("composition-development validation shards are absent")
    expected_source_tasks = set(task_partitions["source_training_tasks"])
    if {shard.environment_id for shard in source_shards} != expected_source_tasks:
        raise RuntimeError("source-training task registry is incomplete")
    expected_development_tasks = set(
        task_partitions["composition_development_tasks"]
    )
    if {
        shard.environment_id for shard in development_shards
    } != expected_development_tasks:
        raise RuntimeError("composition-development task registry is incomplete")
    root_selection_by_path: dict[str, dict[str, Any]] | None = None
    if config.get("root_subset_manifest") is not None:
        subset = load_json(root_path(config["root_subset_manifest"]))
        condition_id = str(config["root_subset_condition_id"])
        condition = next(
            (
                value
                for value in subset["conditions"]
                if value["condition_id"] == condition_id
            ),
            None,
        )
        if condition is None:
            raise RuntimeError("root subset condition is absent")
        root_selection_by_path = {
            str(root_path(value["path"]).resolve()): value for value in condition["entries"]
        }
        source_shards = [
            shard
            for shard in source_shards
            if str(shard.path.resolve()) in root_selection_by_path
        ]
        if not source_shards:
            raise RuntimeError("root subset selected no source shards")
    max_shards = config.get("max_shards_per_task")
    source_shards = limit_shards_per_task(
        source_shards, None if max_shards is None else int(max_shards)
    )

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        distributed.barrier()
    model_config = MODEL_SCALES[str(config["model_scale"])]
    model = EmergentActionUnitLearner(model_config).to(device)
    parameter_count = trainable_parameter_count(model)
    expected_parameters = int(
        architecture["scales"][config["model_scale"]]["exact_trainable_parameters"]
    )
    if parameter_count != expected_parameters:
        raise RuntimeError("instantiated parameter count differs from architecture audit")
    train_model: Any = model
    if world_size > 1:
        train_model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    weights = LossWeights(**config["loss_weights"])
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    micro_steps = 0
    training_windows_local = 0
    latest_metrics: dict[str, float] = {}
    started = time.perf_counter()
    stop = False
    for epoch in range(int(config["epochs"])):
        local_shards = partition_shards(
            source_shards,
            partition=str(config["partition"]),
            rank=rank,
            world_size=world_size,
            seed=seed,
            epoch=epoch,
        )
        for shard_index, shard in enumerate(local_shards):
            arrays = load_training_shard(shard, normalization)
            if root_selection_by_path is not None:
                selection = root_selection_by_path[str(shard.path.resolve())]
                if selection["all_roots"]:
                    indices = np.arange(arrays.state.shape[1], dtype=np.int64)
                else:
                    indices = np.asarray(selection["lane_indices"], dtype=np.int64)
                arrays = select_unit_array_roots(arrays, indices)
            shard_seed = seed + epoch * 10_000_019 + shard_index * 1009 + rank
            batch_iterator = (
                iter_root_window_batches
                if config["window_sampling"]
                == "one_random_window_per_root_per_epoch"
                else iter_shard_batches
            )
            for shard_batch_index, numpy_batch in enumerate(batch_iterator(
                arrays,
                batch_size=int(config["per_gpu_batch_size"]),
                length=int(config["window_length"]),
                stride=int(config["window_stride"]),
                seed=shard_seed,
            )):
                train_model.train()
                batch = torch_batch(numpy_batch, device)
                packing_target = None
                precision_target = None
                if should_run_consistency_probe(
                    optimizer_steps=optimizer_steps,
                    micro_steps=micro_steps,
                    gradient_accumulation_steps=int(
                        config["gradient_accumulation_steps"]
                    ),
                    consistency_every_steps=int(config["consistency_every_steps"]),
                ):
                    packing_target, precision_target = stability_targets(
                        train_model, batch, int(config["consistency_windows"])
                    )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = posterior_forward(train_model, batch)
                    terms = unit_learner_loss(
                        prediction,
                        batch,
                        weights,
                        packing_target=packing_target,
                        precision_target=precision_target,
                    )
                    loss = terms["total"] / int(config["gradient_accumulation_steps"])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                micro_steps += 1
                training_windows_local += len(numpy_batch["state"])
                if micro_steps % int(config["gradient_accumulation_steps"]):
                    continue
                learning_rate = optimizer_learning_rate(config, optimizer_steps)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    train_model.parameters(),
                    float(optimizer_config["gradient_clip_norm"]),
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("nonfinite training gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                latest_metrics = reduced_scalars(terms, world_size)
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "optimizer_step": optimizer_steps,
                                "learning_rate": learning_rate,
                                "gradient_norm": float(gradient_norm),
                                "losses": latest_metrics,
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
                if optimizer_steps >= int(config["max_optimizer_steps"]):
                    stop = True
                    break
                max_batches = config.get("max_batches_per_shard")
                if max_batches is not None and shard_batch_index + 1 >= int(max_batches):
                    break
            del arrays
            if stop:
                break
        if stop:
            break
    if optimizer_steps != int(config["max_optimizer_steps"]):
        raise RuntimeError(
            f"run ended at {optimizer_steps} optimizer steps before max_optimizer_steps"
        )
    if world_size > 1:
        distributed.barrier()

    report: dict[str, Any] | None = None
    if rank == 0:
        validation_result = validation(
            model, development_shards, normalization, config, weights, device
        )
        checkpoint_path = output_dir / "checkpoint_final.pt"
        checkpoint_payload = {
            "model_scale": config["model_scale"],
            "model_config": model_config.__dict__,
            "exact_trainable_parameters": parameter_count,
            "optimizer_steps": optimizer_steps,
            "identity_inference_precision": config["identity_inference_precision"],
            "consistency_every_steps": config["consistency_every_steps"],
            "consistency_windows": config["consistency_windows"],
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        checkpoint_write_seconds = atomic_torch_save(
            checkpoint_payload, checkpoint_path
        )
        checkpoint_read_seconds = verified_checkpoint_read(
            checkpoint_path, str(config["model_scale"]), parameter_count
        )
        wall_seconds = time.perf_counter() - started
        report = {
            "status": "COMPLETE",
            "config": str(config_path),
            "architecture": str(architecture_path),
            "split_manifest": str(split_path),
            "root_subset_condition_id": config.get("root_subset_condition_id"),
            "model_scale": config["model_scale"],
            "exact_trainable_parameters": parameter_count,
            "non_trainable_parameters": sum(
                value.numel() for value in model.parameters() if not value.requires_grad
            ),
            "world_size": world_size,
            "per_gpu_batch_size": config["per_gpu_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "identity_inference_precision": config["identity_inference_precision"],
            "consistency_every_steps": config["consistency_every_steps"],
            "consistency_windows": config["consistency_windows"],
            "effective_batch_size": int(config["per_gpu_batch_size"])
            * world_size
            * int(config["gradient_accumulation_steps"]),
            "optimizer_steps": optimizer_steps,
            "window_sampling": config["window_sampling"],
            "training_windows_local_rank0": training_windows_local,
            "training_independent_root_presentations_local_rank0": (
                training_windows_local
                if config["window_sampling"]
                == "one_random_window_per_root_per_epoch"
                else None
            ),
            "last_reduced_training_losses": latest_metrics,
            "validation": validation_result,
            "checkpoint": {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "write_seconds": checkpoint_write_seconds,
                "verified_read_seconds": checkpoint_read_seconds,
            },
            "wall_seconds": wall_seconds,
            "gpu_hours": world_size * wall_seconds / 3600.0,
            "peak_gpu_memory_bytes_rank0": torch.cuda.max_memory_allocated(device),
            "peak_host_rss_kib_rank0": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        atomic_write_json(output_dir / "run_manifest.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if world_size > 1:
        distributed.barrier()
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
