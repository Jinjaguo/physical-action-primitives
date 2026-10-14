"""Training-shard registry and deterministic lazy-window iteration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from physical_action_primitives.corpus.core import window_starts
from physical_action_primitives.paths import resolve
from physical_action_primitives.corpus.historical_core import required_training_datasets
from physical_action_primitives.backbone.data import (
    FRESH_TRAINING_DATASETS,
    Normalization,
    UnitArrays,
    extract_lazy_window_batch,
    fresh_unit_arrays,
    historical_unit_arrays,
    normalize_unit_arrays,
)


@dataclass(frozen=True)
class TrainingShard:
    source: str
    environment_id: str
    partition: str
    path: Path
    manifest_path: Path | None
    task_specification: dict[str, Any] | None


def validate_training_shard_controller(
    shard: TrainingShard, attributes: Any
) -> None:
    """Check that a shard was collected with the joint-delta-position controller."""

    key = "control_mode" if shard.source == "historical" else "controller_mode"
    value = attributes.get(key)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value != "pd_joint_delta_pos":
        raise RuntimeError(
            f"training shard controller is not pd_joint_delta_pos: {shard.path}"
        )


def root_path(value: str | Path) -> Path:
    """Return an absolute path; relative paths are taken relative to PAP_ROOT."""
    path = Path(value)
    return path if path.is_absolute() else resolve(path)


def normalization_from_manifest(path: Path) -> Normalization:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = Normalization(
        state_mean=np.asarray(manifest["state_mean"], dtype=np.float32),
        state_std=np.asarray(manifest["state_std"], dtype=np.float32),
        action_mean=np.asarray(manifest["action_mean"], dtype=np.float32),
        action_std=np.asarray(manifest["action_std"], dtype=np.float32),
        effect_mean=np.asarray(manifest["effect_mean"], dtype=np.float32),
        effect_std=np.asarray(manifest["effect_std"], dtype=np.float32),
    )
    result.validate()
    return result


def training_shards(
    fresh_release: dict[str, Any],
    historical_release: dict[str, Any],
    branched_release: dict[str, Any],
) -> list[TrainingShard]:
    """Build the shard registry from the fresh, historical, and branched release indices."""
    shards: list[TrainingShard] = []
    for entry in fresh_release["entries"]:
        manifest_path = root_path(entry["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tier1 = manifest["artifacts"]["tier1"]
        shards.append(
            TrainingShard(
                source="fresh",
                environment_id=str(entry["task"]),
                partition=str(manifest["split_assignment"]),
                path=root_path(tier1["path"]),
                manifest_path=manifest_path,
                task_specification=None,
            )
        )
    historical_config = json.loads(
        root_path(historical_release["config"]).read_text(encoding="utf-8")
    )
    task_specs = {
        task["environment_id"]: task for task in historical_config["tasks"]
    }
    for entry in historical_release["entries"]:
        environment_id = str(entry["environment_id"])
        shards.append(
            TrainingShard(
                source="historical",
                environment_id=environment_id,
                partition="source_training",
                path=root_path(entry["path"]),
                manifest_path=root_path(entry["sidecar_path"]),
                task_specification=task_specs[environment_id],
            )
        )
    for entry in branched_release["entries"]:
        manifest_path = root_path(entry["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = manifest["artifact"]
        shards.append(
            TrainingShard(
                source="branched",
                environment_id=str(entry["task"]),
                partition=str(manifest["split_assignment"]),
                path=root_path(artifact["path"]),
                manifest_path=manifest_path,
                task_specification=None,
            )
        )
    return shards


def partition_shards(
    shards: list[TrainingShard],
    *,
    partition: str,
    rank: int,
    world_size: int,
    seed: int,
    epoch: int,
) -> list[TrainingShard]:
    """Balance total work exactly; per-source-task counts differ by at most one."""
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise ValueError("invalid distributed rank")
    by_task: dict[tuple[str, str], list[TrainingShard]] = {}
    for shard in shards:
        if shard.partition == partition:
            by_task.setdefault((shard.source, shard.environment_id), []).append(shard)
    total_shards = sum(len(entries) for entries in by_task.values())
    if total_shards % world_size:
        raise RuntimeError(
            "total training-shard count is not divisible by world size: "
            f"{total_shards} vs {world_size}"
        )
    selected: list[TrainingShard] = []
    next_rank = 0
    for task_index, source_task in enumerate(sorted(by_task)):
        entries = sorted(by_task[source_task], key=lambda item: str(item.path))
        generator = np.random.Generator(
            np.random.PCG64(seed + epoch * 1_000_003 + task_index * 10_007)
        )
        order = generator.permutation(len(entries))
        entries = [entries[int(index)] for index in order]
        selected.extend(
            entry
            for position, entry in enumerate(entries)
            if (next_rank + position) % world_size == rank
        )
        next_rank = (next_rank + len(entries)) % world_size
    if len(selected) != total_shards // world_size:
        raise AssertionError("distributed shard balancing failed")
    generator = np.random.Generator(np.random.PCG64(seed + epoch * 2_000_003))
    order = generator.permutation(len(selected))
    return [selected[int(index)] for index in order]


def load_training_shard(
    shard: TrainingShard,
    normalization: Normalization,
) -> UnitArrays:
    if shard.source in {"fresh", "branched"}:
        names = FRESH_TRAINING_DATASETS
    elif shard.source == "historical" and shard.task_specification is not None:
        names = required_training_datasets(shard.task_specification)
    else:
        raise RuntimeError("unknown training-shard source")
    with h5py.File(shard.path, "r") as handle:
        validate_training_shard_controller(shard, handle.attrs)
        payload = {name: np.asarray(handle[name]) for name in names}
    if shard.source in {"fresh", "branched"}:
        arrays = fresh_unit_arrays(payload)
    else:
        arrays = historical_unit_arrays(payload, shard.task_specification)
    return normalize_unit_arrays(arrays, normalization)


def window_pairs(
    arrays: UnitArrays, *, length: int, stride: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    starts = window_starts(arrays.action.shape[0], length, stride)
    lanes = np.repeat(np.arange(arrays.action.shape[1], dtype=np.int64), len(starts))
    offsets = np.tile(starts.astype(np.int64), arrays.action.shape[1])
    generator = np.random.Generator(np.random.PCG64(seed))
    order = generator.permutation(len(lanes))
    return lanes[order], offsets[order]


def iter_shard_batches(
    arrays: UnitArrays,
    *,
    batch_size: int,
    length: int,
    stride: int,
    seed: int,
) -> Iterator[dict[str, np.ndarray]]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    effective_length = min(length, int(arrays.action.shape[0]))
    lanes, starts = window_pairs(
        arrays, length=effective_length, stride=stride, seed=seed
    )
    for offset in range(0, len(lanes), batch_size):
        yield extract_lazy_window_batch(
            arrays,
            lanes[offset : offset + batch_size],
            starts[offset : offset + batch_size],
            effective_length,
        )


def root_window_pairs(
    arrays: UnitArrays, *, length: int, stride: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select one random lazy window for every root, exactly once per epoch."""

    effective_length = min(length, int(arrays.action.shape[0]))
    starts = window_starts(arrays.action.shape[0], effective_length, stride)
    roots = int(arrays.action.shape[1])
    generator = np.random.Generator(np.random.PCG64(seed))
    lanes = generator.permutation(roots).astype(np.int64, copy=False)
    offsets = generator.choice(starts, size=roots, replace=True).astype(
        np.int64, copy=False
    )
    return lanes, offsets


def iter_root_window_batches(
    arrays: UnitArrays,
    *,
    batch_size: int,
    length: int,
    stride: int,
    seed: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield one independently selected context per trajectory root."""

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    effective_length = min(length, int(arrays.action.shape[0]))
    lanes, starts = root_window_pairs(
        arrays, length=effective_length, stride=stride, seed=seed
    )
    for offset in range(0, len(lanes), batch_size):
        yield extract_lazy_window_batch(
            arrays,
            lanes[offset : offset + batch_size],
            starts[offset : offset + batch_size],
            effective_length,
        )
