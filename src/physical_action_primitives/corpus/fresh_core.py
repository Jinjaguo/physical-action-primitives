"""Pure contracts for fresh-episode corpus collection."""

from __future__ import annotations

from typing import Any

from physical_action_primitives.backbone.split_manifest import validate_split_manifest
from physical_action_primitives.corpus.core import window_starts


def map_array_index(
    tasks: list[dict[str, Any]], array_index: int
) -> tuple[int, dict[str, Any], int]:
    """Map a flat shard index to (task index, task, shard index within the task)."""
    if array_index < 0:
        raise IndexError("negative fresh-corpus array index")
    offset = 0
    for task_index, task in enumerate(tasks):
        next_offset = offset + int(task["shard_count"])
        if array_index < next_offset:
            return task_index, task, array_index - offset
        offset = next_offset
    raise IndexError("fresh-corpus array index exceeds the shard registry")


def total_shard_count(config: dict[str, Any]) -> int:
    return sum(int(task["shard_count"]) for task in config["tasks"])


def fresh_seed_namespaces(split_manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the per-task root-seed ranges reserved for the fresh corpus."""
    return split_manifest["fresh_corpus_seed_namespaces"]


def validate_behavior_config(config: dict[str, Any]) -> None:
    """Check the vector packing, behavior strata and window settings of a config."""
    num_envs = int(config["num_envs"])
    roots_per_shard = int(config["roots_per_shard"])
    batches_per_shard = int(config["batches_per_shard"])
    if num_envs < 1 or roots_per_shard != num_envs * batches_per_shard:
        raise ValueError("roots_per_shard must equal num_envs * batches_per_shard")
    strata = config["behavior_strata"]
    if sum(int(item["lane_count"]) for item in strata) != num_envs:
        raise ValueError("behavior strata must assign every lane exactly once")
    names = [str(item["name"]) for item in strata]
    if len(set(names)) != len(names):
        raise ValueError("behavior stratum names must be unique")
    if any(float(item["gaussian_sigma_controller_space"]) < 0 for item in strata):
        raise ValueError("behavior noise sigma must be non-negative")
    lanes = config["tier2_selected_lanes_per_batch"]
    if len(set(lanes)) != len(lanes) or any(
        int(lane) < 0 or int(lane) >= num_envs for lane in lanes
    ):
        raise ValueError("tier2 lane selection is outside the vector lanes")
    window = config["window"]
    if window.get("storage") != "lazy_index_without_tensor_duplication":
        raise ValueError("windows must be stored as lazy indices")
    for task in config["tasks"]:
        if not window_starts(
            int(task["max_episode_steps"]),
            int(window["length_transitions"]),
            int(window["stride_transitions"]),
        ).size:
            raise ValueError(f"task has no eligible windows: {task['environment_id']}")


def validate_fresh_config(
    config: dict[str, Any], split_manifest: dict[str, Any]
) -> None:
    """Check a fresh-corpus config against the task/seed split."""
    validate_split_manifest(split_manifest)
    validate_behavior_config(config)
    roots_per_shard = int(config["roots_per_shard"])
    tasks = config["tasks"]
    task_ids = [task["environment_id"] for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("fresh task registry has duplicates")
    partitions = split_manifest["task_partitions"]
    if set(partitions["composition_final_tasks"]) & set(task_ids):
        raise ValueError("fresh corpus includes a final-evaluation task")
    source_tasks = set(partitions["source_training_tasks"])
    development_tasks = set(partitions["composition_development_tasks"])
    seed_registry = fresh_seed_namespaces(split_manifest)
    if set(config.get("action_noise_seed_base_by_task", {})) != set(task_ids):
        raise ValueError("action-noise seed registry does not match the task registry")
    for task in tasks:
        env_id = task["environment_id"]
        partition = task["partition"]
        if partition == "source_training" and env_id not in source_tasks:
            raise ValueError(f"source task differs from split: {env_id}")
        if partition == "composition_development" and env_id not in development_tasks:
            raise ValueError(f"development task differs from split: {env_id}")
        root_count = int(task["root_count"])
        if root_count != int(task["shard_count"]) * roots_per_shard:
            raise ValueError(f"task root count is not shard-exact: {env_id}")
        if int(task["root_seed_first"]) != int(seed_registry[env_id]["first"]):
            raise ValueError(f"task seed start differs from split: {env_id}")
        if root_count != int(seed_registry[env_id]["count"]):
            raise ValueError(f"task seed count differs from split: {env_id}")
