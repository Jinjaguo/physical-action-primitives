"""Contracts for the fresh-corpus scale extension (additional unique roots per task)."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from physical_action_primitives.backbone.split_manifest import validate_split_manifest
from physical_action_primitives.corpus.fresh_core import (
    fresh_seed_namespaces,
    validate_behavior_config,
)

TASK_ORDER = (
    "PokeCube-v1",
    "PullCube-v1",
    "PushT-v1",
    "RollBall-v1",
    "StackCube-v1",
)
SHARDS_PER_BLOCK = {
    "PokeCube-v1": 2,
    "PullCube-v1": 2,
    "PushT-v1": 2,
    "RollBall-v1": 2,
    "StackCube-v1": 1,
}
ROOT_SEED_START = {
    "PokeCube-v1": 600_000_000,
    "PullCube-v1": 700_000_000,
    "PushT-v1": 800_000_000,
    "RollBall-v1": 900_000_000,
    "StackCube-v1": 1_500_000_000,
}
NOISE_SEED_START = {
    "PokeCube-v1": 1_600_000_000,
    "PullCube-v1": 1_700_000_000,
    "PushT-v1": 1_800_000_000,
    "RollBall-v1": 1_900_000_000,
    "StackCube-v1": 2_000_000_000,
}
FRESH_EXTENSION_CORPUS_DIRECTORY = "fresh_extension"
FRESH_CORPUS_DIRECTORY = "fresh"


def is_extension_config(config: dict[str, Any]) -> bool:
    return "base_fresh_release_index" in config


def fresh_corpus_directory_name(config: dict[str, Any]) -> str:
    """Directory under ``data/corpus`` that holds the shards of ``config``."""
    if is_extension_config(config):
        return FRESH_EXTENSION_CORPUS_DIRECTORY
    return FRESH_CORPUS_DIRECTORY


def derive_extension_blocks(
    base_release: dict[str, Any], target_combined_bytes: int = 260_000_000_000
) -> tuple[int, dict[str, float]]:
    base_bytes = int(base_release["aggregate"]["artifact_bytes"])
    if base_bytes >= target_combined_bytes:
        raise ValueError("base corpus already meets the combined size target")
    means: dict[str, float] = {}
    for environment_id in TASK_ORDER:
        task = base_release["tasks"][environment_id]
        means[environment_id] = int(task["artifact_bytes"]) / int(task["shards"])
    block_bytes = sum(
        means[environment_id] * SHARDS_PER_BLOCK[environment_id]
        for environment_id in TASK_ORDER
    )
    if block_bytes <= 0:
        raise ValueError("base corpus has no measured extension byte rate")
    minimum_blocks = math.ceil((target_combined_bytes - base_bytes) / block_bytes)
    # Four allocation blocks make each source-task extension a multiple of eight
    # shards, so all 8 DDP ranks receive equal task-specific work.
    blocks = 4 * math.ceil(minimum_blocks / 4)
    if blocks < 1:
        raise AssertionError("extension block calculation failed")
    return blocks, means


def build_extension_config(
    base_config: dict[str, Any],
    base_release: dict[str, Any],
    *,
    base_config_path: str,
    base_release_path: str,
    target_combined_bytes: int = 260_000_000_000,
) -> dict[str, Any]:
    blocks, means = derive_extension_blocks(base_release, target_combined_bytes)
    roots_per_shard = int(base_config["roots_per_shard"])
    tasks = []
    projected_extension_bytes = 0.0
    for source_task in base_config["tasks"]:
        task = deepcopy(source_task)
        environment_id = task["environment_id"]
        shards = blocks * SHARDS_PER_BLOCK[environment_id]
        task["root_seed_first"] = ROOT_SEED_START[environment_id]
        task["shard_count"] = shards
        task["root_count"] = shards * roots_per_shard
        tasks.append(task)
        projected_extension_bytes += means[environment_id] * shards
    base_bytes = int(base_release["aggregate"]["artifact_bytes"])
    return {
        **{
            key: deepcopy(base_config[key])
            for key in (
                "schema_version",
                "shard_hard_limit_bytes",
                "split_manifest",
                "num_envs",
                "roots_per_shard",
                "batches_per_shard",
                "observation_mode",
                "control_mode",
                "reward_mode",
                "simulation_backend",
                "render_backend",
                "official_ppo_source",
                "policy_action_rule",
                "window",
                "tier2_selected_lanes_per_batch",
                "tier2_selection_rule",
                "behavior_strata",
                "model_feature_policy",
            )
        },
        "corpus_id": "fresh_extension",
        "base_fresh_config": base_config_path,
        "base_fresh_release_index": base_release_path,
        "target_combined_bytes": target_combined_bytes,
        "action_noise_seed_base_by_task": NOISE_SEED_START,
        "tasks": tasks,
        "planning_measurements": {
            "base_artifact_bytes": base_bytes,
            "base_task_mean_artifact_bytes_per_shard": means,
            "allocation_block_task_shards": SHARDS_PER_BLOCK,
            "allocation_block_count": blocks,
            "allocation_block_count_multiple_for_eight_rank_balance": 4,
            "projected_extension_bytes": int(math.ceil(projected_extension_bytes)),
            "projected_combined_bytes": int(
                math.ceil(base_bytes + projected_extension_bytes)
            ),
        },
    }


def validate_extension_config(
    config: dict[str, Any],
    split_manifest: dict[str, Any],
    base_release: dict[str, Any],
) -> None:
    validate_split_manifest(split_manifest)
    if not is_extension_config(config):
        raise ValueError("config is not a fresh-extension config")
    if base_release.get("audit_errors"):
        raise ValueError("fresh-extension base corpus has unresolved audit errors")
    validate_behavior_config(config)
    tasks = config["tasks"]
    if tuple(task["environment_id"] for task in tasks) != TASK_ORDER:
        raise ValueError("fresh-extension task order changed")
    roots_per_shard = int(config["roots_per_shard"])
    blocks = int(config["planning_measurements"]["allocation_block_count"])
    expected_shards = {
        task: blocks * SHARDS_PER_BLOCK[task] for task in TASK_ORDER
    }
    seed_intervals = []
    for task in tasks:
        environment_id = task["environment_id"]
        shards = int(task["shard_count"])
        roots = int(task["root_count"])
        if shards != expected_shards[environment_id] or roots != shards * roots_per_shard:
            raise ValueError(f"fresh-extension allocation mismatch: {environment_id}")
        if int(task["root_seed_first"]) != ROOT_SEED_START[environment_id]:
            raise ValueError(f"fresh-extension seed start mismatch: {environment_id}")
        start = int(task["root_seed_first"])
        seed_intervals.append((start, start + roots, environment_id))
    existing_intervals = []
    for environment_id, specification in fresh_seed_namespaces(split_manifest).items():
        start = int(specification["first"])
        existing_intervals.append(
            (start, start + int(specification["count"]), f"base:{environment_id}")
        )
    for environment_id, specification in split_manifest["final_evaluation"]["tasks"].items():
        start = int(specification["first_seed"])
        existing_intervals.append(
            (start, start + int(specification["episode_count"]), f"final:{environment_id}")
        )
    for environment_id, specification in split_manifest["conditions"][
        "interaction_only_exposure"
    ]["root_seed_ranges"].items():
        start = int(specification["first"])
        existing_intervals.append(
            (
                start,
                start + int(specification["count"]),
                f"interaction:{environment_id}",
            )
        )
    all_intervals = sorted(seed_intervals + existing_intervals)
    for left, right in zip(all_intervals, all_intervals[1:]):
        if left[1] > right[0]:
            raise ValueError(f"fresh-extension seed overlap: {left[2]} and {right[2]}")
    if config["action_noise_seed_base_by_task"] != NOISE_SEED_START:
        raise ValueError("fresh-extension noise seed registry mismatch")
