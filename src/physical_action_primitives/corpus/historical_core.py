"""Pure contracts for the historical-source supplement (pre-existing trajectories)."""

from __future__ import annotations

from typing import Any

from physical_action_primitives.corpus.core import window_starts
from physical_action_primitives.backbone.split_manifest import validate_split_manifest


HISTORICAL_TASKS = (
    "PegInsertionSide-v1",
    "PlugCharger-v1",
    "PullCubeTool-v1",
    "StackPyramid-v1",
    "DrawTriangle-v1",
)


def validate_historical_source_config(
    config: dict[str, Any], split_manifest: dict[str, Any]
) -> None:
    """Check a historical-supplement config against the task split."""
    validate_split_manifest(split_manifest)
    tasks = config.get("tasks", [])
    task_ids = tuple(task.get("environment_id") for task in tasks)
    if task_ids != HISTORICAL_TASKS:
        raise ValueError("historical task registry mismatch")
    source_tasks = set(
        split_manifest["task_partitions"]["source_training_tasks"]
    )
    final_tasks = set(
        split_manifest["task_partitions"]["composition_final_tasks"]
    )
    if not set(task_ids) <= source_tasks or set(task_ids) & final_tasks:
        raise ValueError("historical tasks violate the task partition")

    window = config["window"]
    if window.get("storage") != "derived_lazy_index_only":
        raise ValueError("historical windows must be derived lazily")
    for task in tasks:
        if int(task["horizon"]) <= 0 or int(task["action_dimension"]) <= 0:
            raise ValueError("historical tensor dimensions are invalid")
        if not task.get("primary_state") or not task.get("tcp_state"):
            raise ValueError("historical physical-state mapping is incomplete")
        if not task.get("replay_states"):
            raise ValueError("historical replay mapping is incomplete")
        if not window_starts(
            int(task["horizon"]),
            int(window["length_transitions"]),
            int(window["stride_transitions"]),
        ).size:
            raise ValueError("historical task has no eligible windows")

    policy = config["feature_policy"]
    if not all(
        policy[key]
        for key in (
            "task_identity_metadata_only",
            "reward_success_reporting_only",
            "human_phase_or_skill_labels_absent",
            "behavior_family_metadata_only",
            "full_trajectory_available",
        )
    ):
        raise ValueError("historical feature policy must keep task identity and outcomes out of training")


def required_training_datasets(task: dict[str, Any]) -> tuple[str, ...]:
    """Return exact numeric datasets used by the historical training adapter."""
    return tuple(
        dict.fromkeys(
            [
                "actions",
                str(task["primary_state"]),
                *map(str, task["context_states"]),
                str(task["tcp_state"]),
                str(task["robot_qpos"]),
                str(task["robot_qvel"]),
                *map(str, task["replay_states"]),
            ]
        )
    )
