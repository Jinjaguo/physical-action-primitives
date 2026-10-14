"""Validation and expansion helpers for the task-split manifest (source / development / held-out tasks)."""

from __future__ import annotations

from typing import Any


def expanded_final_episode_ids(manifest: dict[str, Any]) -> dict[str, list[str]]:
    final = manifest["final_evaluation"]
    prefix = str(final["episode_id_prefix"])
    result: dict[str, list[str]] = {}
    for environment_id in manifest["task_partitions"]["composition_final_tasks"]:
        specification = final["tasks"][environment_id]
        first = int(specification["first_seed"])
        count = int(specification["episode_count"])
        result[environment_id] = [
            f"{prefix}:{environment_id}:seed-{seed}"
            for seed in range(first, first + count)
        ]
    return result


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    partitions = manifest["task_partitions"]
    source = list(partitions["source_training_tasks"])
    development = list(partitions["composition_development_tasks"])
    final_tasks = list(partitions["composition_final_tasks"])
    if set(source) & set(development) or set(source) & set(final_tasks) or set(development) & set(final_tasks):
        raise ValueError("task partitions overlap")
    if final_tasks != ["PickCube-v1", "PushCube-v1", "LiftPegUpright-v1"]:
        raise ValueError("held-out task list must be PickCube, PushCube, LiftPegUpright")
    strict = manifest["conditions"]["strict_whole_task_holdout"]
    if any(
        strict[key]
        for key in (
            "unit_training_allows_final_task_data",
            "composition_training_allows_final_task_data",
            "checkpoint_selection_allows_final_task_data",
            "hyperparameter_selection_allows_final_task_data",
            "visualization_selection_allows_final_task_data",
        )
    ):
        raise ValueError("strict whole-task holdout must exclude held-out task data everywhere")
    if strict.get("descendants_inherit_parent_split") is not True:
        raise ValueError("descendants must inherit the parent split")
    interaction = manifest["conditions"]["interaction_only_exposure"]
    if interaction.get("complete_trajectory_available_to_model") is not False:
        raise ValueError("interaction-only condition exposes complete trajectories")
    if interaction.get("complete_successful_trajectory_available_to_model") is not False:
        raise ValueError("interaction-only condition exposes successful trajectories")

    expanded = expanded_final_episode_ids(manifest)
    all_final_ids = [episode_id for task in final_tasks for episode_id in expanded[task]]
    final = manifest["final_evaluation"]
    if len(all_final_ids) != int(final["all_episode_count"]) or len(set(all_final_ids)) != len(all_final_ids):
        raise ValueError("final episode IDs are not unique and complete")

    seed_intervals: list[tuple[int, int, str]] = []
    for environment_id, specification in manifest["fresh_corpus_seed_namespaces"].items():
        start = int(specification["first"])
        seed_intervals.append((start, start + int(specification["count"]), f"corpus:{environment_id}"))
    for environment_id, specification in interaction["root_seed_ranges"].items():
        start = int(specification["first"])
        seed_intervals.append((start, start + int(specification["count"]), f"interaction:{environment_id}"))
    for environment_id, specification in final["tasks"].items():
        start = int(specification["first_seed"])
        seed_intervals.append((start, start + int(specification["episode_count"]), f"final:{environment_id}"))
    seed_intervals.sort()
    for left, right in zip(seed_intervals, seed_intervals[1:]):
        if left[1] > right[0]:
            raise ValueError(f"seed namespaces overlap: {left[2]} and {right[2]}")
    if min(start for start, _end, name in seed_intervals if name.startswith("final:")) < 2_000_000_000:
        raise ValueError("held-out seed namespace must stay in the reserved high range")
