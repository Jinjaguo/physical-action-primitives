#!/usr/bin/env python3
"""Collect one branched shard: eight-step branches from shared parent states of a fresh shard.

Run as ``python -m physical_action_primitives.corpus.collect_branched``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

from physical_action_primitives.corpus.branched_core import (
    branch_action_schedule,
    repeat_parent_state,
    select_parent_frames,
    selected_parent_lanes,
    validate_branched_config,
)
from physical_action_primitives.corpus.collection_utils import (
    append_frame,
    common_attributes,
    controller_targets,
    physical_snapshot,
    reporting_outcome,
    stack_arrays,
    string_array,
    target_joint_limit_pass,
    write_h5,
)
from physical_action_primitives.corpus.core import (
    parent_state_max_errors,
    relative_to_root,
    resolve_below,
)
from physical_action_primitives.corpus.fresh_core import map_array_index, validate_fresh_config
from physical_action_primitives.corpus.vector_policy_utils import lane_action_bounds
from physical_action_primitives.manifest import atomic_write_json
from physical_action_primitives.paths import project_root, resolve
from physical_action_primitives.progress import ProgressReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="index of the source fresh shard (default: $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="default: data/corpus/branched",
    )
    parser.add_argument(
        "--heartbeat-root",
        type=Path,
        default=None,
        help="default: logs/corpus/branched",
    )
    args = parser.parse_args()
    if args.array_index is None:
        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_index is None:
            parser.error("--array-index is required (or set SLURM_ARRAY_TASK_ID)")
        args.array_index = int(env_index)
    return args


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def read_selected_source(
    tier1_path: Path, lanes: np.ndarray, task: dict[str, Any]
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    with h5py.File(tier1_path, "r") as handle:
        contact_force = np.asarray(handle["physical/typed_contact_force"][:, lanes])
        contact_mask = np.asarray(
            handle["physical/contact_channel_mask"][:, lanes], dtype=bool
        )
        available = (
            task["contact_observation_status"] == "exact_pairwise_query_available"
        )
        parent_frames, selected_interaction = select_parent_frames(
            contact_force, contact_mask, contact_available=available
        )
        horizon = int(handle["action/processed_controller_space"].shape[0])
        action_frames = np.minimum(parent_frames, horizon - 1)
        source_actions_all = np.asarray(
            handle["action/processed_controller_space"][:, lanes], dtype=np.float32
        )
        parent_actions = source_actions_all[action_frames, np.arange(len(lanes))]
        source_physical: dict[str, np.ndarray] = {}
        for field in (
            "primary",
            "context",
            "context_mask",
            "robot",
            "tcp_pose",
            "qpos",
            "qvel",
        ):
            values = np.asarray(handle[f"physical/{field}"][:, lanes])
            source_physical[field] = values[
                parent_frames, np.arange(len(parent_frames))
            ]
        replay: dict[str, np.ndarray] = {}
        group = handle["replay/sim_state"]
        leaves: list[str] = []
        group.visititems(
            lambda name, value: leaves.append(name)
            if isinstance(value, h5py.Dataset)
            else None
        )
        for leaf in leaves:
            replay[f"replay/sim_state/{leaf}"] = np.asarray(group[leaf][:, lanes])
        parent_ids = decode_strings(handle["metadata/root_id"][lanes])
        parent_seeds = np.asarray(
            handle["metadata/episode_seed"][lanes], dtype=np.int64
        )
    return (
        replay,
        source_physical,
        parent_frames,
        action_frames,
        parent_ids,
        parent_seeds,
        parent_actions,
        selected_interaction,
    )


def restore_errors(
    initial: dict[str, np.ndarray],
    source: dict[str, np.ndarray],
    roots: int,
    branches: int,
) -> tuple[dict[str, float], dict[str, float]]:
    gate = ("primary", "context", "context_mask", "robot", "qpos", "qvel")
    diagnostic = ("tcp_pose",)
    branch_spread = parent_state_max_errors(initial, gate + diagnostic, roots, branches)
    restored: dict[str, float] = {}
    for field in gate + diagnostic:
        first = np.asarray(initial[field]).reshape(roots, branches, -1)[:, 0]
        expected = np.asarray(source[field]).reshape(roots, -1)
        if first.dtype == bool:
            difference = np.logical_xor(first, expected)
        else:
            difference = np.abs(first - expected)
        restored[field] = float(np.max(difference)) if difference.size else 0.0
    return branch_spread, restored


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("branched collection requires a CUDA device")
    root = project_root()
    config_path = resolve_below(root, args.config)
    output_root = resolve_below(
        root, args.output_root or resolve("data", "corpus", "branched")
    )
    heartbeat_root = resolve_below(
        root, args.heartbeat_root or resolve("logs", "corpus", "branched")
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_branched_config(config)
    fresh_config_path = resolve_below(root, resolve(config["fresh_config"]))
    fresh_config = json.loads(fresh_config_path.read_text(encoding="utf-8"))
    split_path = resolve_below(root, resolve(config["split_manifest"]))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_fresh_config(fresh_config, split)
    release_path = resolve_below(root, resolve(config["fresh_release_index"]))
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("audit_errors"):
        raise RuntimeError("branched source release has unresolved audit errors")
    if args.array_index < 0 or args.array_index >= len(release["entries"]):
        raise IndexError("array index outside the source release")
    release_entry = release["entries"][args.array_index]
    if int(release_entry["array_index"]) != args.array_index:
        raise RuntimeError("source release entry ordering changed")
    source_manifest_path = resolve_below(root, resolve(release_entry["manifest"]))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    tier1 = source_manifest["artifacts"]["tier1"]
    tier1_path = resolve_below(root, resolve(tier1["path"]))
    task_index, task, shard_index = map_array_index(
        fresh_config["tasks"], args.array_index
    )
    if task["environment_id"] != release_entry["task"]:
        raise RuntimeError("source task mapping changed")
    if task["environment_id"] in split["task_partitions"]["composition_final_tasks"]:
        raise RuntimeError("final-evaluation task in branched source")

    selection = config["parent_selection"]
    branching = config["branching"]
    lanes = selected_parent_lanes(
        int(source_manifest["independent_root_count"]),
        first_lane=int(selection["within_shard_first_lane"]),
        stride=int(selection["within_shard_lane_stride"]),
    )
    roots = len(lanes)
    branches = int(branching["branches_per_parent"])
    horizon = int(branching["horizon_transitions"])
    if roots != int(selection["parents_per_shard"]):
        raise RuntimeError("selected parent count differs from parents_per_shard")
    reporter = ProgressReporter(
        task=task["environment_id"],
        source=config["corpus_id"],
        shard_id=f"branched-{args.array_index:04d}",
        total=roots * branches,
        unit="branch_trajectories",
        heartbeat_path=heartbeat_root / f"worker_{args.array_index:04d}.json",
        log_every_seconds=30,
    )
    reporter.report(completed=0, transitions=0, output_bytes=0, force=True)
    shard_dir = output_root / task["environment_id"] / f"shard_{shard_index:04d}"
    if (shard_dir / "COMPLETE").exists():
        print(f"shard already complete: {shard_dir}")
        return
    if shard_dir.exists():
        shutil.rmtree(shard_dir)  # discard a partial earlier attempt
    shard_dir.mkdir(parents=True)
    heartbeat_root.mkdir(parents=True, exist_ok=True)
    env: gym.Env | None = None
    started = time.perf_counter()
    try:
        (
            replay,
            source_physical,
            parent_frames,
            action_frames,
            parent_ids,
            parent_seeds,
            parent_actions,
            selected_interaction,
        ) = read_selected_source(tier1_path, lanes, task)
        parent_state = repeat_parent_state(replay, parent_frames, branches)
        raw_env = gym.make(
            task["environment_id"],
            num_envs=roots * branches,
            reconfiguration_freq=0,
            obs_mode=fresh_config["observation_mode"],
            control_mode=fresh_config["control_mode"],
            reward_mode=fresh_config["reward_mode"],
            sim_backend=fresh_config["simulation_backend"],
            render_backend=task.get("render_backend", fresh_config["render_backend"]),
            render_mode=None,
        )
        env = raw_env
        repeated_seeds = np.repeat(parent_seeds, branches)
        env.reset(seed=repeated_seeds.tolist())
        base = env.unwrapped
        base.set_state_dict(parent_state)
        physical_observation = base.get_obs(unflattened=True)
        frames: dict[str, list[np.ndarray]] = defaultdict(list)
        steps: dict[str, list[np.ndarray]] = defaultdict(list)
        initial = physical_snapshot(base, physical_observation, task)
        branch_spread, restored = restore_errors(
            initial, source_physical, roots, branches
        )
        tolerance = float(branching["state_equality_max_abs_tolerance"])
        gate_fields = ("primary", "context", "context_mask", "robot", "qpos", "qvel")
        if max(branch_spread[field] for field in gate_fields) > tolerance:
            raise RuntimeError(f"branch parent-state equality failed: {branch_spread}")
        if max(restored[field] for field in gate_fields) > tolerance:
            raise RuntimeError(f"source parent restoration failed: {restored}")
        # Preserve the exact serialized state supplied to every branch. The
        # simulator round trip is checked separately through the physical
        # snapshot because SAPIEN can introduce sub-tolerance float32 noise in
        # get_state_dict immediately after set_state_dict.
        append_frame(frames, "replay/sim_state", parent_state)
        append_frame(frames, "physical", initial)
        action_low, action_high = lane_action_bounds(env.action_space, roots * branches)
        requested, processed, clipped = branch_action_schedule(
            parent_actions,
            action_low,
            action_high,
            horizon=horizon,
            scale=float(branching["perturbation_scale_controller_space"]),
        )
        rollout_started = time.perf_counter()
        for step in range(horizon):
            _observation, reward, terminated, truncated, info = env.step(
                processed[step]
            )
            physical_observation = base.get_obs(info=info, unflattened=True)
            targets = controller_targets(base)
            append_frame(frames, "replay/sim_state", base.get_state_dict())
            append_frame(
                frames, "physical", physical_snapshot(base, physical_observation, task)
            )
            append_frame(steps, "action/requested_controller_space", requested[step])
            append_frame(steps, "action/processed_controller_space", processed[step])
            append_frame(
                steps,
                "action/perturbation_from_source_controller_space",
                requested[step] - np.repeat(parent_actions, branches, axis=0),
            )
            append_frame(steps, "controller", targets)
            append_frame(
                steps,
                "reporting_only/outcome",
                reporting_outcome(
                    reward, terminated, truncated, info, roots * branches
                ),
            )
            finite = np.all(np.isfinite(processed[step]), axis=1)
            append_frame(
                steps,
                "validity",
                {
                    "finite": finite,
                    "processed_in_controller_support": np.all(
                        (processed[step] >= action_low)
                        & (processed[step] <= action_high),
                        axis=1,
                    ),
                    "action_clipped": clipped[step],
                    "arm_target_in_joint_limits": target_joint_limit_pass(
                        base, targets["arm_target_qpos"]
                    ),
                },
            )
        torch.cuda.synchronize()
        rollout_seconds = time.perf_counter() - rollout_started
        arrays = {**stack_arrays(frames), **stack_arrays(steps)}
        parent_root_ids = [
            parent_ids[root] for root in range(roots) for _branch in range(branches)
        ]
        branch_indices = np.tile(np.arange(branches, dtype=np.int16), roots)
        arrays.update(
            {
                "metadata/root_id": string_array(parent_root_ids),
                "metadata/parent_id": string_array(parent_root_ids),
                "metadata/branch_id": string_array(
                    [
                        f"{parent}:branch:{int(branch)}"
                        for parent, branch in zip(parent_root_ids, branch_indices)
                    ]
                ),
                "metadata/branch_index": branch_indices,
                "metadata/source_shard_lane": np.repeat(lanes, branches),
                "metadata/source_parent_frame": np.repeat(parent_frames, branches),
                "metadata/source_action_frame": np.repeat(action_frames, branches),
                "metadata/source_parent_selected_by_exact_interaction": np.repeat(
                    selected_interaction, branches
                ),
                "metadata/split_assignment": string_array(
                    [source_manifest["split_assignment"]] * roots * branches
                ),
                "temporal/frame_index": np.arange(horizon + 1, dtype=np.int32),
                "temporal/action_index": np.arange(horizon, dtype=np.int32),
            }
        )
        attributes = common_attributes(
            config=config,
            task=task,
            task_index=task_index,
            base=base,
            artifact_level="branched",
        )
        attributes.update(
            {
                "split_assignment": source_manifest["split_assignment"],
                "source_tier1": relative_to_root(tier1_path),
                "fresh_release_index": relative_to_root(release_path),
                "state_equality_max_abs_tolerance": tolerance,
                "contact_observation_status": task["contact_observation_status"],
            }
        )
        output_path = shard_dir / "branches.h5"
        write_seconds, logical_bytes = write_h5(output_path, arrays, attributes)
        artifact_bytes = output_path.stat().st_size
        manifest = {
            "schema_version": 1,
            "status": "COMPLETE",
            "array_index": args.array_index,
            "task_index": task_index,
            "task": task["environment_id"],
            "shard_index": shard_index,
            "split_assignment": source_manifest["split_assignment"],
            "config": relative_to_root(config_path),
            "fresh_release_index": relative_to_root(release_path),
            "source_manifest": relative_to_root(source_manifest_path),
            "source_tier1": relative_to_root(tier1_path),
            "selected_parent_count": roots,
            "branch_trajectory_count": roots * branches,
            "transition_count": roots * branches * horizon,
            "selected_interaction_parent_count": int(selected_interaction.sum()),
            "branch_state_max_abs_errors": branch_spread,
            "source_restore_max_absolute_errors": restored,
            "artifact": {
                "path": relative_to_root(output_path),
                "compressed_bytes": artifact_bytes,
                "logical_bytes": logical_bytes,
                "write_seconds": write_seconds,
            },
            "throughput": {
                "rollout_seconds": rollout_seconds,
                "physical_transitions_per_second": roots
                * branches
                * horizon
                / rollout_seconds,
                "write_bytes_per_second": artifact_bytes / write_seconds,
            },
            "runtime": {
                "hostname": platform.node(),
                "wall_seconds": time.perf_counter() - started,
                "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
        }
        atomic_write_json(shard_dir / "manifest.json", manifest)
        (shard_dir / "COMPLETE").touch(exist_ok=False)
        reporter.report(
            completed=roots * branches,
            transitions=roots * branches * horizon,
            output_bytes=artifact_bytes,
            status="COMPLETE",
            force=True,
        )
    except Exception as error:
        reporter.report(
            completed=0,
            transitions=0,
            output_bytes=0,
            status="FAILED",
            force=True,
            extra={"error": f"{type(error).__name__}: {error}"},
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
