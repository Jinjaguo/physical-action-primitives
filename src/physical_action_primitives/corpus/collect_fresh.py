#!/usr/bin/env python3
"""Collect one packed fresh-episode corpus shard (``python -m physical_action_primitives.corpus.collect_fresh``)."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import shutil
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - register environments
import numpy as np
import torch

from physical_action_primitives.corpus.collection_utils import (
    common_attributes,
    stack_arrays,
    write_h5,
)
from physical_action_primitives.corpus.core import (
    exact_non_support_interaction_mask,
    relative_to_root,
    resolve_below,
    to_numpy,
)
from physical_action_primitives.corpus.fresh_core import map_array_index, validate_fresh_config
from physical_action_primitives.corpus.fresh_extension_core import (
    fresh_corpus_directory_name,
    is_extension_config,
    validate_extension_config,
)
from physical_action_primitives.corpus.policy_utils import (
    PPOFreshRootRecorder,
    initial_sim_state_sha256_by_lane,
    load_official_policy_class,
    policy_tensor,
)
from physical_action_primitives.corpus.vector_policy_utils import (
    behavior_noise_schedule,
    diagnostic_tier2_arrays,
    lane_action_bounds,
    lazy_window_index_arrays,
    packed_tier1_arrays,
)
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
        help="flat shard index over the config's task registry "
        "(default: $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="default: data/corpus/fresh (data/corpus/fresh_extension for an extension config)",
    )
    parser.add_argument(
        "--heartbeat-root",
        type=Path,
        default=None,
        help="default: logs/corpus/<corpus directory>",
    )
    args = parser.parse_args()
    if args.array_index is None:
        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_index is None:
            parser.error("--array-index is required (or set SLURM_ARRAY_TASK_ID)")
        args.array_index = int(env_index)
    return args


def append_batch(
    destination: dict[str, list[np.ndarray]],
    values: dict[str, np.ndarray],
    *,
    axis: int,
) -> None:
    for name, value in values.items():
        array = np.asarray(value)
        if array.ndim <= axis:
            raise RuntimeError(f"cannot concatenate {name} on axis {axis}: {array.shape}")
        destination[name].append(np.ascontiguousarray(array))


def concatenate_batches(
    values: dict[str, list[np.ndarray]], *, axis: int
) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(np.concatenate(sequence, axis=axis))
        for name, sequence in values.items()
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("fresh corpus collection requires a CUDA device")
    root = project_root()
    config_path = resolve_below(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    corpus_name = fresh_corpus_directory_name(config)
    output_root = resolve_below(
        root, args.output_root or resolve("data", "corpus", corpus_name)
    )
    heartbeat_root = resolve_below(
        root, args.heartbeat_root or resolve("logs", "corpus", corpus_name)
    )
    split_path = resolve_below(root, resolve(config["split_manifest"]))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if is_extension_config(config):
        base_release_path = resolve_below(
            root, resolve(config["base_fresh_release_index"])
        )
        base_release = json.loads(base_release_path.read_text(encoding="utf-8"))
        validate_extension_config(config, split_manifest, base_release)
    else:
        validate_fresh_config(config, split_manifest)
    task_index, task, shard_index = map_array_index(
        config["tasks"], args.array_index
    )
    env_id = task["environment_id"]
    checkpoint_path = resolve_below(root, resolve(task["checkpoint"]))
    ppo_source_path = resolve_below(root, resolve(config["official_ppo_source"]))

    shard_dir = output_root / env_id / f"shard_{shard_index:04d}"
    if (shard_dir / "COMPLETE").exists():
        print(f"shard already complete: {shard_dir}")
        return
    if shard_dir.exists():
        shutil.rmtree(shard_dir)  # discard a partial earlier attempt
    shard_dir.mkdir(parents=True)
    heartbeat_root.mkdir(parents=True, exist_ok=True)
    roots_per_shard = int(config["roots_per_shard"])
    num_envs = int(config["num_envs"])
    batches_per_shard = int(config["batches_per_shard"])
    horizon = int(task["max_episode_steps"])
    reporter = ProgressReporter(
        task=env_id,
        source=config["corpus_id"],
        shard_id=f"{env_id}-shard-{shard_index:04d}",
        total=roots_per_shard,
        unit="independent_roots",
        heartbeat_path=heartbeat_root / f"worker_{args.array_index:04d}.json",
        log_every_seconds=30,
    )
    reporter.report(completed=0, transitions=0, output_bytes=0, force=True)

    env: PPOFreshRootRecorder | None = None
    started = time.perf_counter()
    rollout_seconds = 0.0
    reset_seconds = 0.0
    try:
        raw_env = gym.make(
            env_id,
            num_envs=num_envs,
            reconfiguration_freq=0,
            obs_mode=config["observation_mode"],
            control_mode=config["control_mode"],
            reward_mode=config["reward_mode"],
            sim_backend=config["simulation_backend"],
            render_backend=task.get("render_backend", config["render_backend"]),
            render_mode=None,
        )
        env = PPOFreshRootRecorder(raw_env, task)
        device = torch.device("cuda")
        policy: torch.nn.Module | None = None
        action_low: np.ndarray | None = None
        observation_dim: int | None = None

        tier1_frame_batches: dict[str, list[np.ndarray]] = defaultdict(list)
        tier1_step_batches: dict[str, list[np.ndarray]] = defaultdict(list)
        geometry_batches: dict[str, list[np.ndarray]] = defaultdict(list)
        tier2_frame_batches: dict[str, list[np.ndarray]] = defaultdict(list)
        episode_ids: list[str] = []
        episode_seeds: list[int] = []
        initial_hashes: list[str] = []
        behavior_names: list[str] = []
        behavior_sigmas: list[float] = []
        behavior_noise_seeds: list[int] = []
        tier2_episode_ids: list[str] = []
        tier2_episode_seeds: list[int] = []
        tier2_source_lanes: list[int] = []
        total_contacts = 0
        total_any_success = 0
        total_final_success = 0
        total_clipped = 0
        total_joint_limit_failures = 0
        behavior_summary: dict[str, dict[str, int | float]] = {}
        selected_lanes = np.asarray(
            config["tier2_selected_lanes_per_batch"], dtype=np.int64
        )
        shard_root_offset = shard_index * roots_per_shard
        torch.cuda.reset_peak_memory_stats()

        for batch_index in range(batches_per_shard):
            batch_offset = shard_root_offset + batch_index * num_envs
            requested_seeds = np.arange(
                int(task["root_seed_first"]) + batch_offset,
                int(task["root_seed_first"]) + batch_offset + num_envs,
                dtype=np.int64,
            )
            reset_started = time.perf_counter()
            observation, _ = env.reset(seed=requested_seeds.tolist())
            torch.cuda.synchronize()
            reset_seconds += time.perf_counter() - reset_started
            actual_seeds = to_numpy(env.unwrapped._episode_seed).astype(np.int64)
            if not np.array_equal(actual_seeds, requested_seeds):
                raise RuntimeError("fresh ManiSkill seeds differ from requested seeds")
            observation_tensor = policy_tensor(observation, torch.device("cpu"))
            if observation_tensor.shape[0] != num_envs:
                raise RuntimeError("fresh policy observation lost its lane dimension")
            if policy is None:
                observation_dim = int(observation_tensor.shape[1])
                action_low, action_high = lane_action_bounds(
                    env.action_space, num_envs
                )
                policy_spaces = SimpleNamespace(
                    single_observation_space=gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(observation_dim,),
                        dtype=np.float32,
                    ),
                    single_action_space=gym.spaces.Box(
                        low=action_low,
                        high=action_high,
                        dtype=np.float32,
                    ),
                )
                official_policy_class = load_official_policy_class(ppo_source_path)
                policy = official_policy_class(policy_spaces).to(device)
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=True
                )
                policy.load_state_dict(checkpoint, strict=True)
                policy.eval()
            assert action_low is not None
            batch_names, batch_sigmas, batch_noise_seeds, action_noise = (
                behavior_noise_schedule(
                    config,
                    env_id,
                    horizon=horizon,
                    num_envs=num_envs,
                    action_dim=int(action_low.size),
                    noise_seed_offset=batch_offset,
                )
            )
            torch.cuda.synchronize()
            rollout_started = time.perf_counter()
            for step in range(horizon):
                with torch.no_grad():
                    requested = policy.get_action(
                        policy_tensor(observation, device), deterministic=True
                    )
                    requested = requested + torch.as_tensor(
                        action_noise[step], dtype=requested.dtype, device=device
                    )
                observation, _reward, _terminated, truncated, _info = env.step(
                    requested
                )
                if step + 1 < horizon and bool(np.any(to_numpy(truncated))):
                    raise RuntimeError("fresh vector lane truncated before horizon")
            torch.cuda.synchronize()
            rollout_seconds += time.perf_counter() - rollout_started

            frames = stack_arrays(env.frames)
            steps = stack_arrays(env.steps)
            if next(iter(steps.values())).shape[:2] != (horizon, num_envs):
                raise RuntimeError("fresh batch is not time-major [time,lane,...]")
            batch_hashes = initial_sim_state_sha256_by_lane(frames)
            batch_ids = [
                f"{config['corpus_id']}:{env_id}:seed-{int(seed)}"
                for seed in actual_seeds
            ]
            contacts = exact_non_support_interaction_mask(
                frames["physical/typed_contact_force"][1:],
                frames["physical/contact_channel_mask"][1:],
            )
            success = steps["reporting_only/outcome/success"]
            clipped_by_lane = np.sum(
                steps["validity/action_clipped"], axis=0
            ).astype(np.int64)
            joint_failures_by_lane = np.sum(
                ~steps["validity/arm_target_in_joint_limits"], axis=0
            ).astype(np.int64)
            contacts_by_lane = np.sum(contacts, axis=0).astype(np.int64)
            for stratum in sorted(set(batch_names)):
                lanes = np.asarray(
                    [name == stratum for name in batch_names], dtype=bool
                )
                summary = behavior_summary.setdefault(
                    stratum,
                    {
                        "root_count": 0,
                        "noise_sigma_controller_space": float(
                            batch_sigmas[np.flatnonzero(lanes)[0]]
                        ),
                        "any_success_count": 0,
                        "final_success_count": 0,
                        "exact_non_support_contact_transition_count": 0,
                        "clipped_transition_count": 0,
                    },
                )
                summary["root_count"] = int(summary["root_count"]) + int(
                    np.sum(lanes)
                )
                summary["any_success_count"] = int(
                    summary["any_success_count"]
                ) + int(np.sum(np.any(success[:, lanes], axis=0)))
                summary["final_success_count"] = int(
                    summary["final_success_count"]
                ) + int(np.sum(success[-1, lanes]))
                summary["exact_non_support_contact_transition_count"] = int(
                    summary["exact_non_support_contact_transition_count"]
                ) + int(np.sum(contacts[:, lanes]))
                summary["clipped_transition_count"] = int(
                    summary["clipped_transition_count"]
                ) + int(np.sum(clipped_by_lane[lanes]))

            append_batch(
                tier1_frame_batches,
                {
                    name: value
                    for name, value in frames.items()
                    if not name.startswith("teacher_only/")
                },
                axis=1,
            )
            append_batch(tier1_step_batches, steps, axis=1)
            append_batch(geometry_batches, env.geometry, axis=0)
            append_batch(
                tier2_frame_batches,
                {
                    name: np.ascontiguousarray(value[:, selected_lanes])
                    for name, value in frames.items()
                    if name.startswith("teacher_only/")
                },
                axis=1,
            )
            episode_ids.extend(batch_ids)
            episode_seeds.extend(int(seed) for seed in actual_seeds)
            initial_hashes.extend(batch_hashes)
            behavior_names.extend(batch_names)
            behavior_sigmas.extend(float(value) for value in batch_sigmas)
            behavior_noise_seeds.extend(int(value) for value in batch_noise_seeds)
            tier2_episode_ids.extend(batch_ids[int(lane)] for lane in selected_lanes)
            tier2_episode_seeds.extend(
                int(actual_seeds[int(lane)]) for lane in selected_lanes
            )
            tier2_source_lanes.extend(
                batch_index * num_envs + int(lane) for lane in selected_lanes
            )
            total_contacts += int(np.sum(contacts_by_lane))
            total_any_success += int(np.sum(np.any(success, axis=0)))
            total_final_success += int(np.sum(success[-1]))
            total_clipped += int(np.sum(clipped_by_lane))
            total_joint_limit_failures += int(np.sum(joint_failures_by_lane))
            reporter.report(
                completed=(batch_index + 1) * num_envs,
                transitions=(batch_index + 1) * num_envs * horizon,
                output_bytes=0,
                extra={
                    "batch": batch_index + 1,
                    "contacts": total_contacts,
                    "any_successes": total_any_success,
                },
            )
            del frames, steps, contacts, success, action_noise

        if len(episode_ids) != roots_per_shard:
            raise RuntimeError("fresh shard root count changed")
        if len(set(episode_ids)) != roots_per_shard:
            raise RuntimeError("fresh shard root IDs are not unique")
        if len(set(episode_seeds)) != roots_per_shard:
            raise RuntimeError("fresh shard seeds are not unique")
        if len(set(initial_hashes)) != roots_per_shard:
            raise RuntimeError("fresh shard initial states are not unique")
        if policy is None or observation_dim is None or action_low is None:
            raise RuntimeError("fresh teacher initialization did not complete")

        merged_frames = concatenate_batches(tier1_frame_batches, axis=1)
        merged_steps = concatenate_batches(tier1_step_batches, axis=1)
        merged_geometry = concatenate_batches(geometry_batches, axis=0)
        merged_tier2_frames = concatenate_batches(tier2_frame_batches, axis=1)
        tier1_frame_batches.clear()
        tier1_step_batches.clear()
        geometry_batches.clear()
        tier2_frame_batches.clear()
        gc.collect()

        attrs = common_attributes(
            config=config,
            task=task,
            task_index=task_index,
            base=env.unwrapped,
            artifact_level="fresh_packed_tier1",
        )
        attrs.update(
            {
                "split_assignment": task["partition"],
                "split_manifest": config["split_manifest"],
                "teacher_checkpoint": relative_to_root(checkpoint_path),
                "official_ppo_source": relative_to_root(ppo_source_path),
                "teacher_policy_observation_training_excluded": True,
                "contact_observation_status": task["contact_observation_status"],
                "packed_root_count": roots_per_shard,
                "packed_batch_count": batches_per_shard,
                "packed_batch_lane_count": num_envs,
                "time_lane_axis_contract": "datasets are [time,packed_root,...]",
            }
        )
        tier1_payload = packed_tier1_arrays(
            merged_frames,
            merged_steps,
            merged_geometry,
            episode_ids,
            np.asarray(episode_seeds, dtype=np.int64),
            initial_hashes,
            behavior_names,
            np.asarray(behavior_sigmas, dtype=np.float32),
            np.asarray(behavior_noise_seeds, dtype=np.int64),
        )
        tier1_payload["metadata/split_assignment"] = np.asarray(
            [task["partition"].encode("utf-8")] * roots_per_shard,
            dtype=f"S{len(task['partition'])}",
        )
        tier1_path = shard_dir / "tier1_packed_episodes.h5"
        tier1_write_seconds, tier1_logical_bytes = write_h5(
            tier1_path, tier1_payload, attrs
        )

        tier2_payload = diagnostic_tier2_arrays(
            merged_tier2_frames,
            tier2_episode_ids,
            np.asarray(tier2_episode_seeds, dtype=np.int64),
        )
        tier2_payload["metadata/source_tier1_episode_lane"] = np.asarray(
            tier2_source_lanes, dtype=np.int32
        )
        tier2_payload["metadata/split_assignment"] = np.asarray(
            [task["partition"].encode("utf-8")] * len(tier2_episode_ids),
            dtype=f"S{len(task['partition'])}",
        )
        tier2_path = shard_dir / "tier2_teacher_diagnostic.h5"
        tier2_attrs = {
            **attrs,
            "artifact_level": "fresh_teacher_diagnostic_tier2",
            "diagnostic_only": True,
            "source_tier1": relative_to_root(tier1_path),
            "selection_rule": config["tier2_selection_rule"],
        }
        tier2_write_seconds, tier2_logical_bytes = write_h5(
            tier2_path, tier2_payload, tier2_attrs
        )

        window = config["window"]
        window_payload = lazy_window_index_arrays(
            episode_ids,
            horizon,
            int(window["length_transitions"]),
            int(window["stride_transitions"]),
        )
        window_payload["metadata/split_assignment"] = np.asarray(
            [task["partition"].encode("utf-8")]
            * int(window_payload["metadata/window_id"].shape[0]),
            dtype=f"S{len(task['partition'])}",
        )
        window_path = shard_dir / "lazy_window_index.h5"
        window_attrs = {
            **attrs,
            "artifact_level": "fresh_lazy_window_index",
            "source_tier1": relative_to_root(tier1_path),
            "contains_trajectory_tensors": False,
        }
        window_write_seconds, window_logical_bytes = write_h5(
            window_path, window_payload, window_attrs
        )

        artifacts = (tier1_path, tier2_path, window_path)
        artifact_bytes = sum(path.stat().st_size for path in artifacts)
        if artifact_bytes >= int(config["shard_hard_limit_bytes"]):
            raise RuntimeError("fresh shard exceeded its hard byte limit")

        def artifact_record(
            path: Path, logical_bytes: int, write_seconds: float
        ) -> dict[str, Any]:
            return {
                "path": relative_to_root(path),
                "compressed_bytes": path.stat().st_size,
                "logical_bytes": logical_bytes,
                "write_seconds": write_seconds,
            }

        transition_count = roots_per_shard * horizon
        window_count = int(window_payload["metadata/window_id"].shape[0])
        total_write_seconds = (
            tier1_write_seconds + tier2_write_seconds + window_write_seconds
        )
        manifest = {
            "schema_version": 1,
            "status": "COMPLETE",
            "corpus_id": config["corpus_id"],
            "array_index": args.array_index,
            "task_index": task_index,
            "task": task,
            "shard_index": shard_index,
            "shard_key": f"{env_id}:shard-{shard_index:04d}",
            "split_assignment": task["partition"],
            "split_manifest": config["split_manifest"],
            "config": relative_to_root(config_path),
            "independent_root_count": roots_per_shard,
            "unique_seed_count": len(set(episode_seeds)),
            "unique_initial_sim_state_count": len(set(initial_hashes)),
            "first_episode_seed": min(episode_seeds),
            "last_episode_seed": max(episode_seeds),
            "transition_count": transition_count,
            "lazy_window_count": window_count,
            "tier2_root_count": len(tier2_episode_ids),
            "exact_non_support_contact_transition_count": total_contacts,
            "contact_observation_status": task["contact_observation_status"],
            "any_success_count": total_any_success,
            "final_success_count": total_final_success,
            "clipped_transition_count": total_clipped,
            "arm_target_joint_limit_failure_count": total_joint_limit_failures,
            "behavior_summary": behavior_summary,
            "teacher": {
                "policy": config["policy_action_rule"],
                "checkpoint": relative_to_root(checkpoint_path),
                "source": relative_to_root(ppo_source_path),
                "observation_dimension": observation_dim,
                "action_dimension": int(action_low.size),
                "policy_observation_training_excluded": True,
            },
            "artifacts": {
                "tier1": artifact_record(
                    tier1_path, tier1_logical_bytes, tier1_write_seconds
                ),
                "tier2": artifact_record(
                    tier2_path, tier2_logical_bytes, tier2_write_seconds
                ),
                "lazy_window_index": artifact_record(
                    window_path, window_logical_bytes, window_write_seconds
                ),
            },
            "artifact_bytes": artifact_bytes,
            "throughput": {
                "reset_seconds": reset_seconds,
                "rollout_seconds": rollout_seconds,
                "physical_transitions_per_second": transition_count
                / rollout_seconds,
                "all_artifact_write_seconds": total_write_seconds,
                "all_artifact_write_bytes_per_second": artifact_bytes
                / total_write_seconds,
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
            completed=roots_per_shard,
            transitions=transition_count,
            output_bytes=artifact_bytes,
            status="COMPLETE",
            force=True,
            extra={
                "contacts": total_contacts,
                "any_successes": total_any_success,
                "transitions_per_second": manifest["throughput"][
                    "physical_transitions_per_second"
                ],
            },
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
