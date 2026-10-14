"""Vectorized-lane helpers: action bounds, behavior-noise schedule, packed tier arrays."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from physical_action_primitives.corpus.collection_utils import string_array
from physical_action_primitives.corpus.core import window_starts


def lane_action_bounds(space: gym.spaces.Box, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(space.low, dtype=np.float32)
    high = np.asarray(space.high, dtype=np.float32)
    if low.size % num_envs == 0 and low.size > num_envs:
        low = low.reshape(num_envs, -1)
        high = high.reshape(num_envs, -1)
        if not np.all(low == low[:1]) or not np.all(high == high[:1]):
            raise RuntimeError("official PPO lanes expose different action bounds")
        return low[0], high[0]
    return low.reshape(-1), high.reshape(-1)

def behavior_noise_schedule(
    config: dict[str, Any],
    env_id: str,
    *,
    horizon: int,
    num_envs: int,
    action_dim: int,
    noise_seed_offset: int = 0,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic, lane-bound controller-space noise schedule."""

    strata = config.get("behavior_strata")
    if strata is None:
        return (
            ["deterministic_teacher_mean"] * num_envs,
            np.zeros(num_envs, dtype=np.float32),
            np.full(num_envs, -1, dtype=np.int64),
            np.zeros((horizon, num_envs, action_dim), dtype=np.float32),
        )
    names: list[str] = []
    sigmas: list[float] = []
    for item in strata:
        count = int(item["lane_count"])
        names.extend([str(item["name"])] * count)
        sigmas.extend([float(item["gaussian_sigma_controller_space"])] * count)
    if len(names) != num_envs:
        raise RuntimeError("behavior strata do not cover the vector lanes")
    noise_seeds = np.arange(
        int(config["action_noise_seed_base_by_task"][env_id]) + noise_seed_offset,
        int(config["action_noise_seed_base_by_task"][env_id])
        + noise_seed_offset
        + num_envs,
        dtype=np.int64,
    )
    noise = np.empty((horizon, num_envs, action_dim), dtype=np.float32)
    for lane, (seed, sigma) in enumerate(zip(noise_seeds, sigmas)):
        if sigma == 0:
            noise[:, lane] = 0
        else:
            generator = np.random.Generator(np.random.PCG64(int(seed)))
            noise[:, lane] = generator.normal(
                loc=0.0, scale=sigma, size=(horizon, action_dim)
            ).astype(np.float32)
    return names, np.asarray(sigmas, dtype=np.float32), noise_seeds, noise

def packed_tier1_arrays(
    frames: dict[str, np.ndarray],
    steps: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    episode_ids: list[str],
    episode_seeds: np.ndarray,
    initial_hashes: list[str],
    behavior_names: list[str],
    behavior_sigmas: np.ndarray,
    behavior_noise_seeds: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {
        name: value
        for name, value in {**frames, **steps, **geometry}.items()
        if not name.startswith("teacher_only/")
    }
    horizon = next(iter(steps.values())).shape[0]
    lanes = len(episode_ids)
    result.update(
        {
            "metadata/episode_id": string_array(episode_ids),
            "metadata/root_id": string_array(episode_ids),
            "metadata/parent_id": string_array(episode_ids),
            "metadata/episode_seed": np.asarray(episode_seeds, dtype=np.int64),
            "metadata/initial_sim_state_sha256": string_array(initial_hashes),
            "metadata/behavior_process_name": string_array(behavior_names),
            "metadata/behavior_noise_sigma_controller_space": np.asarray(
                behavior_sigmas, dtype=np.float32
            ),
            "metadata/behavior_noise_seed": np.asarray(
                behavior_noise_seeds, dtype=np.int64
            ),
            "metadata/split_assignment": string_array(
                ["unassigned"] * lanes
            ),
            "temporal/frame_index": np.arange(horizon + 1, dtype=np.int32),
            "temporal/action_index": np.arange(horizon, dtype=np.int32),
        }
    )
    if any(name.startswith("teacher_only/") for name in result):
        raise RuntimeError("teacher observation leaked into packed Tier 1")
    return result

def diagnostic_tier2_arrays(
    frames: dict[str, np.ndarray],
    episode_ids: list[str],
    episode_seeds: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {
        name: value for name, value in frames.items() if name.startswith("teacher_only/")
    }
    lanes = len(episode_ids)
    result.update(
        {
            "metadata/episode_id": string_array(episode_ids),
            "metadata/episode_seed": np.asarray(episode_seeds, dtype=np.int64),
            "metadata/episode_lane": np.arange(lanes, dtype=np.int32),
            "metadata/split_assignment": string_array(
                ["unassigned"] * lanes
            ),
        }
    )
    forbidden = ("physical/", "replay/", "action/", "controller/", "validity/")
    if any(name.startswith(forbidden) for name in result):
        raise RuntimeError("physical or replay tensor leaked into diagnostic Tier 2")
    return result

def lazy_window_index_arrays(
    episode_ids: list[str], horizon: int, length: int, stride: int
) -> dict[str, np.ndarray]:
    starts = window_starts(horizon, length, stride)
    lanes = len(episode_ids)
    lane = np.repeat(np.arange(lanes, dtype=np.int32), len(starts))
    start = np.tile(starts, lanes)
    parent_ids = [episode_ids[int(index)] for index in lane]
    result = {
        "metadata/window_id": string_array(
            [
                f"{parent}:window:{int(offset)}:{length}"
                for parent, offset in zip(parent_ids, start)
            ]
        ),
        "metadata/episode_id": string_array(parent_ids),
        "metadata/root_id": string_array(parent_ids),
        "metadata/parent_id": string_array(parent_ids),
        "metadata/episode_lane": lane,
        "metadata/split_assignment": string_array(
            ["unassigned"] * len(lane)
        ),
        "temporal/start_index": start,
        "temporal/end_index_exclusive": start + length,
    }
    forbidden = (
        "training/",
        "physical/",
        "replay/",
        "teacher_only/",
        "action/",
        "controller/",
        "validity/",
        "reporting_only/",
    )
    if any(name.startswith(forbidden) for name in result):
        raise RuntimeError("trajectory tensor leaked into lazy window index")
    return result
