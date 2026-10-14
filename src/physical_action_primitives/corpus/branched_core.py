"""Pure parent-selection and branch-action contracts for branched collection."""

from __future__ import annotations

from typing import Any

import numpy as np

from physical_action_primitives.corpus.core import exact_non_support_interaction_mask


HADAMARD_8 = np.asarray(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, 1, -1, -1, 1, -1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, -1, 1, 1, -1],
    ],
    dtype=np.float32,
)


def validate_branched_config(config: dict[str, Any]) -> None:
    selection = config["parent_selection"]
    branching = config["branching"]
    if int(selection["within_shard_lane_stride"]) < 1:
        raise ValueError("parent lane stride must be positive")
    if int(selection["within_shard_first_lane"]) < 0:
        raise ValueError("first parent lane must be non-negative")
    if int(selection["parents_per_shard"]) < 1:
        raise ValueError("parents_per_shard must be positive")
    if int(branching["branches_per_parent"]) != HADAMARD_8.shape[0]:
        raise ValueError("branches_per_parent must match the 8-row perturbation matrix")
    if int(branching["horizon_transitions"]) < 1:
        raise ValueError("branch horizon must be positive")
    if float(branching["perturbation_scale_controller_space"]) <= 0:
        raise ValueError("perturbation scale must be positive")
    if float(branching["state_equality_max_abs_tolerance"]) <= 0:
        raise ValueError("state equality tolerance must be positive")


def selected_parent_lanes(
    root_count: int, *, first_lane: int = 0, stride: int = 128
) -> np.ndarray:
    if root_count < 1 or first_lane < 0 or stride < 1 or first_lane >= root_count:
        raise ValueError("invalid branched lane selection")
    return np.arange(first_lane, root_count, stride, dtype=np.int64)


def select_parent_frames(
    contact_force: np.ndarray,
    contact_mask: np.ndarray,
    *,
    contact_available: bool,
) -> tuple[np.ndarray, np.ndarray]:
    force = np.asarray(contact_force)
    mask = np.asarray(contact_mask, dtype=bool)
    if force.shape[:2] != mask.shape[:2]:
        raise ValueError("branched contact time/lane axes changed")
    horizon = force.shape[0] - 1
    if horizon < 1:
        raise ValueError("branched source trajectory is empty")
    frames = np.full(force.shape[1], horizon // 2, dtype=np.int32)
    selected_interaction = np.zeros(force.shape[1], dtype=bool)
    if not contact_available:
        return frames, selected_interaction
    interaction = exact_non_support_interaction_mask(force, mask)
    for lane in range(force.shape[1]):
        candidates = np.flatnonzero(interaction[:, lane])
        if candidates.size:
            frames[lane] = min(int(candidates[0]), horizon - 1)
            selected_interaction[lane] = True
    return frames, selected_interaction


def branch_action_schedule(
    parent_actions: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    horizon: int = 8,
    scale: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent = np.asarray(parent_actions, dtype=np.float32)
    if parent.ndim != 2 or not 1 <= parent.shape[1] <= HADAMARD_8.shape[1]:
        raise ValueError(
            "branched source actions must be [parents,action_dim] with 1 <= action_dim <= 8"
        )
    # Use the fixed eight branch rows, projected onto the environment controller axes.
    action_dim = parent.shape[1]
    low = np.broadcast_to(np.asarray(low, dtype=np.float32), parent.shape)
    high = np.broadcast_to(np.asarray(high, dtype=np.float32), parent.shape)
    if np.any(parent < low) or np.any(parent > high):
        raise ValueError("branched source action is outside controller support")
    perturbation = HADAMARD_8[:, :action_dim].copy()
    perturbation[0] = 0
    requested = np.empty((horizon, len(parent), 8, action_dim), dtype=np.float32)
    processed = np.empty_like(requested)
    clipped = np.empty(requested.shape[:-1], dtype=bool)
    for step in range(horizon):
        envelope = np.sin(np.pi * (step + 1) / (horizon + 1))
        values = parent[:, None, :] + np.float32(scale * envelope) * perturbation[None]
        requested[step] = values
        processed[step] = np.clip(values, low[:, None], high[:, None])
        clipped[step] = np.any(processed[step] != requested[step], axis=-1)
    return (
        requested.reshape(horizon, -1, action_dim),
        processed.reshape(horizon, -1, action_dim),
        clipped.reshape(horizon, -1),
    )


def repeat_parent_state(
    source: dict[str, np.ndarray], parent_frames: np.ndarray, branches: int = 8
) -> dict[str, dict[str, np.ndarray]]:
    frames = np.asarray(parent_frames, dtype=np.int64)
    result: dict[str, dict[str, np.ndarray]] = {"actors": {}, "articulations": {}}
    for name, values in source.items():
        if not name.startswith("replay/sim_state/"):
            continue
        relative = name[len("replay/sim_state/") :]
        namespace, field = relative.split("/", 1)
        value = np.asarray(values)
        if value.shape[1] != len(frames):
            raise ValueError("branched replay lane count changed")
        selected = value[frames, np.arange(len(frames))]
        result[namespace][field] = np.repeat(selected, branches, axis=0)
    if not result["actors"] or not result["articulations"]:
        raise ValueError("branched replay state is incomplete")
    return result
