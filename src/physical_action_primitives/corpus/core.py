"""Shared array, contact-channel and window utilities for the corpus package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from physical_action_primitives.paths import project_root

CONTACT_CHANNELS = (
    {"index": 0, "name": "robot_finger1_on_primary", "slice": [0, 3], "units": "N"},
    {"index": 1, "name": "robot_finger2_on_primary", "slice": [3, 6], "units": "N"},
    {"index": 2, "name": "robot_finger1_on_control_actor", "slice": [6, 9], "units": "N"},
    {"index": 3, "name": "robot_finger2_on_control_actor", "slice": [9, 12], "units": "N"},
    {"index": 4, "name": "control_actor_on_primary", "slice": [12, 15], "units": "N"},
    {"index": 5, "name": "primary_on_distinct_context_actor", "slice": [15, 18], "units": "N"},
    {"index": 6, "name": "primary_on_support_table", "slice": [18, 21], "units": "N"},
)

TRAINING_WINDOW_PREFIXES = (
    "training/state/",
    "training/action/",
    "training/controller/",
)

FORBIDDEN_TRAINING_TOKENS = (
    "task",
    "environment",
    "reward",
    "success",
    "goal",
    "primitive",
    "phase",
    "skill",
    "language",
    "rgb",
)

def relative_to_root(path: Path) -> str:
    """Return ``path`` relative to ``PAP_ROOT`` (POSIX form) when it lies inside it."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_root()).as_posix()
    except ValueError:
        return str(resolved)

def resolve_below(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"path escapes project root: {resolved}")
    return resolved

def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError("object arrays are forbidden in physical tensors")
    return np.ascontiguousarray(array)

def flatten_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, dict):
        result: dict[str, np.ndarray] = {}
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            name = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten_arrays(child, name))
        return result
    return {prefix: to_numpy(value)}

def join_flat_name(namespace: str, leaf: str) -> str:
    """Join a flattening namespace without creating an empty HDF5 leaf name."""

    if not namespace:
        return leaf
    return f"{namespace}/{leaf}" if leaf else namespace

def exact_non_support_interaction_mask(
    contact_force: np.ndarray, contact_channel_mask: np.ndarray
) -> np.ndarray:
    """Return exact nonzero manipulation-contact observations, without a threshold.

    The support-table channel is intentionally excluded: otherwise a resting
    object would count as an interaction on essentially every frame.
    """

    force = np.asarray(contact_force)
    mask = np.asarray(contact_channel_mask, dtype=bool)
    if force.shape[-1] != 21 or mask.shape[-1] != 7:
        raise ValueError("expected 21D contact and seven-channel availability mask")
    reshaped = force.reshape(*force.shape[:-1], 7, 3)
    available_nonzero = np.any(reshaped[..., :6, :] != 0, axis=-1) & mask[..., :6]
    return np.any(available_nonzero, axis=-1)

def parent_state_max_errors(
    snapshot: dict[str, np.ndarray],
    field_names: tuple[str, ...],
    roots: int,
    branches: int,
) -> dict[str, float]:
    """Measure the spread across branches of numeric and boolean parent-state fields."""

    errors: dict[str, float] = {}
    for name in field_names:
        source = np.asarray(snapshot[name])
        value = source.reshape(roots, branches, *source.shape[1:])
        if value.dtype == bool:
            difference = np.logical_xor(value, value[:, :1])
        else:
            difference = np.abs(value - value[:, :1])
        errors[name] = float(np.max(difference)) if difference.size else 0.0
    return errors

def plain_h5_attribute(value: Any) -> Any:
    """Normalize scalar subclasses before passing values to h5py attributes."""

    if isinstance(value, (str, np.str_)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value

def fixed_utf8_array(values: list[str]) -> np.ndarray:
    """Encode metadata strings explicitly, avoiding NumPy Unicode datasets."""

    encoded = [str(value).encode("utf-8") for value in values]
    width = max(1, *(len(value) for value in encoded))
    return np.asarray(encoded, dtype=f"S{width}")

def assert_training_feature_policy(dataset_names: list[str]) -> None:
    names = [name for name in dataset_names if name.startswith(TRAINING_WINDOW_PREFIXES)]
    if not names:
        raise ValueError("no training tensors found")
    leaked = [
        name
        for name in names
        if any(token in name.lower() for token in FORBIDDEN_TRAINING_TOKENS)
    ]
    if leaked:
        raise ValueError(f"forbidden training-tensor names: {leaked}")

def window_starts(horizon: int, length: int, stride: int) -> np.ndarray:
    if horizon < length or length < 1 or stride < 1:
        raise ValueError("invalid horizon/window/stride")
    return np.arange(0, horizon - length + 1, stride, dtype=np.int32)

def windowize_time_lane(
    array: np.ndarray,
    starts: np.ndarray,
    *,
    length: int,
    state_like: bool,
) -> np.ndarray:
    """Convert [time,lane,...] to independent [window,time,...] records."""

    value = np.asarray(array)
    span = length + 1 if state_like else length
    if value.ndim < 2:
        raise ValueError("window input must be [time,lane,...]")
    windows = [
        value[int(start) : int(start) + span, lane]
        for lane in range(value.shape[1])
        for start in starts
    ]
    return np.ascontiguousarray(np.stack(windows, axis=0))

def logical_nbytes(arrays: dict[str, np.ndarray]) -> int:
    return int(sum(np.asarray(value).nbytes for value in arrays.values()))
