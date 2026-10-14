"""Task-agnostic physical features, normalization, and lazy-window extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


STATE_DIMENSION = 108
ACTION_DIMENSION = 8
EFFECT_DIMENSION = 39
CONTACT_DIMENSION = 7
CONTEXT_SLOTS = 3
ENTITY_DIMENSION = 13
STATE_PASSTHROUGH_INDICES = np.asarray(
    [52, 53, 54, 101, 102, 103, 104, 105, 106, 107], dtype=np.int64
)
FRESH_TRAINING_DATASETS = (
    "physical/primary",
    "physical/context",
    "physical/context_mask",
    "physical/tcp_pose",
    "physical/qpos",
    "physical/qvel",
    "physical/typed_contact_force",
    "physical/contact_channel_mask",
    "action/processed_controller_space",
    "validity/finite",
)


@dataclass(frozen=True)
class UnitArrays:
    state: np.ndarray
    action: np.ndarray
    effect: np.ndarray
    contact_active: np.ndarray
    validity: np.ndarray
    action_dimension_mask: np.ndarray
    effect_dimension_mask: np.ndarray
    contact_observation_mask: np.ndarray

    def validate(self) -> None:
        frames, roots, state_dimension = self.state.shape
        transitions = frames - 1
        expected = {
            "state": (frames, roots, STATE_DIMENSION),
            "action": (transitions, roots, ACTION_DIMENSION),
            "effect": (transitions, roots, EFFECT_DIMENSION),
            "contact_active": (transitions, roots, CONTACT_DIMENSION),
            "validity": (transitions, roots),
            "action_dimension_mask": (transitions, roots, ACTION_DIMENSION),
            "effect_dimension_mask": (transitions, roots, EFFECT_DIMENSION),
            "contact_observation_mask": (
                transitions,
                roots,
                CONTACT_DIMENSION,
            ),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"unit array shape mismatch {name}: {getattr(self, name).shape}")
        for name in ("state", "action", "effect"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"nonfinite unit array: {name}")
        if state_dimension != STATE_DIMENSION:
            raise ValueError("unit state dimension changed")


@dataclass(frozen=True)
class Normalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    effect_mean: np.ndarray
    effect_std: np.ndarray

    def validate(self) -> None:
        expected = {
            "state_mean": (STATE_DIMENSION,),
            "state_std": (STATE_DIMENSION,),
            "action_mean": (ACTION_DIMENSION,),
            "action_std": (ACTION_DIMENSION,),
            "effect_mean": (EFFECT_DIMENSION,),
            "effect_std": (EFFECT_DIMENSION,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid normalization field: {name}")
            if name.endswith("_std") and np.any(value <= 0):
                raise ValueError(f"nonpositive normalization scale: {name}")


class RunningMoments:
    """Float64 parallel moments for stable shard-streamed normalization."""

    def __init__(self, dimension: int) -> None:
        self.count = np.zeros(dimension, dtype=np.int64)
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.m2 = np.zeros(dimension, dtype=np.float64)

    def update(self, values: np.ndarray, mask: np.ndarray | None = None) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1, self.mean.size)
        if not np.isfinite(flat).all():
            raise ValueError("nonfinite normalization input")
        if mask is None:
            flat_mask = np.ones(flat.shape, dtype=bool)
        else:
            flat_mask = np.broadcast_to(np.asarray(mask, dtype=bool), values.shape).reshape(
                flat.shape
            )
        batch_count = flat_mask.sum(axis=0, dtype=np.int64)
        if not np.any(batch_count):
            return
        batch_mean = np.divide(
            (flat * flat_mask).sum(axis=0),
            batch_count,
            out=np.zeros_like(self.mean),
            where=batch_count > 0,
        )
        batch_m2 = (((flat - batch_mean) ** 2) * flat_mask).sum(axis=0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        update = batch_count > 0
        self.mean[update] += (
            delta[update] * batch_count[update] / total[update]
        )
        self.m2[update] += batch_m2[update] + (
            np.square(delta[update])
            * self.count[update]
            * batch_count[update]
            / total[update]
        )
        self.count = total

    def finalize(
        self, minimum_std: float = 1e-6, *, allow_unobserved: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        observed = self.count >= 2
        if np.any(self.count == 1):
            raise ValueError("normalization dimension has exactly one sample")
        if not allow_unobserved and not np.all(observed):
            raise ValueError("insufficient samples for normalization")
        if not np.any(observed):
            raise ValueError("normalization has no observed dimensions")
        mean = self.mean.copy()
        variance = np.zeros_like(self.mean)
        variance[observed] = self.m2[observed] / self.count[observed]
        std = np.ones_like(self.mean)
        std[observed] = np.sqrt(np.maximum(variance[observed], minimum_std**2))
        mean[~observed] = 0.0
        return mean.astype(np.float32), std.astype(np.float32)


def select_unit_array_roots(arrays: UnitArrays, roots: np.ndarray) -> UnitArrays:
    """Select independent trajectory lanes while preserving every time axis."""
    indices = np.asarray(roots, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 1:
        raise ValueError("unit root selection must be a nonempty vector")
    if np.any(indices < 0) or np.any(indices >= arrays.state.shape[1]):
        raise IndexError("unit root selection is outside the shard")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("unit root selection contains duplicates")
    result = UnitArrays(
        state=np.ascontiguousarray(arrays.state[:, indices]),
        action=np.ascontiguousarray(arrays.action[:, indices]),
        effect=np.ascontiguousarray(arrays.effect[:, indices]),
        contact_active=np.ascontiguousarray(arrays.contact_active[:, indices]),
        validity=np.ascontiguousarray(arrays.validity[:, indices]),
        action_dimension_mask=np.ascontiguousarray(
            arrays.action_dimension_mask[:, indices]
        ),
        effect_dimension_mask=np.ascontiguousarray(
            arrays.effect_dimension_mask[:, indices]
        ),
        contact_observation_mask=np.ascontiguousarray(
            arrays.contact_observation_mask[:, indices]
        ),
    )
    result.validate()
    return result


def canonicalize_quaternion(values: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("invalid near-zero quaternion")
    result = quaternion / norm
    return np.where(result[..., :1] < 0, -result, result).astype(np.float32)


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    scalar = left[..., :1] * right[..., :1] - np.sum(
        left[..., 1:] * right[..., 1:], axis=-1, keepdims=True
    )
    vector = (
        left[..., :1] * right[..., 1:]
        + right[..., :1] * left[..., 1:]
        + np.cross(left[..., 1:], right[..., 1:])
    )
    return np.concatenate((scalar, vector), axis=-1)


def quaternion_delta_rotvec_wxyz(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    initial = canonicalize_quaternion(initial)
    final = canonicalize_quaternion(final)
    conjugate = initial.copy()
    conjugate[..., 1:] *= -1
    relative = canonicalize_quaternion(quaternion_multiply_wxyz(final, conjugate))
    vector = relative[..., 1:]
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(relative[..., :1], 0.0, 1.0))
    scale = np.where(norm > 1e-7, angle / np.maximum(norm, 1e-12), 2.0)
    return (vector * scale).astype(np.float32)


def signed_log1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.sign(values) * np.log1p(np.abs(values))


def entity_state(
    values: np.ndarray, presence_mask: np.ndarray | None = None
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] not in (7, 13):
        raise ValueError(f"physical entity must have 7 or 13 channels: {values.shape}")
    result = np.zeros((*values.shape[:-1], ENTITY_DIMENSION), dtype=np.float32)
    result[..., : values.shape[-1]] = values
    if presence_mask is None:
        result[..., 3:7] = canonicalize_quaternion(result[..., 3:7])
    else:
        presence = np.asarray(presence_mask, dtype=bool)
        if presence.shape != values.shape[:-1]:
            raise ValueError("entity presence mask does not match its time/lane axes")
        quaternion = result[..., 3:7]
        if np.any(presence):
            quaternion[presence] = canonicalize_quaternion(quaternion[presence])
        result[~presence] = 0.0
    return result


def pad_robot(values: np.ndarray, dimension: int = 9) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] > dimension:
        raise ValueError("robot vector exceeds canonical dimension")
    result = np.zeros((*values.shape[:-1], dimension), dtype=np.float32)
    result[..., : values.shape[-1]] = values
    return result


def assemble_state(
    primary: np.ndarray,
    contexts: list[np.ndarray],
    tcp_pose: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    contact_force: np.ndarray,
    contact_mask: np.ndarray,
    context_masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    primary = entity_state(primary)
    if len(contexts) > CONTEXT_SLOTS:
        raise ValueError("too many physical context actors")
    if context_masks is None:
        context_masks = [
            np.ones(primary.shape[:-1], dtype=bool) for _ in contexts
        ]
    if len(context_masks) != len(contexts):
        raise ValueError("context-mask count does not match context-actor count")
    context_values = np.zeros(
        (*primary.shape[:-1], CONTEXT_SLOTS, ENTITY_DIMENSION), dtype=np.float32
    )
    context_presence = np.zeros((*primary.shape[:-1], CONTEXT_SLOTS), dtype=np.float32)
    for index, (context, presence) in enumerate(zip(contexts, context_masks)):
        value = entity_state(context, presence)
        if value.shape[:-1] != primary.shape[:-1]:
            raise ValueError("context time/lane axes do not match primary")
        context_values[..., index, :] = value
        context_presence[..., index] = np.asarray(presence, dtype=np.float32)
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
    if tcp_pose.shape != (*primary.shape[:-1], 7):
        raise ValueError("TCP pose shape changed")
    tcp_pose = tcp_pose.copy()
    tcp_pose[..., 3:7] = canonicalize_quaternion(tcp_pose[..., 3:7])
    contact_force = np.asarray(contact_force, dtype=np.float32)
    contact_mask = np.asarray(contact_mask, dtype=bool)
    if contact_force.shape != (*primary.shape[:-1], 21):
        raise ValueError("typed contact-force shape changed")
    if contact_mask.shape != (*primary.shape[:-1], CONTACT_DIMENSION):
        raise ValueError("contact-channel mask shape changed")
    state = np.concatenate(
        (
            primary,
            context_values.reshape(*primary.shape[:-1], -1),
            context_presence,
            tcp_pose,
            pad_robot(qpos),
            pad_robot(qvel),
            signed_log1p(contact_force),
            contact_mask.astype(np.float32),
        ),
        axis=-1,
    )
    if state.shape[-1] != STATE_DIMENSION:
        raise AssertionError("canonical state dimension changed")
    return np.ascontiguousarray(state, dtype=np.float32)


def structured_effect(
    primary: np.ndarray,
    tcp_pose: np.ndarray,
    contact_force: np.ndarray,
) -> np.ndarray:
    primary = entity_state(primary)
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32).copy()
    tcp_pose[..., 3:7] = canonicalize_quaternion(tcp_pose[..., 3:7])
    effect = np.concatenate(
        (
            primary[1:, ..., :3] - primary[:-1, ..., :3],
            quaternion_delta_rotvec_wxyz(
                primary[:-1, ..., 3:7], primary[1:, ..., 3:7]
            ),
            primary[1:, ..., 7:10],
            primary[1:, ..., 10:13],
            tcp_pose[1:, ..., :3] - tcp_pose[:-1, ..., :3],
            quaternion_delta_rotvec_wxyz(
                tcp_pose[:-1, ..., 3:7], tcp_pose[1:, ..., 3:7]
            ),
            signed_log1p(contact_force[1:]),
        ),
        axis=-1,
    )
    if effect.shape[-1] != EFFECT_DIMENSION:
        raise AssertionError("canonical effect dimension changed")
    return np.ascontiguousarray(effect, dtype=np.float32)


def canonical_action(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values, dtype=np.float32)
    if source.ndim < 1 or not 1 <= source.shape[-1] <= ACTION_DIMENSION:
        raise ValueError("controller action dimension must be in [1,8]")
    action = np.zeros((*source.shape[:-1], ACTION_DIMENSION), dtype=np.float32)
    action[..., : source.shape[-1]] = source
    mask = np.zeros(action.shape, dtype=bool)
    mask[..., : source.shape[-1]] = True
    return action, mask


def fresh_unit_arrays(payload: dict[str, np.ndarray]) -> UnitArrays:
    primary = payload["physical/primary"]
    context_mask = np.asarray(payload["physical/context_mask"], dtype=bool)
    context = np.asarray(payload["physical/context"], dtype=np.float32)
    contact_force = payload["physical/typed_contact_force"]
    contact_mask = np.asarray(payload["physical/contact_channel_mask"], dtype=bool)
    state = assemble_state(
        primary,
        [context],
        payload["physical/tcp_pose"],
        payload["physical/qpos"],
        payload["physical/qvel"],
        contact_force,
        contact_mask,
        context_masks=[context_mask[..., 0]],
    )
    action, action_mask = canonical_action(
        payload["action/processed_controller_space"]
    )
    transitions, roots = action.shape[:2]
    availability = contact_mask[1:]
    effect_mask = np.ones((transitions, roots, EFFECT_DIMENSION), dtype=bool)
    effect_mask[..., 18:] = np.repeat(availability, 3, axis=-1)
    result = UnitArrays(
        state=state,
        action=action,
        effect=structured_effect(primary, payload["physical/tcp_pose"], contact_force),
        contact_active=(
            contact_force[1:].reshape(transitions, roots, CONTACT_DIMENSION, 3) != 0
        ).any(axis=-1),
        validity=np.asarray(payload["validity/finite"], dtype=bool),
        action_dimension_mask=action_mask,
        effect_dimension_mask=effect_mask,
        contact_observation_mask=availability,
    )
    result.validate()
    return result


def historical_unit_arrays(
    payload: dict[str, np.ndarray], task: dict[str, Any]
) -> UnitArrays:
    primary = payload[str(task["primary_state"])]
    contexts = [payload[str(name)] for name in task["context_states"]]
    tcp = payload[str(task["tcp_state"])]
    frames, roots = primary.shape[:2]
    contact_force = np.zeros((frames, roots, 21), dtype=np.float32)
    contact_mask = np.zeros((frames, roots, CONTACT_DIMENSION), dtype=bool)
    state = assemble_state(
        primary,
        contexts,
        tcp,
        payload[str(task["robot_qpos"])],
        payload[str(task["robot_qvel"])],
        contact_force,
        contact_mask,
    )
    action, action_mask = canonical_action(payload["actions"])
    transitions = frames - 1
    effect_mask = np.ones((transitions, roots, EFFECT_DIMENSION), dtype=bool)
    effect_mask[..., 18:] = False
    result = UnitArrays(
        state=state,
        action=action,
        effect=structured_effect(primary, tcp, contact_force),
        contact_active=np.zeros((transitions, roots, CONTACT_DIMENSION), dtype=bool),
        validity=np.ones((transitions, roots), dtype=bool),
        action_dimension_mask=action_mask,
        effect_dimension_mask=effect_mask,
        contact_observation_mask=np.zeros(
            (transitions, roots, CONTACT_DIMENSION), dtype=bool
        ),
    )
    result.validate()
    return result


def normalize_unit_arrays(arrays: UnitArrays, normalization: Normalization) -> UnitArrays:
    normalization.validate()
    state = (arrays.state - normalization.state_mean) / normalization.state_std
    action = (arrays.action - normalization.action_mean) / normalization.action_std
    effect = (arrays.effect - normalization.effect_mean) / normalization.effect_std
    context_masks = arrays.state[..., 52:55].astype(bool)
    for index in range(CONTEXT_SLOTS):
        start = 13 + index * ENTITY_DIMENSION
        state[..., start : start + ENTITY_DIMENSION] *= context_masks[
            ..., index : index + 1
        ]
    contact_masks = arrays.state[..., 101:108].astype(bool)
    state[..., 80:101] *= np.repeat(contact_masks, 3, axis=-1)
    action *= arrays.action_dimension_mask
    effect *= arrays.effect_dimension_mask
    return UnitArrays(
        state=np.ascontiguousarray(state, dtype=np.float32),
        action=np.ascontiguousarray(action, dtype=np.float32),
        effect=np.ascontiguousarray(effect, dtype=np.float32),
        contact_active=arrays.contact_active,
        validity=arrays.validity,
        action_dimension_mask=arrays.action_dimension_mask,
        effect_dimension_mask=arrays.effect_dimension_mask,
        contact_observation_mask=arrays.contact_observation_mask,
    )


def extract_lazy_window_batch(
    arrays: UnitArrays,
    lanes: np.ndarray,
    starts: np.ndarray,
    length: int = 16,
) -> dict[str, np.ndarray]:
    lanes = np.asarray(lanes, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    if lanes.ndim != 1 or starts.shape != lanes.shape:
        raise ValueError("lazy-window lanes and starts must be aligned vectors")
    state_time = starts[:, None] + np.arange(length + 1)[None]
    step_time = state_time[:, :-1]
    state = arrays.state[state_time, lanes[:, None]]

    def steps(value: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(value[step_time, lanes[:, None]])

    return {
        "state": np.ascontiguousarray(state[:, :-1]),
        "action": steps(arrays.action),
        "effect": steps(arrays.effect),
        "contact_active": steps(arrays.contact_active),
        "state_delta": np.ascontiguousarray(state[:, 1:] - state[:, :-1]),
        "validity": steps(arrays.validity),
        "action_dimension_mask": steps(arrays.action_dimension_mask),
        "effect_dimension_mask": steps(arrays.effect_dimension_mask),
        "contact_observation_mask": steps(arrays.contact_observation_mask),
    }
