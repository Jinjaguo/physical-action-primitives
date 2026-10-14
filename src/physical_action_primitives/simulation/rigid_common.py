"""Shared definitions for the 12D rigid response law: response extraction, affine head, cache loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


TASKS = ("PushT-v1", "PokeCube-v1", "RollBall-v1")
SOURCE_INDICES = np.asarray([12, 2, 22, 10, 14, 0, 24], dtype=np.int64)
HELD_INDICES = np.asarray([i for i in range(25) if i not in SOURCE_INDICES], dtype=np.int64)
ACTIVE_INDICES = np.asarray([0, 1, 5, 6, 7, 11], dtype=np.int64)
OFF_PLANE_INDICES = np.asarray([2, 3, 4, 8, 9, 10], dtype=np.int64)
RESPONSE_SCALE = np.asarray(
    [
        0.005,
        0.005,
        0.005,
        np.deg2rad(5.0),
        np.deg2rad(5.0),
        np.deg2rad(5.0),
        0.05,
        0.05,
        0.05,
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)
PLANAR_RESPONSE_SCALE = np.asarray(
    [0.005, 0.005, np.deg2rad(5.0), 0.05, 0.05, 1.0], dtype=np.float32
)


def canonicalize_quaternion(values: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-10):
        raise ValueError("near-zero quaternion")
    result = quaternion / norm
    return np.where(result[..., :1] < 0.0, -result, result)


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
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
    conjugate[..., 1:] *= -1.0
    relative = canonicalize_quaternion(quaternion_multiply_wxyz(final, conjugate))
    vector = relative[..., 1:]
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(relative[..., :1], 0.0, 1.0))
    scale = np.where(norm > 1e-8, angle / np.maximum(norm, 1e-12), 2.0)
    return (vector * scale).astype(np.float32)


def quaternion_yaw_wxyz(values: np.ndarray) -> np.ndarray:
    quaternion = canonicalize_quaternion(values)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_to_local_vector(values: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Rotate XYZ vectors about world Z into lateral/forward/up coordinates."""
    values = np.asarray(values, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    while direction.ndim < values.ndim - 1:
        direction = direction[None]
    cosine = np.cos(direction)
    sine = np.sin(direction)
    result = values.copy()
    result[..., 0] = values[..., 0] * cosine + values[..., 1] * sine
    result[..., 1] = -values[..., 0] * sine + values[..., 1] * cosine
    return result


def response12_from_object_trace(object_trace: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return exact endpoint 12D response from [..., time, lane, object13]."""
    trace = np.asarray(object_trace, dtype=np.float32)
    if trace.shape[-1] != 13 or trace.shape[-3] < 2:
        raise ValueError(f"expected [...,time,lane,13], got {trace.shape}")
    initial = trace[..., 0, :, :]
    final = trace[..., -1, :, :]
    delta_position = world_to_local_vector(final[..., :3] - initial[..., :3], direction)
    delta_rotation = world_to_local_vector(
        quaternion_delta_rotvec_wxyz(initial[..., 3:7], final[..., 3:7]), direction
    )
    # The off-plane coordinates are the local x/y components of the exact
    # endpoint relative rotation vector. The z coordinate is the wrapped
    # endpoint heading change; for rolling objects rotvec-z and heading delta
    # differ because rotations do not commute.
    yaw_delta = quaternion_yaw_wxyz(final[..., 3:7]) - quaternion_yaw_wxyz(initial[..., 3:7])
    delta_rotation[..., 2] = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    linear_velocity = world_to_local_vector(final[..., 7:10], direction)
    angular_velocity = world_to_local_vector(final[..., 10:13], direction)
    return np.concatenate(
        (delta_position, delta_rotation, linear_velocity, angular_velocity), axis=-1
    ).astype(np.float32)


def responses12_from_branches(branches: dict[str, np.ndarray]) -> np.ndarray:
    object_trace = np.asarray(branches["object"][:25], dtype=np.float32)
    response = response12_from_object_trace(object_trace, np.asarray(branches["direction"]))
    # [probe,time,lane,13] -> response [probe,lane,12] -> [lane,probe,12]
    return response.transpose(1, 0, 2)


def planar_active_response(response12: np.ndarray) -> np.ndarray:
    return np.asarray(response12)[..., ACTIVE_INDICES]


def mlp(inputs: int, outputs: int, width: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, width),
        nn.SiLU(),
        nn.Linear(width, width),
        nn.SiLU(),
        nn.Linear(width, outputs),
    )


class RigidLawHead(nn.Module):
    """Affine 12D response head conditioned on a source identity vector."""

    def __init__(self, identity_dim: int = 32, context_dim: int = 12, width: int = 128):
        super().__init__()
        self.identity_dim = identity_dim
        self.context_dim = context_dim
        self.width = width
        self.network = mlp(identity_dim + context_dim, 12 * 3, width)

    def parameters12(self, identity: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.network(torch.cat((identity, context), dim=-1))
        return value[:, :12], value[:, 12:].reshape(-1, 12, 2)

    def forward(
        self, identity: torch.Tensor, context: torch.Tensor, intervention: torch.Tensor
    ) -> torch.Tensor:
        bias, jacobian = self.parameters12(identity, context)
        return bias[:, None, :] + torch.einsum("bri,bki->bkr", jacobian, intervention)


def candidate_grid(bound: float) -> np.ndarray:
    return np.asarray(
        [(x, y) for x in np.linspace(-bound, bound, 9) for y in np.linspace(-bound, bound, 9)],
        dtype=np.float32,
    )


def alpha12(dx_m: float, dy_m: float) -> np.ndarray:
    target = np.zeros(12, dtype=np.float32)
    target[0] = dx_m
    target[1] = dy_m
    return target


def load_cache(root: str | Path) -> dict[str, np.ndarray | list[str]]:
    rows: list[dict[str, np.ndarray]] = []
    tasks: list[str] = []
    for path in sorted(Path(root).glob("*.npz")):
        data = dict(np.load(path))
        rows.append(data)
        tasks.extend([path.stem] * len(data["u"]))
    if not rows:
        raise FileNotFoundError(f"no cache files in {root}")
    keys = ("u", "h", "xi", "grid", "y12", "y12_physical")
    result: dict[str, np.ndarray | list[str]] = {
        key: np.concatenate([row[key] for row in rows]) for key in keys
    }
    result["tasks"] = tasks
    result["source_indices"] = np.asarray(rows[0]["source_indices"], dtype=np.int64)
    return result
