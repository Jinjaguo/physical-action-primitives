"""Shared helpers for collecting corpus shards: physical snapshots, HDF5 writing, attributes."""

from __future__ import annotations

import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import mani_skill.envs  # noqa: F401 - register environments
import numpy as np
import torch

from physical_action_primitives.corpus.core import (
    CONTACT_CHANNELS,
    fixed_utf8_array,
    flatten_arrays,
    join_flat_name,
    logical_nbytes,
    plain_h5_attribute,
    to_numpy,
)


def append_frame(
    destination: dict[str, list[np.ndarray]], namespace: str, value: Any
) -> None:
    for name, array in flatten_arrays(value).items():
        destination[join_flat_name(namespace, name)].append(array)

def stack_arrays(values: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(np.stack(sequence, axis=0))
        for name, sequence in values.items()
    }

def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"

def controller_targets(base: Any) -> dict[str, np.ndarray]:
    controllers = base.agent.controller.controllers
    arm = controllers["arm"]
    result = {"arm_target_qpos": to_numpy(arm._target_qpos).astype(np.float32)}
    gripper = controllers.get("gripper")
    if gripper is not None and hasattr(gripper, "_target_qpos"):
        result["gripper_target_qpos"] = to_numpy(gripper._target_qpos).astype(np.float32)
    else:
        result["gripper_target_qpos"] = np.zeros((int(base.num_envs), 0), dtype=np.float32)
    return result

def target_joint_limit_pass(base: Any, arm_target: np.ndarray) -> np.ndarray:
    arm = base.agent.controller.controllers["arm"]
    limits = to_numpy(base.agent.robot.get_qlimits()).astype(np.float32)
    indices = to_numpy(arm.active_joint_indices).astype(np.int64)
    lower = limits[:, indices, 0]
    upper = limits[:, indices, 1]
    return np.all((arm_target >= lower) & (arm_target <= upper), axis=1)

def typed_contact(base: Any, task: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    num_envs = int(base.num_envs)
    force = np.zeros((num_envs, 21), dtype=np.float32)
    mask = np.zeros((num_envs, 7), dtype=bool)
    robot = base.agent
    primary = getattr(base, task["primary_attr"])
    if hasattr(robot, "finger1_link") and hasattr(robot, "finger2_link"):
        force[:, 0:3] = to_numpy(
            base.scene.get_pairwise_contact_forces(robot.finger1_link, primary)
        )
        force[:, 3:6] = to_numpy(
            base.scene.get_pairwise_contact_forces(robot.finger2_link, primary)
        )
        mask[:, 0:2] = True
        if task["control_actor"] != task["primary_actor"]:
            control = getattr(base, task["control_attr"])
            force[:, 6:9] = to_numpy(
                base.scene.get_pairwise_contact_forces(robot.finger1_link, control)
            )
            force[:, 9:12] = to_numpy(
                base.scene.get_pairwise_contact_forces(robot.finger2_link, control)
            )
            force[:, 12:15] = to_numpy(
                base.scene.get_pairwise_contact_forces(control, primary)
            )
            mask[:, 2:5] = True
    if task.get("context_attr") is not None:
        context = getattr(base, task["context_attr"])
        force[:, 15:18] = to_numpy(
            base.scene.get_pairwise_contact_forces(primary, context)
        )
        mask[:, 5] = True
    table_builder = getattr(base, "table_scene", None)
    if table_builder is None:
        table_builder = getattr(base, "scene_builder", None)
    if table_builder is None or not hasattr(table_builder, "table"):
        raise AttributeError("task has no support-table builder")
    force[:, 18:21] = to_numpy(
        base.scene.get_pairwise_contact_forces(primary, table_builder.table)
    )
    mask[:, 6] = True
    return force, mask

def physical_snapshot(
    base: Any, observation: dict[str, Any], task: dict[str, Any]
) -> dict[str, np.ndarray]:
    state = base.get_state_dict()
    primary = to_numpy(state["actors"][task["primary_actor"]]).astype(np.float32)
    context_actor = task.get("context_actor")
    if context_actor is None:
        context = np.zeros_like(primary)
        context_mask = np.zeros((int(base.num_envs), 1), dtype=bool)
    else:
        context = to_numpy(state["actors"][context_actor]).astype(np.float32)
        context_mask = np.ones((int(base.num_envs), 1), dtype=bool)
    robot_name = next(iter(state["articulations"]))
    qpos = to_numpy(base.agent.robot.get_qpos()).astype(np.float32)
    qvel = to_numpy(base.agent.robot.get_qvel()).astype(np.float32)
    contact_force, contact_mask = typed_contact(base, task)
    snapshot = {
        "primary": primary,
        "context": context,
        "context_mask": context_mask,
        "robot": to_numpy(state["articulations"][robot_name]).astype(np.float32),
        "tcp_pose": to_numpy(observation["extra"]["tcp_pose"]).astype(np.float32),
        "qpos": qpos,
        "qvel": qvel,
        "gripper_qpos": qpos[:, -2:],
        "gripper_qvel": qvel[:, -2:],
        "typed_contact_force": contact_force,
        "contact_channel_mask": contact_mask,
    }
    for name, value in snapshot.items():
        if value.dtype != bool and not np.isfinite(value).all():
            raise RuntimeError(f"non-finite physical field {name}")
    return snapshot

def reporting_outcome(
    reward: Any, terminated: Any, truncated: Any, info: dict[str, Any], num_envs: int
) -> dict[str, np.ndarray]:
    success = info.get("success", np.zeros(num_envs, dtype=bool))
    return {
        "reward": to_numpy(reward).astype(np.float32),
        "success": to_numpy(success).astype(bool),
        "terminated": to_numpy(terminated).astype(bool),
        "truncated": to_numpy(truncated).astype(bool),
    }

def h5_dataset(handle: h5py.File, name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    group_name, dataset_name = name.rsplit("/", 1) if "/" in name else ("", name)
    parent = handle.require_group(group_name) if group_name else handle
    options: dict[str, Any] = {}
    if array.size and array.dtype.kind in "buifc":
        options.update(compression="lzf", shuffle=True, chunks=True)
    parent.create_dataset(dataset_name, data=array, **options)

def write_h5(
    path: Path, arrays: dict[str, np.ndarray], attributes: dict[str, Any]
) -> tuple[float, int]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"leftover partial artifact: {partial}")
    started = time.perf_counter()
    try:
        with h5py.File(partial, "w") as handle:
            for name, value in sorted(attributes.items()):
                handle.attrs[name] = plain_h5_attribute(value)
            for name, value in sorted(arrays.items()):
                try:
                    h5_dataset(handle, name, value)
                except Exception as error:
                    array = np.asarray(value)
                    raise RuntimeError(
                        f"failed HDF5 dataset {name}: shape={array.shape} dtype={array.dtype}"
                    ) from error
            handle.flush()
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return time.perf_counter() - started, logical_nbytes(arrays)

def string_array(values: list[str]) -> np.ndarray:
    return fixed_utf8_array(values)

def geometry_arrays(observation: dict[str, Any], task: dict[str, Any]) -> dict[str, np.ndarray]:
    extra = observation["extra"]
    result: dict[str, np.ndarray] = {}
    for name in task.get("geometry_observation_keys", []):
        if name not in extra:
            raise KeyError(f"configured geometry observation is absent: {name}")
        result[f"source_geometry/{name}"] = to_numpy(extra[name])
    return result

def common_attributes(
    *,
    config: dict[str, Any],
    task: dict[str, Any],
    task_index: int,
    base: Any,
    artifact_level: str,
) -> dict[str, Any]:
    return {
        "schema_version": int(config["schema_version"]),
        "corpus_id": config["corpus_id"],
        "artifact_level": artifact_level,
        "environment_id": task["environment_id"],
        "task_index": task_index,
        "mechanism_metadata_only": task["mechanism"],
        "controller_mode": str(getattr(base, "control_mode", "unknown")),
        "observation_mode": str(getattr(base, "obs_mode", "unknown")),
        "simulation_backend": str(getattr(base, "sim_backend", "unknown")),
        "render_backend": str(getattr(base, "render_backend", "unknown")),
        "contact_registry_json": json.dumps(CONTACT_CHANNELS, sort_keys=True),
        "task_replay_spec_json": json.dumps(task, sort_keys=True),
        "environment_class": f"{base.__class__.__module__}.{base.__class__.__qualname__}",
        "mani_skill_version": getattr(mani_skill, "__version__", package_version("mani-skill")),
        "gymnasium_version": package_version("gymnasium"),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "sapien_version": package_version("sapien"),
        "task_identity_is_metadata_only": True,
        "reward_success_are_reporting_only": True,
        "rgb_present": False,
        "language_present": False,
        "human_phase_or_skill_labels_present": False,
    }
