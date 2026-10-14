"""Policy-rollout helpers: teacher PPO loading and the vectorized root recorder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from physical_action_primitives.corpus.collection_utils import (
    append_frame,
    controller_targets,
    geometry_arrays,
    physical_snapshot,
    reporting_outcome,
    target_joint_limit_pass,
)
from physical_action_primitives.corpus.core import to_numpy


def initial_sim_state_sha256_by_lane(
    frames: dict[str, np.ndarray],
) -> list[str]:
    """Digest the serialized initial simulator state of each lane (used to check uniqueness)."""
    names = sorted(name for name in frames if name.startswith("replay/sim_state/"))
    if not names:
        raise RuntimeError("episode has no serialized initial simulator state")
    lane_count = int(frames[names[0]].shape[1])
    digests = [hashlib.sha256() for _ in range(lane_count)]
    for name in names:
        initial = np.asarray(frames[name][0])
        if initial.shape[0] != lane_count:
            raise RuntimeError("serialized initial-state lane count changed")
        for lane, digest in enumerate(digests):
            value = np.ascontiguousarray(initial[lane])
            digest.update(name.encode("utf-8"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
            digest.update(value.tobytes(order="C"))
    return [digest.hexdigest() for digest in digests]

def load_official_policy_class(source_path: Path) -> type[torch.nn.Module]:
    module_name = "_official_mani_skill_ppo"
    specification = importlib.util.spec_from_file_location(module_name, source_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the official PPO source")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module.Agent

def policy_tensor(observation: Any, device: torch.device) -> torch.Tensor:
    if isinstance(observation, torch.Tensor):
        value = observation.to(device=device, dtype=torch.float32)
    else:
        value = torch.as_tensor(observation, dtype=torch.float32, device=device)
    if value.ndim == 0:
        return value.reshape(1, 1)
    if value.ndim == 1:
        return value.reshape(1, -1)
    return value.reshape(value.shape[0], -1)

class PPOFreshRootRecorder(gym.Wrapper):
    """Preserve policy observations separately from model-eligible telemetry."""

    def __init__(self, env: gym.Env, task: dict[str, Any]) -> None:
        super().__init__(env)
        self.task = task
        self.frames: dict[str, list[np.ndarray]] = defaultdict(list)
        self.steps: dict[str, list[np.ndarray]] = defaultdict(list)
        self.geometry: dict[str, np.ndarray] = {}

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        policy_observation, info = self.env.reset(**kwargs)
        self.frames = defaultdict(list)
        self.steps = defaultdict(list)
        base = self.unwrapped
        physical_observation = base.get_obs(info=info, unflattened=True)
        append_frame(self.frames, "replay/sim_state", base.get_state_dict())
        append_frame(
            self.frames,
            "physical",
            physical_snapshot(base, physical_observation, self.task),
        )
        append_frame(
            self.frames,
            "teacher_only/policy_observation",
            to_numpy(policy_observation).astype(np.float32),
        )
        self.geometry = geometry_arrays(physical_observation, self.task)
        return policy_observation, info

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        num_envs = int(self.unwrapped.num_envs)
        requested = to_numpy(action).astype(np.float32).reshape(num_envs, -1)
        action_dim = requested.shape[1]
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        if low.size == action_dim:
            low = low.reshape(1, action_dim)
            high = high.reshape(1, action_dim)
        elif low.size == num_envs * action_dim:
            low = low.reshape(num_envs, action_dim)
            high = high.reshape(num_envs, action_dim)
        else:
            raise RuntimeError("vector action-space shape does not match policy action")
        processed = np.clip(requested, low, high).astype(np.float32)
        policy_observation, reward, terminated, truncated, info = self.env.step(
            processed
        )
        base = self.unwrapped
        physical_observation = base.get_obs(info=info, unflattened=True)
        targets = controller_targets(base)
        append_frame(self.frames, "replay/sim_state", base.get_state_dict())
        append_frame(
            self.frames,
            "physical",
            physical_snapshot(base, physical_observation, self.task),
        )
        append_frame(
            self.frames,
            "teacher_only/policy_observation",
            to_numpy(policy_observation).astype(np.float32),
        )
        append_frame(self.steps, "action/requested_teacher_controller_space", requested)
        append_frame(self.steps, "action/processed_controller_space", processed)
        append_frame(self.steps, "controller", targets)
        append_frame(
            self.steps,
            "reporting_only/outcome",
            reporting_outcome(reward, terminated, truncated, info, num_envs),
        )
        finite = (
            np.all(np.isfinite(requested), axis=1)
            & np.all(np.isfinite(processed), axis=1)
            & np.all(np.isfinite(targets["arm_target_qpos"]), axis=1)
        )
        append_frame(
            self.steps,
            "validity",
            {
                "finite": finite,
                "processed_in_controller_support": np.all(
                    (processed >= low) & (processed <= high), axis=1
                ),
                "action_clipped": np.any(processed != requested, axis=1),
                "arm_target_in_joint_limits": target_joint_limit_pass(
                    base, targets["arm_target_qpos"]
                ),
            },
        )
        return policy_observation, reward, terminated, truncated, info
