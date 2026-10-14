#!/usr/bin/env python3
"""Validate one branched shard (shapes, parent-state equality, perturbation schedule).

Run as ``python -m physical_action_primitives.corpus.audit_branched``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from physical_action_primitives.corpus.branched_core import HADAMARD_8, validate_branched_config
from physical_action_primitives.corpus.core import (
    parent_state_max_errors,
    relative_to_root,
    resolve_below,
)
from physical_action_primitives.manifest import atomic_write_json
from physical_action_primitives.paths import project_root, resolve

FINAL_TASKS = {"PickCube-v1", "PushCube-v1", "LiftPegUpright-v1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="default: data/manifests/branched_audit_plan.json",
    )
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="entry of the plan to audit (default: $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: data/manifests/audits/branched/shard_XXXX.json",
    )
    args = parser.parse_args()
    if args.array_index is None:
        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_index is None:
            parser.error("--array-index is required (or set SLURM_ARRAY_TASK_ID)")
        args.array_index = int(env_index)
    return args


def dataset_names(handle: h5py.File | h5py.Group) -> list[str]:
    names: list[str] = []
    handle.visititems(
        lambda name, value: names.append(name)
        if isinstance(value, h5py.Dataset)
        else None
    )
    return sorted(names)


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = resolve_below(root, args.config)
    plan_path = resolve_below(
        root, args.plan or resolve("data", "manifests", "branched_audit_plan.json")
    )
    output_path = resolve_below(
        root,
        args.output
        or resolve(
            "data", "manifests", "audits", "branched", f"shard_{args.array_index:04d}.json"
        ),
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_branched_config(config)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "COMPLETE":
        raise RuntimeError("branched audit plan is incomplete")
    if args.array_index < 0 or args.array_index >= len(plan["entries"]):
        raise IndexError("audit index outside the plan")
    entry = plan["entries"][args.array_index]
    if int(entry["array_index"]) != args.array_index:
        raise RuntimeError("audit plan ordering changed")
    manifest_path = resolve_below(root, resolve(entry["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["task"] in FINAL_TASKS:
        raise RuntimeError("final-evaluation task found in a branched shard")
    artifact = manifest["artifact"]
    artifact_path = resolve_below(root, resolve(artifact["path"]))
    if artifact_path.stat().st_size != int(artifact["compressed_bytes"]):
        raise RuntimeError("branched artifact size mismatch")
    selection = config["parent_selection"]
    branching = config["branching"]
    roots = int(manifest["selected_parent_count"])
    branches = int(manifest["branch_trajectory_count"]) // roots
    horizon = int(manifest["transition_count"]) // (roots * branches)
    if (roots, branches, horizon) != (
        int(selection["parents_per_shard"]),
        int(branching["branches_per_parent"]),
        int(branching["horizon_transitions"]),
    ):
        raise RuntimeError("branched shard dimensions differ from the config")
    tolerance = float(branching["state_equality_max_abs_tolerance"])
    with h5py.File(artifact_path, "r") as handle:
        names = dataset_names(handle)
        for name in names:
            dataset = handle[name]
            if dataset.dtype.kind in "fc" and not np.isfinite(np.asarray(dataset)).all():
                raise RuntimeError(f"nonfinite branched dataset: {name}")
        for name in names:
            if name.startswith(("physical/", "replay/sim_state/")):
                if handle[name].shape[:2] != (horizon + 1, roots * branches):
                    raise RuntimeError(f"branched frame shape mismatch: {name}")
            if name.startswith(("action/", "controller/", "validity/", "reporting_only/")):
                if handle[name].shape[:2] != (horizon, roots * branches):
                    raise RuntimeError(f"branched transition shape mismatch: {name}")
        root_ids = decode_strings(handle["metadata/root_id"][:])
        parent_ids = decode_strings(handle["metadata/parent_id"][:])
        branch_ids = decode_strings(handle["metadata/branch_id"][:])
        branch_indices = np.asarray(handle["metadata/branch_index"])
        if root_ids != parent_ids or len(set(root_ids)) != roots:
            raise RuntimeError("branched root/parent identity mismatch")
        if len(set(branch_ids)) != roots * branches:
            raise RuntimeError("branch IDs are not unique")
        if not np.array_equal(
            branch_indices.reshape(roots, branches),
            np.broadcast_to(np.arange(branches), (roots, branches)),
        ):
            raise RuntimeError("branch ordering changed")
        source_lanes = np.asarray(handle["metadata/source_shard_lane"])
        expected_lanes = np.repeat(
            int(selection["within_shard_first_lane"])
            + int(selection["within_shard_lane_stride"]) * np.arange(roots),
            branches,
        )
        if not np.array_equal(source_lanes, expected_lanes):
            raise RuntimeError("source-lane selection differs from the config")

        gate_fields = ("primary", "context", "context_mask", "robot", "qpos", "qvel")
        initial = {name: np.asarray(handle[f"physical/{name}"][0]) for name in gate_fields}
        equality = parent_state_max_errors(initial, gate_fields, roots, branches)
        if max(equality.values()) > tolerance:
            raise RuntimeError(f"branch physical-state equality failed: {equality}")
        replay_initial = {
            name: np.asarray(handle[f"replay/sim_state/{name}"][0])
            for name in dataset_names(handle["replay/sim_state"])
        }
        replay_equality = parent_state_max_errors(
            replay_initial, tuple(replay_initial), roots, branches
        )
        if max(replay_equality.values()) != 0.0:
            raise RuntimeError("serialized branch inputs are not exactly equal")

        perturbation = np.asarray(
            handle["action/perturbation_from_source_controller_space"]
        )
        action_dim = perturbation.shape[-1]
        perturbation = perturbation.reshape(horizon, roots, branches, action_dim)
        expected = np.empty((horizon, branches, action_dim), dtype=np.float32)
        matrix = HADAMARD_8[:, :action_dim].copy()
        matrix[0] = 0
        for step in range(horizon):
            envelope = np.sin(np.pi * (step + 1) / (horizon + 1))
            expected[step] = (
                float(branching["perturbation_scale_controller_space"])
                * envelope
                * matrix
            )
        if not np.allclose(perturbation, expected[:, None], atol=2e-7, rtol=2e-6):
            raise RuntimeError("branch perturbation schedule differs from the config")
        if not np.asarray(handle["validity/processed_in_controller_support"]).all():
            raise RuntimeError("processed branch action left controller support")
        if str(handle.attrs["source_tier1"]) != manifest["source_tier1"]:
            raise RuntimeError("branched HDF5 source link mismatch")
        if not bool(handle.attrs["task_identity_is_metadata_only"]):
            raise RuntimeError("branched shard task identity policy changed")
        if not bool(handle.attrs["reward_success_are_reporting_only"]):
            raise RuntimeError("branched shard outcome policy changed")

    audit = {
        "schema_version": 1,
        "status": "COMPLETE",
        "array_index": args.array_index,
        "task": manifest["task"],
        "shard_index": manifest["shard_index"],
        "audit_plan": relative_to_root(plan_path),
        "manifest": relative_to_root(manifest_path),
        "measurements": {
            "selected_parents": roots,
            "branch_trajectories": roots * branches,
            "branch_transitions": roots * branches * horizon,
            "selected_interaction_parents": manifest[
                "selected_interaction_parent_count"
            ],
            "artifact_bytes": artifact["compressed_bytes"],
            "branch_physical_max_abs": max(equality.values()),
            "branch_replay_max_abs": max(replay_equality.values()),
        },
        "checks": {
            "artifact_size_match": True,
            "final_tasks_absent": True,
            "branch_physical_equality": True,
            "branch_serialized_input_exact_equality": True,
            "branch_identity_complete": True,
            "parent_selection_match": True,
            "perturbation_schedule_match": True,
            "time_lane_alignment_valid": True,
            "floating_payloads_finite": True,
            "feature_policy_match": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, audit)
    print(json.dumps(audit["measurements"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
