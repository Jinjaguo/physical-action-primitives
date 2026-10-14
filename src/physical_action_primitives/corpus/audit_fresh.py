#!/usr/bin/env python3
"""Validate one packed fresh shard on CPU (shapes, seeds, finiteness, tier separation).

Run as ``python -m physical_action_primitives.corpus.audit_fresh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from physical_action_primitives.corpus.core import (
    exact_non_support_interaction_mask,
    relative_to_root,
    resolve_below,
    window_starts,
)
from physical_action_primitives.corpus.fresh_core import map_array_index, validate_fresh_config
from physical_action_primitives.corpus.fresh_extension_core import (
    fresh_corpus_directory_name,
    is_extension_config,
    validate_extension_config,
)
from physical_action_primitives.manifest import atomic_write_json
from physical_action_primitives.paths import project_root, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--plan", type=Path)
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="entry of --plan to audit (default: $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: data/manifests/audits/<fresh|fresh_extension>/shard_XXXX.json",
    )
    args = parser.parse_args()
    if args.plan is not None and args.array_index is None:
        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_index is None:
            parser.error("--plan requires --array-index (or SLURM_ARRAY_TASK_ID)")
        args.array_index = int(env_index)
    if args.plan is None and args.array_index is not None:
        parser.error("--array-index is only valid with --plan")
    return args


def dataset_names(handle: h5py.File | h5py.Group) -> list[str]:
    names: list[str] = []
    handle.visititems(
        lambda name, value: names.append(name)
        if isinstance(value, h5py.Dataset)
        else None
    )
    return sorted(names)


def decode_strings(dataset: h5py.Dataset) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(dataset).reshape(-1)
    ]


def initial_hashes_by_lane(handle: h5py.File) -> list[str]:
    group = handle["replay/sim_state"]
    leaves = dataset_names(group)
    lane_count = int(group[leaves[0]].shape[1])
    digests = [hashlib.sha256() for _ in range(lane_count)]
    for leaf in leaves:
        name = f"replay/sim_state/{leaf}"
        initial = np.asarray(group[leaf][0])
        if initial.shape[0] != lane_count:
            raise RuntimeError("fresh initial-state lane count changed")
        for lane, digest in enumerate(digests):
            value = np.ascontiguousarray(initial[lane])
            digest.update(name.encode("utf-8"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
            digest.update(value.tobytes(order="C"))
    return [digest.hexdigest() for digest in digests]


def require_finite(handle: h5py.File) -> None:
    for name in dataset_names(handle):
        value = np.asarray(handle[name])
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise RuntimeError(f"non-finite fresh shard payload: {name}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def benchmark_training_reads(tier1_path: Path, repetitions: int = 3) -> dict[str, Any]:
    prefixes = ("physical/", "action/", "controller/")
    elapsed = 0.0
    logical_bytes = 0
    checksum = 0.0
    for _ in range(repetitions):
        with h5py.File(tier1_path, "r") as handle:
            names = [name for name in dataset_names(handle) if name.startswith(prefixes)]
            started = time.perf_counter()
            pass_bytes = 0
            for name in names:
                value = np.asarray(handle[name])
                pass_bytes += value.nbytes
                if value.size and value.dtype.kind in "biufc":
                    checksum += float(np.asarray(value).reshape(-1)[0])
            elapsed += time.perf_counter() - started
            logical_bytes += pass_bytes
    return {
        "repetitions": repetitions,
        "training_dataset_count": len(names),
        "logical_bytes_read": logical_bytes,
        "elapsed_seconds": elapsed,
        "logical_bytes_per_second": logical_bytes / elapsed,
        "anti_optimization_checksum": checksum,
        "scope": "cached sequential task-agnostic physical/action/controller tensors",
    }


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = resolve_below(root, args.config)
    plan_path: Path | None = None
    if args.plan is not None:
        plan_path = resolve_below(root, args.plan)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("status") != "COMPLETE":
            raise RuntimeError("shard audit plan is incomplete")
        if int(args.array_index) < 0 or int(args.array_index) >= len(plan["entries"]):
            raise IndexError("audit plan index is outside its entries")
        entry = plan["entries"][int(args.array_index)]
        if int(entry["array_index"]) != int(args.array_index):
            raise RuntimeError("audit plan ordering changed")
        manifest_path = resolve_below(root, resolve(entry["manifest"]))
    else:
        manifest_path = resolve_below(root, args.manifest)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_path = resolve_below(root, resolve(config["split_manifest"]))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if is_extension_config(config):
        base_release_path = resolve_below(
            root, resolve(config["base_fresh_release_index"])
        )
        validate_extension_config(
            config,
            split_manifest,
            json.loads(base_release_path.read_text(encoding="utf-8")),
        )
    else:
        validate_fresh_config(config, split_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("fresh shard is incomplete")
    task_index, task, shard_index = map_array_index(
        config["tasks"], int(manifest["array_index"])
    )
    if manifest["task"] != task or int(manifest["task_index"]) != task_index:
        raise RuntimeError("fresh shard task mapping failed")
    if int(manifest["shard_index"]) != shard_index:
        raise RuntimeError("fresh shard index mapping failed")
    if task["environment_id"] in split_manifest["task_partitions"]["composition_final_tasks"]:
        raise RuntimeError("final-evaluation task found in a fresh shard")
    if args.output is not None:
        output_path = resolve_below(root, args.output)
    else:
        output_path = resolve_below(
            root,
            resolve(
                "data",
                "manifests",
                "audits",
                fresh_corpus_directory_name(config),
                f"shard_{int(manifest['array_index']):04d}.json",
            ),
        )

    artifact_paths = {
        key: resolve_below(root, resolve(record["path"]))
        for key, record in manifest["artifacts"].items()
    }
    for key, path in artifact_paths.items():
        if path.stat().st_size != int(manifest["artifacts"][key]["compressed_bytes"]):
            raise RuntimeError(f"fresh artifact byte count mismatch: {key}")
    tier1_path = artifact_paths["tier1"]
    tier2_path = artifact_paths["tier2"]
    window_path = artifact_paths["lazy_window_index"]
    roots = int(config["roots_per_shard"])
    horizon = int(task["max_episode_steps"])
    first_seed = int(task["root_seed_first"]) + shard_index * roots
    expected_seeds = np.arange(first_seed, first_seed + roots, dtype=np.int64)

    with h5py.File(tier1_path, "r") as tier1:
        names = dataset_names(tier1)
        if any(name.startswith("teacher_only/") for name in names):
            raise RuntimeError("teacher policy observation leaked into fresh Tier 1")
        if str(tier1.attrs["split_assignment"]) != task["partition"]:
            raise RuntimeError("fresh Tier 1 split assignment failed")
        require_finite(tier1)
        for name in names:
            shape = tier1[name].shape
            if name.startswith(("physical/", "replay/sim_state/")) and shape[:2] != (
                horizon + 1,
                roots,
            ):
                raise RuntimeError(f"fresh frame alignment failed: {name}:{shape}")
            if name.startswith(
                ("action/", "controller/", "validity/", "reporting_only/")
            ) and shape[:2] != (horizon, roots):
                raise RuntimeError(f"fresh step alignment failed: {name}:{shape}")
        seeds = np.asarray(tier1["metadata/episode_seed"], dtype=np.int64)
        if not np.array_equal(seeds, expected_seeds):
            raise RuntimeError("fresh shard exact seed interval failed")
        root_ids = decode_strings(tier1["metadata/root_id"])
        expected_ids = [
            f"{config['corpus_id']}:{task['environment_id']}:seed-{int(seed)}"
            for seed in expected_seeds
        ]
        if root_ids != expected_ids or len(set(root_ids)) != roots:
            raise RuntimeError("fresh shard root-ID registry failed")
        split_values = set(decode_strings(tier1["metadata/split_assignment"]))
        if split_values != {task["partition"]}:
            raise RuntimeError("fresh Tier 1 root split values failed")
        stored_hashes = decode_strings(tier1["metadata/initial_sim_state_sha256"])
        recomputed_hashes = initial_hashes_by_lane(tier1)
        if stored_hashes != recomputed_hashes or len(set(stored_hashes)) != roots:
            raise RuntimeError("fresh initial simulator-state hashes failed")
        expected_names: list[str] = []
        expected_sigmas: list[float] = []
        for _batch in range(int(config["batches_per_shard"])):
            for stratum in config["behavior_strata"]:
                expected_names.extend(
                    [str(stratum["name"])] * int(stratum["lane_count"])
                )
                expected_sigmas.extend(
                    [float(stratum["gaussian_sigma_controller_space"])]
                    * int(stratum["lane_count"])
                )
        if decode_strings(tier1["metadata/behavior_process_name"]) != expected_names:
            raise RuntimeError("fresh behavior lane registry failed")
        if not np.allclose(
            np.asarray(tier1["metadata/behavior_noise_sigma_controller_space"]),
            expected_sigmas,
            rtol=0,
            atol=1e-7,
        ):
            raise RuntimeError("fresh behavior sigma registry failed")
        expected_noise_seeds = np.arange(
            int(config["action_noise_seed_base_by_task"][task["environment_id"]])
            + shard_index * roots,
            int(config["action_noise_seed_base_by_task"][task["environment_id"]])
            + (shard_index + 1) * roots,
            dtype=np.int64,
        )
        if not np.array_equal(
            np.asarray(tier1["metadata/behavior_noise_seed"], dtype=np.int64),
            expected_noise_seeds,
        ):
            raise RuntimeError("fresh behavior noise-seed registry failed")
        contacts = exact_non_support_interaction_mask(
            np.asarray(tier1["physical/typed_contact_force"])[1:],
            np.asarray(tier1["physical/contact_channel_mask"])[1:],
        )
        success = np.asarray(tier1["reporting_only/outcome/success"], dtype=bool)
        if int(np.sum(contacts)) != int(
            manifest["exact_non_support_contact_transition_count"]
        ):
            raise RuntimeError("fresh contact count failed")
        if int(np.sum(np.any(success, axis=0))) != int(manifest["any_success_count"]):
            raise RuntimeError("fresh any-success count failed")
        if int(np.sum(success[-1])) != int(manifest["final_success_count"]):
            raise RuntimeError("fresh final-success count failed")

    selected_per_batch = np.asarray(
        config["tier2_selected_lanes_per_batch"], dtype=np.int32
    )
    source_lanes = np.concatenate(
        [
            batch * int(config["num_envs"]) + selected_per_batch
            for batch in range(int(config["batches_per_shard"]))
        ]
    )
    with h5py.File(tier2_path, "r") as tier2:
        names = dataset_names(tier2)
        if not any(name.startswith("teacher_only/") for name in names):
            raise RuntimeError("fresh diagnostic Tier 2 lacks teacher observation")
        forbidden = ("physical/", "replay/", "action/", "controller/", "validity/")
        if any(name.startswith(forbidden) for name in names):
            raise RuntimeError("physical tensor leaked into fresh diagnostic Tier 2")
        if str(tier2.attrs["source_tier1"]) != relative_to_root(tier1_path):
            raise RuntimeError("fresh Tier 2 source link failed")
        if not np.array_equal(
            np.asarray(tier2["metadata/source_tier1_episode_lane"], dtype=np.int32),
            source_lanes,
        ):
            raise RuntimeError("fresh Tier 2 stratified source lanes failed")
        if decode_strings(tier2["metadata/episode_id"]) != [
            root_ids[int(lane)] for lane in source_lanes
        ]:
            raise RuntimeError("fresh Tier 2 root links failed")
        require_finite(tier2)

    starts = window_starts(
        horizon,
        int(config["window"]["length_transitions"]),
        int(config["window"]["stride_transitions"]),
    )
    expected_windows = roots * len(starts)
    with h5py.File(window_path, "r") as windows:
        names = dataset_names(windows)
        if any(not name.startswith(("metadata/", "temporal/")) for name in names):
            raise RuntimeError("trajectory tensor leaked into fresh lazy index")
        if str(windows.attrs["source_tier1"]) != relative_to_root(tier1_path):
            raise RuntimeError("fresh lazy-index source link failed")
        if windows["metadata/window_id"].shape != (expected_windows,):
            raise RuntimeError("fresh lazy-window count failed")
        if decode_strings(windows["metadata/episode_id"])[:: len(starts)] != root_ids:
            raise RuntimeError("fresh lazy-window root ordering failed")
        if set(decode_strings(windows["metadata/split_assignment"])) != {
            task["partition"]
        }:
            raise RuntimeError("fresh lazy-window split assignment failed")

    read_benchmark = benchmark_training_reads(tier1_path)
    audit = {
        "schema_version": 1,
        "status": "COMPLETE",
        "manifest": relative_to_root(manifest_path),
        "audit_plan": relative_to_root(plan_path) if plan_path is not None else None,
        "shard_key": manifest["shard_key"],
        "array_index": manifest["array_index"],
        "task": task["environment_id"],
        "split_assignment": task["partition"],
        "checks": {
            "artifact_sizes_match": True,
            "config_and_task_mapping_match": True,
            "final_tasks_absent": True,
            "root_ids_seeds_and_initial_states_unique": True,
            "time_lane_alignment_valid": True,
            "teacher_observation_absent_from_tier1": True,
            "physical_tensors_absent_from_tier2": True,
            "trajectory_tensors_absent_from_lazy_index": True,
            "behavior_and_tier2_strata_match": True,
            "source_tier_links_match": True,
            "floating_payloads_finite": True,
        },
        "measurements": {
            "independent_roots": roots,
            "transitions": int(manifest["transition_count"]),
            "lazy_windows": expected_windows,
            "tier2_roots": len(source_lanes),
            "artifact_bytes": int(manifest["artifact_bytes"]),
            "exact_non_support_contacts": int(np.sum(contacts)),
            "any_successes": int(np.sum(np.any(success, axis=0))),
            "final_successes": int(np.sum(success[-1])),
            "collection_physical_transitions_per_second": manifest["throughput"][
                "physical_transitions_per_second"
            ],
            "write_bytes_per_second": manifest["throughput"][
                "all_artifact_write_bytes_per_second"
            ],
            "read_benchmark": read_benchmark,
            "audit_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, audit)
    lines = [
        "# Fresh corpus shard audit",
        "",
        f"- Shard: `{audit['shard_key']}`",
        f"- Roots / transitions / lazy windows: {roots} / {manifest['transition_count']} / {expected_windows}",
        f"- Artifact bytes: {manifest['artifact_bytes']}",
        f"- Collection throughput: {manifest['throughput']['physical_transitions_per_second']:.3f} transitions/s",
        f"- Sequential cached training read throughput: {read_benchmark['logical_bytes_per_second']:.3f} logical bytes/s",
        "- Result: shard passed all checks.",
    ]
    atomic_write_text(output_path.with_suffix(".md"), "\n".join(lines) + "\n")
    print(json.dumps(audit["measurements"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
