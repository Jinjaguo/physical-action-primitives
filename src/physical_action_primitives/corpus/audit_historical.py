#!/usr/bin/env python3
"""Validate and index the historical-source shards usable for backbone training.

Shards are read from ``data/corpus/historical/<task>/*.h5``; each shard has a
JSON sidecar with the same stem whose ``completion_status`` is ``COMPLETE``.
Run as ``python -m physical_action_primitives.corpus.audit_historical``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from physical_action_primitives.corpus.core import (
    relative_to_root,
    resolve_below,
    window_starts,
)
from physical_action_primitives.corpus.historical_core import (
    HISTORICAL_TASKS,
    required_training_datasets,
    validate_historical_source_config,
)
from physical_action_primitives.manifest import atomic_write_json
from physical_action_primitives.paths import project_root, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="default: data/corpus/historical",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: data/manifests/historical_release.json",
    )
    return parser.parse_args()


def assert_finite(dataset: h5py.Dataset, name: str) -> None:
    if dataset.dtype.kind not in "biufc":
        raise RuntimeError(f"training dataset is not numeric: {name}")
    if dataset.dtype.kind in "fc":
        rows = int(dataset.shape[0]) if dataset.shape else 1
        for start in range(0, rows, 16):
            block = dataset[start : min(rows, start + 16)] if dataset.shape else dataset[()]
            if not np.isfinite(block).all():
                raise RuntimeError(f"nonfinite historical training tensor: {name}")


def audit_shard(path: Path, task: dict[str, Any]) -> dict[str, Any]:
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise RuntimeError(f"missing historical sidecar: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("completion_status") != "COMPLETE":
        raise RuntimeError(f"historical sidecar incomplete: {sidecar_path}")

    horizon = int(task["horizon"])
    action_dimension = int(task["action_dimension"])
    with h5py.File(path, "r") as handle:
        episodes = int(handle["actions"].shape[1])
        if handle["actions"].shape != (horizon, episodes, action_dimension):
            raise RuntimeError(f"historical action shape mismatch: {path}")
        if handle["terminated"].shape != (horizon, episodes):
            raise RuntimeError(f"historical termination shape mismatch: {path}")
        if handle["truncated"].shape != (horizon, episodes):
            raise RuntimeError(f"historical truncation shape mismatch: {path}")
        for name in required_training_datasets(task):
            if name not in handle:
                raise RuntimeError(f"missing historical training dataset {name}: {path}")
            dataset = handle[name]
            expected_time = horizon if name == "actions" else horizon + 1
            if dataset.ndim < 2 or dataset.shape[:2] != (expected_time, episodes):
                raise RuntimeError(f"historical time/lane contract failed {name}: {path}")
            assert_finite(dataset, name)
        seeds = np.asarray(handle["metadata/episode_seed"][:], dtype=np.int64)
        parent_ids = handle["metadata/parent_group_id"][:]
        if seeds.shape != (episodes,) or len(set(map(int, seeds))) != episodes:
            raise RuntimeError(f"historical shard seeds are not unique: {path}")
        if parent_ids.shape != (episodes,):
            raise RuntimeError(f"historical parent IDs mismatch: {path}")
    if "episode_count" in sidecar and int(sidecar["episode_count"]) != episodes:
        raise RuntimeError(f"historical sidecar episode count mismatch: {path}")
    starts = window_starts(
        horizon,
        int(task["window_length"]),
        int(task["window_stride"]),
    )
    return {
        "environment_id": task["environment_id"],
        "shard_id": path.stem,
        "path": relative_to_root(path),
        "sidecar_path": relative_to_root(sidecar_path),
        "independent_roots": episodes,
        "transitions": episodes * horizon,
        "lazy_windows": episodes * len(starts),
        "source_bytes": int(path.stat().st_size),
        "seed_min": int(seeds.min()),
        "seed_max": int(seeds.max()),
    }


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = resolve_below(root, args.config)
    shard_root = resolve_below(
        root, args.root or resolve("data", "corpus", "historical")
    )
    output_path = resolve_below(
        root,
        args.output or resolve("data", "manifests", "historical_release.json"),
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_path = resolve_below(root, resolve(config["split_manifest"]))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_historical_source_config(config, split)

    task_specs = {task["environment_id"]: task for task in config["tasks"]}
    records: list[dict[str, Any]] = []
    all_root_ids: set[str] = set()
    task_totals: dict[str, dict[str, int]] = {}
    window = config["window"]
    for environment_id in HISTORICAL_TASKS:
        paths = sorted((shard_root / environment_id).glob("*.h5"))
        if not paths:
            raise RuntimeError(f"no historical shards found: {environment_id}")
        task = {
            **task_specs[environment_id],
            "window_length": int(window["length_transitions"]),
            "window_stride": int(window["stride_transitions"]),
        }
        task_records = [audit_shard(path, task) for path in paths]
        task_root_ids: set[str] = set()
        for path in paths:
            with h5py.File(path, "r") as handle:
                for seed in handle["metadata/episode_seed"][:]:
                    root_id = f"historical:{environment_id}:seed-{int(seed)}"
                    if root_id in task_root_ids or root_id in all_root_ids:
                        raise RuntimeError(f"duplicate historical root ID: {root_id}")
                    task_root_ids.add(root_id)
        records.extend(task_records)
        all_root_ids.update(task_root_ids)
        task_totals[environment_id] = {
            "shards": len(task_records),
            "independent_roots": sum(x["independent_roots"] for x in task_records),
            "transitions": sum(x["transitions"] for x in task_records),
            "lazy_windows": sum(x["lazy_windows"] for x in task_records),
            "source_bytes": sum(x["source_bytes"] for x in task_records),
        }

    totals = {
        "tasks": len(task_totals),
        "shards": len(records),
        "independent_roots": sum(x["independent_roots"] for x in records),
        "transitions": sum(x["transitions"] for x in records),
        "lazy_windows": sum(x["lazy_windows"] for x in records),
        "source_bytes": sum(x["source_bytes"] for x in records),
    }
    release = {
        "schema_version": 1,
        "status": "COMPLETE",
        "config": relative_to_root(config_path),
        "split_manifest": config["split_manifest"],
        "totals": totals,
        "tasks": task_totals,
        "entries": records,
        "audit_errors": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, release)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
