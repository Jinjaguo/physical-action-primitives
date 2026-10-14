#!/usr/bin/env python3
"""Stream source-training shards to compute physical-feature normalization statistics."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from physical_action_primitives.corpus.historical_core import required_training_datasets
from physical_action_primitives.manifest import atomic_write_json
from physical_action_primitives.paths import resolve
from physical_action_primitives.backbone.training_data import root_path
from physical_action_primitives.backbone.data import (
    ACTION_DIMENSION,
    EFFECT_DIMENSION,
    FRESH_TRAINING_DATASETS,
    STATE_DIMENSION,
    STATE_PASSTHROUGH_INDICES,
    Normalization,
    RunningMoments,
    fresh_unit_arrays,
    historical_unit_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh-release-index", type=Path, default=resolve("data/manifests/fresh_release.json")
    )
    parser.add_argument(
        "--historical-release-index",
        type=Path,
        default=resolve("data/manifests/historical_release.json"),
    )
    parser.add_argument(
        "--branched-release-index",
        type=Path,
        default=resolve("data/manifests/branched_release.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=resolve("data/manifests/normalization.json")
    )
    return parser.parse_args()


def load_h5(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {name: np.asarray(handle[name]) for name in names}


def state_observation_mask(state: np.ndarray) -> np.ndarray:
    mask = np.ones(state.shape, dtype=bool)
    for index in range(3):
        start = 13 + index * 13
        mask[..., start : start + 13] = state[..., 52 + index : 53 + index].astype(
            bool
        )
    mask[..., 80:101] = np.repeat(state[..., 101:108].astype(bool), 3, axis=-1)
    return mask


def update_moments(
    arrays: Any,
    state_moments: RunningMoments,
    action_moments: RunningMoments,
    effect_moments: RunningMoments,
) -> None:
    state_moments.update(arrays.state, state_observation_mask(arrays.state))
    action_moments.update(arrays.action, arrays.action_dimension_mask)
    effect_moments.update(arrays.effect, arrays.effect_dimension_mask)


def read_fresh_arrays(entry: dict[str, Any]) -> Any:
    manifest = json.loads(root_path(entry["manifest"]).read_text(encoding="utf-8"))
    path = root_path(manifest["artifacts"]["tier1"]["path"])
    return fresh_unit_arrays(load_h5(path, FRESH_TRAINING_DATASETS))


def main() -> None:
    args = parse_args()
    fresh_path = root_path(args.fresh_release_index)
    historical_path = root_path(args.historical_release_index)
    branched_path = root_path(args.branched_release_index)
    output_path = root_path(args.output)
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    branched = json.loads(branched_path.read_text(encoding="utf-8"))

    state_moments = RunningMoments(STATE_DIMENSION)
    action_moments = RunningMoments(ACTION_DIMENSION)
    effect_moments = RunningMoments(EFFECT_DIMENSION)
    started = time.perf_counter()
    completed = 0
    fresh_excluded_development_shards = 0
    for entry in fresh["entries"]:
        manifest = json.loads(root_path(entry["manifest"]).read_text(encoding="utf-8"))
        if manifest["split_assignment"] != "source_training":
            fresh_excluded_development_shards += 1
            continue
        update_moments(
            read_fresh_arrays(entry), state_moments, action_moments, effect_moments
        )
        completed += 1
        if completed % 10 == 0:
            print(
                json.dumps(
                    {
                        "fresh_shards_complete": completed,
                        "fresh_shards_total": len(fresh["entries"]),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    historical_config_path = root_path(historical["config"])
    historical_config = json.loads(
        historical_config_path.read_text(encoding="utf-8")
    )
    task_specs = {
        task["environment_id"]: task for task in historical_config["tasks"]
    }
    for entry in historical["entries"]:
        task = task_specs[entry["environment_id"]]
        path = root_path(entry["path"])
        names = required_training_datasets(task)
        arrays = historical_unit_arrays(load_h5(path, names), task)
        update_moments(arrays, state_moments, action_moments, effect_moments)

    branched_completed = 0
    branched_excluded_development_shards = 0
    for entry in branched["entries"]:
        manifest_path = root_path(entry["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["split_assignment"] != "source_training":
            branched_excluded_development_shards += 1
            continue
        artifact = manifest["artifact"]
        path = root_path(artifact["path"])
        arrays = fresh_unit_arrays(load_h5(path, FRESH_TRAINING_DATASETS))
        update_moments(arrays, state_moments, action_moments, effect_moments)
        branched_completed += 1

    if completed < 1 or branched_completed < 1:
        raise RuntimeError("source-training normalization registry is incomplete")

    state_unobserved = np.flatnonzero(state_moments.count == 0)
    action_unobserved = np.flatnonzero(action_moments.count == 0)
    effect_unobserved = np.flatnonzero(effect_moments.count == 0)
    state_mean, state_std = state_moments.finalize(allow_unobserved=True)
    action_mean, action_std = action_moments.finalize(allow_unobserved=True)
    effect_mean, effect_std = effect_moments.finalize(allow_unobserved=True)
    state_mean[STATE_PASSTHROUGH_INDICES] = 0.0
    state_std[STATE_PASSTHROUGH_INDICES] = 1.0
    normalization = Normalization(
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        effect_mean=effect_mean,
        effect_std=effect_std,
    )
    normalization.validate()
    result = {
        "partition": "source_training",
        "fresh_source_shard_count": completed,
        "fresh_excluded_development_shard_count": fresh_excluded_development_shards,
        "branched_source_shard_count": branched_completed,
        "branched_excluded_development_shard_count": (
            branched_excluded_development_shards
        ),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "state_observation_count": state_moments.count.tolist(),
        "state_unobserved_indices": state_unobserved.tolist(),
        "state_passthrough_indices": STATE_PASSTHROUGH_INDICES.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "action_observation_count": action_moments.count.tolist(),
        "action_unobserved_indices": action_unobserved.tolist(),
        "effect_mean": effect_mean.tolist(),
        "effect_std": effect_std.tolist(),
        "effect_observation_count": effect_moments.count.tolist(),
        "effect_unobserved_indices": effect_unobserved.tolist(),
        "unobserved_dimension_policy": (
            "canonical padded dimensions with zero masked observations "
            "are set to mean=0 and std=1; downstream observation masks keep "
            "them excluded from losses"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, result)
    print(json.dumps({"output": str(output_path), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
