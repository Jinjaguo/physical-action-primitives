#!/usr/bin/env python3
"""Enumerate the completed fresh shards and write the shard audit plan.

Run as ``python -m physical_action_primitives.corpus.plan_fresh_audit``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physical_action_primitives.corpus.core import relative_to_root, resolve_below
from physical_action_primitives.corpus.fresh_core import (
    map_array_index,
    total_shard_count,
    validate_fresh_config,
)
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
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="default: data/corpus/fresh (data/corpus/fresh_extension for an extension config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: data/manifests/fresh_audit_plan.json "
        "(fresh_extension_audit_plan.json for an extension config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = resolve_below(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    name = fresh_corpus_directory_name(config)
    corpus_root = resolve_below(
        root, args.corpus_root or resolve("data", "corpus", name)
    )
    output_path = resolve_below(
        root, args.output or resolve("data", "manifests", f"{name}_audit_plan.json")
    )
    split_path = resolve_below(root, resolve(config["split_manifest"]))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if is_extension_config(config):
        base_release_path = resolve_below(
            root, resolve(config["base_fresh_release_index"])
        )
        validate_extension_config(
            config,
            split,
            json.loads(base_release_path.read_text(encoding="utf-8")),
        )
    else:
        validate_fresh_config(config, split)

    by_array_index: dict[int, tuple[Path, dict[str, Any]]] = {}
    ignored_incomplete = 0
    for manifest_path in sorted(corpus_root.glob("*/shard_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE" or not (
            manifest_path.parent / "COMPLETE"
        ).is_file():
            ignored_incomplete += 1
            continue
        array_index = int(manifest["array_index"])
        if array_index in by_array_index:
            raise RuntimeError(f"duplicate complete fresh shard: {array_index}")
        task_index, task, shard_index = map_array_index(config["tasks"], array_index)
        if (
            manifest["task"] != task
            or int(manifest["task_index"]) != task_index
            or int(manifest["shard_index"]) != shard_index
        ):
            raise RuntimeError(f"fresh manifest mapping failed: {manifest_path}")
        by_array_index[array_index] = (manifest_path, manifest)

    expected_count = total_shard_count(config)
    missing = sorted(set(range(expected_count)) - set(by_array_index))
    unexpected = sorted(set(by_array_index) - set(range(expected_count)))
    if missing or unexpected:
        raise RuntimeError(
            f"audit plan requires complete coverage; missing={missing[:20]} "
            f"({len(missing)} total), unexpected={unexpected[:20]}"
        )
    entries: list[dict[str, Any]] = []
    for array_index in range(expected_count):
        manifest_path, manifest = by_array_index[array_index]
        entries.append(
            {
                "array_index": array_index,
                "task": manifest["task"]["environment_id"],
                "shard_index": manifest["shard_index"],
                "shard_key": manifest["shard_key"],
                "manifest": relative_to_root(manifest_path),
                "artifact_bytes": int(manifest["artifact_bytes"]),
                "independent_roots": int(manifest["independent_root_count"]),
                "transitions": int(manifest["transition_count"]),
                "lazy_windows": int(manifest["lazy_window_count"]),
            }
        )
    plan = {
        "schema_version": 1,
        "status": "COMPLETE",
        "config": relative_to_root(config_path),
        "split_manifest": config["split_manifest"],
        "corpus_root": relative_to_root(corpus_root),
        "entry_count": len(entries),
        "ignored_incomplete_manifest_count": ignored_incomplete,
        "totals": {
            "artifact_bytes": sum(entry["artifact_bytes"] for entry in entries),
            "independent_roots": sum(entry["independent_roots"] for entry in entries),
            "transitions": sum(entry["transitions"] for entry in entries),
            "lazy_windows": sum(entry["lazy_windows"] for entry in entries),
        },
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, plan)
    print(json.dumps(plan["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
