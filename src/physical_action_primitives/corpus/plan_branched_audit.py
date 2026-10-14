#!/usr/bin/env python3
"""Enumerate the completed branched shards and write the shard audit plan.

Run as ``python -m physical_action_primitives.corpus.plan_branched_audit``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physical_action_primitives.corpus.branched_core import validate_branched_config
from physical_action_primitives.corpus.core import relative_to_root, resolve_below
from physical_action_primitives.manifest import atomic_write_json
from physical_action_primitives.paths import project_root, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="default: data/corpus/branched",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: data/manifests/branched_audit_plan.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = resolve_below(root, args.config)
    corpus_root = resolve_below(
        root, args.corpus_root or resolve("data", "corpus", "branched")
    )
    output_path = resolve_below(
        root,
        args.output or resolve("data", "manifests", "branched_audit_plan.json"),
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_branched_config(config)
    release_path = resolve_below(root, resolve(config["fresh_release_index"]))
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("status") != "COMPLETE" or release.get("audit_errors"):
        raise RuntimeError("branched source release is not clean")

    observed: dict[int, tuple[Path, dict[str, Any]]] = {}
    incomplete = 0
    for path in sorted(corpus_root.glob("*/shard_*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE" or not (path.parent / "COMPLETE").is_file():
            incomplete += 1
            continue
        index = int(manifest["array_index"])
        if index in observed:
            raise RuntimeError(f"duplicate branched shard: {index}")
        observed[index] = (path, manifest)
    expected_count = len(release["entries"])
    missing = sorted(set(range(expected_count)) - set(observed))
    unexpected = sorted(set(observed) - set(range(expected_count)))
    if missing or unexpected:
        raise RuntimeError(
            f"branched coverage incomplete missing={missing[:20]} ({len(missing)}) "
            f"unexpected={unexpected[:20]}"
        )
    entries = []
    for index in range(expected_count):
        path, manifest = observed[index]
        source_entry = release["entries"][index]
        if (
            manifest["task"] != source_entry["task"]
            or int(manifest["shard_index"]) != int(source_entry["shard_index"])
            or manifest["source_manifest"] != source_entry["manifest"]
        ):
            raise RuntimeError(f"branched source mapping failed: {index}")
        entries.append(
            {
                "array_index": index,
                "task": manifest["task"],
                "shard_index": manifest["shard_index"],
                "manifest": relative_to_root(path),
                "source_manifest": manifest["source_manifest"],
                "selected_parents": manifest["selected_parent_count"],
                "branch_trajectories": manifest["branch_trajectory_count"],
                "branch_transitions": manifest["transition_count"],
                "artifact_bytes": manifest["artifact"]["compressed_bytes"],
            }
        )
    totals = {
        "selected_parents": sum(int(x["selected_parents"]) for x in entries),
        "branch_trajectories": sum(int(x["branch_trajectories"]) for x in entries),
        "branch_transitions": sum(int(x["branch_transitions"]) for x in entries),
        "artifact_bytes": sum(int(x["artifact_bytes"]) for x in entries),
    }
    plan = {
        "schema_version": 1,
        "status": "COMPLETE",
        "config": relative_to_root(config_path),
        "fresh_release_index": relative_to_root(release_path),
        "entry_count": len(entries),
        "ignored_incomplete_manifest_count": incomplete,
        "totals": totals,
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, plan)
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
