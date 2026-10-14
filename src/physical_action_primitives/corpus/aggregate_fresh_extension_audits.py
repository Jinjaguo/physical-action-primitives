#!/usr/bin/env python3
"""Aggregate the extension shard audits and combine them with the base fresh release index.

Run as ``python -m physical_action_primitives.corpus.aggregate_fresh_extension_audits``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from physical_action_primitives.corpus.core import relative_to_root, resolve_below
from physical_action_primitives.corpus.fresh_core import total_shard_count
from physical_action_primitives.corpus.fresh_extension_core import validate_extension_config
from physical_action_primitives.manifest import atomic_write_json
from physical_action_primitives.paths import project_root, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="default: data/manifests/fresh_extension_audit_plan.json",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=None,
        help="default: data/manifests/audits/fresh_extension",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: data/manifests/fresh_release.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = resolve_below(root, args.config)
    plan_path = resolve_below(
        root,
        args.plan or resolve("data", "manifests", "fresh_extension_audit_plan.json"),
    )
    audit_root = resolve_below(
        root,
        args.audit_root or resolve("data", "manifests", "audits", "fresh_extension"),
    )
    output_path = resolve_below(
        root,
        args.output or resolve("data", "manifests", "fresh_release.json"),
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_path = resolve_below(root, resolve(config["split_manifest"]))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    base_path = resolve_below(root, resolve(config["base_fresh_release_index"]))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    validate_extension_config(config, split, base)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "COMPLETE":
        raise RuntimeError("fresh extension audit plan is incomplete")
    audits: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(audit_root.glob("shard_*.json")):
        audit = json.loads(path.read_text(encoding="utf-8"))
        index = int(audit["array_index"])
        if index in audits:
            raise RuntimeError(f"duplicate fresh extension audit: {index}")
        audits[index] = (path, audit)
    expected_count = total_shard_count(config)
    if len(plan["entries"]) != expected_count:
        raise RuntimeError("fresh extension audit-plan entry count differs from the config")
    missing = sorted(set(range(expected_count)) - set(audits))
    if missing:
        raise RuntimeError(f"missing fresh extension audits: {missing[:20]} ({len(missing)})")

    extension_totals = {
        "independent_roots": 0,
        "transitions": 0,
        "lazy_windows": 0,
        "tier2_roots": 0,
        "artifact_bytes": 0,
        "exact_non_support_contacts": 0,
        "any_successes": 0,
        "final_successes": 0,
    }
    extension_tasks: dict[str, dict[str, int]] = defaultdict(
        lambda: {"shards": 0, **{name: 0 for name in extension_totals}}
    )
    extension_entries = []
    for plan_entry in plan["entries"]:
        index = int(plan_entry["array_index"])
        audit_path, audit = audits[index]
        if audit.get("status") != "COMPLETE":
            raise RuntimeError(f"fresh extension audit incomplete: {index}")
        if audit.get("manifest") != plan_entry["manifest"]:
            raise RuntimeError(f"fresh extension audit manifest mismatch: {index}")
        if not all(audit["checks"].values()):
            raise RuntimeError(f"fresh extension audit check failed: {index}")
        measurements = audit["measurements"]
        for name in extension_totals:
            extension_totals[name] += int(measurements[name])
        task = str(audit["task"])
        extension_tasks[task]["shards"] += 1
        for name in extension_totals:
            extension_tasks[task][name] += int(measurements[name])
        extension_entries.append(
            {
                **plan_entry,
                "corpus_partition": "extension",
                "audit": relative_to_root(audit_path),
            }
        )
    expected_roots = sum(int(task["root_count"]) for task in config["tasks"])
    if extension_totals["independent_roots"] != expected_roots:
        raise RuntimeError("fresh extension root count differs from the config")
    if extension_totals["artifact_bytes"] != int(plan["totals"]["artifact_bytes"]):
        raise RuntimeError("fresh extension artifact bytes differ from the plan")

    combined_aggregate = dict(base["aggregate"])
    for name, value in extension_totals.items():
        if name in combined_aggregate:
            combined_aggregate[name] = int(combined_aggregate[name]) + value
    combined_tasks = {name: dict(values) for name, values in base["tasks"].items()}
    for task, values in extension_tasks.items():
        target = combined_tasks[task]
        for name, value in values.items():
            target[name] = int(target.get(name, 0)) + int(value)
    base_entries = [{**entry, "corpus_partition": "base"} for entry in base["entries"]]
    release = {
        "schema_version": 1,
        "status": "COMPLETE",
        "release_structure": "base_plus_extension",
        "base_release_index": relative_to_root(base_path),
        "extension_config": relative_to_root(config_path),
        "extension_audit_plan": relative_to_root(plan_path),
        "all_shards_independently_audited": True,
        "aggregate": combined_aggregate,
        "base_aggregate": base["aggregate"],
        "extension_aggregate": extension_totals,
        "tasks": dict(sorted(combined_tasks.items())),
        "entries": base_entries + extension_entries,
        "audit_errors": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, release)
    print(
        json.dumps(
            {
                "combined_artifact_bytes": int(combined_aggregate["artifact_bytes"]),
                "combined_independent_roots": combined_aggregate["independent_roots"],
                "extension": extension_totals,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
