#!/usr/bin/env python3
"""Aggregate the per-shard branched audits into the branched release index.

Run as ``python -m physical_action_primitives.corpus.aggregate_branched_audits``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
        "--plan",
        type=Path,
        default=None,
        help="default: data/manifests/branched_audit_plan.json",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=None,
        help="default: data/manifests/audits/branched",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: data/manifests/branched_release.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = resolve_below(root, args.config)
    plan_path = resolve_below(
        root, args.plan or resolve("data", "manifests", "branched_audit_plan.json")
    )
    audit_root = resolve_below(
        root, args.audit_root or resolve("data", "manifests", "audits", "branched")
    )
    output_path = resolve_below(
        root,
        args.output or resolve("data", "manifests", "branched_release.json"),
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_branched_config(config)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "COMPLETE":
        raise RuntimeError("branched audit plan is incomplete")
    audits: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(audit_root.glob("shard_*.json")):
        audit = json.loads(path.read_text(encoding="utf-8"))
        index = int(audit["array_index"])
        if index in audits:
            raise RuntimeError(f"duplicate branched audit: {index}")
        audits[index] = (path, audit)
    expected_count = len(plan["entries"])
    missing = sorted(set(range(expected_count)) - set(audits))
    if missing:
        raise RuntimeError(f"missing branched audits: {missing[:20]} ({len(missing)})")

    totals = {
        "selected_parents": 0,
        "branch_trajectories": 0,
        "branch_transitions": 0,
        "selected_interaction_parents": 0,
        "artifact_bytes": 0,
    }
    task_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"shards": 0, **{name: 0 for name in totals}}
    )
    entries = []
    for plan_entry in plan["entries"]:
        index = int(plan_entry["array_index"])
        audit_path, audit = audits[index]
        if audit.get("status") != "COMPLETE":
            raise RuntimeError(f"branched audit incomplete: {index}")
        if audit.get("manifest") != plan_entry["manifest"]:
            raise RuntimeError(f"branched audit manifest mismatch: {index}")
        if not all(audit["checks"].values()):
            raise RuntimeError(f"branched audit check failed: {index}")
        measurements = audit["measurements"]
        for name in totals:
            totals[name] += int(measurements[name])
        task = str(audit["task"])
        task_totals[task]["shards"] += 1
        for name in totals:
            task_totals[task][name] += int(measurements[name])
        entries.append({**plan_entry, "audit": relative_to_root(audit_path)})
    if totals["artifact_bytes"] != int(plan["totals"]["artifact_bytes"]):
        raise RuntimeError("branched aggregate artifact bytes differ from the plan")
    release = {
        "schema_version": 1,
        "status": "COMPLETE",
        "config": relative_to_root(config_path),
        "audit_plan": relative_to_root(plan_path),
        "fresh_release_index": plan["fresh_release_index"],
        "all_shards_independently_audited": True,
        "totals": totals,
        "tasks": dict(sorted(task_totals.items())),
        "entries": entries,
        "audit_errors": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, release)
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
