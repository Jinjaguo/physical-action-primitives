#!/usr/bin/env python3
"""Aggregate the per-shard fresh audits into the fresh release index.

Run as ``python -m physical_action_primitives.corpus.aggregate_fresh_audits``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from physical_action_primitives.corpus.core import relative_to_root, resolve_below
from physical_action_primitives.corpus.fresh_core import total_shard_count, validate_fresh_config
from physical_action_primitives.manifest import atomic_write_json
from physical_action_primitives.paths import project_root, resolve

SUM_KEYS = (
    "independent_roots",
    "transitions",
    "lazy_windows",
    "tier2_roots",
    "artifact_bytes",
    "exact_non_support_contacts",
    "any_successes",
    "final_successes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="default: data/manifests/fresh_audit_plan.json",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=None,
        help="default: data/manifests/audits/fresh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: data/manifests/fresh_base_release.json",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = resolve_below(root, args.config)
    plan_path = resolve_below(
        root, args.plan or resolve("data", "manifests", "fresh_audit_plan.json")
    )
    audit_root = resolve_below(
        root, args.audit_root or resolve("data", "manifests", "audits", "fresh")
    )
    output_path = resolve_below(
        root, args.output or resolve("data", "manifests", "fresh_base_release.json")
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_path = resolve_below(root, resolve(config["split_manifest"]))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_fresh_config(config, split)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "COMPLETE":
        raise RuntimeError("fresh corpus audit plan is incomplete")
    expected_count = total_shard_count(config)
    if len(plan["entries"]) != expected_count:
        raise RuntimeError("fresh corpus audit-plan entry count differs from the config")

    audits_by_index: dict[int, tuple[Path, dict[str, Any]]] = {}
    for audit_path in sorted(audit_root.glob("shard_*.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        array_index = int(audit["array_index"])
        if array_index in audits_by_index:
            raise RuntimeError(f"duplicate fresh shard audit: {array_index}")
        audits_by_index[array_index] = (audit_path, audit)
    missing = sorted(set(range(expected_count)) - set(audits_by_index))
    if missing:
        raise RuntimeError(f"missing fresh shard audits: {missing[:20]} ({len(missing)} total)")

    aggregate: dict[str, int | float] = {
        **{key: 0 for key in SUM_KEYS},
        "collection_rollout_equivalent_seconds": 0.0,
        "artifact_write_equivalent_seconds": 0.0,
    }
    task_totals: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"shards": 0, **{key: 0 for key in SUM_KEYS}}
    )
    index_entries: list[dict[str, Any]] = []
    for entry in plan["entries"]:
        array_index = int(entry["array_index"])
        audit_path, audit = audits_by_index[array_index]
        if audit.get("status") != "COMPLETE":
            raise RuntimeError(f"fresh shard audit incomplete: {array_index}")
        if not all(audit["checks"].values()):
            raise RuntimeError(f"fresh shard audit check failed: {array_index}")
        if audit.get("manifest") != entry["manifest"]:
            raise RuntimeError(f"fresh shard audit manifest mismatch: {array_index}")
        measurements = audit["measurements"]
        for key in SUM_KEYS:
            aggregate[key] = int(aggregate[key]) + int(measurements[key])
        collection_rate = float(
            measurements["collection_physical_transitions_per_second"]
        )
        write_rate = float(measurements["write_bytes_per_second"])
        aggregate["collection_rollout_equivalent_seconds"] = float(
            aggregate["collection_rollout_equivalent_seconds"]
        ) + int(measurements["transitions"]) / collection_rate
        aggregate["artifact_write_equivalent_seconds"] = float(
            aggregate["artifact_write_equivalent_seconds"]
        ) + int(measurements["artifact_bytes"]) / write_rate
        task_summary = task_totals[str(audit["task"])]
        task_summary["shards"] = int(task_summary["shards"]) + 1
        for key in SUM_KEYS:
            task_summary[key] = int(task_summary[key]) + int(measurements[key])
        index_entries.append({**entry, "audit": relative_to_root(audit_path)})

    expected_roots = sum(int(task["root_count"]) for task in config["tasks"])
    if int(aggregate["independent_roots"]) != expected_roots:
        raise RuntimeError("fresh aggregate root count differs from the config")
    if int(aggregate["artifact_bytes"]) != int(plan["totals"]["artifact_bytes"]):
        raise RuntimeError("fresh aggregate artifact byte count differs from the plan")
    aggregate["sum_transitions_per_sum_rollout_second"] = int(
        aggregate["transitions"]
    ) / float(aggregate["collection_rollout_equivalent_seconds"])
    aggregate["sum_artifact_bytes_per_sum_write_second"] = int(
        aggregate["artifact_bytes"]
    ) / float(aggregate["artifact_write_equivalent_seconds"])
    release_index = {
        "schema_version": 1,
        "status": "COMPLETE",
        "config": relative_to_root(config_path),
        "split_manifest": config["split_manifest"],
        "audit_plan": relative_to_root(plan_path),
        "shard_audit_root": relative_to_root(audit_root),
        "shard_count": expected_count,
        "all_shards_independently_audited": True,
        "aggregate": aggregate,
        "tasks": dict(sorted(task_totals.items())),
        "entries": index_entries,
        "audit_errors": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, release_index)
    lines = [
        "# Fresh corpus aggregate audit",
        "",
        f"- Shards: {expected_count}",
        f"- Independent roots: {aggregate['independent_roots']}",
        f"- Physical transitions: {aggregate['transitions']}",
        f"- Lazy windows: {aggregate['lazy_windows']}",
        f"- Tier 2 roots: {aggregate['tier2_roots']}",
        f"- Compressed bytes: {aggregate['artifact_bytes']}",
        "- Audit errors: none",
    ]
    atomic_write_text(output_path.with_suffix(".md"), "\n".join(lines) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
