"""Project-root resolution.

All data, checkpoints, and results live under a single root directory that is
selected with the ``PAP_ROOT`` environment variable (default: current directory).

Layout under the root::

    data/corpus/{fresh,fresh_extension,historical,branched}/   pretraining interaction corpus
    data/manifests/                            corpus and split manifests
    data/organization/                         physical-organization study data
    data/query_invoke/                         Query/Invoke study data
    checkpoints/backbone/{50m,200m,800m}/      pretrained backbones
    checkpoints/teachers/<task>/               PPO teacher policies used for collection
    checkpoints/pap/                           PAP response models
    results/                                   evaluation outputs
"""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(os.environ.get("PAP_ROOT", ".")).resolve()


def resolve(*parts: str | os.PathLike) -> Path:
    """Return ``PAP_ROOT/<parts>``."""
    return project_root().joinpath(*parts)


def require_within_project(path: Path) -> Path:
    """Resolve ``path`` and ensure it stays inside the project root."""
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(project_root())
    except ValueError as exc:
        raise RuntimeError(f"path escapes project root: {resolved!s}") from exc
    return resolved
