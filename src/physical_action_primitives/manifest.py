from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .paths import require_within_project


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish metadata only after the payload is complete."""
    destination = require_within_project(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
