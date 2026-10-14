from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .manifest import atomic_write_json


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(seconds)))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    if hours:
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


class ProgressReporter:
    """Publish human-readable stdout and atomic machine-readable heartbeats."""

    def __init__(
        self,
        *,
        task: str,
        source: str,
        shard_id: str,
        total: int,
        unit: str,
        heartbeat_path: Path,
        log_every_seconds: float = 30.0,
    ) -> None:
        if total <= 0:
            raise ValueError("total must be positive")
        if log_every_seconds <= 0:
            raise ValueError("log_every_seconds must be positive")
        self.task = task
        self.source = source
        self.shard_id = shard_id
        self.total = total
        self.unit = unit
        self.heartbeat_path = heartbeat_path
        self.log_every_seconds = log_every_seconds
        self.started_monotonic = time.monotonic()
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.last_log_monotonic = float("-inf")

    def report(
        self,
        *,
        completed: int,
        transitions: int,
        output_bytes: int,
        status: str = "RUNNING",
        force: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if completed < 0 or completed > self.total:
            raise ValueError(f"completed must be in [0, {self.total}]")
        if transitions < 0 or output_bytes < 0:
            raise ValueError("transitions and output_bytes must be non-negative")

        now_monotonic = time.monotonic()
        elapsed = now_monotonic - self.started_monotonic
        rate = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
        eta_seconds = (
            (self.total - completed) / rate if rate > 0 and completed < self.total else 0.0
            if completed == self.total
            else None
        )
        payload: dict[str, Any] = {
            "status": status,
            "task": self.task,
            "source": self.source,
            "shard_id": self.shard_id,
            "unit": self.unit,
            "completed": completed,
            "total": self.total,
            "percent": 100.0 * completed / self.total,
            "transitions": transitions,
            "output_bytes": output_bytes,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
            "units_per_second": rate,
            "started_at": self.started_at,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hostname": os.uname().nodename,
        }
        if extra:
            payload["extra"] = extra
        atomic_write_json(self.heartbeat_path, payload)

        should_log = (
            force
            or completed == 0
            or completed == self.total
            or now_monotonic - self.last_log_monotonic >= self.log_every_seconds
        )
        if should_log:
            print(
                "[progress] "
                f"task={self.task} source={self.source} shard={self.shard_id} "
                f"{self.unit}={completed}/{self.total} ({payload['percent']:.1f}%) "
                f"transitions={transitions} elapsed={format_duration(elapsed)} "
                f"eta={format_duration(eta_seconds)} rate={rate:.3f}_{self.unit}/s "
                f"output_bytes={output_bytes} status={status}",
                flush=True,
            )
            self.last_log_monotonic = now_monotonic
        return payload
