"""Persistent wall-clock cap shared by every experiment in one Stage A run."""
import json
import os
import time
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


def fsync_directory(path):
    if os.name != "nt":
        descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


class Budget:
    def __init__(self, path, minutes=105, resume=False, initial_spent_seconds=0.):
        if not 0 < minutes <= 120:
            raise ValueError("Stage A shared budget must be >0 and <=120 minutes")
        self.path = Path(path)
        now = time.time()
        if self.path.exists():
            if not resume:
                raise ValueError("Run already exists: use --resume without changing its budget/configuration")
            record = json.loads(self.path.read_text(encoding="utf-8"))
            if record["limit_seconds"] != minutes * 60:
                raise ValueError("Budget cannot be changed on resume")
            self.spent = record["spent_seconds"]
            if record["active"]:
                # A killed job may have kept consuming GPU time after its last heartbeat.
                self.spent += max(0, now - record["heartbeat_utc"])
            else:
                self.spent += initial_spent_seconds
        else:
            if not 0 <= initial_spent_seconds < minutes * 60:
                raise ValueError("Startup allocation already exhausted this step's cap")
            self.spent = float(initial_spent_seconds)
        self.limit = minutes * 60
        self.started = time.monotonic()
        self.last_write = 0.
        self.closed = False
        self.heartbeat(force=True)

    @property
    def elapsed(self):
        return self.spent + time.monotonic() - self.started

    @property
    def remaining(self):
        return max(0., self.limit - self.elapsed)

    def heartbeat(self, force=False):
        if force or time.monotonic() - self.last_write > 15:
            write_json(self.path, {"limit_seconds": self.limit, "spent_seconds": self.elapsed,
                                  "active": not self.closed, "heartbeat_utc": time.time()})
            self.last_write = time.monotonic()

    def check(self, reserve=30):
        self.heartbeat()
        if self.remaining < reserve:
            raise BudgetExceeded("Shared Stage A wall-clock budget exhausted; incomplete evaluations are not selectable")

    def close(self):
        self.closed = True
        self.heartbeat(force=True)
