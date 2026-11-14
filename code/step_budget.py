"""Independent per-step runner timer. Estimates are never Kaggle charge claims."""
import json
import math
import os
from pathlib import Path
import time

from budget import BudgetExceeded, write_json


class StepBudget:
    def __init__(self, path, seconds, scope="stage_b_step1", resume=False, allocated_gpus=0, startup_seconds=0.):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Step wall-time cap must be positive and finite")
        self.path, self.scope = Path(path), scope
        self.limit, self.allocated_gpus = float(seconds), int(allocated_gpus)
        if not math.isfinite(startup_seconds) or startup_seconds < 0:
            raise ValueError("Startup estimate must be finite and nonnegative")
        self.spent = startup_seconds
        if self.path.exists():
            if not resume:
                raise ValueError("Step ledger exists; preserve it and use exact --resume, never reset it")
            old = json.loads(self.path.read_text(encoding="utf-8"))
            if old.get("scope") != scope or old.get("limit_seconds") != self.limit:
                raise ValueError("Resume cannot change step scope or cap")
            if old.get("allocated_gpus") != self.allocated_gpus:
                raise ValueError("Resume cannot change the fixed allocation")
            if old.get("active"):
                raise ValueError("Unclean step termination: reconcile the saved ledger before resume")
            self.spent += float(old["runner_elapsed_seconds"])
            if not math.isfinite(self.spent) or self.spent < 0:
                raise ValueError("Invalid saved step elapsed time")
        elif resume:
            raise ValueError("Cannot resume without this step's own ledger")
        self.started = time.monotonic()
        self.closed = False
        self.stopped_elapsed = None
        self.last_write = self.started
        self.events = self.path.parent / "events.jsonl"
        self.heartbeat(force=True)
        self.emit("step_budget_started", new_step=not resume, initial_runner_seconds=self.spent,
                  wall_cap_seconds=self.limit, accounting_label="runner estimate")

    @property
    def elapsed(self):
        return self.stopped_elapsed if self.closed else self.spent + time.monotonic() - self.started

    @property
    def remaining(self):
        return max(0., self.limit - self.elapsed)

    def emit(self, event, **values):
        if "scope" in values:
            values["timing_scope"] = values.pop("scope")
        item = {"event": event, "scope": self.scope, "runner_seconds": self.elapsed, **values}
        line = json.dumps(item, allow_nan=False)
        self.events.parent.mkdir(parents=True, exist_ok=True)
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(line, flush=True)

    def heartbeat(self, force=False):
        if force or time.monotonic() - self.last_write >= 2:
            write_json(self.path, {"scope": self.scope, "limit_seconds": self.limit,
                       "runner_elapsed_seconds": self.elapsed, "active": not self.closed,
                       "allocated_gpus": self.allocated_gpus,
                       "runner_estimate_gpu_minutes": self.elapsed * self.allocated_gpus / 60,
                       "verified_kaggle_charge": False})
            self.last_write = time.monotonic()

    def check(self, reserve=5):
        self.heartbeat()
        if self.remaining <= reserve:
            raise BudgetExceeded(f"{self.scope}: STEP BUDGET EXHAUSTED; runner elapsed {self.elapsed:.3f}s / {self.limit:.3f}s; stop (exit 124)")

    def close(self):
        if not self.closed:
            self.stopped_elapsed = self.elapsed
            self.closed = True
            self.heartbeat(force=True)


def fresh_session_budget(directory,seconds=28800,scope="session",allocated_gpus=1,startup_seconds=0.):
    """Start a new session wall clock; cumulative time is reporting only."""
    import uuid
    directory=Path(directory);sessions=directory/'sessions';sessions.mkdir(parents=True,exist_ok=True)
    paths=list(sessions.glob('*/step_budget.json'))
    paths.extend(directory/name for name in ['step_budget.json','wall_budget.json'] if (directory/name).exists())
    prior=0.
    for path in paths:
        elapsed=float(json.loads(path.read_text())['runner_elapsed_seconds'])
        if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid prior session elapsed time')
        prior+=elapsed
    current=sessions/('session_'+uuid.uuid4().hex);current.mkdir()
    budget=StepBudget(current/'step_budget.json',seconds,scope=scope,allocated_gpus=allocated_gpus,startup_seconds=startup_seconds)
    write_json(directory/'session_summary.json',{'current_session':current.name,'prior_cumulative_runner_seconds':prior,'prior_time_used_for_stopping':False})
    return budget,prior
