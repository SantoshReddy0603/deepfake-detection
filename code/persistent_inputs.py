"""Read training inputs directly; never copy initialization or caches into working."""
import hashlib
from pathlib import Path
import time

import torch

from data import load_split
from validation_cache import verify_cache


def load_inputs(data_root, split, cache_dir, initialization, budget, smoke=False, verified_input=False):
    timings = {}
    began = time.perf_counter()
    budget.emit("phase_started", phase="data_discovery")
    training, validation = load_split(data_root, split, smoke=smoke)
    timings["data_discovery_seconds"] = time.perf_counter() - began
    budget.emit("phase_completed", phase="data_discovery", seconds=timings["data_discovery_seconds"],
                train_images=len(training), validation_images=len(validation))
    budget.emit("phase_started", phase="cache_load", path=str(Path(cache_dir).resolve()), writes_to_input=False)
    smoke_indices = None
    if smoke and not verified_input:
        # Production verifies all images; smoke rehashes just the128 it uses.
        selected = {i for label in [0, 1] for i in validation[validation.label == label].sample(64, random_state=42).index}
        smoke_indices = sorted(selected)
    manifest = verify_cache(validation, cache_dir, check_deadline=budget.check, timings=timings,
                            smoke_media_indices=smoke_indices, verified_input=verified_input,
                            progress=lambda message: budget.emit("cache_verification", message=message))
    budget.emit("phase_completed", phase="cache_load", seconds=timings["cache_load_seconds"],
                conditions=len(manifest["conditions"]), cache_key=manifest["key"])
    if verified_input:
        print(f"STATUS: mounted cache load and verification {timings['cache_load_seconds']:.3f}s "
              f"({len(manifest['conditions'])} conditions, {Path(cache_dir).resolve()})", flush=True)
    began = time.perf_counter()
    path = Path(initialization).resolve()
    budget.emit("phase_started", phase="initialization_input_load", path=str(path), writes_to_input=False)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            budget.check()
            digest.update(chunk)
    sha256 = digest.hexdigest()
    if not sha256.startswith("983f1562"):
        raise ValueError("Expected the permitted ConvNeXt-Tiny ImageNet initialization hash")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or tuple(state["classifier.2.weight"].shape) != (1000, 768):
        raise ValueError("Unexpected ConvNeXt-Tiny ImageNet state structure")
    del state
    budget.check()
    timings["initialization_input_load_seconds"] = time.perf_counter() - began
    budget.emit("phase_completed", phase="initialization_input_load", seconds=timings["initialization_input_load_seconds"], sha256=sha256)
    return training, validation, manifest, timings, sha256
