"""Create 16 deterministic disk-backed validation arrays once; epochs only read them."""
import hashlib
import json
from pathlib import Path
import os
import time
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
from augment import CONDITIONS, CACHE_RECIPE_VERSION, degrade
from data import fingerprint, open_rgb


def expected_contract(rows, seed, check_deadline=None, cached_media=None, media_indices=None):
    media = []
    for index, row in enumerate(rows.itertuples()):
        if check_deadline:
            check_deadline()
        if media_indices is None or index in media_indices:
            sha256 = fingerprint(row.path)
        else:
            sample = cached_media[index]
            if sample["image_id"] != row.image_id or sample["label"] != int(row.label):
                raise ValueError("Cached sample ID/label alignment changed")
            sha256 = sample["sha256"]
        media.append({"image_id": row.image_id, "label": int(row.label), "sha256": sha256})
    content = {"recipe_version": CACHE_RECIPE_VERSION, "conditions": list(CONDITIONS), "seed": seed,
               "pillow": PIL.__version__, "numpy": np.__version__, "samples": media, "size": 224}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return content, digest


def verify_cache(rows, directory, seed=42, progress=print, check_deadline=None, timings=None, smoke_media_indices=None, verified_input=False):
    """Read-only verification; never creates arrays or writes into a mounted input."""
    started = time.perf_counter()
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Persistent validation cache manifest missing; run CPU preparation separately")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if smoke_media_indices is not None:
        if not smoke_media_indices or len(set(smoke_media_indices)) != len(smoke_media_indices) or any(
                type(i) is not int or not 0 <= i < len(rows) for i in smoke_media_indices):
            raise ValueError("Invalid engineering-smoke media indices")
    if verified_input:
        # Mounted, previously verified arrays already contain the rendered pixels.
        # Bind IDs/labels and recipe without opening source JPEGs or requiring the build libraries.
        samples = manifest["samples"]
        if len(samples) != len(rows):
            raise ValueError("Mounted cache sample count changed")
        contract, _ = expected_contract(rows, seed, check_deadline, samples, set())
        contract.update(pillow=manifest["pillow"], numpy=manifest["numpy"])
        key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    else:
        contract, key = expected_contract(rows, seed, check_deadline, manifest.get("samples"),
                                         set(smoke_media_indices) if smoke_media_indices is not None else None)
    contract_seconds = time.perf_counter() - started
    if manifest["key"] != key or any(manifest.get(k) != v for k, v in contract.items()):
        raise ValueError("Validation cache content/seed/library/recipe changed; use a new cache directory")
    checksums_started = time.perf_counter()
    for name in CONDITIONS:
        if check_deadline:
            check_deadline()
        condition_started = time.perf_counter()
        array = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        size = (directory / f"{name}.npy").stat().st_size
        if (array.shape != (len(rows), 224, 224, 3) or array.dtype != np.uint8
                or size != array.offset + array.nbytes):
            raise ValueError(f"Invalid cached array: {name}")
        del array
        digest = hashlib.sha256()
        with (directory / f"{name}.npy").open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                if check_deadline:
                    check_deadline()
                digest.update(chunk)
        if digest.hexdigest() != manifest.get("array_sha256", {}).get(name):
            raise ValueError(f"Validation cache checksum mismatch: {name}")
        progress(f"Verified cached condition {name}: {time.perf_counter() - condition_started:.3f}s")
    if timings is not None:
        timings.update(image_contract_seconds=contract_seconds,
                       image_media_hashes_verified=0 if verified_input else (len(rows) if smoke_media_indices is None else len(smoke_media_indices)),
                       verification_scope=("mounted cache size and SHA256 only" if verified_input else
                                           ("full validation partition" if smoke_media_indices is None else "engineering smoke subset; all16 array hashes still checked")),
                       array_checksum_seconds=time.perf_counter() - checksums_started,
                       cache_load_seconds=time.perf_counter() - started)
    return manifest


def build_cache(rows, directory, seed=42, progress=print, check_deadline=None):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        return verify_cache(rows, directory, seed, progress, check_deadline)
    contract, key = expected_contract(rows, seed, check_deadline)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Incomplete validation cache: preserve it for diagnosis and choose a fresh directory")
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype="uint8",
                                             shape=(len(rows), 224, 224, 3)) for name in CONDITIONS}
    for i, row in enumerate(rows.itertuples()):
        if check_deadline:
            check_deadline()
        image = open_rgb(row.path)
        for name in CONDITIONS:
            arrays[name][i] = np.asarray(degrade(image, name, row.image_id, seed), dtype=np.uint8)
        if (i + 1) % 100 == 0:
            progress(f"Cached {i + 1}/{len(rows)} validation images across all 16 conditions")
    for array in arrays.values():
        array.flush()
    del arrays
    checksums = {}
    for name in CONDITIONS:
        if check_deadline:
            check_deadline()
        checksums[name] = fingerprint(directory / f"{name}.npy")
    manifest = {"key": key, **contract, "array_sha256": checksums,
                "bytes_uint8_payload": len(rows) * 16 * 224 * 224 * 3}
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


class CachedDataset(Dataset):
    def __init__(self, directory, condition):
        self.directory, self.condition = str(directory), condition
        manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
        if condition not in manifest["conditions"]:
            raise ValueError("Condition not in cache")
        self.ids = [r["image_id"] for r in manifest["samples"]]
        self.labels = [r["label"] for r in manifest["samples"]]
        self._array = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self._array is None:
            self._array = np.load(Path(self.directory) / f"{self.condition}.npy", mmap_mode="r", allow_pickle=False)
        return torch.from_numpy(np.array(self._array[index], copy=True)).permute(2, 0, 1), self.labels[index], self.ids[index]
