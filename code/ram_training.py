"""Decode and verify training images once, retaining Stage A augmentation semantics."""
import hashlib
import io
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from augment import TrainAugment
from budget import write_json


def _records(rows):
    if not {"image_id", "label", "path"}.issubset(rows.columns) or rows.empty:
        raise ValueError("Training rows require nonempty image_id, label and path columns")
    records = list(rows[["image_id", "label", "path"]].itertuples(index=False, name=None))
    seen = set()
    for image_id, label, path in records:
        if (not isinstance(image_id, str) or not image_id or "/" in image_id
                or "\\" in image_id or image_id in {".", ".."} or image_id in seen):
            raise ValueError("Training image IDs must be unique, nonempty safe strings")
        if isinstance(label, (bool, np.bool_)) or label not in (0, 1):
            raise ValueError(f"Non-binary training label for {image_id}")
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError(f"Missing training image path for {image_id}")
        seen.add(image_id)
    return records


def _expected(value):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("version") != 1
            or value.get("decode") != "exif_rgb_bilinear_224"
            or not isinstance(value.get("samples"), list)):
        raise ValueError("Invalid RAM training manifest")
    seen = set()
    for sample in value["samples"]:
        if not isinstance(sample, dict) or set(sample) != {"image_id", "label", "sha256"}:
            raise ValueError("Invalid RAM training sample manifest")
        image_id, digest = sample["image_id"], sample["sha256"]
        if (not isinstance(image_id, str) or image_id in seen or not image_id
                or isinstance(sample["label"], bool) or sample["label"] not in (0, 1)
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Invalid RAM training sample identity/checksum")
        seen.add(image_id)
    return value


@dataclass
class RAMTrainingStore:
    images: torch.Tensor
    manifest: dict
    preload_seconds: float

    @property
    def pixels(self):
        return self.images

    @property
    def train_media_sha256(self):
        return {sample["image_id"]: sample["sha256"] for sample in self.manifest["samples"]}


def preload_training(rows, manifest_path=None, expected_manifest=None,
                     check_deadline=None, progress=None, shared_memory=False, decode_workers=1):
    """Read each JPEG once for both SHA256 and decode into contiguous uint8 NHWC.

    Existing manifest_path is validated, never silently replaced. An explicit
    expected_manifest may also be supplied when the input manifest is read-only.
    Default storage uses fork copy-on-write on Linux, avoiding a 1 GiB tmpfs
    allocation for 7,000 images. Use fork DataLoader workers or zero workers.
    Explicit shared tensors also support spawn when shared memory is sufficient.
    """
    started = time.perf_counter()
    records = _records(rows)
    expected = _expected(expected_manifest) if expected_manifest is not None else None
    manifest_path = Path(manifest_path) if manifest_path is not None else None
    if manifest_path is not None and manifest_path.exists():
        stored = _expected(manifest_path)
        if expected is not None and stored != expected:
            raise ValueError("Existing and supplied RAM training manifests differ")
        expected = stored
    expected_by_id = None
    if expected is not None:
        expected_by_id = {sample["image_id"]: sample for sample in expected["samples"]}
        if set(expected_by_id) != {record[0] for record in records}:
            raise ValueError("RAM training manifest IDs differ from requested images")
    if check_deadline is not None:
        check_deadline()
    required_bytes = len(records) * 224 * 224 * 3
    if shared_memory and os.name != "nt":
        shared_root = Path("/dev/shm")
        if not shared_root.is_dir() or shutil.disk_usage(shared_root).free < required_bytes + 16 * 1024**2:
            raise ValueError("Insufficient /dev/shm for shared image tensor; use fork storage")
    images = torch.empty((len(records), 224, 224, 3), dtype=torch.uint8)
    if shared_memory:
        images.share_memory_()
    destination = images.numpy()
    if isinstance(decode_workers, bool) or not isinstance(decode_workers, int) or not 1 <= decode_workers <= 4:
        raise ValueError("Decode workers must be between one and four")

    def decode(record):
        image_id, label, path = record
        try:
            encoded = Path(path).read_bytes()
            digest = hashlib.sha256(encoded).hexdigest()
            sample = {"image_id": image_id, "label": int(label), "sha256": digest}
            if expected_by_id is not None and expected_by_id[image_id] != sample:
                raise ValueError("Image bytes or label differ from expected manifest")
            with Image.open(io.BytesIO(encoded)) as image:
                decoded = ImageOps.exif_transpose(image).convert("RGB").resize(
                    (224, 224), Image.Resampling.BILINEAR)
                return sample, np.asarray(decoded, dtype=np.uint8)
        except Exception as exc:
            raise ValueError(f"Cannot preload training sample {image_id}: {exc}") from exc

    samples = []
    with ThreadPoolExecutor(max_workers=decode_workers) as executor:
        for index, (sample, pixels) in enumerate(executor.map(decode, records)):
            if check_deadline is not None:
                check_deadline()
            destination[index] = pixels
            samples.append(sample)
            if progress is not None and ((index + 1) % 256 == 0 or index + 1 == len(records)):
                progress({"completed": index + 1, "total": len(records),
                          "elapsed_seconds": time.perf_counter() - started})
    if check_deadline is not None:
        check_deadline()
    manifest = {"version": 1, "decode": "exif_rgb_bilinear_224", "samples": samples}
    if manifest_path is not None and not manifest_path.exists():
        write_json(manifest_path, manifest)
    return RAMTrainingStore(images, manifest, time.perf_counter() - started)


class RAMTrainDataset(Dataset):
    """Select fold rows by sample ID, applying the unchanged PIL TrainAugment."""
    def __init__(self, store, rows=None, degradation=False, colour="strong"):
        self.store = store
        samples = store.manifest["samples"]
        if (store.images.dtype != torch.uint8 or not store.images.is_contiguous()
                or tuple(store.images.shape) != (len(samples), 224, 224, 3)):
            raise ValueError("Invalid RAM training image tensor")
        lookup = {sample["image_id"]: index for index, sample in enumerate(samples)}
        if len(lookup) != len(samples):
            raise ValueError("Duplicate RAM training image IDs")
        if rows is None:
            self.indices = list(range(len(samples)))
        else:
            records = _records(rows)
            self.indices = []
            for image_id, label, _ in records:
                if image_id not in lookup or samples[lookup[image_id]]["label"] != label:
                    raise ValueError(f"Fold image ID/label absent from RAM store: {image_id}")
                self.indices.append(lookup[image_id])
        self.augment = TrainAugment(degradation, colour)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        position = self.indices[index]
        sample = self.store.manifest["samples"][position]
        image = Image.fromarray(self.store.images[position].numpy())
        image = self.augment(image)
        return (torch.from_numpy(image).permute(2, 0, 1),
                torch.tensor(float(sample["label"])), sample["image_id"])


def worker_count(available_cores=None):
    """Keep two augmentation workers where the CPU allocation permits it."""
    cores = available_cores if available_cores is not None else (os.cpu_count() or 1)
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise ValueError("CPU core count must be a positive integer")
    return min(2, cores) if os.name != "nt" else 0
