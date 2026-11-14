"""Official training adapter. This module does not open or inspect test images."""
import hashlib
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from augment import TrainAugment


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(data_root, split_csv, allow_small=False, smoke=False):
    root = Path(data_root).resolve()
    if not allow_small and any((p / ".software_test_only").exists() for p in [root, *root.parents]):
        raise ValueError("Software fixtures cannot be selected for official training")
    metadata_path = root / "train_metadata.csv"
    if not metadata_path.is_file():
        raise ValueError("data-root must contain official train_metadata.csv and train/real,fake")
    metadata = pd.read_csv(metadata_path, dtype={"image_id": str}, keep_default_na=False)
    if set(metadata.columns) != {"image_id", "label"} or metadata.empty:
        raise ValueError("Expected official image_id,label metadata")
    if metadata.image_id.duplicated().any() or not metadata.label.isin([0, 1]).all():
        raise ValueError("Duplicate IDs or non-binary train labels")
    if not allow_small and (len(metadata) != 7000 or metadata.label.value_counts().to_dict() != {0: 3478, 1: 3522}):
        raise ValueError("Official train count/class contract changed; inspect rather than silently adapting")
    rows = []
    for row in metadata.itertuples():
        if not row.image_id or Path(row.image_id).name != row.image_id or "/" in row.image_id or "\\" in row.image_id:
            raise ValueError("Unsafe image ID")
        path = (root / "train" / ("fake" if row.label else "real") / f"{row.image_id}.jpg")
        if not smoke:
            path = path.resolve()
        if not smoke and (not path.is_relative_to(root / "train") or not path.is_file()):
            raise ValueError(f"Missing/escaping official train file: {row.image_id}")
        rows.append(str(path))
    metadata["path"] = rows
    fold = pd.read_csv(split_csv, dtype=str, keep_default_na=False)
    if set(fold.columns) != {"image_id", "partition", "group_id"} or fold.image_id.duplicated().any():
        raise ValueError("Invalid fixed fold schema/IDs")
    if set(fold.image_id) != set(metadata.image_id) or not fold.partition.isin(["train", "val"]).all() or fold.group_id.eq("").any():
        raise ValueError("Fixed fold must cover training IDs only, with explicit source-candidate groups")
    result = metadata.merge(fold, on="image_id", validate="one_to_one").sort_values("image_id").reset_index(drop=True)
    training, validation = (result[result.partition == p].reset_index(drop=True) for p in ["train", "val"])
    if set(training.group_id) & set(validation.group_id):
        raise ValueError("Source-candidate group leakage")
    if any(frame.label.nunique() != 2 for frame in [training, validation]):
        raise ValueError("Both partitions must contain both classes")
    if not allow_small and (len(training), len(validation)) != (5600, 1400):
        raise ValueError("Fixed fold0 count changed")
    if smoke:
        # Keep full metadata/group validation, but touch only media used by smoke.
        for frame, count in [(training, 256), (validation, 128)]:
            for label in [0, 1]:
                for row in frame[frame.label == label].sample(count // 2, random_state=42).itertuples():
                    path = Path(row.path).resolve()
                    if not path.is_relative_to(root / "train") or not path.is_file():
                        raise ValueError(f"Missing/escaping smoke file: {row.image_id}")
    return training, validation


def open_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)


class TrainDataset(Dataset):
    def __init__(self, rows, degradation=False, colour="strong"):
        self.rows = rows
        self.augment = TrainAugment(degradation, colour)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        try:
            image = self.augment(open_rgb(row.path))
        except Exception as exc:
            raise ValueError(f"Cannot read training sample {row.image_id}: {exc}") from exc
        return torch.from_numpy(image).permute(2, 0, 1), torch.tensor(float(row.label)), row.image_id


def seed_worker(worker):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
