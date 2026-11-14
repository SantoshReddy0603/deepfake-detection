"""Offline image inference. Predictions come only from exported model weights and input pixels."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from model_def import load_export, normalize


class InputImages(Dataset):
    def __init__(self, directory):
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError("test_dir must be an existing directory")
        self.paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.ids = [p.stem for p in self.paths]
        if not self.paths:
            raise ValueError("test_dir contains no supported images")
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Duplicate image stems across file extensions")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        error = ""
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                array = np.array(image, dtype=np.uint8, copy=True)
        except (OSError, ValueError) as exc:
            error = f"{path.name}: {type(exc).__name__}: {exc}"
            array = np.zeros((224, 224, 3), dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1), self.ids[index], error


class LoaderFailure(RuntimeError):
    pass


def safe_batches(loader):
    try:
        iterator = iter(loader)
    except (RuntimeError, OSError) as exc:
        raise LoaderFailure(str(exc)) from exc
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            return
        except (RuntimeError, OSError) as exc:
            raise LoaderFailure(str(exc)) from exc
        yield batch


@torch.inference_mode()
def model_probabilities(model, x, device, backoffs):
    try:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(normalize(x, device)).reshape(-1).float()
            probabilities = output if getattr(model, "outputs_probabilities", False) else output.sigmoid()
        return probabilities.cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        if device.type != "cuda" or len(x) == 1:
            raise
        backoffs.append(len(x))
        torch.cuda.empty_cache()
        middle = len(x) // 2
        return np.concatenate([model_probabilities(model, x[:middle], device, backoffs),
                               model_probabilities(model, x[middle:], device, backoffs)])


def validate_rows(rows, expected_ids, threshold):
    if len(rows) != len(expected_ids) or [r[0] for r in rows] != list(expected_ids):
        raise ValueError("Prediction count/order/IDs differ from actual input directory")
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    for image_id, label, confidence in rows:
        if type(label) is not int or label not in {0, 1} or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid prediction label/probability")
        if label != int(confidence >= threshold):
            raise ValueError("Threshold mismatch")


def infer(test_dir, weights, output, batch_size=64, threshold=None, workers=2, device=None):
    started = time.perf_counter()
    if batch_size < 1 or workers < 0:
        raise ValueError("Batch size must be positive and workers nonnegative")
    if Path(weights).stat().st_size >= 5_000_000_000:
        raise ValueError("Weights violate the 5 GB competition limit")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, metadata = load_export(weights, device)
    threshold = float(metadata["threshold"] if threshold is None else threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0,1]")
    dataset = InputImages(test_dir)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    backoffs, errors = [], []
    def predict(worker_count):
        rows, issues = [], []
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=worker_count, shuffle=False,
                            pin_memory=device.type == "cuda", persistent_workers=False)
        for x, ids, failures in safe_batches(loader):
            probabilities = model_probabilities(model, x, device, backoffs)
            for image_id, p, failure in zip(ids, probabilities, failures):
                value = .5 if failure else float(p)
                if failure:
                    issues.append(failure)
                rows.append((image_id, int(value >= threshold), value))
        return rows, issues
    try:
        rows, errors = predict(workers)
    except LoaderFailure as exc:
        if workers == 0:
            raise
        warnings.warn(f"DataLoader worker failure; retrying deterministically with workers=0: {exc}")
        rows, errors = predict(0)
    validate_rows(rows, dataset.ids, threshold)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "label", "confidence"])
        writer.writerows((image_id, label, format(p, ".17g")) for image_id, label, p in rows)
    temporary.replace(output)
    with open(output, newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if next(reader) != ["image_id", "label", "confidence"]:
            raise ValueError("Saved submission has incorrect header order")
        reloaded = [(r[0], int(r[1]), float(r[2])) for r in reader]
    validate_rows(reloaded, dataset.ids, threshold)
    for issue in errors:
        warnings.warn("Unreadable image; emitted neutral confidence 0.5: " + issue)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {"images": len(rows), "fake_count": sum(r[1] for r in rows), "threshold": threshold,
               "device": str(device), "precision": "CUDA autocast FP16" if device.type == "cuda" else "CPU FP32",
               "elapsed_seconds_including_model_load_and_csv": elapsed,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "corrupt_image_count": len(errors), "oom_batch_backoffs": backoffs,
               "export_purpose": metadata.get("purpose"), "output": str(output)}
    print(json.dumps(summary), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    infer(args.test_dir, args.weights, args.output, args.batch_size, args.threshold, args.workers)


if __name__ == "__main__":
    main()
