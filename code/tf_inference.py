"""TensorFlow inference with the SavedModels written by export_tf.py.

    python tf_inference.py --test_dir test --weights tf_export --output predictions_tf.csv

Same contract as inference_script.py (image_id,label,confidence with confidence = P(fake)),
using only TensorFlow, NumPy and Pillow.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_image(path, size):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--weights", required=True, help="folder written by export_tf.py")
    p.add_argument("--output", default="predictions.csv")
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args(argv)
    root = Path(a.weights)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    mean = np.array(manifest["mean"], dtype=np.float32)
    std = np.array(manifest["std"], dtype=np.float32)
    members = []
    for member in manifest["members"]:
        fn = tf.saved_model.load(str(root / member["saved_model"])).signatures["serving_default"]
        members.append((fn, list(fn.structured_input_signature[1])[0]))
    paths = sorted(q for q in Path(a.test_dir).iterdir() if q.suffix.lower() in SUFFIXES)
    rows = []
    for path in paths:
        try:
            x = (load_image(path, manifest["size"]) - mean) / std
            views = [x[None], x[None, :, ::-1]]
            logits = [float(list(fn(**{name: tf.constant(np.ascontiguousarray(v))}).values())[0].numpy().reshape(-1)[0])
                      for fn, name in members for v in views]
            confidence = 1 / (1 + np.exp(-np.mean(logits)))
        except (OSError, ValueError):
            confidence = 0.5
        rows.append((path.stem, int(confidence >= a.threshold), f"{confidence:.6f}"))
    with open(a.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "label", "confidence"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.output}")


if __name__ == "__main__":
    main()
