"""Export a PyTorch bundle to TensorFlow SavedModels and check that both give the same scores.

    python export_tf.py --weights model_weights.pt --out tf_export [--check_dir some_images]

Each ensemble member goes PyTorch -> ONNX -> TensorFlow SavedModel (one folder per member,
NHWC float input, already normalised, one logit out). tf_export/manifest.json lists the
members and the fusion rule, so tf_inference.py can reproduce the PyTorch predictions:
mean member logit over the original and mirrored image, then a sigmoid.

Tested with tensorflow 2.19, tf_keras 2.19, onnx 1.17, onnx2tf 1.29 (plus onnxsim,
onnx_graphsurgeon, sng4onnx, psutil).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from data import open_rgb
from model_def import NORMALIZATION, build_model


def load_members(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 3:
        raise ValueError("Expected a logit-ensemble bundle (format 3)")
    for index, member in enumerate(payload["members"]):
        model = build_model(member["architecture"], pretrained=False, dropout=member["dropout"])
        model.load_state_dict({k: v.float() if v.is_floating_point() else v
                               for k, v in member["state_dict"].items()}, strict=True)
        yield f"{index}_{member['architecture']}", model.eval()


class ExportGELU(torch.nn.Module):
    """Exact GELU written with Mul/Erf only, which the converter handles reliably."""

    def __init__(self):
        super().__init__()
        self.register_buffer("gelu_half", torch.full((1,), 0.5))
        self.register_buffer("gelu_scale", torch.full((1,), 0.7071067811865476))

    def forward(self, x):
        y = x * self.gelu_half
        return y + y * torch.erf(x * self.gelu_scale)


def make_exportable(model):
    for module in model.modules():
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU) or type(child).__name__ == "GELU":  # timm has its own class
                setattr(module, name, ExportGELU())
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    return model


def preprocess(paths):
    mean = np.array(NORMALIZATION["mean"], dtype=np.float32)
    std = np.array(NORMALIZATION["std"], dtype=np.float32)
    batch = np.stack([np.asarray(open_rgb(p), dtype=np.float32) / 255. for p in paths])
    return (batch - mean) / std  # NHWC


def tf_logits(saved_model_dir, nhwc):
    import tensorflow as tf
    fn = tf.saved_model.load(str(saved_model_dir)).signatures["serving_default"]
    name = list(fn.structured_input_signature[1])[0]
    out = [list(fn(**{name: tf.constant(nhwc[i:i + 1])}).values())[0].numpy().reshape(-1)[0]
           for i in range(len(nhwc))]
    return np.array(out, dtype=np.float64)


def skip_tflite():
    """The converter also writes TFLite files after the SavedModel. They are not needed here, and
    building them takes most of the time and memory, so that step is replaced by a no-op."""
    import types

    import tensorflow as tf

    class Skipped:
        def __init__(self, *args, **kwargs):
            self.target_spec = types.SimpleNamespace()

        def convert(self):
            return b""

    for factory in ("from_concrete_functions", "from_keras_model", "from_saved_model"):
        setattr(tf.lite.TFLiteConverter, factory, staticmethod(lambda *args, **kwargs: Skipped()))


def convert_member(a, index, nhwc):
    """Convert one member and return its PyTorch and TensorFlow logits on the check inputs."""
    import onnx
    import onnx2tf
    import onnxsim

    out = Path(a.out)
    name, model = next(m for i, m in enumerate(load_members(a.weights)) if i == index)
    nchw = torch.from_numpy(nhwc).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        reference = model(nchw).reshape(-1).double().numpy()
    make_exportable(model)
    onnx_path = out / f"{name}.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(onnx_path), opset_version=a.opset,
                      input_names=["image"], output_names=["logit"], dynamo=False)
    simplified, ok = onnxsim.simplify(onnx.load(str(onnx_path)))  # folds the shape arithmetic
    if not ok:
        raise RuntimeError(f"ONNX simplification failed for {name}")
    onnx.save(simplified, str(onnx_path))
    saved = out / name
    if not (saved / "saved_model.pb").exists():
        skip_tflite()
        onnx2tf.convert(input_onnx_file_path=str(onnx_path), output_folder_path=str(saved),
                        output_signaturedefs=True, non_verbose=True)
    onnx_path.unlink()
    for extra in saved.glob("*.tflite"):
        extra.unlink()
    converted = tf_logits(saved, nhwc)
    result = {"name": name, "torch": reference.tolist(), "tf": converted.tolist()}
    (out / f"{name}.check.json").write_text(json.dumps(result), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--check_dir", help="folder of images used for the PyTorch/TensorFlow parity check")
    p.add_argument("--check_images", type=int, default=32)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--member", type=int, help="convert only this member (used internally, one process each)")
    a = p.parse_args(argv)

    out = Path(a.out).resolve()
    a.out, a.weights = str(out), str(Path(a.weights).resolve())
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    if a.check_dir:
        a.check_dir = str(Path(a.check_dir).resolve())
        paths = sorted(q for q in Path(a.check_dir).iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
        nhwc = preprocess(paths[:a.check_images])
    else:
        nhwc = rng.standard_normal((4, 224, 224, 3)).astype(np.float32)

    if a.member is not None:
        # the converter checks itself on a small image sample that it otherwise downloads
        sample = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
        if not sample.exists():
            np.save(sample, rng.random((20, 128, 128, 3), dtype=np.float32))
        convert_member(a, a.member, nhwc)
        return

    payload = torch.load(a.weights, map_location="cpu", weights_only=True)
    count = len(payload["members"])
    del payload
    manifest = {"input": "NHWC float32, RGB/255 then ImageNet mean/std", "size": NORMALIZATION["size"],
                "mean": NORMALIZATION["mean"], "std": NORMALIZATION["std"],
                "fusion": "sigmoid(mean member logit over identity and horizontal flip)", "members": []}
    torch_total, tf_total = 0, 0
    with tempfile.TemporaryDirectory() as scratch:  # one process per member keeps memory flat
        for index in range(count):
            command = [sys.executable, str(Path(__file__).resolve()), "--weights", a.weights, "--out", a.out,
                       "--check_images", str(a.check_images), "--opset", str(a.opset), "--member", str(index)]
            if a.check_dir:
                command += ["--check_dir", a.check_dir]
            subprocess.run(command, check=True, cwd=scratch, stdout=subprocess.DEVNULL)
            check = next(out.glob(f"{index}_*.check.json"))
            result = json.loads(check.read_text(encoding="utf-8"))
            check.unlink()
            reference, converted = np.array(result["torch"]), np.array(result["tf"])
            gap = float(np.abs(reference - converted).max())
            manifest["members"].append({"name": result["name"], "saved_model": result["name"], "max_logit_gap": gap})
            torch_total, tf_total = torch_total + reference, tf_total + converted
            print(f"{result['name']}: max |logit gap| {gap:.2e} on {len(nhwc)} inputs "
                  f"(mean |logit| {np.abs(reference).mean():.2f})", flush=True)
    p_torch, p_tf = 1 / (1 + np.exp(-torch_total / count)), 1 / (1 + np.exp(-tf_total / count))
    manifest["parity"] = {"inputs": int(len(nhwc)), "real_images": bool(a.check_dir),
                          "max_probability_gap": float(np.abs(p_torch - p_tf).max())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"ensemble: max |probability gap| {manifest['parity']['max_probability_gap']:.2e}; wrote {out}")


if __name__ == "__main__":
    main()
