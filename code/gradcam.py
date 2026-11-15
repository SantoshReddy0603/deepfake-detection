"""Grad-CAM heatmaps for every member of an exported bundle.

    python gradcam.py --weights model_weights.pt --images some_dir --out gradcam/ [--limit 16]

For each image it writes one PNG: the input followed by one overlay per member, plus the
fused fake probability in the file name. The heatmap shows which face regions pushed the
member's logit towards "fake".
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import open_rgb
from model_def import build_model, normalize

CHANNELS_LAST_OUTPUT = {"swin_t", "swin_small_in22k"}
TOKEN_OUTPUT = {"vit_base_in21k", "deit3_base_in22k"}


def target_layer(model, architecture):
    if hasattr(model, "net"):
        model = model.net
        if hasattr(model, "stages"):
            return model.stages[-1]
        return model.conv_head
    if architecture == "convnext_base_in22k":
        return model.stages[-1]
    if architecture == "resnet50":
        return model.layer4
    if architecture == "maxvit_t":
        return model.blocks[-1]
    if architecture in TOKEN_OUTPUT:
        return model.blocks[-1].norm1
    if architecture == "swin_small_in22k":
        return model.layers[-1]
    if architecture in ("convnext_small_in22k", "convnextv2_tiny_in22k"):
        return model.stages[-1]
    return model.features[-1]


def as_feature_map(tensor, architecture):
    if architecture in CHANNELS_LAST_OUTPUT:
        return tensor.permute(0, 3, 1, 2)
    if architecture in TOKEN_OUTPUT:
        patches = tensor[:, tensor.shape[1] - 196:]
        return patches.transpose(1, 2).reshape(tensor.shape[0], -1, 14, 14)
    return tensor


def members_from(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") not in (2, 3):
        raise ValueError("Expected an ensemble bundle")
    for member in payload["members"]:
        model = build_model(member["architecture"], pretrained=False, dropout=member["dropout"])
        model.load_state_dict({k: v.float() if v.is_floating_point() else v
                               for k, v in member["state_dict"].items()}, strict=True)
        yield member["architecture"], model.eval()


def grad_cam(model, architecture, x):
    store = {}
    layer = target_layer(model, architecture)
    forward = layer.register_forward_hook(lambda m, i, o: store.update(act=o))
    backward = layer.register_full_backward_hook(lambda m, gi, go: store.update(grad=go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logit = model(x).reshape(-1)[0]
        logit.backward()
    finally:
        forward.remove()
        backward.remove()
    act = as_feature_map(store["act"].detach(), architecture)
    grad = as_feature_map(store["grad"].detach(), architecture)
    cam = torch.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(1))[0]
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)
    return cam[0, 0].numpy(), float(logit.detach())


def jet(values):
    v = np.clip(values, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=16)
    a = p.parse_args(argv)
    members = list(members_from(a.weights))
    paths = sorted(q for q in Path(a.images).iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"})[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = open_rgb(path)
        base = np.asarray(image, dtype=np.uint8)
        x = normalize(torch.from_numpy(base.copy()).permute(2, 0, 1)[None], torch.device("cpu"))
        panels, logits = [base], []
        for architecture, model in members:
            cam, logit = grad_cam(model, architecture, x.clone().requires_grad_(True))
            logits.append(logit)
            panels.append((0.55 * base + 0.45 * jet(cam)).astype(np.uint8))
        probability = 1 / (1 + np.exp(-np.mean(logits)))
        Image.fromarray(np.concatenate(panels, axis=1)).save(out / f"{path.stem}_p{probability:.3f}.png")
    print(f"wrote {len(paths)} panels ({len(members)} members each) to {out}")


if __name__ == "__main__":
    main()
