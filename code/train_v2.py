"""Five-fold group cross-validation and all-data refit for any supported backbone.

Recipe: degradation + mosaic augmentation, AdamW with a warm-up epoch and cosine decay,
smoothed BCE, EMA weights, mixed precision. Every epoch the EMA model scores the held-out
fold with flip test-time augmentation, so stop epochs and ensemble members can later be
chosen on pooled out-of-fold predictions of all 7,000 training images.

    python train_v2.py --data_root DATA --split fold0.csv --arch convnext_small --fold 0 --out runs
    python train_v2.py --data_root DATA --split fold0.csv --arch convnext_small --fold all \
        --save_epochs 6 7 8 --out runs
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from augment import CONDITIONS, TrainAugment, degrade, pixelate
from data import open_rgb, seed_worker
from model_def import ARCHITECTURES, EMA, build_model, head_parameters, normalize
from run_cv import group_folds


def load_rows(data_root, split_csv):
    root = Path(data_root)
    meta = pd.read_csv(root / "train_metadata.csv", dtype={"image_id": str}, keep_default_na=False)
    groups = pd.read_csv(split_csv, dtype=str, keep_default_na=False)[["image_id", "group_id"]]
    rows = meta.merge(groups, on="image_id", validate="one_to_one")
    if len(rows) != len(meta) or rows.group_id.eq("").any():
        raise ValueError("Every training image needs a group id")
    rows["path"] = [str(root / "train" / ("fake" if label else "real") / f"{image_id}.jpg")
                    for image_id, label in zip(rows.image_id, rows.label)]
    return rows.sort_values("image_id").reset_index(drop=True)


def conflict_groups(rows):
    """Near-duplicate groups that carry both labels: at least one label in them is wrong."""
    counts = rows.groupby("group_id").label.nunique()
    return set(counts[counts > 1].index)


ROBUST_CONDITIONS = CONDITIONS + ("pixelate_8", "pixelate_16", "pixelate_24")


def apply_condition(image, condition, image_id):
    if condition.startswith("pixelate_"):
        return pixelate(image, int(condition.split("_")[1]))
    return degrade(image, condition, image_id)


class Images(Dataset):
    def __init__(self, rows, augment=None, condition=None):
        self.rows, self.augment, self.condition = rows.reset_index(drop=True), augment, condition

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image = open_rgb(row.path)
        if self.condition:
            image = apply_condition(image, self.condition, row.image_id)
        array = self.augment(image) if self.augment else np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1), torch.tensor(float(row.label))


def schedule(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        x = normalize(x, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logit = 0.5 * (model(x).reshape(-1).float() + model(x.flip(3)).reshape(-1).float())
        out.append(logit.cpu().numpy())
    return np.concatenate(out)


def auc(labels, scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    sorted_scores = scores[order]
    start = 0
    for end in range(1, len(scores) + 1):
        if end == len(scores) or sorted_scores[end] != sorted_scores[start]:
            ranks[order[start:end]] = (start + end + 1) / 2
            start = end
    positives = labels.sum()
    negatives = len(labels) - positives
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", required=True, help="CSV with image_id,group_id (near-duplicate groups)")
    p.add_argument("--arch", required=True, choices=ARCHITECTURES)
    p.add_argument("--fold", required=True, help="fold index, or 'all' for the all-data refit")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12, help="cosine horizon")
    p.add_argument("--stop", type=int, default=None, help="last epoch actually trained (default: --epochs)")
    p.add_argument("--save_epochs", type=int, nargs="*", default=[], help="refit: epochs whose EMA weights are saved")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--backbone_lr", type=float, default=1e-4)
    p.add_argument("--head_lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--smoothing", type=float, default=0.05)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--mosaic", type=float, default=0.5)
    p.add_argument("--mosaic_max", type=int, default=16, help="largest mosaic block in pixels")
    p.add_argument("--exclude", help="CSV with an image_id column: these images are never trained on")
    p.add_argument("--train_frac", type=float, default=1.0, help="train on this fraction of training groups")
    p.add_argument("--robust", action="store_true", help="after the last epoch, score the held-out fold "
                   "under every degradation condition")
    p.add_argument("--keep_conflicts", action="store_true", help="train on mixed-label groups too")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="smoke test: use only this many images")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    torch.backends.cudnn.benchmark = True

    rows = load_rows(a.data_root, a.split)
    if a.limit:
        rows = pd.concat([rows[rows.label == label].sample(a.limit // 2, random_state=a.seed) for label in (0, 1)])
        rows = rows.sort_values("image_id").reset_index(drop=True)
    folds = group_folds(rows[["image_id", "label", "group_id"]], seed=42, folds=a.folds)
    rows = rows.merge(folds[["image_id", "fold"]], on="image_id", validate="one_to_one")
    refit = a.fold == "all"
    if refit:
        train_rows, val_rows = rows, rows.iloc[:0]
    else:
        fold = int(a.fold)
        train_rows, val_rows = rows[rows.fold != fold], rows[rows.fold == fold]
    dropped = 0
    if not a.keep_conflicts:
        bad = conflict_groups(train_rows)
        dropped = int(train_rows.group_id.isin(bad).sum())
        train_rows = train_rows[~train_rows.group_id.isin(bad)]
    excluded = 0
    if a.exclude:
        drop = set(pd.read_csv(a.exclude, dtype={"image_id": str}).image_id)
        excluded = int(train_rows.image_id.isin(drop).sum())
        train_rows = train_rows[~train_rows.image_id.isin(drop)]
    if a.train_frac < 1:
        groups = np.array(sorted(train_rows.group_id.unique()))
        keep = np.random.default_rng(a.seed).choice(groups, round(len(groups) * a.train_frac), replace=False)
        train_rows = train_rows[train_rows.group_id.isin(set(keep))]
    stop = a.stop or a.epochs
    if refit and (not a.save_epochs or max(a.save_epochs) > stop):
        raise ValueError("Refit needs --save_epochs within the trained epochs")

    tag = f"{a.arch}_s{a.seed}_f{a.fold}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(a.arch, pretrained=True, dropout=a.dropout).to(device).to(memory_format=torch.channels_last)
    head = head_parameters(model)
    head_ids = {id(t) for t in head}
    optimizer = torch.optim.AdamW([
        {"params": [t for t in model.parameters() if id(t) not in head_ids], "lr": a.backbone_lr},
        {"params": head, "lr": a.head_lr}], weight_decay=a.weight_decay)
    loader_options = {"num_workers": a.workers, "worker_init_fn": seed_worker, "pin_memory": device.type == "cuda",
                      "persistent_workers": a.workers > 0}
    train_loader = DataLoader(Images(train_rows, TrainAugment(True, "strong", a.mosaic, a.mosaic_max)), batch_size=a.batch_size,
                              shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(a.seed),
                              **loader_options)
    val_loader = DataLoader(Images(val_rows), batch_size=128, shuffle=False, **loader_options) if len(val_rows) else None
    steps = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: schedule(s, steps, steps * a.epochs))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    ema = EMA(model, a.ema_decay)

    oof = val_rows[["image_id", "label", "fold"]].copy() if val_loader else None
    history = []
    for epoch in range(1, stop + 1):
        model.train()
        started, total = time.time(), 0.0
        for x, y in train_loader:
            x = normalize(x, device).contiguous(memory_format=torch.channels_last)
            y = y.to(device) * (1 - a.smoothing) + a.smoothing / 2
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x).reshape(-1)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            total += loss.item()
        record = {"epoch": epoch, "train_loss": total / steps, "seconds": round(time.time() - started, 1)}
        if val_loader:
            logits = predict_logits(ema.model, val_loader, device)
            oof[f"e{epoch}"] = logits
            record["val_auc"] = auc(oof.label.to_numpy(), logits)
            oof.to_csv(out / f"oof_{tag}.csv", index=False, float_format="%.6f")
        if refit and epoch in a.save_epochs:
            state = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                     for k, v in ema.model.state_dict().items()}
            torch.save({"architecture": a.arch, "dropout": a.dropout, "epoch": epoch, "seed": a.seed,
                        "state_dict": state}, out / f"member_{a.arch}_s{a.seed}_e{epoch}.pt")
        history.append(record)
        print(json.dumps({"run": tag, **record}), flush=True)
    if a.robust and val_loader:
        robust = val_rows[["image_id", "label", "fold"]].copy()
        for condition in ROBUST_CONDITIONS:
            loader = DataLoader(Images(val_rows, condition=condition), batch_size=128, shuffle=False,
                                num_workers=a.workers, worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
            robust[condition] = predict_logits(ema.model, loader, device)
            print(json.dumps({"run": tag, "condition": condition,
                              "auc": auc(robust.label.to_numpy(), robust[condition].to_numpy())}), flush=True)
        robust.to_csv(out / f"robust_{tag}.csv", index=False, float_format="%.6f")
    summary = {"run": tag, "args": vars(a), "train_count": len(train_rows), "val_count": len(val_rows),
               "conflict_images_dropped": dropped,
               "excluded_images": excluded, "train_frac": a.train_frac, "history": history}
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
