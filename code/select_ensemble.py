"""Choose stop epochs and ensemble members from pooled out-of-fold predictions, then bundle.

    python select_ensemble.py --runs runs [more_runs ...] --baseline oof_predictions.csv --out chosen.json
    python select_ensemble.py --runs runs [more_runs ...] --chosen chosen.json --bundle model_weights.pt

Step 1 reads every oof_<arch>_s<seed>_f<k>.csv produced by train_v2.py, pools the folds
(each image scored once, by the model that never trained on it), takes each run's best
epoch (or its last one with --last_epoch, which avoids picking the epoch on the same
predictions that are then reported), and adds members greedily while the mean-logit AUC
improves, up to --max_members. The gain over the
baseline OOF scores is reported with a paired bootstrap interval.
Step 2 packs the all-data refit members saved at the chosen epochs into one bundle.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model_def import export_logit_ensemble
from train_v2 import auc


def pooled_runs(runs, folds):
    pieces = defaultdict(list)
    for path in sorted(q for r in runs for q in Path(r).glob("oof_*_f[0-9]*.csv")):
        match = re.fullmatch(r"oof_(.+)_s(\d+)_f(\d+)\.csv", path.name)
        pieces[(match[1], int(match[2]))].append(pd.read_csv(path, dtype={"image_id": str}))
    pooled = {}
    for key, frames in pieces.items():
        if len(frames) != folds:
            print(f"skip {key}: {len(frames)}/{folds} folds")
            continue
        common = sorted(set.intersection(*(set(c for c in f.columns if c.startswith("e")) for f in frames)),
                        key=lambda c: int(c[1:]))
        frame = pd.concat([f[["image_id", "label"] + common] for f in frames]).sort_values("image_id")
        pooled[key] = frame.reset_index(drop=True)
    return pooled


def bootstrap(labels, base, new, draws=1000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        i = rng.integers(0, len(labels), len(labels))
        if labels[i].min() == labels[i].max():
            continue
        diffs.append(auc(labels[i], new[i]) - auc(labels[i], base[i]))
    return np.percentile(diffs, [2.5, 97.5])


def select(args):
    pooled = pooled_runs(args.runs, args.folds)
    if not pooled:
        raise SystemExit("no complete runs found")
    ids = next(iter(pooled.values())).image_id
    labels = next(iter(pooled.values())).label.to_numpy()
    candidates = {}
    fixed = {k: int(v) for k, v in (item.split("=") for item in args.epoch or [])}
    for (arch, seed), frame in pooled.items():
        if not frame.image_id.equals(ids):
            raise ValueError("runs cover different images")
        scores = {int(c[1:]): auc(labels, frame[c].to_numpy()) for c in frame.columns if c.startswith("e")}
        epoch = max(scores) if args.last_epoch else max(scores, key=scores.get)
        epoch = fixed.get(f"{arch}_s{seed}", epoch)
        candidates[f"{arch}_s{seed}"] = {"architecture": arch, "seed": seed, "epoch": epoch, "auc": scores[epoch],
                                         "logit": frame[f"e{epoch}"].to_numpy(), "by_epoch": scores}
        print(f"{arch} s{seed}: epoch {epoch}, pooled OOF AUC {scores[epoch]:.5f}")
    chosen, total, best = [], None, -1.0
    remaining = dict(candidates)
    while remaining and len(chosen) < args.max_members:
        trial = {name: auc(labels, (c["logit"] if total is None else total + c["logit"]) / (len(chosen) + 1))
                 for name, c in remaining.items()}
        name = max(trial, key=trial.get)
        if trial[name] <= best + args.min_gain:
            break
        best = trial[name]
        chosen.append(name)
        total = remaining.pop(name)["logit"] + (0 if total is None else total)
        print(f"+ {name}: ensemble OOF AUC {best:.5f}")
    ensemble = total / len(chosen)
    every = sum(c["logit"] for c in candidates.values()) / len(candidates)
    report = {"pooled_images": int(len(labels)), "selected_auc": best, "all_members_auc": auc(labels, every),
              "members": [{k: candidates[n][k] for k in ("architecture", "seed", "epoch", "auc")} for n in chosen]}
    print(f"all {len(candidates)} members: {report['all_members_auc']:.5f}")
    if args.baseline:
        base = pd.read_csv(args.baseline, dtype={"image_id": str}).set_index("image_id").loc[ids]
        base_score = base[args.baseline_column].to_numpy()
        lo, hi = bootstrap(labels, base_score, ensemble)
        report["baseline_auc"] = auc(labels, base_score)
        report["gain_95ci"] = [float(lo), float(hi)]
        print(f"baseline {report['baseline_auc']:.5f}; gain {best - report['baseline_auc']:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


def bundle(args):
    report = json.loads(Path(args.chosen).read_text(encoding="utf-8"))
    members = []
    for m in report["members"]:
        name = f"member_{m['architecture']}_s{m['seed']}_e{m['epoch']}.pt"
        path = next((Path(r) / name for r in args.runs if (Path(r) / name).exists()), None)
        if path is None:
            raise FileNotFoundError(name)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        members.append((payload["architecture"], payload["dropout"], payload["state_dict"]))
        print(f"packed {path.name}")
    export_logit_ensemble(members, args.bundle, threshold=args.threshold, purpose="final_submission",
                          tta=True, half=not args.fp32)
    print(f"wrote {args.bundle} ({Path(args.bundle).stat().st_size / 1e6:.0f} MB)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, nargs="+", help="one or more folders written by train_v2.py")
    p.add_argument("--last_epoch", action="store_true", help="use each run's last epoch instead of its best")
    p.add_argument("--max_members", type=int, default=99)
    p.add_argument("--epoch", nargs="*", help="fix a run's epoch, e.g. convnext_small_s7=8")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--baseline", help="CSV with image_id and a probability column for comparison")
    p.add_argument("--baseline_column", default="probability")
    p.add_argument("--min_gain", type=float, default=0.0002)
    p.add_argument("--out", default="chosen.json")
    p.add_argument("--chosen", help="report from step 1; with --bundle, packs refit members")
    p.add_argument("--bundle")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fp32", action="store_true")
    a = p.parse_args(argv)
    bundle(a) if a.bundle else select(a)


if __name__ == "__main__":
    main()
