"""Group-aware cross-validation with final-suite EMA predictions and OOF losses."""
import time
PROCESS_STARTED = time.monotonic()

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import threading

import numpy as np
import pandas as pd

from augment import CONDITIONS
from budget import BudgetExceeded, fsync_directory, write_json
from data import fingerprint, load_split
from metrics import binary_metrics
from ram_training import preload_training
from run_stage_b import configuration
from step_budget import StepBudget
from train import experiment_epoch
from validation_cache import build_cache

ROOT = Path(__file__).resolve().parent


def group_folds(rows, seed=42, folds=3):
    """Balance class counts and total sizes without splitting candidate families."""
    required = {"image_id", "label", "group_id"}
    if not required.issubset(rows.columns) or rows.empty:
        raise ValueError("Fold input requires nonempty IDs, labels and candidate groups")
    if (rows.image_id.duplicated().any() or rows.group_id.eq("").any()
            or not rows.label.isin([0, 1]).all() or rows.label.nunique() != 2):
        raise ValueError("Invalid fold IDs, labels or candidate groups")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("At least two folds are required")
    groups = rows.groupby("group_id", sort=True).label.agg(["size", "sum"])
    if len(groups) < folds:
        raise ValueError("Too few candidate groups for the requested folds")
    rng = np.random.default_rng(seed)
    groups["tie"] = rng.random(len(groups))
    groups = groups.sort_values(["size", "tie"], ascending=[False, True])
    counts = np.zeros((folds, 2), dtype=np.int64)
    total = np.array([(rows.label == 0).sum(), (rows.label == 1).sum()], dtype=float)
    assignments = {}
    initial = rng.permutation(folds)
    for index, (group_id, group) in enumerate(groups.iterrows()):
        addition = np.array([group["size"] - group["sum"], group["sum"]], dtype=np.int64)
        if index < folds:
            chosen = int(initial[index])
        else:
            candidates = []
            for fold in range(folds):
                proposed = counts.copy()
                proposed[fold] += addition
                class_imbalance = np.mean(np.std(proposed / total, axis=0))
                size_imbalance = np.std(proposed.sum(axis=1) / len(rows))
                candidates.append((class_imbalance + size_imbalance, counts[fold].sum(), fold))
            chosen = int(min(candidates)[2])
        counts[chosen] += addition
        assignments[group_id] = chosen
    result = rows.copy().sort_values("image_id").reset_index(drop=True)
    result["fold"] = result.group_id.map(assignments).astype(int)
    for fold in range(folds):
        validation = result[result.fold == fold]
        if validation.empty or validation.label.nunique() != 2:
            raise ValueError("Every validation fold must contain both classes")
        if set(validation.group_id) & set(result[result.fold != fold].group_id):
            raise ValueError("Candidate-family leakage")
    return result


def f1_threshold(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (labels.ndim != 1 or labels.shape != probabilities.shape or not len(labels)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError("Expected aligned binary labels and finite probabilities in [0,1]")
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    positives = np.r_[0, np.cumsum(labels[order])]
    thresholds = np.unique(np.r_[0., .5, 1., probabilities])
    predicted_counts = np.searchsorted(-sorted_probabilities, -thresholds, side="right")
    true_positives = positives[predicted_counts]
    denominators = predicted_counts + labels.sum()
    scores = 2 * true_positives / np.maximum(denominators, 1)
    best = scores.max()
    tied = np.flatnonzero(np.isclose(scores, best, rtol=0, atol=1e-12))
    winner = min(tied, key=lambda i: (abs(thresholds[i] - .5), thresholds[i]))
    threshold = float(thresholds[winner])
    return {"threshold": threshold, "objective": "maximum clean OOF F1",
            "tie_break": "closest to0.5, then lower threshold", "metrics": binary_metrics(labels, probabilities, threshold)}


def assemble_oof(assignments, prediction_paths):
    """Require each held-out ID exactly once, with matching labels and raw BCE."""
    if (not {"image_id", "label", "fold"}.issubset(assignments.columns)
            or assignments.empty or assignments.image_id.duplicated().any()):
        raise ValueError("Invalid OOF assignments")
    if set(prediction_paths) != set(assignments.fold.unique()):
        raise ValueError("Prediction files must cover every fold exactly once")
    pieces = []
    for fold, path in sorted(prediction_paths.items()):
        expected = assignments[assignments.fold == fold].set_index("image_id")
        with np.load(path, allow_pickle=False) as saved:
            fields = {name: np.array(saved[name], copy=True) for name in
                      ("image_id", "label", "probability", "logit", "loss")}
        length = len(expected)
        if any(value.ndim != 1 or len(value) != length for value in fields.values()):
            raise ValueError("OOF prediction array shape mismatch")
        frame = pd.DataFrame(fields)
        if frame.image_id.duplicated().any() or set(frame.image_id) != set(expected.index):
            raise ValueError("OOF validation IDs differ from the held-out fold")
        aligned_labels = expected.loc[frame.image_id, "label"].to_numpy()
        if not np.array_equal(frame.label.to_numpy(), aligned_labels):
            raise ValueError("OOF labels differ from training metadata")
        probabilities = frame.probability.to_numpy(dtype=float)
        logits = frame.logit.to_numpy(dtype=float)
        losses = frame.loss.to_numpy(dtype=float)
        if (not np.isfinite(np.c_[probabilities, logits, losses]).all()
                or (probabilities < 0).any() or (probabilities > 1).any() or (losses < -1e-6).any()):
            raise ValueError("Invalid OOF probability, logit or loss")
        sigmoid = np.exp(-np.logaddexp(0., -logits))
        expected_loss = np.logaddexp(0., logits) - aligned_labels * logits
        if (not np.allclose(probabilities, sigmoid, rtol=1e-5, atol=2e-6)
                or not np.allclose(losses, expected_loss, rtol=1e-5, atol=2e-6)):
            raise ValueError("OOF probability or unsmoothed loss inconsistent with logits")
        frame["fold"] = int(fold)
        pieces.append(frame)
    result = pd.concat(pieces).sort_values("image_id").reset_index(drop=True)
    if result.image_id.duplicated().any() or set(result.image_id) != set(assignments.image_id):
        raise ValueError("OOF coverage must include every training image once")
    return result


def write_csv(path, frame):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--original-split", default=str(ROOT / "splits/fold0.csv"))
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--fold0-training-only", action="store_true", help="Cross-fit only original fold0 training rows for noise weights")
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--session-hours", type=float, default=8.)
    parser.add_argument("--session-start-unix", type=float)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.75)
    parser.add_argument("--yield-on-other-gpu-process", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 100 or not 0 <= args.workers <= 4:
        parser.error("Invalid epoch count or worker limit")
    if not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= .8:
        parser.error("GPU memory fraction must be positive and at most0.8")
    if not math.isfinite(args.session_hours) or not 0 < args.session_hours <= 8:
        parser.error("Session wall limit must be positive and at most eight hours")
    startup = time.monotonic() - PROCESS_STARTED
    if args.session_start_unix is not None:
        if not math.isfinite(args.session_start_unix) or args.session_start_unix > time.time():
            parser.error("Invalid session start timestamp")
        startup = max(startup, time.time() - args.session_start_unix)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    budget = StepBudget(out / "step_budget.json", args.session_hours * 3600,
                        scope="group_aware_cv", resume=args.resume, allocated_gpus=1,
                        startup_seconds=startup)
    def hard_stop():
        print("CV session wall deadline reached; exit124. Completed epoch files remain available.", flush=True)
        os._exit(124)
    watchdog = threading.Timer(budget.remaining, hard_stop)
    watchdog.daemon = True
    watchdog.start()
    status, exit_code = "running", 0
    fold_results = []
    try:
        if args.yield_on_other_gpu_process:
            from resource_guard import ResourceGuard
            gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
                raise ValueError("Shared-host guard requires one explicit GPU UUID")
            budget = ResourceGuard(budget, gpu_uuid=gpu_uuid)
            budget.check()
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly the one authorized GPU before launching CV")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        torch.set_num_threads(4)
        original_train, original_validation = load_split(args.data_root, args.original_split)
        pool = original_train if args.fold0_training_only else pd.concat([original_train, original_validation])
        rows = pool.sort_values("image_id").reset_index(drop=True)
        if args.fold0_training_only and set(rows.image_id) & set(original_validation.image_id):
            raise ValueError("Noise-weight cross-fitting must exclude original fold0 validation rows")
        assignments = group_folds(rows, seed=42, folds=3)
        manifest_path = out / "training_media_manifest.json"
        store = preload_training(rows, manifest_path=manifest_path, check_deadline=budget.check,
                                 decode_workers=4,
                                 progress=lambda value: budget.emit("ram_preload_progress", **value))
        signature = {"epochs": args.epochs, "seed": 42, "folds": 3,
                     "pool": "original_fold0_training_only" if args.fold0_training_only else "all_official_training",
                     "split_sha256": fingerprint(args.original_split),
                     "initialization_sha256": fingerprint(args.initialization),
                     "training_media_sha256": hashlib.sha256(json.dumps(store.manifest, sort_keys=True).encode()).hexdigest(),
                     "assignment": assignments[["image_id", "label", "group_id", "fold"]].to_dict(orient="records"),
                     "source_sha256": {p.name: fingerprint(p) for p in sorted(ROOT.glob("*.py"))}}
        contract_path = out / "run_contract.json"
        if contract_path.is_file() and (not args.resume or json.loads(contract_path.read_text()) != signature):
            raise ValueError("CV recipe, source, input or fold assignment changed")
        write_json(contract_path, signature)
        write_csv(out / "fold_assignments.csv", assignments[["image_id", "label", "group_id", "fold"]])
        prediction_paths = {}
        for fold in range(3):
            budget.check()
            training = assignments[assignments.fold != fold].reset_index(drop=True)
            validation = assignments[assignments.fold == fold].reset_index(drop=True)
            fold_dir = out / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)
            manifest = build_cache(validation, fold_dir / "validation_cache", seed=42,
                                   check_deadline=budget.check,
                                   progress=lambda message: budget.emit("validation_cache", fold=fold, message=message))
            config = configuration(args.workers, initialization=args.initialization)
            config.update(name=f"convnext_tiny_cv_fold_{fold}", device=args.device,
                          max_epochs=args.epochs, patience=args.epochs + 1,
                          validation_policy="ema_clean_until_final", artifact_purpose="cv_validation",
                          ram_training=True, channels_last=True)
            while True:
                budget.check()
                state = experiment_epoch(config, training, fold_dir / "validation_cache", manifest["key"],
                                         fold_dir / "experiment", budget, training_store=store)
                if state["done"]:
                    break
            final = state["history"][-1]
            if final["epoch"] + 1 != args.epochs or not final["ema"]["selection_eligible"]:
                raise ValueError("CV requires the frozen final epoch and complete16-condition evaluation")
            prediction_dir = fold_dir / "experiment/predictions" / f"epoch_{args.epochs:02}_ema"
            saved = json.loads((prediction_dir / "manifest.json").read_text())
            if saved["cache_key"] != manifest["key"] or set(saved["conditions"]) != set(CONDITIONS):
                raise ValueError("CV prediction manifest does not match the full validation suite")
            for details in saved["conditions"].values():
                if fingerprint(prediction_dir / details["file"]) != details["sha256"]:
                    raise ValueError("CV prediction checksum mismatch")
            prediction_paths[fold] = prediction_dir / "clean.npz"
            fold_results.append({"fold": fold, "train_count": len(training), "validation_count": len(validation),
                                 "epochs": args.epochs, **{key: final["ema"][key] for key in
                                                         ("mean_auc", "minimum_auc", "clean_auc")}})
            write_json(out / "fold_results.json", fold_results)
            budget.emit("fold_complete", **fold_results[-1])
        oof = assemble_oof(assignments, prediction_paths)
        write_csv(out / "oof_predictions.csv", oof)
        scores = [row["mean_auc"] for row in fold_results]
        summary = {"status": "completed", "folds": fold_results, "oof_count": len(oof),
                   "pool_count": len(rows),
                   "pool": "original_fold0_training_only" if args.fold0_training_only else "all_official_training",
                   "excluded_original_fold0_validation_count": len(original_validation) if args.fold0_training_only else 0,
                   "noise_weight_eligible_for_original_fold0": args.fold0_training_only,
                   "mean_fold_selection_auc": float(np.mean(scores)),
                   "std_fold_selection_auc": float(np.std(scores)),
                   "std_definition": "population standard deviation across3folds",
                   "label_threshold": f1_threshold(oof.label.to_numpy(), oof.probability.to_numpy()),
                   "oof_metrics_at_half": binary_metrics(oof.label.to_numpy(), oof.probability.to_numpy()),
                   "selection": "final frozen epoch EMA mean16 AUC, minimum AUC reported",
                   "wall_seconds": budget.elapsed}
        write_json(out / "cv_results.json", summary)
        budget.emit("cv_complete", oof_count=len(oof), mean_auc=summary["mean_fold_selection_auc"],
                    std_auc=summary["std_fold_selection_auc"])
        status = "completed"
    except BudgetExceeded as exc:
        status, exit_code = str(exc), 124
        budget.emit("wall_timeout", message=status, exit_code=exit_code)
    except KeyboardInterrupt:
        status, exit_code = "interrupted", 130
        budget.emit("interrupted", exit_code=exit_code)
    except Exception as exc:
        status, exit_code = f"failed: {type(exc).__name__}: {exc}", 1
        budget.emit("failure", message=status, exit_code=exit_code)
    finally:
        write_json(out / "status.json", {"status": status, "exit_code": exit_code,
                   "completed_folds": len(fold_results), "wall_seconds": budget.elapsed})
        budget.close()
        budget.emit("cv_finished", status=status, exit_code=exit_code)
        watchdog.cancel()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
