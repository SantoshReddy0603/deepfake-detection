"""Down-weight consistently high-loss training samples from restricted cross-fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from budget import write_json
from data import fingerprint
from run_cv import assemble_oof, write_csv


RULE = {"epochs": "last3completedepochs", "consistent_top_fraction": .10,
        "maximum_selected_fraction": .025, "selected_weight": .25,
        "ranking": "descending unsmoothed clean BCE loss; ties use image_id ascending",
        "final_ranking": "descending mean loss across3epochs; ties use image_id ascending",
        "padding": "never add samples outside the consistent-high-loss intersection"}


def _validated_losses(restricted_cv_root, original_split, epochs, final_full_train=False):
    root = Path(restricted_cv_root)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 3:
        raise ValueError("At least three completed cross-fit epochs are required")
    split_path = Path(original_split)
    split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    if (set(split.columns) != {"image_id", "partition", "group_id"}
            or split.image_id.duplicated().any() or split.image_id.eq("").any() or split.group_id.eq("").any()
            or not split.partition.isin(["train", "val"]).all()):
        raise ValueError("Invalid original fold0 split")
    training_ids = set(split.loc[split.partition == "train", "image_id"])
    validation_ids = set(split.loc[split.partition == "val", "image_id"])
    if len(training_ids) != 5600 or len(validation_ids) != 1400:
        raise ValueError("Noise weights require the official5600training/1400validation split")
    if training_ids & validation_ids:
        raise ValueError("Original training and validation IDs overlap")
    if final_full_train:
        training_ids |= validation_ids
        validation_ids = set()
    expected_count = 7000 if final_full_train else 5600
    expected_pool = "all_official_training" if final_full_train else "original_fold0_training_only"
    excluded_count = 0 if final_full_train else 1400
    contract_path = root / "run_contract.json"
    summary_path = root / "cv_results.json"
    assignment_path = root / "fold_assignments.csv"
    media_path = root / "training_media_manifest.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (contract.get("pool") != expected_pool
            or contract.get("epochs") != epochs or contract.get("folds") != 3
            or contract.get("split_sha256") != fingerprint(split_path)):
        raise ValueError("Cross-fit run must use the same original fold0 training pool, split and epochs")
    if (summary.get("status") != "completed" or summary.get("pool") != contract["pool"]
            or summary.get("noise_weight_eligible_for_original_fold0") is not (not final_full_train)
            or summary.get("oof_count") != expected_count or summary.get("pool_count") != expected_count
            or summary.get("excluded_original_fold0_validation_count") != excluded_count):
        raise ValueError("Cross-fit completion/validation-isolation evidence is missing")
    completed_folds = summary.get("folds", [])
    if (len(completed_folds) != 3 or {fold.get("fold") for fold in completed_folds} != {0, 1, 2}
            or any(fold.get("epochs") != epochs for fold in completed_folds)):
        raise ValueError("Every restricted cross-fit fold must complete the frozen epoch count")
    assignments = pd.read_csv(assignment_path, dtype={"image_id": str, "group_id": str}, keep_default_na=False)
    columns = ["image_id", "label", "group_id", "fold"]
    if (set(assignments.columns) != set(columns) or assignments.image_id.duplicated().any()
            or set(assignments.image_id) != training_ids or set(assignments.image_id) & validation_ids
            or not assignments.label.isin([0, 1]).all() or set(assignments.fold) != {0, 1, 2}):
        raise ValueError("Cross-fit assignments must cover original training IDs only")
    original_groups = split.set_index("image_id").group_id
    if (not np.array_equal(assignments.group_id.to_numpy(), original_groups.loc[assignments.image_id].to_numpy())
            or assignments.groupby("group_id").fold.nunique().ne(1).any()):
        raise ValueError("Cross-fit candidate groups differ or cross fold boundaries")
    assignments = assignments.sort_values("image_id").reset_index(drop=True)
    expected = pd.DataFrame(contract.get("assignment", []))
    if set(expected.columns) != set(columns):
        raise ValueError("Cross-fit assignment contract is missing or malformed")
    expected = expected.sort_values("image_id").reset_index(drop=True)
    if assignments[columns].to_dict("records") != expected[columns].to_dict("records"):
        raise ValueError("Cross-fit assignments do not match their run contract")
    media = json.loads(media_path.read_text(encoding="utf-8"))
    media_sha = hashlib.sha256(json.dumps(media, sort_keys=True).encode()).hexdigest()
    if media_sha != contract.get("training_media_sha256"):
        raise ValueError("Cross-fit training-media manifest checksum mismatch")
    media_samples = pd.DataFrame(media.get("samples", []))
    if (not {"image_id", "label", "sha256"}.issubset(media_samples.columns)
            or media_samples.image_id.duplicated().any() or set(media_samples.image_id) != training_ids):
        raise ValueError("Cross-fit media manifest must cover original training IDs only")
    media_labels = media_samples.set_index("image_id").loc[assignments.image_id, "label"].to_numpy()
    if not np.array_equal(assignments.label.to_numpy(), media_labels):
        raise ValueError("Cross-fit assignment labels differ from the media manifest")
    source_checksums = {"run_contract.json": fingerprint(contract_path),
                        "cv_results.json": fingerprint(summary_path),
                        "fold_assignments.csv": fingerprint(assignment_path),
                        "training_media_manifest.json": fingerprint(media_path)}
    result = assignments[["image_id", "label", "fold"]].copy()
    selected_epochs = list(range(epochs - 2, epochs + 1))
    for epoch in selected_epochs:
        prediction_paths = {}
        for fold in range(3):
            prediction_dir = root / f"fold_{fold}" / "experiment/predictions" / f"epoch_{epoch:02}_ema"
            manifest_path = prediction_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cache_path = root / f"fold_{fold}/validation_cache/manifest.json"
            cache_manifest = json.loads(cache_path.read_text(encoding="utf-8"))
            details = manifest.get("conditions", {}).get("clean", {})
            expected_count = int((assignments.fold == fold).sum())
            if (details.get("file") != "clean.npz" or details.get("count") != expected_count
                    or not isinstance(cache_manifest.get("key"), str) or not cache_manifest["key"]
                    or manifest.get("cache_key") != cache_manifest["key"]
                    or manifest.get("validation_indices") is not None):
                raise ValueError("Clean prediction manifest differs from its restricted fold")
            cache_samples = cache_manifest.get("samples", [])
            expected_fold = assignments[assignments.fold == fold].set_index("image_id")
            cached = pd.DataFrame(cache_samples)
            if (not {"image_id", "label"}.issubset(cached.columns) or cached.image_id.duplicated().any()
                    or set(cached.image_id) != set(expected_fold.index)
                    or not np.array_equal(cached.label.to_numpy(), expected_fold.loc[cached.image_id, "label"].to_numpy())):
                raise ValueError("Validation cache IDs/labels differ from the restricted cross-fit fold")
            prediction_path = prediction_dir / "clean.npz"
            if fingerprint(prediction_path) != details.get("sha256"):
                raise ValueError("Clean prediction checksum mismatch")
            prediction_paths[fold] = prediction_path
            source_checksums[manifest_path.relative_to(root).as_posix()] = fingerprint(manifest_path)
            source_checksums[cache_path.relative_to(root).as_posix()] = fingerprint(cache_path)
            source_checksums[prediction_path.relative_to(root).as_posix()] = details["sha256"]
        oof = assemble_oof(assignments, prediction_paths)
        aligned = oof.set_index("image_id").loc[result.image_id]
        result[f"loss_epoch_{epoch:02}"] = aligned.loss.to_numpy(dtype=float)
    provenance = {"restricted_cv_root": str(root.resolve()), "original_split_sha256": fingerprint(split_path),
                  "source_checksums": source_checksums,
                  "cv_contract": {key: contract.get(key) for key in
                                  ("epochs", "seed", "folds", "pool", "initialization_sha256",
                                   "training_media_sha256", "source_sha256")},
                  "assignment_sha256": hashlib.sha256(json.dumps(contract["assignment"], sort_keys=True).encode()).hexdigest(),
                  "pool": expected_pool, "excluded_validation_count": excluded_count}
    return result, selected_epochs, provenance


def restricted_losses(restricted_cv_root, original_split, epochs):
    return _validated_losses(restricted_cv_root, original_split, epochs)


def full_training_losses(cv_root, original_split, epochs):
    return _validated_losses(cv_root, original_split, epochs, final_full_train=True)


def weight_map_digest(weights):
    return hashlib.sha256(json.dumps({str(k):float(v) for k,v in weights.items()},sort_keys=True).encode()).hexdigest()


def consistent_weights(losses, epochs):
    if len(epochs) != 3 or epochs != list(range(epochs[0], epochs[0] + 3)):
        raise ValueError("Consistency requires exactly three consecutive epochs")
    columns = [f"loss_epoch_{epoch:02}" for epoch in epochs]
    if (not {"image_id", "label", "fold", *columns}.issubset(losses.columns)
            or losses.empty or losses.image_id.duplicated().any()
            or not losses.image_id.map(lambda value: isinstance(value, str) and bool(value)).all()
            or not losses.label.isin([0, 1]).all()):
        raise ValueError("Invalid aligned training loss table")
    values = losses[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Per-image losses must be finite and nonnegative")
    result = losses.copy().sort_values("image_id").reset_index(drop=True)
    count = len(result)
    top_count = math.ceil(.10 * count)
    candidate_ids = set(result.image_id)
    for column in columns:
        ranked = result.sort_values([column, "image_id"], ascending=[False, True])
        candidate_ids &= set(ranked.head(top_count).image_id)
    result["mean_loss"] = result[columns].mean(axis=1)
    result["consistently_high_loss"] = result.image_id.isin(candidate_ids)
    target = math.ceil(.025 * count)
    ranked = result[result.consistently_high_loss].sort_values(["mean_loss", "image_id"], ascending=[False, True])
    selected = ranked.head(target).image_id.tolist()
    result["weight"] = np.where(result.image_id.isin(selected), .25, 1.)
    summary = {"train_count": count, "target_selected_count": target,
               "consistent_candidate_count": len(candidate_ids), "selected_count": len(selected),
               "selected_fraction": len(selected) / count, "top_count_per_epoch": top_count,
               "selection_coverage": len(selected) / target, "epochs": epochs}
    return result, selected, summary


def build_weights(restricted_cv_root, original_split, output, epochs, final_full_train=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    losses, selected_epochs, provenance = _validated_losses(restricted_cv_root, original_split, epochs, final_full_train)
    table, selected, summary = consistent_weights(losses, selected_epochs)
    write_csv(output / "sample_losses_and_weights.csv", table)
    document = {"version": 1, "weights": dict(zip(table.image_id, table.weight.astype(float))),
                "selected_ids": selected, "selection_rule": RULE, "summary": summary,
                "provenance": provenance, "csv_sha256": fingerprint(output / "sample_losses_and_weights.csv"),
                "validation_labels_modified": False, "labels_dropped_or_flipped": False}
    document["purpose"] = "full_train_refit" if final_full_train else "fold0_training_ablation"
    document["weight_map_sha256"] = weight_map_digest(document["weights"])
    write_json(output / "noise_weights.json", document)
    return document


def load_final_weights(path, original_split, rows, expected_digest):
    path=Path(path)
    document=json.loads(path.read_text())
    if (document.get("purpose")!="full_train_refit" or document.get("version")!=1
            or document.get("selection_rule")!=RULE
            or document.get("validation_labels_modified") is not False
            or document.get("labels_dropped_or_flipped") is not False
            or document.get("provenance",{}).get("pool")!="all_official_training"
            or document.get("provenance",{}).get("excluded_validation_count")!=0
            or document.get("csv_sha256")!=fingerprint(path.parent/"sample_losses_and_weights.csv")):
        raise ValueError("Final noise weights require verified all7000 training OOF provenance")
    provenance=document["provenance"]
    losses,epochs,verified=full_training_losses(provenance["restricted_cv_root"],original_split,15)
    table,selected,summary=consistent_weights(losses,epochs)
    weights=dict(zip(table.image_id,table.weight.astype(float)))
    if (len(rows)!=7000 or set(weights)!=set(rows.image_id)
            or document.get("weights")!=weights or document.get("selected_ids")!=selected
            or document.get("summary")!=summary or provenance!=verified
            or document.get("weight_map_sha256")!=weight_map_digest(weights)
            or expected_digest!=weight_map_digest(weights)):
        raise ValueError("Frozen all7000 weight map differs from validated OOF losses")
    expected=rows.set_index("image_id").label
    if not np.array_equal(table.label.to_numpy(),expected.loc[table.image_id].to_numpy()):
        raise ValueError("Final noise-weight labels differ from official training labels")
    return weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restricted-cv-root", required=True)
    parser.add_argument("--original-split", required=True)
    parser.add_argument("--output", required=True, help="Directory for JSON weights and per-image loss CSV")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--full-train-refit", action="store_true", help="Use all7000 training OOF losses only after the noise recipe wins validation")
    args = parser.parse_args(argv)
    result = build_weights(args.restricted_cv_root, args.original_split, args.output, args.epochs, args.full_train_refit)
    print(json.dumps(result["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
