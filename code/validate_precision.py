"""Check export storage quality on the untouched cached validation suite."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

STARTED = time.monotonic()

import numpy as np

from budget import BudgetExceeded, write_json
from metrics import auc_roc, summarize


def fingerprint(path, check=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            if check: check()
            digest.update(chunk)
    return digest.hexdigest()


def verify_cache(directory, conditions, check, expected_count=1400, image_size=224):
    """Check all rendered arrays once without opening any source images."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    if (len(conditions) != 16 or tuple(manifest.get("conditions", [])) != tuple(conditions)
            or manifest.get("recipe_version") != 1 or manifest.get("seed") != 42
            or manifest.get("size") != image_size or len(samples) != expected_count
            or len({s.get("image_id") for s in samples}) != expected_count
            or any(not isinstance(s.get("image_id"), str) or not s["image_id"]
                   or type(s.get("label")) is not int or s["label"] not in (0, 1)
                   or not isinstance(s.get("sha256"), str) or len(s["sha256"]) != 64
                   or any(c not in "0123456789abcdef" for c in s["sha256"]) for s in samples)
            or {s["label"] for s in samples} != {0, 1}
            or set(manifest.get("array_sha256", {})) != set(conditions)):
        raise ValueError("Invalid complete sixteen-condition validation cache manifest")
    fields = ("recipe_version", "conditions", "seed", "pillow", "numpy", "samples", "size")
    contract = {key: manifest[key] for key in fields}
    if hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest() != manifest.get("key"):
        raise ValueError("Validation cache manifest contract checksum mismatch")
    arrays = {}
    for condition in conditions:
        check()
        started = time.monotonic()
        path = directory / f"{condition}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (array.shape != (expected_count, image_size, image_size, 3) or array.dtype != np.uint8
                or path.stat().st_size != array.offset + array.nbytes):
            raise ValueError(f"Invalid validation cache array size/shape/dtype: {condition}")
        del array
        digest = fingerprint(path, check)
        if digest != manifest["array_sha256"][condition]:
            raise ValueError(f"Validation cache array checksum mismatch: {condition}")
        arrays[condition] = {"sha256": digest, "bytes": path.stat().st_size,
                             "verification_seconds": time.monotonic() - started}
    return manifest, arrays


def compare_predictions(original_dir, rounded_dir, cache_manifest, conditions, auc_tolerance=.001, probability_tolerance=.003, storage_precision="float16"):
    ids = np.asarray([s["image_id"] for s in cache_manifest["samples"]])
    labels = np.asarray([s["label"] for s in cache_manifest["samples"]])
    streams = []
    for directory in [Path(original_dir), Path(rounded_dir)]:
        manifest = json.loads((directory / "manifest.json").read_text())
        if (manifest.get("cache_key") != cache_manifest["key"] or manifest.get("selection_eligible") is not True
                or manifest.get("validation_indices") is not None or set(manifest.get("conditions", {})) != set(conditions)):
            raise ValueError("Prediction manifest must cover the same complete validation suite")
        predictions = {}
        for condition in conditions:
            details = manifest["conditions"][condition]
            if details.get("file") != f"{condition}.npz" or details.get("count") != len(ids):
                raise ValueError("Prediction manifest file/count mismatch")
            path = directory / details["file"]
            if fingerprint(path) != details.get("sha256"):
                raise ValueError("Prediction checksum mismatch")
            with np.load(path, allow_pickle=False) as saved:
                probabilities = np.asarray(saved["probability"], dtype=np.float64)
                if (not np.array_equal(saved["image_id"], ids) or not np.array_equal(saved["label"], labels)
                        or probabilities.shape != labels.shape or not np.isfinite(probabilities).all()
                        or (probabilities < 0).any() or (probabilities > 1).any()):
                    raise ValueError("Prediction IDs/labels/probabilities do not align with cached validation")
                predictions[condition] = probabilities
        streams.append(predictions)
    comparisons, metrics = {}, [{}, {}]
    failed = []
    for condition in conditions:
        a, b = streams[0][condition], streams[1][condition]
        original_auc, rounded_auc = auc_roc(labels, a), auc_roc(labels, b)
        if original_auc is None or rounded_auc is None:
            raise ValueError("Precision validation requires defined AUC in every condition")
        drop = original_auc - rounded_auc
        maximum = float(np.max(np.abs(a - b)))
        passed = drop <= auc_tolerance + 1e-12 and maximum <= probability_tolerance + 1e-12
        comparisons[condition] = {"fp32_stored_auc": original_auc, ("fp16_stored_auc" if storage_precision == "float16" else "fp32_export_auc"): rounded_auc,
                                  "auc_drop": drop, "maximum_absolute_probability_difference": maximum,
                                  "passed": passed}
        for values, auc in zip(metrics, [original_auc, rounded_auc]): values[condition] = {"auc": auc}
        if not passed: failed.append(condition)
    return {"passed": not failed, "failed_conditions": failed, "conditions": comparisons,
            "fp32_stored": summarize(metrics[0], conditions), ("fp16_stored" if storage_precision == "float16" else "fp32_export"): summarize(metrics[1], conditions),
            "maximum_auc_drop": max(item["auc_drop"] for item in comparisons.values()),
            "maximum_absolute_probability_difference": max(item["maximum_absolute_probability_difference"] for item in comparisons.values()),
            "auc_drop_tolerance": auc_tolerance, "probability_tolerance": probability_tolerance}


def validate_selected_storage(payload):
    import torch
    allowed = {"stage_a_validation", "stage_b_fold0_validation", "ablation_validation"}
    if payload.get("format_version") == 2:
        members = payload.get("members", [])
        if (payload.get("architecture") != "probability_ensemble" or payload.get("aggregation") != "uniform_probability"
                or not isinstance(members, list) or len(members) != 2
                or {member.get("architecture") for member in members} != {"convnext_tiny", "efficientnet_b0"}):
            raise ValueError("Expected the selected two-model probability ensemble")
        states = [member["state_dict"] for member in members]
    elif payload.get("format_version") == 1:
        states = [payload["state_dict"]]
    else:
        raise ValueError("Unknown selected validation export format")
    if payload.get("purpose") not in allowed:
        raise ValueError("Expected a selected validation-purpose export")
    for state in states:
        floating = [value for value in state.values() if value.is_floating_point()]
        if not floating or any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in floating):
            raise ValueError("Every selected model must contain finite FP32-stored floating tensors")


def export_rounded(original, payload, path, storage_precision="float16"):
    from model_def import export_model, export_ensemble
    if payload["format_version"] == 2:
        members = [{"format_version": 1, "architecture": member["architecture"], "dropout": member["dropout"],
                    "state_dict": member["state_dict"], "preprocessing": payload["preprocessing"],
                    "label_mapping": payload["label_mapping"], "threshold": payload["threshold"],
                    "purpose": payload["purpose"]} for member in payload["members"]]
        export_ensemble(members, path, payload["threshold"], "precision_validation", half=storage_precision == "float16")
    else:
        export_model(original, payload["architecture"], path, threshold=payload["threshold"],
                     purpose="precision_validation", half=storage_precision == "float16")


def selected_decision_binding(weights, frozen_decision=None):
    selected_sha = fingerprint(weights)
    decision_sha = None
    if frozen_decision:
        decision = json.loads(Path(frozen_decision).read_text())
        if decision.get("evidence", {}).get("selected_weights_sha256") != selected_sha:
            raise ValueError("Selected validation weights differ from frozen decision")
        decision_sha = fingerprint(frozen_decision)
    return selected_sha, decision_sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--session-hours", type=float, default=8.)
    parser.add_argument("--frozen-decision", help="Frozen recipe whose selected validation export is being accepted")
    parser.add_argument("--storage-precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 128 or not math.isfinite(args.session_hours) or not 0 < args.session_hours <= 8:
        parser.error("Invalid batch size or session wall limit (maximum8hours)")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    import multiprocessing
    def hard_stop():
        print("Precision validation wall deadline reached; exit124", flush=True)
        try:
            for process in multiprocessing.active_children():
                try:
                    process.terminate()
                    process.join(timeout=.25)
                    if process.is_alive(): process.kill()
                except (OSError, ValueError): pass
        finally:
            os._exit(124)
    watchdog = threading.Timer(max(.001, args.session_hours * 3600 - (time.monotonic() - STARTED)), hard_stop)
    watchdog.daemon = True
    watchdog.start()
    budget, status, code = None, "running", 0
    try:
        from step_budget import StepBudget
        from resource_guard import ResourceGuard
        uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not uuid.startswith("GPU-") or "," in uuid:
            raise ValueError("Precision validation requires one explicit authorized GPU UUID")
        budget = ResourceGuard(StepBudget(output / "wall_budget.json", args.session_hours * 3600,
                                         scope="storage_validation", allocated_gpus=1,
                                         startup_seconds=time.monotonic() - STARTED), uuid)
        budget.check()
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "4"
        import torch
        from augment import CONDITIONS
        from model_def import load_export
        from train import evaluate, sync_export
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one GPU before precision validation")
        torch.cuda.set_per_process_memory_fraction(.75, 0)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        started = time.monotonic()
        manifest, arrays = verify_cache(args.cache_dir, CONDITIONS, budget.check)
        cache_seconds = time.monotonic() - started
        budget.emit("cache_verified", cache_load_seconds=cache_seconds, conditions=16, images=1400)
        selected_sha, decision_hash = selected_decision_binding(args.weights, args.frozen_decision)
        payload = torch.load(args.weights, map_location="cpu", weights_only=True)
        validate_selected_storage(payload)
        started = time.monotonic()
        original, metadata = load_export(args.weights, args.device)
        original.to(memory_format=torch.channels_last)
        if args.frozen_decision and json.loads(Path(args.frozen_decision).read_text()).get("storage_precision", "float16") != args.storage_precision:
            raise ValueError("Storage precision differs from frozen decision")
        export_stream = args.storage_precision + "_export"
        rounded_path = output / (export_stream + ".pt")
        export_rounded(original, payload, rounded_path, args.storage_precision)
        sync_export(rounded_path)
        rounded, _ = load_export(rounded_path, args.device)
        rounded.to(memory_format=torch.channels_last)
        model_seconds = time.monotonic() - started
        device = torch.device(args.device)
        results, times = {}, {}
        for name, model in [("fp32_stored", original), (export_stream, rounded)]:
            started = time.monotonic()
            results[name] = evaluate(model, args.cache_dir, device, args.batch_size, 2, budget,
                                     prediction_dir=output / name, channels_last=True)
            torch.cuda.synchronize(device)
            times[name] = time.monotonic() - started
            budget.emit("precision_stream_complete", stream=name, seconds=times[name],
                        **{key: results[name][key] for key in ("clean_auc", "mean_auc", "minimum_auc")})
        comparison = compare_predictions(output / "fp32_stored", output / export_stream, manifest, CONDITIONS, storage_precision=args.storage_precision)
        status = "passed" if comparison["passed"] else ("FP32_FALLBACK_REQUIRED" if args.storage_precision == "float16" else "STORAGE_PARITY_FAILED")
        code = 0 if comparison["passed"] else 2
        report = {"status": status, "fp32_fallback_required": not comparison["passed"],
                  "storage_precision": args.storage_precision, "comparison": comparison, "streams": results, "cache_arrays": arrays,
                  "cache_manifest_sha256": fingerprint(Path(args.cache_dir) / "manifest.json"),
                  "selected_weights_sha256": selected_sha, "frozen_decision_sha256": decision_hash, "export_weights_sha256": fingerprint(rounded_path),
                  "selected_weights_bytes": Path(args.weights).stat().st_size, "export_weights_bytes": rounded_path.stat().st_size,
                  "cache_verification_seconds": cache_seconds, "model_load_export_seconds": model_seconds,
                  "validation_seconds": times, "hardware": torch.cuda.get_device_name(0), "t4_timing_verified": False,
                  "compute_precision": "both streams use the existing CUDA autocast FP16 validation recipe; storage rounding is the only precision change",
                  "channels_last": True, "workers": 2, "torch_threads": 4, "allocated_gpus": 1,
                  "wall_seconds": time.monotonic() - STARTED, "exit_code": code}
        write_json(output / "precision_validation.json", report)
        budget.emit("precision_validation_complete", status=status, exit_code=code, comparison=comparison)
    except BudgetExceeded as exc:
        status, code = str(exc), 124
    except KeyboardInterrupt:
        status, code = "interrupted", 130
    except Exception as exc:
        status, code = f"failed: {type(exc).__name__}: {exc}", 1
    finally:
        if budget is not None:
            budget.emit("precision_validation_stopped", status=status, exit_code=code)
            budget.close()
        watchdog.cancel()
        write_json(output / "runner_status.json", {"status": status, "exit_code": code,
                                                   "wall_seconds": time.monotonic() - STARTED})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
