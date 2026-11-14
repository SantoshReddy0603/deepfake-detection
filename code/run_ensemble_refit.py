"""Refit the frozen two-member ensemble sequentially using one verified RAM store."""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import threading
import time

STARTED = time.monotonic()

import torch

from budget import BudgetExceeded, write_json
from data import fingerprint
from model_def import NORMALIZATION, export_ensemble, initialization_provenance
from noise_weights import RULE, load_final_weights
from ram_training import preload_training
from run_refit import all_training_rows, refit, restore_epoch_views
from run_stage_b import configuration
from step_budget import StepBudget,fresh_session_budget
from train import save_checkpoint, sync_export

ROOT = Path(__file__).resolve().parent
ARCHITECTURES = ("convnext_tiny", "efficientnet_b0")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def frozen_recipe(path):
    decision = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"architecture", "aggregation", "degradation", "ema_decay", "schedule_epochs",
                "label_threshold", "evidence_no_test_used", "members", "evidence"}
    if (not isinstance(decision, dict) or set(decision) not in (required, required | {"storage_precision"})
            or decision.get("storage_precision", "float16") not in ("float16", "float32")
            or decision["architecture"] != "probability_ensemble" or decision["aggregation"] != "uniform_probability"
            or decision["degradation"] is not True or decision["ema_decay"] != .99
            or type(decision["schedule_epochs"]) is not int or decision["schedule_epochs"] != 15
            or decision["evidence_no_test_used"] is not True or not isinstance(decision["evidence"], dict)
            or isinstance(decision["label_threshold"], bool) or not isinstance(decision["label_threshold"], (int, float))
            or not math.isfinite(decision["label_threshold"]) or not 0 <= decision["label_threshold"] <= 1
            or not isinstance(decision["members"], list) or len(decision["members"]) != 2):
        raise ValueError("Frozen ensemble decision differs from the qualified training-only recipe")
    for index, (architecture, epochs) in enumerate(zip(ARCHITECTURES, (5, 8))):
        member = decision["members"][index]
        fields = {"architecture", "stop_epochs", "layer_lr_decay"}
        if index == 0: fields |= {"noise_weight_map_sha256", "noise_rule"}
        if (not isinstance(member, dict) or set(member) != fields or member["architecture"] != architecture
                or type(member["stop_epochs"]) is not int or member["stop_epochs"] != epochs
                or isinstance(member["layer_lr_decay"], bool) or member["layer_lr_decay"] != 1):
            raise ValueError("Expected ConvNeXt5 with noise only, then unweighted B0-8; no untested layer decay")
        if index == 0:
            digest = member["noise_weight_map_sha256"]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest) or member["noise_rule"] != RULE):
                raise ValueError("ConvNeXt requires the frozen full-training noise rule and map SHA256")
    return decision


def member_configs(decision, initializations, weights, workers=2, device="cuda:0"):
    configs = []
    for member in decision["members"]:
        architecture = member["architecture"]
        config = configuration(workers, initialization=initializations[architecture])
        config.update(name=architecture + "_ensemble_full_refit", architecture=architecture,
                      artifact_purpose="final_submission", max_epochs=15, schedule_epochs=15,
                      stop_epochs=member["stop_epochs"], threshold=decision["label_threshold"],
                      layer_lr_decay=1., device=device, ram_training=True, channels_last=True)
        if decision.get("storage_precision") == "float32": config["export_half"] = False
        if architecture == "convnext_tiny": config["sample_weights"] = weights
        configs.append(config)
    result = {"architecture": "probability_ensemble", "aggregation": "uniform_probability",
              "threshold": decision["label_threshold"], "members": configs}
    if decision.get("storage_precision") == "float32": result["storage_precision"] = "float32"
    return result


def qualified_weights(path, split, rows, expected_digest):
    weights = load_final_weights(path, split, rows, expected_digest)
    if (len(weights) != 7000 or set(weights) != set(rows.image_id) or set(weights.values()) != {.25, 1.}
            or sum(value == .25 for value in weights.values()) != 175):
        raise ValueError("Qualified final ConvNeXt recipe requires exactly175 down-weighted samples among7000")
    return weights


def member_signature(root_signature, config):
    return canonical_hash({"run_signature": root_signature, "architecture": config["architecture"], "config": config})


def source_hashes():
    paths = {path.name: path for path in ROOT.glob("*.py")}
    for name in ("budget", "data", "model_def", "noise_weights", "ram_training", "run_refit", "run_stage_b",
                 "step_budget", "train", "augment", "metrics", "validation_cache", "resource_guard"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            paths[path.name] = path
    return {name: fingerprint(path) for name, path in sorted(paths.items())}


def validate_checkpoint(checkpoint, config, signature, software_test, train_count):
    epoch = checkpoint.get("epoch")
    history = checkpoint.get("history", [])
    if (checkpoint.get("format_version") != 1 or checkpoint.get("signature") != signature
            or checkpoint.get("config") != config or checkpoint.get("software_test") is not software_test
            or type(epoch) is not int or not 1 <= epoch <= config["stop_epochs"]
            or len(history) != epoch or [record.get("epoch") for record in history] != list(range(1, epoch + 1))
            or any(record.get("train_count") != train_count for record in history)):
        raise ValueError("Member checkpoint signature/config/history differs from the frozen ensemble")


def validate_member_export(payload, checkpoint, config, software_test):
    if (payload.get("format_version") != 1 or payload.get("architecture") != config["architecture"]
            or payload.get("preprocessing") != NORMALIZATION or payload.get("label_mapping") != {"0": "real", "1": "fake"}
            or payload.get("threshold") != config["threshold"] or payload.get("dropout") != config["dropout"]
            or payload.get("purpose") != ("software_test" if software_test else "final_submission")
            or set(payload.get("state_dict", {})) != set(checkpoint["ema"]["model"])):
        raise ValueError("Member export metadata/state keys differ from its final EMA")
    for key, expected in checkpoint["ema"]["model"].items():
        expected = (expected.half() if config.get("export_half", True) else expected.float()) if expected.is_floating_point() else expected
        actual = payload["state_dict"][key]
        if actual.dtype != expected.dtype or not actual.equal(expected):
            raise ValueError("Member export differs from exact rounded final EMA")


def run_members(config, rows, store, output, signature, budget, decision_sha256, software_test=False):
    output = Path(output)
    histories, references, payloads = {}, [], []
    for training in config["members"]:
        architecture = training["architecture"]
        directory = output / architecture
        directory.mkdir(exist_ok=True)
        latest, exported = directory / "latest.pt", directory / ("model_weights.pt" if not training.get("export_half", True) else "model_fp16.pt")
        signed = member_signature(signature, training)
        previous = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
        if previous: validate_checkpoint(previous, training, signed, software_test, len(rows))
        complete = previous is not None and previous["epoch"] == training["stop_epochs"] and exported.exists()
        budget.check()
        budget.emit("member_started", architecture=architecture, reuse_completed=complete)
        try:
            if not complete:
                refit(training, rows, store, directory, signed, budget, software_test=software_test)
            elif previous:
                restore_epoch_views(directory, previous["history"])
        finally:
            if latest.exists():
                snapshot = torch.load(latest, map_location="cpu", weights_only=False)
                validate_checkpoint(snapshot, training, signed, software_test, len(rows))
                histories[architecture] = [{"architecture": architecture, **record} for record in snapshot["history"]]
                write_json(output / "history.json", [record for name in ARCHITECTURES for record in histories.get(name, [])])
                del snapshot
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, training, signed, software_test, len(rows))
        if checkpoint["epoch"] != training["stop_epochs"]:
            raise ValueError("Member did not complete the frozen epoch count")
        payload = torch.load(exported, map_location="cpu", weights_only=True)
        validate_member_export(payload, checkpoint, training, software_test)
        payloads.append(payload)
        references.append({"architecture": architecture, "checkpoint": f"{architecture}/latest.pt",
                           "checkpoint_sha256": fingerprint(latest), "signature": signed})
        budget.emit("member_complete", architecture=architecture, epochs=training["stop_epochs"])
        del checkpoint, previous
        gc.collect()
        if not software_test: torch.cuda.empty_cache()
    budget.check()
    export_path = output / ("model_weights.pt" if config.get("storage_precision") == "float32" else "model_fp16.pt")
    export_ensemble(payloads, export_path, config["threshold"], "software_test" if software_test else "final_submission", half=config.get("storage_precision") != "float32")
    sync_export(export_path)
    reference = {"format_version": 2, "ensemble_reference": True, "signature": signature, "config": config,
                 "members": references, "software_test": software_test, "train_count": len(rows),
                 "history": [record for name in ARCHITECTURES for record in histories[name]],
                 "preprocessing": NORMALIZATION, "frozen_decision_sha256": decision_sha256}
    save_checkpoint(output / "latest.pt", reference)
    summary = {"status": "completed", "architecture": "probability_ensemble", "aggregation": "uniform_probability",
               "train_count": len(rows), "stop_epochs": {member["architecture"]: member["stop_epochs"] for member in config["members"]},
               "schedule_epochs": 15, "threshold": config["threshold"], "signature": signature,
               "frozen_decision_sha256": decision_sha256, "export": export_path.name,
               "export_sha256": fingerprint(export_path), "members": references,
               "validation_used": False, "test_data_used": False, "wall_seconds": budget.elapsed}
    write_json(output / "refit_results.json", summary)
    budget.emit("ensemble_refit_complete", **summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "original-split", "conv-initialization", "b0-initialization", "noise-weights", "frozen-decision", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--session-hours", type=float, default=8.)
    parser.add_argument("--session-start-unix", type=float)
    parser.add_argument("--gpu-memory-fraction", type=float, choices=[.5, .75], default=.75)
    parser.add_argument("--workers", type=int, choices=[2], default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.session_hours) or not 0 < args.session_hours <= 8:
        parser.error("Session wall limit must be positive and at most8hours")
    startup = time.monotonic() - STARTED
    if args.session_start_unix is not None:
        if not math.isfinite(args.session_start_unix) or args.session_start_unix > time.time():
            parser.error("Invalid session start timestamp")
        startup = max(startup, time.time() - args.session_start_unix)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    run_lock=None
    if os.name=='posix':
        import fcntl
        run_lock=(output/'.run.lock').open('a');fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget,prior_seconds=fresh_session_budget(output,args.session_hours*3600,scope="ensemble_full_refit",allocated_gpus=1,startup_seconds=startup)
    import multiprocessing
    def hard_stop():
        print("Ensemble refit wall deadline reached; completed checkpoints preserved; exit124", flush=True)
        try:
            for process in multiprocessing.active_children():
                try:
                    process.terminate()
                    process.join(timeout=.25)
                    if process.is_alive(): process.kill()
                except (OSError, ValueError): pass
        finally: os._exit(124)
    timer = threading.Timer(max(.001, budget.remaining), hard_stop)
    timer.daemon = True
    timer.start()
    status, code = "running", 0
    try:
        decision = frozen_recipe(args.frozen_decision)
        from resource_guard import ResourceGuard
        uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not uuid.startswith("GPU-") or "," in uuid:
            raise ValueError("Ensemble refit requires one explicit authorized GPU UUID")
        budget = ResourceGuard(budget, uuid)
        budget.check()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one authorized GPU before ensemble refit")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        torch.set_num_threads(4)
        rows = all_training_rows(args.data_root, args.original_split)
        weights = qualified_weights(args.noise_weights, args.original_split, rows, decision["members"][0]["noise_weight_map_sha256"])
        store = preload_training(rows, manifest_path=output / "training_media_manifest.json", check_deadline=budget.check,
                                 decode_workers=4, progress=lambda event: budget.emit("ram_preload_progress", **event))
        budget.emit("ram_preload_complete", train_count=len(rows), preload_seconds=store.preload_seconds)
        initializations = {"convnext_tiny": args.conv_initialization, "efficientnet_b0": args.b0_initialization}
        config = member_configs(decision, initializations, weights)
        contract = {"config": config, "frozen_decision_sha256": fingerprint(args.frozen_decision),
                    "split_sha256": fingerprint(args.original_split), "noise_weights_sha256": fingerprint(args.noise_weights),
                    "initialization_sha256": {name: fingerprint(path) for name, path in initializations.items()},
                    "media": store.manifest, "source_sha256": source_hashes(),
                    "torch": str(torch.__version__), "torchvision": str(sys.modules["torchvision"].__version__),
                    "device_name": torch.cuda.get_device_name(0)}
        path = output / "run_contract.json"
        if args.resume and not path.exists(): raise ValueError("Cannot resume without the original root run contract")
        if path.exists() and json.loads(path.read_text()) != contract: raise ValueError("Ensemble recipe/source/inputs changed on resume")
        write_json(path, contract)
        for architecture, initialization in initializations.items():
            directory = output / architecture
            directory.mkdir(exist_ok=True)
            if not args.resume: write_json(directory / "initialization.json", initialization_provenance(architecture, initialization))
        run_members(config, rows, store, output, canonical_hash(contract), budget, contract["frozen_decision_sha256"])
        status = "completed"
    except BudgetExceeded as exc:
        status, code = str(exc), 124
    except KeyboardInterrupt:
        status, code = "interrupted", 130
    except Exception as exc:
        status, code = f"failed: {type(exc).__name__}: {exc}", 1
    finally:
        budget.emit("ensemble_refit_stopped", status=status, exit_code=code)
        budget.close()
        timer.cancel()
        write_json(output / "runner_status.json", {"status": status, "exit_code": code, "runner_wall_seconds": budget.elapsed})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
