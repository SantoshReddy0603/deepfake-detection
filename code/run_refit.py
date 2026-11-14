"""Refit a frozen recipe on all official training images without validation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

PROCESS_STARTED = time.monotonic()

import pandas as pd
import torch
from torch.utils.data import DataLoader

from budget import BudgetExceeded, write_json
from data import fingerprint, load_split, seed_worker
from model_def import EMA, NORMALIZATION, build_model, export_model, initialization_provenance
from ram_training import RAMTrainDataset, preload_training
from run_stage_b import configuration
from step_budget import StepBudget
from train import normalized_batch, optimizer_groups, training_weights, restore_rng, rng_state, save_checkpoint, seed_all, sync_export

ROOT = Path(__file__).resolve().parent


def frozen_recipe(path, schedule_epochs, stop_epochs, threshold):
    decision = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"architecture", "degradation", "ema_decay", "schedule_epochs", "stop_epochs",
                "label_threshold", "evidence_no_test_used"}
    allowed = required | {"layer_lr_decay", "evidence", "noise_weight_map_sha256", "noise_rule"}
    if not isinstance(decision, dict) or not required.issubset(decision) or set(decision) - allowed:
        raise ValueError("Frozen decision must contain only supported recipe and evidence fields")
    if "evidence" in decision and not isinstance(decision["evidence"], dict):
        raise ValueError("Frozen decision evidence must be a dictionary")
    value = decision.get("label_threshold")
    value = value.get("threshold") if isinstance(value, dict) else value
    if (decision.get("architecture") != "convnext_tiny" or decision.get("degradation") is not True
            or decision.get("ema_decay") != .99 or decision.get("evidence_no_test_used") is not True
            or decision.get("schedule_epochs") != schedule_epochs or decision.get("stop_epochs") != stop_epochs
            or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1 or value != threshold):
        raise ValueError("CLI recipe/threshold must match the frozen training-only decision")
    decay = decision.get("layer_lr_decay", 1.)
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 < decay <= 1:
        raise ValueError("Frozen layer_lr_decay must be in (0,1]")
    if "noise_weight_map_sha256" in decision or "noise_rule" in decision:
        if decay != 1.:raise ValueError("Noise weighting and layer decay were not jointly validated")
        from noise_weights import RULE
        digest=decision.get("noise_weight_map_sha256","")
        if (not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest)
                or decision.get("noise_rule")!=RULE):
            raise ValueError("Frozen noise recipe requires the exact rule and SHA256 weight map")
    return decision


def cosine_factor(step, steps_per_epoch, schedule_epochs):
    warmup = steps_per_epoch
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1., (step - warmup) / max(1, steps_per_epoch * schedule_epochs - warmup))
    return .5 * (1 + math.cos(math.pi * progress))


def all_training_rows(data_root, original_split):
    train, validation = load_split(data_root, original_split)
    rows = pd.concat([train, validation]).sort_values("image_id").reset_index(drop=True)
    if len(rows) != 7000 or rows.image_id.duplicated().any() or not rows.label.isin([0, 1]).all():
        raise ValueError("Final refit requires exactly all7000 official training images")
    return rows


def restore_epoch_views(output, history):
    for record in history:
        write_json(output / "epochs" / f"epoch_{record['epoch']:03}.json", record)
    write_json(output / "history.json", history)


def refit(config, rows, store, output, signature, budget, software_test=False):
    """The latest checkpoint is canonical; incomplete epochs are replayed on resume."""
    output = Path(output)
    sample_weights=training_weights(config,rows)
    latest = output / "latest.pt"
    device = torch.device(config["device"])
    if device.type != "cuda" and not software_test:
        raise ValueError("Final refit requires the fixed CUDA accelerator")
    previous = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    if previous and (previous["signature"] != signature or previous["software_test"] != software_test):
        raise ValueError("Refit resume recipe, inputs, source or runtime changed")
    seed_all(config["seed"], use_cuda=device.type == "cuda")
    started = time.perf_counter()
    options = {"pretrained": False, "dropout": config["dropout"]}
    if previous is None and not software_test:
        options["initialization_path"] = config["initialization_path"]
    model = build_model(config["architecture"], **options).to(device)
    if config["channels_last"]:
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config), weight_decay=config["weight_decay"])
    steps_per_epoch = math.ceil(math.ceil(len(rows) / config["batch_size"]) / config["accumulation"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_factor(step, steps_per_epoch, config["schedule_epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, config["ema_decay"])
    history, step, first_epoch = [], 0, 1
    if previous:
        model.load_state_dict(previous["model"], strict=True)
        ema.load_state_dict(previous["ema"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        restore_rng(previous["rng"], use_cuda=device.type == "cuda")
        history, step, first_epoch = previous["history"], previous["global_step"], previous["epoch"] + 1
        restore_epoch_views(output, history)
    budget.emit("model_init_complete", model_init_seconds=time.perf_counter() - started)
    for epoch in range(first_epoch, config["stop_epochs"] + 1):
        budget.check()
        epoch_started = time.perf_counter()
        dataset = RAMTrainDataset(store, rows, config["degradation"], config["colour"])
        options = {"prefetch_factor": 2, "multiprocessing_context": "fork"} if config["workers"] else {}
        generator = torch.Generator().manual_seed(config["seed"] + epoch - 1)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, generator=generator,
                            num_workers=config["workers"], worker_init_fn=seed_worker,
                            pin_memory=device.type == "cuda", **options)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen, updates, skipped = 0., 0, 0, 0
        first_loader = first_compute = None
        loader_started = time.perf_counter()
        for index, (x, y, image_ids) in enumerate(loader):
            if index == 0:
                first_loader = time.perf_counter() - loader_started
            compute_started = time.perf_counter()
            budget.check()
            x = normalized_batch(x, device, channels_last=config["channels_last"])
            y = y.to(device)
            target = y * (1 - config["smoothing"]) + config["smoothing"] * .5
            acc = config["accumulation"]
            window_samples = min(config["batch_size"] * acc,
                                 len(rows) - (index // acc) * config["batch_size"] * acc)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x).reshape(-1)
                if sample_weights is None:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                else:
                    weights=torch.tensor([sample_weights[str(i)] for i in image_ids],device=device,dtype=logits.dtype)
                    loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction="none")*weights).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite full-train refit loss")
            scaler.scale(loss * len(y) / window_samples).backward()
            total_loss += float(loss.detach()) * len(y)
            seen += len(y)
            if (index + 1) % acc == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not scaler.is_enabled() or scaler.get_scale() >= old_scale:
                    scheduler.step()
                    ema.update(model)
                    step += 1
                    updates += 1
                else:
                    skipped += 1
                optimizer.zero_grad(set_to_none=True)
            if index == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                first_compute = time.perf_counter() - compute_started
                budget.emit("first_batch_complete", epoch=epoch, loader_seconds=first_loader,
                            compute_seconds=first_compute)
        if not updates:
            raise FloatingPointError("No optimizer update completed in refit epoch")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - epoch_started
        record = {"epoch": epoch, "global_step": step, "train_loss": total_loss / seen,
                  "train_count": seen, "train_batches": len(loader), "train_seconds": duration,
                  "seconds_per_batch": duration / len(loader), "first_batch_loader_seconds": first_loader,
                  "first_batch_compute_seconds": first_compute, "successful_updates": updates,
                  "skipped_amp_updates": skipped, "ema_updates": ema.updates,
                  "learning_rates": [group["lr"] for group in optimizer.param_groups],
                  "validation": None, "selection": "frozen epoch; no full-train validation"}
        history.append(record)
        save_checkpoint(latest, {"format_version": 1, "signature": signature, "config": config,
                                "software_test": software_test, "model": model.state_dict(),
                                "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                                "epoch": epoch, "global_step": step, "history": history,
                                "rng": rng_state(use_cuda=device.type == "cuda"), "preprocessing": NORMALIZATION})
        restore_epoch_views(output, history)
        budget.emit("epoch_complete", **record)
    if not history or history[-1]["epoch"] != config["stop_epochs"]:
        raise ValueError("Refit did not finish its frozen epoch count")
    budget.check()
    export_path = output / ("model_weights.pt" if not config.get("export_half", True) else "model_fp16.pt")
    export_model(ema.model, config["architecture"], export_path, threshold=config["threshold"],
                 purpose="software_test" if software_test else "final_submission", half=config.get("export_half", True))
    sync_export(export_path)
    result = {"status": "completed", "epochs": config["stop_epochs"],
              "schedule_epochs": config["schedule_epochs"], "train_count": len(rows),
              "threshold": config["threshold"], "signature": signature, "export": export_path.name,
              "export_sha256": fingerprint(export_path), "validation_used": False,
              "test_data_used": False, "wall_seconds": budget.elapsed}
    write_json(output / "refit_results.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--original-split", required=True)
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--frozen-decision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schedule-epochs", type=int, default=15)
    parser.add_argument("--stop-epochs", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--session-hours", type=float, default=8.)
    parser.add_argument("--session-start-unix", type=float)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.75)
    parser.add_argument("--yield-on-other-gpu-process", action="store_true",
                        help="Accepted explicitly for launchers; shared-host guarding is always enabled")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--noise-weights", help="Validated all7000 OOF weights for a frozen noise-winning recipe")
    args = parser.parse_args(argv)
    if not 1 <= args.stop_epochs <= args.schedule_epochs <= 100 or not 0 <= args.workers <= 4:
        parser.error("Invalid frozen epochs or worker count")
    if not math.isfinite(args.session_hours) or not 0 < args.session_hours <= 8:
        parser.error("Session wall limit must be positive and at most8hours")
    if not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= .8:
        parser.error("GPU memory fraction must be positive and at most0.8")
    decision = frozen_recipe(args.frozen_decision, args.schedule_epochs, args.stop_epochs, args.threshold)
    startup = time.monotonic() - PROCESS_STARTED
    if args.session_start_unix is not None:
        if not math.isfinite(args.session_start_unix) or args.session_start_unix > time.time():
            parser.error("Invalid session start timestamp")
        startup = max(startup, time.time() - args.session_start_unix)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    budget = StepBudget(output / "step_budget.json", args.session_hours * 3600,
                        scope="final_full_train_refit", resume=args.resume, allocated_gpus=1,
                        startup_seconds=startup)
    def hard_stop():
        print("Refit session wall deadline reached; completed epoch checkpoints preserved; exit124", flush=True)
        os._exit(124)
    watchdog = threading.Timer(budget.remaining, hard_stop)
    watchdog.daemon = True
    watchdog.start()
    status, code = "running", 0
    try:
        from resource_guard import ResourceGuard
        uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not uuid.startswith("GPU-") or "," in uuid:
            raise ValueError("Shared-host guard requires one explicit GPU UUID")
        budget = ResourceGuard(budget, uuid)
        budget.check()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one authorized GPU before refit")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        torch.set_num_threads(4)
        rows = all_training_rows(args.data_root, args.original_split)
        store = preload_training(rows, manifest_path=output / "training_media_manifest.json",
                                 check_deadline=budget.check, decode_workers=4,
                                 progress=lambda value: budget.emit("ram_preload_progress", **value))
        budget.emit("ram_preload_complete", preload_seconds=store.preload_seconds, train_count=len(rows))
        config = configuration(args.workers, initialization=args.initialization)
        config.update(name="convnext_tiny_final_refit", artifact_purpose="final_submission",
                      max_epochs=args.schedule_epochs, device=args.device,
                      schedule_epochs=args.schedule_epochs, stop_epochs=args.stop_epochs,
                      threshold=args.threshold, channels_last=True, ram_training=True,
                      layer_lr_decay=decision.get("layer_lr_decay", 1.))
        if decision.get("noise_weight_map_sha256"):
            if not args.noise_weights:raise ValueError("Frozen noise recipe requires --noise-weights")
            from noise_weights import load_final_weights
            config["sample_weights"]=load_final_weights(args.noise_weights,args.original_split,rows,decision["noise_weight_map_sha256"])
        elif args.noise_weights:
            raise ValueError("Unweighted frozen recipe cannot silently accept noise weights")
        contract = {"config": config, "frozen_decision_sha256": fingerprint(args.frozen_decision),
                    "split_sha256": fingerprint(args.original_split),
                    "initialization_sha256": fingerprint(args.initialization),
                    "media": store.manifest,
                    "source_sha256": {path.name: fingerprint(path) for path in sorted(ROOT.glob("*.py"))},
                    "torch": str(torch.__version__), "device_name": torch.cuda.get_device_name(0)}
        path = output / "run_contract.json"
        if path.exists() and (not args.resume or json.loads(path.read_text()) != contract):
            raise ValueError("Existing refit contract differs or --resume is missing")
        write_json(path, contract)
        signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        if not args.resume and (output / "latest.pt").exists():
            raise ValueError("Existing checkpoint requires explicit --resume")
        if not args.resume:
            write_json(output / "initialization.json", initialization_provenance("convnext_tiny", args.initialization))
        refit(config, rows, store, output, signature, budget)
        status = "completed"
    except BudgetExceeded as exc:
        status, code = str(exc), 124
    except KeyboardInterrupt:
        status, code = "interrupted", 130
    except Exception as exc:
        status, code = f"failed: {type(exc).__name__}: {exc}", 1
    finally:
        budget.emit("refit_stopped", status=status, exit_code=code)
        budget.close()
        watchdog.cancel()
        write_json(output / "runner_status.json", {"status": status, "exit_code": code,
                                                   "runner_wall_seconds": budget.elapsed})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
