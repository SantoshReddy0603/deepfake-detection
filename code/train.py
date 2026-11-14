"""One epoch + full cached live/EMA validation, with resumable epoch boundaries."""
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from augment import CONDITIONS
from budget import fsync_directory, write_json
from data import TrainDataset, seed_worker, fingerprint
from metrics import binary_metrics, summarize
from model_def import EMA, build_model, export_model, normalize, NORMALIZATION, initialization_provenance
from validation_cache import CachedDataset


def training_weights(config, rows):
    weights = config.get("sample_weights")
    if weights is None:
        return None
    if not isinstance(weights, dict) or set(weights) != set(rows.image_id):
        raise ValueError("Training weights must match exactly the current training IDs")
    if any(not isinstance(w, (float, int)) or isinstance(w, bool)
           or not math.isfinite(w) or not 0 < w <= 1 for w in weights.values()):
        raise ValueError("Training weights must be finite and in (0, 1]")
    return weights


def optimizer_groups(model, config):
    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    decay = config.get("layer_lr_decay", 1.)
    if not 0 < decay <= 1:
        raise ValueError("Layer learning-rate decay must be in (0, 1]")
    groups = []
    if decay != 1.:
        if config["architecture"] != "convnext_tiny":
            raise ValueError("Layer decay is defined for ConvNeXt-Tiny only")
        for index, layer in enumerate(model.features):
            params = [p for p in layer.parameters() if id(p) not in head_ids]
            if params:
                groups.append({"params": params, "lr": config["backbone_lr"] * decay ** (len(model.features)-1-index)})
    else:
        groups.append({"params": [p for p in model.parameters() if id(p) not in head_ids],
                       "lr": config["backbone_lr"]})
    groups.append({"params": head, "lr": config["head_lr"]})
    if sorted(id(p) for group in groups for p in group["params"]) != sorted(id(p) for p in model.parameters()):
        raise ValueError("Optimizer parameter partition incomplete or duplicated")
    return groups


def training_stop(config):
    stop=config.get("stop_epochs",config["max_epochs"])
    if isinstance(stop,bool) or not isinstance(stop,int) or not 1<=stop<=config["max_epochs"]:
        raise ValueError("Frozen stop epoch must lie within the unchanged cosine horizon")
    return stop


def seed_all(seed, use_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed also queues CUDA seeding on CUDA-capable installations.
    # Seed the CPU generator directly so a CPU smoke never touches CUDA RNGs.
    torch.random.default_generator.manual_seed(seed)
    if use_cuda:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise ValueError("CUDA seeding requested without a CUDA device")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rng_state(use_cuda=False):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.random.default_generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if use_cuda else None}


def restore_rng(state, use_cuda=False):
    if state["cuda"] is not None:
        if not use_cuda:
            raise ValueError("Cannot restore CUDA RNG state into a CPU run")
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the saved CUDA RNG topology")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.default_generator.set_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def sync_export(path):
    # Windows _commit requires a writable descriptor for FlushFileBuffers.
    with Path(path).open("r+b") as handle:
        os.fsync(handle.fileno())
    fsync_directory(Path(path).parent)


def checkpoint_alias(source, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(".pt.tmp")
    shutil.copyfile(source, temporary)
    sync_export(temporary)
    temporary.replace(destination)
    fsync_directory(destination.parent)


def reconcile_epoch_files(directory, payload):
    """Restore JSON views and aliases from our canonical latest checkpoint, without a model."""
    directory = Path(directory)
    history, best = payload["history"], payload["best"]
    recovered = {"rewritten_json": [], "repaired_aliases": [], "missing_exports": []}
    def reconcile_json(path, expected):
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
            actual = None
        if actual != expected:
            write_json(path, expected)
            recovered["rewritten_json"].append(path.relative_to(directory).as_posix())
    reconcile_json(directory / "history.json", history)
    reconcile_json(directory / "best.json", best)
    for record in history:
        reconcile_json(directory / "epochs" / f"epoch_{record['epoch'] + 1:03}.json", record)
    diagnostic = payload.get("diagnostic_best")
    if diagnostic is not None:
        reconcile_json(directory / "diagnostic_best.json", diagnostic)
    for selected, alias in [(best.get("ema"), "best.pt"), (diagnostic, "diagnostic_best.pt")]:
        if not selected or not selected.get("checkpoint"):
            continue
        name = selected["checkpoint"]
        if Path(name).name != name:
            raise ValueError("Canonical checkpoint export must be a filename within its run directory")
        source, destination = directory / name, directory / alias
        if not source.is_file():
            recovered["missing_exports"].append(name)
        elif not destination.is_file() or fingerprint(source) != fingerprint(destination):
            checkpoint_alias(source, destination)
            recovered["repaired_aliases"].append(alias)
    return recovered


def normalized_batch(x, device, image_size=None, channels_last=False):
    x = normalize(x, device)
    if image_size is not None:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x.contiguous(memory_format=torch.channels_last) if channels_last else x


def log_event(budget, event, **values):
    if hasattr(budget, "emit"):
        budget.emit(event, **values)
    else:
        print(json.dumps({"event": event, **values}), flush=True)


@torch.inference_mode()
def evaluate(model, cache, device, batch_size, workers, budget, conditions=CONDITIONS,
             prediction_dir=None, image_size=None, validation_indices=None, channels_last=False):
    model.eval()
    results = {}
    prediction_dir = Path(prediction_dir) if prediction_dir is not None else None
    saved = {}
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
        (prediction_dir / "manifest.json").unlink(missing_ok=True)
    for condition in conditions:
        budget.check()
        dataset = CachedDataset(cache, condition)
        if validation_indices is not None:
            if (not validation_indices or len(set(validation_indices)) != len(validation_indices) or
                    any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(dataset)
                        for i in validation_indices)):
                raise ValueError("Validation indices must be unique, nonempty, in-range integers")
            dataset = Subset(dataset, [int(i) for i in validation_indices])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda", persistent_workers=False)
        labels, probabilities, identifiers, raw_logits = [], [], [], []
        for x, y, ids in loader:
            budget.check()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(normalized_batch(x, device, image_size, channels_last)).reshape(-1)
            if not torch.isfinite(output).all():
                raise FloatingPointError("Nonfinite validation outputs")
            if getattr(model,"outputs_probabilities",False):
                if not ((output>=0)&(output<=1)).all():raise ValueError("Invalid ensemble probability")
                probabilities_tensor=output.float()
                logits=torch.logit(probabilities_tensor.clamp(1e-7,1-1e-7))
            else:
                logits=output
                probabilities_tensor=logits.float().sigmoid()
            p = probabilities_tensor.cpu().numpy()
            labels.extend(y.numpy().tolist())
            probabilities.extend(p.tolist())
            identifiers.extend(ids)
            raw_logits.extend(logits.float().cpu().numpy().tolist())
        results[condition] = binary_metrics(labels, probabilities)
        if prediction_dir is not None:
            logits_array = np.asarray(raw_logits, dtype=np.float32)
            labels_array = np.asarray(labels, dtype=np.int64)
            path = prediction_dir / f"{condition}.npz"
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, image_id=np.asarray(identifiers, dtype=str),
                                    label=labels_array, logit=logits_array,
                                    probability=np.asarray(probabilities, dtype=np.float32),
                                    loss=np.logaddexp(0., logits_array) - labels_array * logits_array)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            fsync_directory(path.parent)
            saved[condition] = {"file": path.name, "sha256": fingerprint(path), "count": len(labels)}
    full = list(conditions) == list(CONDITIONS)
    result = {"conditions": results, "selection_eligible": full}
    if full:
        result.update(summarize(results, CONDITIONS))
    else:
        result["clean_auc"] = results["clean"]["auc"]
    if prediction_dir is not None:
        manifest = json.loads((Path(cache) / "manifest.json").read_text(encoding="utf-8"))
        write_json(prediction_dir / "manifest.json", {"format_version": 1,
                   "cache_key": manifest["key"], "selection_eligible": full,
                   "validation_indices": [int(i) for i in validation_indices] if validation_indices is not None else None,
                   "label_mapping": {"0": "real", "1": "fake"}, "conditions": saved})
    return result


def experiment_epoch(config, rows, cache, cache_key, directory, budget, software_test=False, training_store=None):
    """A deadline/interruption loses only the unfinished epoch; reload latest.pt to retry."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    smoke_cpu = config.get("artifact_purpose") == "smoke_cpu"
    smoke_gpu = config.get("artifact_purpose") == "smoke_gpu"
    if smoke_gpu and device.type != "cuda":
        raise ValueError("GPU smoke requires CUDA")
    image_size = config.get("smoke_image_size")
    if smoke_cpu:
        if device.type != "cpu" or image_size not in (32, 64):
            raise ValueError("CPU smoke requires explicit 32 or 64 pixel smoke_image_size")
    elif image_size is not None:
        raise ValueError("Reduced resolution is reserved for explicit CPU smoke runs")
    if config.get("validation_indices") is not None and not (smoke_cpu or smoke_gpu or software_test):
        raise ValueError("Validation subsampling is reserved for smoke and software fixtures")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no real Stage A CPU fallback training")
    if device.type != "cuda" and not software_test and not smoke_cpu:
        raise ValueError("CPU training is reserved for isolated software fixtures")
    contract_started = time.perf_counter()
    log_event(budget, "training_contract_started", experiment=config["name"], train_images=len(rows))
    stop_epoch=training_stop(config)
    sample_weights = training_weights(config, rows)
    media_hashes = None
    if config.get("ram_training"):
        if training_store is None:
            raise ValueError("RAM training requires a verified preloaded store")
        source_hashes = training_store.train_media_sha256
        media_hashes = [source_hashes[image_id] for image_id in rows.image_id]
    elif not config.get("mounted_training_input"):
        media_hashes = []
        for index, media_path in enumerate(rows.path):
            budget.check()
            media_hashes.append(fingerprint(media_path))
            if (index + 1) % 256 == 0:
                log_event(budget, "training_contract_progress", verified_images=index + 1, train_images=len(rows))
    identity = {"config": config, "cache_key": cache_key,
                "train": rows[["image_id", "label", "group_id"]].to_dict(orient="records"),
                "train_media_sha256": media_hashes,
                "torch": str(torch.__version__), "torchvision": torchvision.__version__, "device": str(device)}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    training_contract_seconds = time.perf_counter() - contract_started
    log_event(budget, "training_contract_complete", experiment=config["name"],
              training_contract_seconds=training_contract_seconds, train_images=len(rows))
    path = directory / "latest.pt"
    # Full RNG/optimizer checkpoints are our own trusted run files, not arbitrary downloaded models.
    previous = torch.load(path, map_location="cpu", weights_only=False) if path.is_file() else None
    if previous and (previous["signature"] != signature or previous["software_test"] != software_test):
        raise ValueError("Resume recipe, data, cache, runtime or artifact purpose changed")
    if previous:
        recovered = reconcile_epoch_files(directory, previous)
        log_event(budget, "epoch_files_reconciled", **recovered)
        if previous["epoch"] + 1 >= stop_epoch or previous["bad_epochs"] >= config["patience"]:
            return {"done": True, "history": previous["history"], "best": previous["best"]}
    seed_all(config["seed"], use_cuda=device.type == "cuda")
    initialization_started = time.perf_counter()
    log_event(budget, "model_init_started", experiment=config["name"])
    initialization_path = config.get("initialization_path")
    model_options = {"pretrained": previous is None and not software_test and not smoke_cpu,
                     "dropout": config["dropout"]}
    if initialization_path is not None and previous is None and not software_test and not smoke_cpu:
        model_options.update(pretrained=False, initialization_path=initialization_path)
    model = build_model(config["architecture"], **model_options).to(device)
    if config.get("channels_last"):
        model.to(memory_format=torch.channels_last)
    provenance = previous["initialization"] if previous else (
        {"source": "random software fixture initialization"} if software_test else (
            {"source": "random initialization for CPU software smoke only", "input_size": image_size,
             "competition_selection_eligible": False} if smoke_cpu else (
                 initialization_provenance(config["architecture"], path=initialization_path)
                 if initialization_path is not None else initialization_provenance(config["architecture"]))))
    write_json(directory / "initialization.json", provenance)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config), weight_decay=config["weight_decay"])
    steps_per_epoch = math.ceil(math.ceil(len(rows) / config["batch_size"]) / config["accumulation"])
    total, warmup = steps_per_epoch * config["max_epochs"], steps_per_epoch
    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1., (step - warmup) / max(1, total - warmup))
        return .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, config["ema_decay"])
    epoch, step, history = 0, 0, []
    best, bad_epochs = {"live": None, "ema": None}, 0
    diagnostic_best = previous.get("diagnostic_best") if previous else None
    if previous:
        model.load_state_dict(previous["model"], strict=True)
        ema.load_state_dict(previous["ema"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        restore_rng(previous["rng"], use_cuda=device.type == "cuda")
        epoch, step, history = previous["epoch"] + 1, previous["global_step"], previous["history"]
        best, bad_epochs = previous["best"], previous["bad_epochs"]
    model_init_seconds = time.perf_counter() - initialization_started
    log_event(budget, "model_init_complete", experiment=config["name"], model_init_seconds=model_init_seconds,
              scope="model, optimizer, EMA and checkpoint restoration")
    if epoch >= stop_epoch or bad_epochs >= config["patience"]:
        return {"done": True, "history": history, "best": best}
    budget.check()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator().manual_seed(config["seed"] + epoch)
    if config.get("ram_training"):
        from ram_training import RAMTrainDataset
        dataset = RAMTrainDataset(training_store, rows, config["degradation"], config["colour"])
    else:
        dataset = TrainDataset(rows, config["degradation"], config["colour"])
    loader_options = {}
    if config["workers"]:
        loader_options.update(prefetch_factor=2)
        if config.get("ram_training"):
            loader_options["multiprocessing_context"] = "fork"
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, generator=generator,
                        num_workers=config["workers"], worker_init_fn=seed_worker, pin_memory=device.type == "cuda", **loader_options)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total, seen, updates, skipped = 0., 0, 0, 0
    criterion = torch.nn.BCEWithLogitsLoss()
    first_batch_loader_seconds = first_batch_compute_seconds = None
    loader_started = time.perf_counter()
    for index, (x, y, image_ids) in enumerate(loader):
        if index == 0:
            first_batch_loader_seconds = time.perf_counter() - loader_started
        batch_started = time.perf_counter()
        budget.check()
        x, y = normalized_batch(x, device, image_size, config.get("channels_last", False)), y.to(device)
        target = y * (1 - config["smoothing"]) + config["smoothing"] * .5
        acc = config["accumulation"]
        window_samples = min(config["batch_size"] * acc,
                             len(rows) - (index // acc) * config["batch_size"] * acc)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x).reshape(-1)
            if sample_weights is None:
                loss = criterion(logits, target)
            else:
                weights = torch.tensor([sample_weights[str(i)] for i in image_ids], device=device, dtype=logits.dtype)
                loss = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        scaler.scale(loss * len(y) / window_samples).backward()
        loss_total += float(loss.detach()) * len(y)
        seen += len(y)
        if (index + 1) % acc == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled())
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            successful = not scaler.is_enabled() or scaler.get_scale() >= old_scale
            if successful:
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
            first_batch_compute_seconds = time.perf_counter() - batch_started
            log_event(budget, "first_batch_complete", experiment=config["name"], epoch=epoch + 1,
                      loader_seconds=first_batch_loader_seconds, compute_seconds=first_batch_compute_seconds,
                      scope="loader includes fetch/augmentation; compute includes normalization, forward/backward and any scheduled optimizer update")
    if not updates:
        raise FloatingPointError("No optimizer update completed this epoch")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    log_event(budget, "training_complete", train_seconds=train_seconds, train_batches=len(loader),
              mean_seconds_per_batch=train_seconds / len(loader),
              steady_seconds_per_batch=((train_seconds - first_batch_loader_seconds - first_batch_compute_seconds) / (len(loader) - 1)) if len(loader) > 1 else None)
    ema_only = config.get("validation_policy") in ("ema_clean3_full_after", "ema_full", "ema_clean_until_final", "ema_clean_only")
    conditions = ("clean",) if config.get("validation_policy") in ("ema_clean3_full_after", "live_ema_clean3_full_after", "live_clean_ema_clean3_full_after") and epoch < 3 else CONDITIONS
    if config.get("validation_policy") == "ema_clean_until_final" and epoch < stop_epoch - 1:
        conditions = ("clean",)
    if config.get("validation_policy")=="ema_clean_only":conditions=("clean",)
    live_start = time.perf_counter()
    live = None if ema_only else evaluate(model, cache, device, config["eval_batch_size"], config["workers"], budget,
                                         conditions=("clean",) if config.get("validation_policy") == "live_clean_ema_clean3_full_after" else conditions, image_size=image_size,
                                         validation_indices=config.get("validation_indices"), channels_last=config.get("channels_last", False))
    live_seconds = time.perf_counter() - live_start
    ema_start = time.perf_counter()
    prediction_dir = directory / "predictions" / f"epoch_{epoch + 1:02}_ema" if config.get("save_validation_predictions") else None
    averaged = evaluate(ema.model, cache, device, config["eval_batch_size"], config["workers"], budget,
                        conditions=conditions, prediction_dir=prediction_dir, image_size=image_size,
                        validation_indices=config.get("validation_indices"), channels_last=config.get("channels_last", False))
    ema_seconds = time.perf_counter() - ema_start
    improved = False
    for name, candidate, state_model in [("live", live, model), ("ema", averaged, ema.model)]:
        if candidate is None or not candidate["selection_eligible"]:
            continue
        score = (candidate["mean_auc"], candidate["minimum_auc"])
        if best[name] is None or score > (best[name]["mean_auc"], best[name]["minimum_auc"]):
            best[name] = {"epoch": epoch, **{k: v for k, v in candidate.items() if k != "conditions"}}
            checkpoint_name = f"best_epoch_{epoch:02}_{name}.pt"
            best[name]["checkpoint"] = checkpoint_name
            if name == "ema" and prediction_dir is not None:
                best[name]["predictions"] = (prediction_dir / "manifest.json").relative_to(directory).as_posix()
            export_model(state_model, config["architecture"], directory / checkpoint_name,
                         purpose="software_test" if software_test else config.get("artifact_purpose", "stage_a_validation"), half=False)
            sync_export(directory / checkpoint_name)
            if name == "ema":
                checkpoint_alias(directory / checkpoint_name, directory / "best.pt")
            improved = True
    if not averaged["selection_eligible"]:
        clean_auc = averaged["clean_auc"]
        if diagnostic_best is None or clean_auc > diagnostic_best["clean_auc"]:
            diagnostic_best = {"epoch": epoch, "clean_auc": clean_auc, "selection_eligible": False,
                               "reason": "clean-only diagnostic; never a full16 selected model",
                               "checkpoint": f"diagnostic_epoch_{epoch:02}_ema.pt"}
            diagnostic_path = directory / diagnostic_best["checkpoint"]
            export_model(ema.model, config["architecture"], diagnostic_path,
                         purpose="software_test" if software_test else "clean_only_diagnostic", half=False)
            sync_export(diagnostic_path)
            checkpoint_alias(diagnostic_path, directory / "diagnostic_best.pt")
            write_json(directory / "diagnostic_best.json", diagnostic_best)
    bad_epochs = (0 if improved else bad_epochs + 1) if averaged["selection_eligible"] else bad_epochs
    record = {"epoch": epoch, "global_step": step, "train_loss": loss_total / seen, "successful_updates": updates,
              "skipped_amp_updates": skipped, "ema_updates": ema.updates, "train_seconds": train_seconds,
              "train_batches": len(loader), "seconds_per_batch": train_seconds / len(loader),
              "steady_seconds_per_batch": ((train_seconds - first_batch_loader_seconds - first_batch_compute_seconds) / (len(loader) - 1)) if len(loader) > 1 else None,
              "live_validation_seconds": live_seconds, "ema_validation_seconds": ema_seconds,
              "epoch_seconds": time.perf_counter() - started, "live": live, "ema": averaged,
              "model_init_seconds": model_init_seconds,
              "training_contract_seconds": training_contract_seconds,
              "first_batch_loader_seconds": first_batch_loader_seconds,
              "first_batch_compute_seconds": first_batch_compute_seconds,
              "first_batch_seconds": first_batch_loader_seconds + first_batch_compute_seconds,
              "validation_seconds": live_seconds + ema_seconds,
              "smoke_image_size": image_size,
              "competition_selection_eligible": not (smoke_cpu or smoke_gpu) and averaged["selection_eligible"],
              "learning_rates": [group["lr"] for group in optimizer.param_groups],
              "validation_scope": "full16" if averaged["selection_eligible"] else "clean_only_not_selectable",
              "best_so_far": {k: dict(v) if v is not None else None for k, v in best.items()},
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    history.append(record)
    payload = {"format_version": 1, "signature": signature, "config": config, "cache_key": cache_key,
               "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
               "global_step": step, "rng": rng_state(use_cuda=device.type == "cuda"), "history": history, "best": best,
               "bad_epochs": bad_epochs, "preprocessing": NORMALIZATION,
               "diagnostic_best": diagnostic_best,
               "label_mapping": {"0": "real", "1": "fake"}, "software_test": software_test}
    payload["initialization"] = provenance
    save_checkpoint(path, payload)
    write_json(directory / "history.json", history)
    write_json(directory / "best.json", best)
    write_json(directory / "epochs" / f"epoch_{epoch + 1:03}.json", record)
    budget.heartbeat(force=True)
    log_event(budget, "epoch_committed", experiment=config["name"], epoch=epoch,
              epoch_seconds=record["epoch_seconds"], train_seconds=train_seconds,
              live_validation_seconds=live_seconds, ema_validation_seconds=ema_seconds,
              live_clean_auc=live.get("clean_auc") if live is not None else None,
              live_mean_auc=live.get("mean_auc") if live is not None else None,
              live_minimum_auc=live.get("minimum_auc") if live is not None else None,
              ema_minimum_auc=averaged.get("minimum_auc"),
              ema_mean_auc=averaged.get("mean_auc"), ema_clean_auc=averaged["clean_auc"],
              model_init_seconds=model_init_seconds, first_batch_seconds=record["first_batch_seconds"],
              training_contract_seconds=training_contract_seconds,
              remaining_shared_minutes=budget.remaining / 60)
    return {"done": epoch + 1 >= stop_epoch or bad_epochs >= config["patience"], "history": history, "best": best}
