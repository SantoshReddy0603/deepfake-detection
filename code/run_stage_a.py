"""Bounded one-fold factorial comparison. No CV, ensembles, noise filtering or test inference."""
import argparse
import hashlib
import json
import platform
import os
import sys
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
from augment import CONDITIONS
from budget import Budget, BudgetExceeded, write_json
from data import fingerprint, load_split
from train import experiment_epoch
from validation_cache import build_cache

ROOT = Path(__file__).resolve().parent


def environment():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": torchvision.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "pillow": PIL.__version__, "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "gpu_free_total_bytes": list(torch.cuda.mem_get_info()) if torch.cuda.is_available() else None}


def experiments(args):
    shared = {"device": "cuda", "seed": 42, "ema_decay": .99, "dropout": .2, "backbone_lr": 1e-4,
              "head_lr": 1e-3, "weight_decay": .05, "smoothing": .05, "max_epochs": args.max_epochs,
              "patience": 3, "eval_batch_size": args.eval_batch_size, "workers": args.workers, "colour": "strong"}
    rows = []
    for architecture, batch, acc in [("efficientnet_b0", 32, 1), ("convnext_tiny", 16, 2)]:
        for degradation in [False, True]:
            rows.append({**shared, "name": architecture + ("_on" if degradation else "_off"),
                         "architecture": architecture, "batch_size": batch, "accumulation": acc,
                         "degradation": degradation})
    return rows


def collect_report(out, configs, elapsed, status, cache_seconds, colour_status):
    reports, flat = {}, []
    for config in configs:
        path = out / config["name"] / "history.json"
        history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        reports[config["name"]] = {"config": config, "epochs_completed": len(history), "history": history}
        for epoch in history:
            for stream in ["live", "ema"]:
                for condition, metrics in epoch[stream]["conditions"].items():
                    flat.append({"experiment": config["name"], "epoch": epoch["epoch"], "stream": stream,
                                 "condition": condition, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
    primary = [c for c in configs if not c.get("colour_ablation")]
    common = min((reports[c["name"]]["epochs_completed"] for c in primary), default=0)
    comparisons = []
    for config in configs:
        history = reports[config["name"]]["history"]
        eligible = history[:common] if not config.get("colour_ablation") else history
        for stream in ["live", "ema"]:
            if eligible:
                winner = max(eligible, key=lambda h: (h[stream]["mean_auc"], h[stream]["minimum_auc"]))
                comparisons.append({"experiment": config["name"], "stream": stream, "best_epoch": winner["epoch"],
                                    "epochs_compared": len(eligible), "matched_primary_epochs": not config.get("colour_ablation", False),
                                    **{k: v for k, v in winner[stream].items() if k != "conditions"}})
    document = {"scope": "Stage A one fold only; no competition test predictions", "status": status,
                "elapsed_shared_seconds": elapsed, "cache_build_or_verification_seconds": cache_seconds,
                "condition_count": len(CONDITIONS), "common_primary_epochs": common,
                "colour_check": colour_status, "experiments": reports, "comparisons": comparisons,
                "next_action": "STOP. User go/no-go required before CV, ensemble, noise-handling or final refits."}
    write_json(out / "stage_a_results.json", document)
    if flat:
        pd.DataFrame(flat).to_csv(out / "per_condition_metrics.csv", index=False)
    if comparisons:
        pd.DataFrame(comparisons).to_csv(out / "comparison.csv", index=False)
    lines = ["# Stage A results", "", f"Status: {status}. Shared elapsed: {elapsed / 60:.2f} minutes.",
             f"Cached validation conditions: 16. Common primary epochs: {common}.",
             "", "Selection uses all 16 AUCs; live and EMA are reported separately.",
             "Comparisons use the same completed epoch budget across all four primary cells; optional colour ablations show their own coverage.",
             "", "| Experiment | Stream | Epochs compared | Mean AUC | Minimum AUC | Clean AUC | Clean-degraded gap |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['experiment']} | {r['stream']} | {r['epochs_compared']} | {r['mean_auc']:.5f} | {r['minimum_auc']:.5f} | {r['clean_auc']:.5f} | {r['clean_degraded_gap']:.5f} |")
    lines += ["", "## First complete epoch profiles", "", "| Experiment | Train s | Live 16-condition val s | EMA 16-condition val s | Epoch s |", "|---|---:|---:|---:|---:|"]
    for name, report in reports.items():
        if report["history"]:
            h = report["history"][0]
            lines.append(f"| {name} | {h['train_seconds']:.2f} | {h['live_validation_seconds']:.2f} | {h['ema_validation_seconds']:.2f} | {h['epoch_seconds']:.2f} |")
    lines += ["", f"Colour check: {colour_status}", "", document["next_action"],
              "No hidden/test-image labels, normalization, matching, pseudo-labels or test BN adaptation are used.",
              "The runner wall-clock cap is not a measurement of Kaggle session billing before/after this process; stop unused GPU sessions."]
    (out / "STAGE_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default=str(ROOT / "splits/fold0.csv"))
    parser.add_argument("--cache-dir")
    parser.add_argument("--budget-minutes", type=float, default=105)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_epochs <= 12 or args.workers < 0 or args.eval_batch_size < 1 or not 0 < args.budget_minutes <= 120:
        parser.error("Invalid epochs/workers/evaluation batch/shared budget")
    training, validation = load_split(args.data_root, args.split)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir) if args.cache_dir else out / "validation_cache"
    configs = experiments(args)
    if args.prepare_only:
        manifest = build_cache(validation, cache)
        write_json(out / "cache_ready.json", {"key": manifest["key"], "bytes": manifest["bytes_uint8_payload"]})
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Use --prepare-only for CPU cache creation; real Stage A training requires Kaggle GPU")
    run_contract = {"split_sha256": fingerprint(args.split), "metadata_sha256": fingerprint(Path(args.data_root) / "train_metadata.csv"),
                    "experiments": configs, "purpose": "stage_a_one_fold"}
    contract_path = out / "run_contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text()) != run_contract:
        raise ValueError("Run configuration changed; do not mix results or reset a Stage A budget")
    if contract_path.is_file() and not args.resume:
        raise ValueError("Use --resume for an existing run")
    write_json(contract_path, run_contract)
    write_json(out / "environment.json", environment())
    allocated_gpus = max(1, torch.cuda.device_count())
    # Conservatively charge every visible GPU, even though this runner trains only on cuda:0.
    # A T4 x2 allocation therefore receives half the wall time of a single-GPU allocation.
    budget = Budget(out / "budget.json", args.budget_minutes / allocated_gpus, args.resume)
    write_json(out / "allocation_budget.json", {"requested_gpu_minutes": args.budget_minutes,
               "visible_allocated_gpus": allocated_gpus, "training_gpus_used": 1,
               "wall_minutes_cap": args.budget_minutes / allocated_gpus,
               "scope": "conservative process bound; external session setup/idle billing is not measured"})
    watchdog = None
    status, cache_seconds, colour_status = "running", 0., "not triggered"
    done, states = set(), {}
    colour_path = out / "colour_ablation_configs.json"
    if colour_path.is_file():
        configs.extend(json.loads(colour_path.read_text()))
        colour_status = "triggered; optional ablation coverage recorded below"
    try:
        budget.check()
        def hard_stop():
            print("Hard Stage A deadline reached; terminating this training process (exit 124).", file=sys.stderr, flush=True)
            os._exit(124)
        watchdog = threading.Timer(budget.remaining, hard_stop)
        watchdog.daemon = True
        watchdog.start()
        began = time.perf_counter()
        manifest = build_cache(validation, cache, check_deadline=budget.check)
        cache_seconds = time.perf_counter() - began
        while True:
            advanced = False
            for config in list(configs):
                if config["name"] in done:
                    continue
                budget.check()
                path = out / config["name"] / "history.json"
                if path.is_file():
                    history = json.loads(path.read_text())
                    if history and budget.remaining < history[-1]["epoch_seconds"] * 1.3 + 45:
                        raise BudgetExceeded("Insufficient time for another complete, selectable epoch")
                state = experiment_epoch(config, training, cache, manifest["key"], out / config["name"], budget)
                states[config["name"]] = state
                if state["done"]:
                    done.add(config["name"])
                advanced = True
                collect_report(out, configs, budget.elapsed, status, cache_seconds, colour_status)
            # Detect the colour/general degradation gap after every primary has real evidence.
            if not colour_path.exists() and len(states) >= 4:
                candidates = []
                for c in configs[:4]:
                    if c["degradation"]:
                        for stream in ["live", "ema"]:
                            b = states[c["name"]]["best"][stream]
                            if b and max(b["clean_degraded_gap"], b["colour_gap"]) >= .03:
                                candidates.append((b["mean_auc"], c))
                if candidates:
                    _, selected = max(candidates, key=lambda row: row[0])
                    estimate = states[selected["name"]]["history"][-1]["epoch_seconds"]
                    if budget.remaining > 2 * estimate * 1.3 + 60:
                        extras = [{**selected, "name": selected["architecture"] + "_colour_" + colour,
                                   "colour": colour, "colour_ablation": True} for colour in ["none", "grayscale"]]
                        write_json(colour_path, extras)
                        configs.extend(extras)
                        colour_status = "large-gap trigger >=0.03; no-colour/grayscale ablations scheduled within shared cap"
                    else:
                        colour_status = "large gap triggered, ablation budget-blocked; needs user decision at Stage A go/no-go"
            if not advanced or len(done) == len(configs):
                status = "completed_planned_epochs_or_early_stopping"
                break
    except BudgetExceeded as exc:
        status = str(exc)
    except Exception as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        if watchdog is not None:
            watchdog.cancel()
        budget.close()
        report = collect_report(out, configs, budget.elapsed, status, cache_seconds, colour_status)
        print(json.dumps({"status": status, "common_primary_epochs": report["common_primary_epochs"],
                          "report": str(out / "STAGE_A_REPORT.md"), "next_action": report["next_action"]}), flush=True)


if __name__ == "__main__":
    main()
