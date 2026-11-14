"""Verify final weights in sequential isolated offline GPU processes."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

PROCESS_STARTED = time.monotonic()

from budget import write_json

ROOT = Path(__file__).resolve().parent


class VerificationTerminated(BaseException):
    pass


def termination_requested(signum, frame):
    # Repeated supervisor signals must not interrupt cleanup of our child group.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise VerificationTerminated("Controller received SIGTERM; stopping its active child")


def arm_child_watchdog(limit_seconds, startup=PROCESS_STARTED):
    import multiprocessing
    def hard_stop():
        print("Verification child wall deadline reached; exit124", file=sys.stderr, flush=True)
        try:
            # active_children contains only processes created by this process, never peers.
            for process in multiprocessing.active_children():
                try:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=.25)
                        if process.is_alive():
                            process.kill()
                except (OSError, ValueError):
                    pass
        finally:
            os._exit(124)
    timer = threading.Timer(max(.001, limit_seconds - (time.monotonic() - startup)), hard_stop)
    timer.daemon = True
    timer.start()
    return timer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_recipe(config):
    recipe = {key: config[key] for key in ("architecture", "degradation", "ema_decay", "schedule_epochs",
                                          "stop_epochs", "threshold", "dropout")}
    recipe["layer_lr_decay"] = config.get("layer_lr_decay", 1.)
    if config.get("export_half") is False: recipe["storage_precision"] = "float32"
    if config.get("sample_weights") is not None:
        recipe["noise_weight_map_sha256"] = hashlib.sha256(json.dumps({str(k): float(v) for k, v in config["sample_weights"].items()}, sort_keys=True).encode()).hexdigest()
    return recipe


def contract_anchor(contract):
    decision_hash = contract.get("frozen_decision_sha256")
    if (not isinstance(decision_hash, str) or len(decision_hash) != 64
            or any(character not in "0123456789abcdef" for character in decision_hash)):
        raise ValueError("Refit contract has no valid frozen decision SHA256")
    config = contract["config"]
    if config.get("architecture") == "probability_ensemble":
        members = config.get("members", [])
        if (config.get("aggregation") != "uniform_probability" or len(members) != 2
                or [member.get("architecture") for member in members] != ["convnext_tiny", "efficientnet_b0"]
                or any(member.get("threshold") != config["threshold"] for member in members)
                or members[1].get("sample_weights") is not None):
            raise ValueError("Invalid frozen ensemble member configuration")
        recipe = {"architecture": "probability_ensemble", "aggregation": "uniform_probability",
                  "threshold": config["threshold"], "members": [scalar_recipe(member) for member in members]}
    else:
        recipe = scalar_recipe(config)
    if config.get("storage_precision") == "float32": recipe["storage_precision"] = "float32"
    return {"frozen_decision_sha256": decision_hash, "refit_recipe": recipe,
            "refit_contract_signature": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()}


def bind_reference_contract(trusted, contract):
    anchor = contract_anchor(contract)
    if trusted.get("signature") != anchor["refit_contract_signature"] or trusted.get("config") != contract["config"]:
        raise ValueError("Reference checkpoint signature/config differs from its refit contract")
    if trusted.get("ensemble_reference") is True and trusted.get("frozen_decision_sha256") != anchor["frozen_decision_sha256"]:
        raise ValueError("Ensemble reference decision hash differs from its refit contract")
    return anchor


def check_prediction_hash(directory, result):
    if result.get("predictions_csv_sha256") != sha256(Path(directory) / "predictions.csv"):
        raise ValueError("Predictions CSV checksum differs from the generated child output")


def runtime_sources(source_dir, modules=None):
    paths = {name: (Path(source_dir) / name).resolve() for name in ("inference_script.py", "model_def.py")}
    if modules is not None:
        if set(modules) != set(paths) or any(Path(modules[name].__file__).resolve() != path for name, path in paths.items()):
            raise ValueError("Imported inference/model modules do not match the requested source directory")
    return {name: sha256(path) for name, path in paths.items()}


def parity_difference(reference, exported, tolerance=.003):
    if len(reference) != 256 or len(exported) != 256:
        raise ValueError("Parity requires exactly256 cached clean images")
    if any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in [*reference, *exported]):
        raise ValueError("Parity probabilities must be finite and in[0,1]")
    difference = max(abs(float(a) - float(b)) for a, b in zip(reference, exported))
    if difference > tolerance:
        raise ValueError(f"FP16 probability parity failed: {difference} > {tolerance}")
    return difference


def read_predictions(path, expected_ids, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Submission threshold must be finite and in[0,1]")
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image_id", "label", "confidence"]:
            raise ValueError("Submission header must be image_id,label,confidence")
        rows = []
        for row in reader:
            if None in row or None in row.values() or row["label"] not in {"0", "1"}:
                raise ValueError("Malformed submission row")
            rows.append((row["image_id"], int(row["label"]), float(row["confidence"])))
    if ([row[0] for row in rows] != list(expected_ids) or len(set(expected_ids)) != len(expected_ids)
            or any(not math.isfinite(p) or not 0 <= p <= 1 or label != int(p >= threshold)
                   for _, label, p in rows)):
        raise ValueError("Submission IDs, probabilities or threshold labels are invalid")
    return rows


def repeat_difference(first, second, tolerance=1e-6):
    if not first or [(i, y) for i, y, _ in first] != [(i, y) for i, y, _ in second]:
        raise ValueError("Repeated prediction IDs or labels differ")
    if any(not math.isfinite(p) for _, _, p in first + second):
        raise ValueError("Nonfinite repeat probabilities")
    difference = max(abs(a[2] - b[2]) for a, b in zip(first, second))
    if difference > tolerance:
        raise ValueError(f"Repeat probability difference {difference} exceeds {tolerance}")
    return difference


def block_network():
    import socket
    if getattr(socket.socket, "_offline_verification_block", False):
        return
    def denied(*args, **kwargs):
        raise RuntimeError("Network access blocked during offline verification")
    original = socket.socket
    class OfflineSocket(original):
        _offline_verification_block = True

        def __new__(cls, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
            if family in (socket.AF_INET, socket.AF_INET6):
                return denied()
            return super().__new__(cls, family, type, proto, fileno=fileno)
    socket.socket = OfflineSocket
    socket.create_connection = denied
    socket.getaddrinfo = denied


def offline_worker_init(worker_id):
    """Spawned DataLoader workers need their own internet block; local IPC stays usable."""
    block_network()


def validate_export_identity(payload, trusted):
    config = trusted["config"]
    if (payload.get("purpose") != "final_submission" or trusted.get("software_test") is not False
            or config.get("artifact_purpose") != "final_submission"
            or trusted["epoch"] != config["stop_epochs"]
            or trusted["history"][-1]["train_count"] != 7000
            or config["architecture"] != payload["architecture"]
            or config["threshold"] != payload["threshold"] or config["dropout"] != payload["dropout"]):
        raise ValueError("Export metadata differs from the completed full-training EMA checkpoint")
    expected, actual = trusted["ema"]["model"], payload["state_dict"]
    if set(expected) != set(actual):
        raise ValueError("Export state keys differ from reference EMA")
    floating_count = 0
    for key, value in expected.items():
        if value.is_floating_point():
            value = value.half() if config.get("export_half", True) else value.float()
            floating_count += 1
        if actual[key].dtype != value.dtype or not actual[key].equal(value):
            raise ValueError(f"Export tensor differs from the exact storage-converted reference EMA: {key}")
    if not floating_count:
        raise ValueError("Reference EMA has no floating-point model tensors")


def verified_member_paths(descriptors, config, root_signature, directory):
    directory = Path(directory).resolve()
    configs = config.get("members", [])
    if (not isinstance(descriptors, list) or len(descriptors) != 2 or len(configs) != 2
            or [member.get("architecture") for member in configs] != ["convnext_tiny", "efficientnet_b0"]):
        raise ValueError("Ensemble reference requires its two ordered member descriptors")
    paths = []
    for descriptor, member, epochs in zip(descriptors, configs, [5, 8]):
        architecture = member["architecture"]
        expected_signature = hashlib.sha256(json.dumps({"run_signature": root_signature, "architecture": architecture,
                                                       "config": member}, sort_keys=True).encode()).hexdigest()
        if (not isinstance(descriptor, dict) or set(descriptor) != {"architecture", "checkpoint", "checkpoint_sha256", "signature"}
                or descriptor.get("architecture") != architecture or descriptor.get("checkpoint") != f"{architecture}/latest.pt"
                or descriptor.get("signature") != expected_signature or member.get("stop_epochs") != epochs):
            raise ValueError("Ensemble member path/signature/order differs from the frozen contract")
        path = (directory / descriptor["checkpoint"]).resolve()
        if not path.is_relative_to(directory) or path == directory:
            raise ValueError("Ensemble member checkpoint escapes the reference directory")
        if sha256(path) != descriptor.get("checkpoint_sha256"):
            raise ValueError("Ensemble member checkpoint SHA256 mismatch")
        paths.append(path)
    return paths


def validate_ensemble_identity(payload, trusted, directory, loader):
    config = trusted["config"]
    if (payload.get("format_version") != 2 or trusted.get("format_version") != 2
            or trusted.get("ensemble_reference") is not True or trusted.get("software_test") is not False
            or trusted.get("train_count") != 7000 or config.get("architecture") != "probability_ensemble"
            or payload.get("architecture") != config["architecture"] or payload.get("aggregation") != "uniform_probability"
            or config.get("aggregation") != "uniform_probability" or payload.get("threshold") != config["threshold"]
            or payload.get("purpose") != "final_submission" or payload.get("preprocessing") != trusted.get("preprocessing")
            or payload.get("label_mapping") != {"0": "real", "1": "fake"}
            or not isinstance(payload.get("members"), list) or len(payload["members"]) != 2):
        raise ValueError("Ensemble export/reference metadata differs from the completed full refit")
    paths = verified_member_paths(trusted.get("members"), config, trusted["signature"], directory)
    validated = []
    for exported, member, descriptor, path in zip(payload["members"], config["members"], trusted["members"], paths):
        checkpoint = loader(path)
        history = checkpoint.get("history", [])
        if (set(exported) != {"architecture", "dropout", "state_dict"} or exported["architecture"] != member["architecture"]
                or checkpoint.get("format_version") != 1 or checkpoint.get("signature") != descriptor["signature"]
                or checkpoint.get("config") != member or len(history) != member["stop_epochs"]
                or checkpoint.get("preprocessing") != trusted["preprocessing"]
                or [record.get("epoch") for record in history] != list(range(1, member["stop_epochs"] + 1))
                or any(record.get("train_count") != 7000 for record in history)
                or any(str(value.dtype) != "torch.float32" for value in checkpoint["ema"]["model"].values() if value.is_floating_point())):
            raise ValueError("Trusted ensemble member signature/config/completion differs from root reference")
        single = {"purpose": payload["purpose"], "architecture": exported["architecture"], "threshold": payload["threshold"],
                  "dropout": exported["dropout"], "state_dict": exported["state_dict"]}
        validate_export_identity(single, checkpoint)
        validated.append({"config": member, "ema": checkpoint["ema"]})
    return validated, trusted["members"]


def child(args):
    """Only this function imports GPU/ML dependencies; the controller stays CPU-only."""
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
    watchdog = arm_child_watchdog(args.child_wall_seconds)
    block_network()
    sys.path.insert(0, str(Path(args.source_dir).resolve()))
    import numpy as np
    import torch
    import inference_script as inference
    import model_def
    from model_def import build_model, load_export
    from resource_guard import ResourceGuard
    from step_budget import StepBudget
    source_hashes = runtime_sources(args.source_dir, {"inference_script.py": inference, "model_def.py": model_def})
    directory = Path(args.output)
    home = Path(os.environ["TORCH_HOME"])
    if not home.is_dir() or any(home.iterdir()):
        raise ValueError("Each verification child needs an empty private TORCH_HOME")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid:
        raise ValueError("Verification requires one explicit authorized GPU UUID")
    budget = ResourceGuard(StepBudget(directory / "child_wall.json", 8 * 3600,
                                     scope=f"verification_{args.child}", allocated_gpus=1), uuid)
    try:
        budget.check()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible GPU is required")
        torch.cuda.set_per_process_memory_fraction(.75, 0)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats(0)
        # infer() resets its own timer statistics; retain the whole-child peak here.
        torch.cuda.reset_peak_memory_stats = lambda *a, **kw: None
        hardware = torch.cuda.get_device_name(0)
        original_loader = inference.DataLoader
        def offline_loader(*loader_args, **loader_kwargs):
            loader_kwargs["worker_init_fn"] = offline_worker_init
            return original_loader(*loader_args, **loader_kwargs)
        inference.DataLoader = offline_loader
        original_batches = inference.safe_batches
        def guarded_batches(loader):
            budget.check()
            try:
                for batch in original_batches(loader):
                    budget.check()
                    yield batch
            except inference.LoaderFailure as exc:
                raise RuntimeError("Verification requires the configured two-worker loader without fallback") from exc
        inference.safe_batches = guarded_batches
        if args.child == "parity":
            payload = torch.load(args.weights, map_location="cpu", weights_only=True)
            trusted = torch.load(args.reference, map_location="cpu", weights_only=False)
            contract_path = Path(args.reference).parent / "run_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            anchor = bind_reference_contract(trusted, contract)
            reference_binding = {}
            if payload.get("format_version") == 2:
                checkpoints, descriptors = validate_ensemble_identity(payload, trusted, Path(args.reference).parent,
                    lambda path: torch.load(path, map_location="cpu", weights_only=False))
                reference_binding["reference_members"] = descriptors
            else:
                validate_export_identity(payload, trusted)
            config = trusted["config"]
            manifest = json.loads((Path(args.cache_dir) / "manifest.json").read_text())
            clean_path = Path(args.cache_dir) / "clean.npy"
            if sha256(clean_path) != manifest.get("array_sha256", {}).get("clean"):
                raise ValueError("Clean parity cache checksum mismatch")
            images = np.load(clean_path, mmap_mode="r", allow_pickle=False)
            if (images.dtype != np.uint8 or images.shape != (len(manifest["samples"]), 224, 224, 3)
                    or len(images) < 256 or clean_path.stat().st_size != images.offset + images.nbytes):
                raise ValueError("Invalid clean parity cache")
            if payload.get("format_version") == 2:
                models = []
                for checkpoint in checkpoints:
                    member = checkpoint["config"]
                    model = build_model(member["architecture"], pretrained=False, dropout=member["dropout"])
                    model.load_state_dict(checkpoint["ema"]["model"], strict=True)
                    models.append(model)
                reference = model_def.ProbabilityEnsemble(models).to("cuda:0").eval()
            else:
                reference = build_model(config["architecture"], pretrained=False, dropout=config["dropout"])
                reference.load_state_dict(trusted["ema"]["model"], strict=True)
                reference = reference.to("cuda:0").eval()
            exported, metadata = load_export(args.weights, "cuda:0")
            predictions = [[], []]
            with torch.inference_mode():
                for start in range(0, 256, args.batch_size):
                    budget.check()
                    pixels = torch.from_numpy(np.array(images[start:min(start + args.batch_size, 256)], copy=True)).permute(0, 3, 1, 2)
                    for values, model in zip(predictions, [reference, exported]):
                        values.extend(inference.model_probabilities(model, pixels, torch.device("cuda:0"), []).tolist())
            difference = parity_difference(*predictions)
            report = {"parity_images": 256, "maximum_absolute_probability_difference": difference,
                      "tolerance": .003, "storage_tensor_check": True, "storage_precision": config.get("storage_precision", "float16"), "exact_ema_state_identity": True,
                      "parity_precision": "same CUDA autocast FP16 execution for original EMA and storage-converted export",
                      "reference_sha256": sha256(args.reference), "cache_clean_sha256": sha256(clean_path),
                      "refit_contract_sha256": sha256(contract_path), **anchor, **reference_binding}
        else:
            dataset = inference.InputImages(args.test_dir)
            if len(dataset) != 3000:
                raise ValueError("Official test verification requires exactly3000 input images")
            report = inference.infer(args.test_dir, args.weights, directory / "predictions.csv",
                                     batch_size=args.batch_size, workers=2, device="cuda:0")
            if report.get("export_purpose") != "final_submission" or report["corrupt_image_count"] != 0:
                raise ValueError("Official inference requires final export and zero corrupt inputs")
            rows = read_predictions(directory / "predictions.csv", dataset.ids, report["threshold"])
            inference.validate_rows(rows, dataset.ids, report["threshold"])
            report.update(expected_ids=dataset.ids, schema_check=True, workers=2,
                          predictions_csv_sha256=sha256(directory / "predictions.csv"))
        torch.cuda.synchronize(0)
        report.update(status="passed", hardware=hardware, weights_sha256=sha256(args.weights),
                      runtime_source_sha256=source_hashes,
                      network_blocked=True, empty_torch_home=True, allocated_gpus=1,
                      network_policy="IPv4/IPv6 sockets, DNS and create_connection blocked; local IPC allowed",
                      gpu_memory_fraction=.75, torch_threads=4, batch_size=args.batch_size,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0))
        if any(home.iterdir()):
            raise ValueError("Offline verification unexpectedly wrote to TORCH_HOME")
        if runtime_sources(args.source_dir) != source_hashes:
            raise ValueError("Runtime inference/model sources changed during child verification")
        write_json(directory / "child_result.json", report)
    finally:
        budget.close()
        watchdog.cancel()


def stop_child(process):
    if os.name == "nt":
        if process.poll() is not None:
            return
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # A failed or exited leader can leave its workers behind in our own PGID.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def run_child(command, directory, environment, timeout):
    started = time.monotonic()
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        process = None
        try:
            # Defer SIGTERM until the new process handle is assigned so cleanup owns it.
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM}) if hasattr(signal, "pthread_sigmask") else None
            try:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment, **options)
            finally:
                if previous_mask is not None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            code = process.wait(timeout=max(.01, timeout))
            stop_child(process)
        except BaseException:
            if process is not None:
                stop_child(process)
            raise
    elapsed = time.monotonic() - started
    if code:
        raise RuntimeError(f"Verification child exited {code}; inspect {directory / 'stderr.log'}")
    result = json.loads((directory / "child_result.json").read_text())
    if result.get("status") != "passed":
        raise ValueError("Verification child did not report a passing result")
    result["full_process_wall_seconds_including_imports_and_exit"] = elapsed
    write_json(directory / "process_result.json", result)
    return result


def verify(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    previous_handler = signal.signal(signal.SIGTERM, termination_requested)
    try:
        source_hashes = runtime_sources(args.source_dir)
        if Path(args.weights).stat().st_size >= 5_000_000_000:
            raise ValueError("Weights exceed the competition size limit")
        for name in ["parity", "inference", "repeat"]:
            directory = output / name
            directory.mkdir()
            home = directory / "torch_home"
            home.mkdir()
            environment = dict(os.environ, TORCH_HOME=str(home), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            command = [sys.executable, str(Path(__file__).resolve()), "--child", name,
                       "--weights", str(Path(args.weights).resolve()), "--reference", str(Path(args.reference).resolve()),
                       "--cache-dir", str(Path(args.cache_dir).resolve()), "--test-dir", str(Path(args.test_dir).resolve()),
                       "--source-dir", str(Path(args.source_dir).resolve()), "--output", str(directory),
                       "--batch-size", str(args.batch_size)]
            # Leave time to terminate our process group and flush the final status.
            remaining = args.session_hours * 3600 - (time.monotonic() - started) - 15
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, args.session_hours * 3600)
            command.extend(["--child-wall-seconds", str(remaining)])
            result = run_child(command, directory, environment, remaining)
            if result.get("runtime_source_sha256") != source_hashes:
                raise ValueError("Runtime source hashes differ between the controller and child")
            if name != "parity":
                check_prediction_hash(directory, result)
            results[name] = result
            write_json(output / "progress.json", results)
        a, b = results["inference"], results["repeat"]
        if (runtime_sources(args.source_dir) != source_hashes
                or any(result.get("runtime_source_sha256") != source_hashes for result in results.values())):
            raise ValueError("Runtime inference/model source hashes changed before acceptance")
        check_prediction_hash(output / "inference", a)
        check_prediction_hash(output / "repeat", b)
        contract_path = Path(args.reference).parent / "run_contract.json"
        anchor = contract_anchor(json.loads(contract_path.read_text(encoding="utf-8")))
        parity = results["parity"]
        if (any(parity.get(key) != value for key, value in anchor.items())
                or parity.get("refit_contract_sha256") != sha256(contract_path)
                or parity.get("reference_sha256") != sha256(args.reference)):
            raise ValueError("Refit provenance changed after parity verification")
        reference_binding = {}
        if anchor["refit_recipe"]["architecture"] == "probability_ensemble":
            current_contract = json.loads(contract_path.read_text(encoding="utf-8"))
            verified_member_paths(parity.get("reference_members"), current_contract["config"],
                                  anchor["refit_contract_signature"], Path(args.reference).parent)
            reference_binding["reference_members"] = parity["reference_members"]
        if (len(a["expected_ids"]) != 3000 or a["expected_ids"] != b["expected_ids"]
                or a["threshold"] != b["threshold"]):
            raise ValueError("Repeat input identity or threshold changed")
        first = read_predictions(output / "inference/predictions.csv", a["expected_ids"], a["threshold"])
        second = read_predictions(output / "repeat/predictions.csv", a["expected_ids"], a["threshold"])
        difference = repeat_difference(first, second)
        hardware = {result["hardware"] for result in results.values()}
        hashes = {result["weights_sha256"] for result in results.values()}
        if len(hardware) != 1 or hashes != {sha256(args.weights)}:
            raise ValueError("Hardware or weights changed between verification children")
        report = {"status": "passed", "checks": results, "repeat_maximum_absolute_probability_difference": difference,
                  "repeat_tolerance": 1e-6, "weights_sha256": sha256(args.weights),
                  "weights_bytes": Path(args.weights).stat().st_size, "hardware": next(iter(hardware)),
                  "runtime_source_sha256": source_hashes,
                  **anchor, **reference_binding, "refit_contract_sha256": parity["refit_contract_sha256"],
                  "reference_sha256": parity["reference_sha256"],
                  "t4_timing_verified": False, "test_images": 3000,
                  "timing_caveat": "Fresh Python processes; OS filesystem and GPU driver caches may be warm. Hardware is measured, not a T4 equivalence claim.",
                  "wall_seconds": time.monotonic() - started}
        write_json(output / "verification.json", report)
        print(json.dumps({key: value for key, value in report.items() if key != "checks"}), flush=True)
        return report
    except BaseException as exc:
        write_json(output / "verification.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                                  "completed_checks": list(results), "wall_seconds": time.monotonic() - started})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["weights", "reference", "cache-dir", "test-dir", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--source-dir", default=str(ROOT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--session-hours", type=float, default=8.)
    parser.add_argument("--child", choices=["parity", "inference", "repeat"], help=argparse.SUPPRESS)
    parser.add_argument("--child-wall-seconds", type=float, default=8 * 3600, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 128 or not math.isfinite(args.session_hours) or not 0 < args.session_hours <= 8:
        parser.error("Batch size must be1..128 and session wall limit positive and at most8hours")
    if args.child:
        if not math.isfinite(args.child_wall_seconds) or not 0 < args.child_wall_seconds <= 8 * 3600:
            parser.error("Child wall deadline must be positive and at most8hours")
        child(args)
    else:
        try:
            verify(args)
        except VerificationTerminated:
            return 143
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
