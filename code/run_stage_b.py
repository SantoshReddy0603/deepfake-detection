"""Stage B step 1 or bounded CPU checks; no session/accelerator control."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import zipfile

from budget import BudgetExceeded, write_json
from step_budget import StepBudget

ROOT = Path(__file__).resolve().parent


def configuration(workers=2, eval_batch_size=64, smoke=False, initialization=None):
    config = {'name': 'convnext_tiny_cpu_smoke' if smoke else 'convnext_tiny_on_15',
              'architecture': 'convnext_tiny', 'device': 'cpu' if smoke else 'cuda',
              'seed': 42, 'ema_decay': .99, 'dropout': .2, 'backbone_lr': 1e-4, 'head_lr': 1e-3,
              'weight_decay': .05, 'smoothing': .05, 'max_epochs': 1 if smoke else 15, 'patience': 16,
              'batch_size': 8 if smoke else 16, 'accumulation': 2, 'eval_batch_size': eval_batch_size,
              'workers': workers, 'colour': 'strong', 'degradation': True,
              'validation_policy': 'ema_full' if smoke else 'ema_clean3_full_after',
              'save_validation_predictions': True,
              'artifact_purpose': 'smoke_cpu' if smoke else 'stage_b_fold0_validation'}
    if smoke:
        config['smoke_image_size'] = 32
    elif initialization is not None:
        config['initialization_path'] = str(Path(initialization).resolve())
    return config


def report(out, status, runner_estimate_gpu_minutes=0., allocated_gpus=0,
           planned_epochs=15, quota_snapshot=None, timings=None):
    out = Path(out)
    path = out / 'experiment/history.json'
    history = json.loads(path.read_text()) if path.is_file() else []
    eligible = [r for r in history if r['ema']['selection_eligible']]
    score = lambda r: (r['ema']['mean_auc'], r['ema']['minimum_auc'])
    winner = max(eligible, key=score) if eligible else None
    improving = None
    if len(history) == 15 and planned_epochs == 15:
        earlier = [r for r in eligible if r['epoch'] < 12]
        tail = [r for r in eligible if r['epoch'] >= 12]
        improving = bool(earlier and tail and max(map(score, tail)) > max(map(score, earlier)) and score(tail[-1]) > score(earlier[-1]))
    smoke = planned_epochs == 1
    result = {'step': 1, 'status': status, 'completed_epochs': len(history), 'planned_epochs': planned_epochs,
              'engineering_smoke_only': smoke, 'competition_result': bool(history) and not smoke,
              'best': None if winner is None else {'epoch_one_based': winner['epoch'] + 1, **winner['ema']},
              'still_improving_at_15': improving, 'timings': timings or {},
              'accounting': {'label': 'runner estimate', 'gpu_minutes': runner_estimate_gpu_minutes,
                             'allocated_gpus': allocated_gpus, 'verified_kaggle_charge': False,
                             'quota_snapshot_from_user': quota_snapshot,
                             'scope': 'includes startup when a launch timestamp is supplied; reconcile real Kaggle quota separately'},
              'next_action': 'STOP and wait for user. End any Kaggle session; no later step or accelerator switch.'}
    columns = ['epoch', 'train_loss', 'model_init_seconds', 'first_batch_seconds', 'train_seconds',
               'validation_seconds', 'clean_auc', 'mean_auc', 'minimum_auc', 'validation_scope',
               'live_clean_auc', 'live_mean_auc', 'live_minimum_auc']
    curves, conditions = [], []
    for r in history:
        curves.append({'epoch': r['epoch'] + 1, **{k: r.get(k) for k in columns[1:6]},
                       'clean_auc': r['ema']['clean_auc'], 'mean_auc': r['ema'].get('mean_auc'),
                       'minimum_auc': r['ema'].get('minimum_auc'), 'validation_scope': r['validation_scope'],
                       **{f'live_{key}': (r.get('live') or {}).get(key) for key in ('clean_auc', 'mean_auc', 'minimum_auc')}})
        for stream in ('live', 'ema'):
            conditions.extend({'epoch': r['epoch'] + 1, 'stream': stream, 'condition': name, **metrics}
                              for name, metrics in (r.get(stream) or {}).get('conditions', {}).items())
    result['curves'] = curves
    write_json(out / 'stage_b_step1_results.json', result)
    with (out / 'epoch_curves.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(curves)
    if conditions:
        with (out / 'per_condition_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(conditions[0]))
            writer.writeheader()
            writer.writerows(conditions)
    lines = ['# Smoke result' if smoke else '# Stage B step 1 result', '',
             f'Status: {status}. Completed {len(history)}/{planned_epochs} epochs.',
             ('GPU smoke uses the production resolution and initialization on a subset.' if allocated_gpus else
              'CPU smoke uses random initialization and 32-pixel inputs.') if smoke else
             'Live uses clean validation every epoch. EMA uses clean in epochs 1-3 and all 16 conditions from epoch 4. Selection uses EMA mean AUC then minimum AUC.', '',
             '| Epoch | Live clean | Live mean | Live min | EMA clean | EMA mean | EMA min |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    metric_columns = ('live_clean_auc', 'live_mean_auc', 'live_minimum_auc', 'clean_auc', 'mean_auc', 'minimum_auc')
    for r in curves:
        values = ['–' if r[key] is None else f"{r[key]:.6f}" for key in metric_columns]
        lines.append(f"| {r['epoch']} | " + ' | '.join(values) + ' |')
    lines += ['', f'Runner estimate: {runner_estimate_gpu_minutes:.6f} GPU-minutes. This is not a verified charge.',
              'Session wall limit is eight hours, including startup.', '', result['next_action']]
    (out / 'STAGE_B_STEP1_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def preserve(out):
    out = Path(out)
    target = out.parent / (out.name + '_recovery.zip')
    temporary = target.with_suffix('.zip.tmp')
    best_path = out / 'experiment/best.json'
    best = json.loads(best_path.read_text()) if best_path.is_file() else {}
    selected = (best.get('ema') or {}).get('checkpoint')
    keep = {'latest.pt', 'best.pt', 'diagnostic_best.pt', selected}
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(out.rglob('*')):
            if not path.is_file() or path.name.endswith('.tmp') or 'validation_cache' in path.parts:
                continue
            if path.suffix == '.pt' and path.name not in keep:
                continue
            archive.write(path, path.relative_to(out).as_posix())
    temporary.replace(target)
    return target


def stratified_subset(frame, count):
    import pandas as pd
    if count % 2 or min(frame.label.value_counts()) < count // 2:
        raise ValueError('Smoke subset requires enough examples of both classes')
    return pd.concat([frame[frame.label == label].sample(count // 2, random_state=42) for label in [0, 1]]).sort_values('image_id').reset_index(drop=True)


def step_one_forecast(first, measured_epoch_seconds, elapsed_seconds, allocated):
    # Three clean epochs, then twelve full suites. Extrapolate the measured clean
    # condition to sixteen passes until a full production suite is measured.
    clean = first['validation_seconds']
    clean_epoch = measured_epoch_seconds
    full_epoch = clean_epoch + 15 * clean
    return {'measured_epoch_seconds': clean_epoch, 'measured_clean_validation_seconds': clean,
            'estimated_full_epoch_seconds': full_epoch,
            'forecast_gpu_minutes': (elapsed_seconds + 2 * clean_epoch + 12 * full_epoch) * allocated / 60,
            'accounting_label': 'runner estimate'}


def main(argv=None):
    entry_started = time.monotonic()
    entry_unix = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--input-root', default='/kaggle/input')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.75)
    parser.add_argument('--yield-on-other-gpu-process', action='store_true')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--initialization', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default=str(ROOT / 'splits/fold0.csv'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='CPU engineering smoke')
    mode.add_argument('--gpu-smoke', action='store_true', help='Production-recipe subset, maximum 180 wall seconds')
    mode.add_argument('--preflight', action='store_true')
    parser.add_argument('--allocated-gpus', type=int)
    parser.add_argument('--step-one-approved', action='store_true')
    parser.add_argument('--session-hours', type=float, default=8.)
    parser.add_argument('--wall-minutes', type=float, help='CPU-only cap; smoke maximum3 minutes')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--ram-training', action='store_true')
    parser.add_argument('--ram-smoke-subset', action='store_true')
    parser.add_argument('--channels-last', action='store_true')
    parser.add_argument('--torch-threads', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--session-start-unix', type=float, help='Conservative timestamp immediately before starting the Kaggle session')
    args = parser.parse_args(argv)
    if args.workers < 0 or args.eval_batch_size < 1 or args.cpu_threads < 1:
        parser.error('Invalid loader/CPU settings')
    if args.ram_smoke_subset and not (args.gpu_smoke and args.ram_training):
        parser.error("RAM smoke subset is only valid for the GPU RAM smoke")
    smoke = args.smoke or args.gpu_smoke
    cpu = args.smoke or args.preflight
    if cpu:
        wall = args.wall_minutes if args.wall_minutes is not None else (3. if args.smoke else 15.)
        if not math.isfinite(wall) or wall <= 0 or (args.smoke and wall > 3):
            parser.error('CPU smoke cap must be >0 and <=3 minutes')
        allocated, quota = 0, None
    else:
        if args.allocated_gpus not in (1, 2):
            parser.error('Declare the fixed one- or two-GPU allocation')
        allocated = args.allocated_gpus
        if not math.isfinite(args.session_hours) or not 0 < args.session_hours <= 8:
            parser.error('Session wall limit must be positive and at most eight hours')
        if args.gpu_smoke and args.resume:
            parser.error('Smoke must start fresh')
        if not args.gpu_smoke and not args.step_one_approved:
            parser.error('Step 1 requires user approval')
        wall = 3. if args.gpu_smoke else args.session_hours * 60
        quota = None
    startup_seconds = 0.
    if not cpu and not args.resume and (not args.gpu_smoke or args.session_start_unix is not None):
        if args.session_start_unix is None or not math.isfinite(args.session_start_unix) or args.session_start_unix > entry_unix:
            parser.error('A valid session launch timestamp is required for the startup-inclusive cap')
        startup_seconds = entry_unix - args.session_start_unix
    scope = ('stage_b_gpu_smoke' if args.gpu_smoke else 'stage_b_cpu_smoke' if args.smoke else
             'stage_b_cpu_preflight' if args.preflight else 'stage_b_step1')
    watchdog = threading.Timer(max(0., wall * 60 - startup_seconds - (time.monotonic() - entry_started)),
        lambda: (print(f'{scope}: HARD STEP DEADLINE; exit124. Completed epoch files remain usable.', file=sys.stderr, flush=True), os._exit(124)))
    watchdog.daemon = True
    watchdog.start()
    try:
        out = Path(args.output).resolve()
        out.mkdir(parents=True, exist_ok=True)
        forecast_path = out / 'epoch_one_forecast.json'
        budget = StepBudget(out / 'step_budget.json', wall * 60, scope, args.resume, allocated,
                            startup_seconds + time.monotonic() - entry_started)
    except BaseException:
        watchdog.cancel()
        raise
    # Resume gets only the remaining step time, including this invocation's setup.
    watchdog.cancel()
    watchdog = threading.Timer(max(0., budget.remaining),
        lambda: (print(f'{scope}: HARD STEP DEADLINE; exit124. Completed epoch files remain usable.', file=sys.stderr, flush=True), os._exit(124)))
    watchdog.daemon = True
    watchdog.start()
    timings = {'startup_seconds': startup_seconds}
    exit_code, status = 0, 'running'
    try:
        budget.check()
        imports_started = time.perf_counter()
        budget.emit('phase_started', phase='runtime_imports')
        import numpy as np
        import pandas as pd
        import PIL
        import torch
        import torchvision
        from data import fingerprint
        from model_def import build_model
        from persistent_inputs import load_inputs
        from train import experiment_epoch
        timings['runtime_import_seconds'] = time.perf_counter() - imports_started
        budget.emit('phase_completed', phase='runtime_imports', seconds=timings['runtime_import_seconds'])
        budget.check()
        if cpu:
            torch.set_num_threads(args.cpu_threads)
        elif not torch.cuda.is_available() or torch.cuda.device_count() != allocated:
            raise RuntimeError('Fixed GPU allocation does not match --allocated-gpus; stop without switching accelerators')
        if not cpu and any(not Path(value).resolve().is_relative_to(Path(args.input_root).resolve())
                           for value in (args.data_root, args.cache_dir, args.initialization)):
            raise ValueError('GPU inputs must remain within the declared input root')
        if not cpu:
            if not 0 < args.gpu_memory_fraction <= .8:
                raise ValueError('GPU memory fraction must be positive and at most .8')
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
            if args.yield_on_other_gpu_process:
                from resource_guard import ResourceGuard
                budget = ResourceGuard(budget, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
                budget.check()
        if args.torch_threads is not None:
            if not 1 <= args.torch_threads <= 4:
                raise ValueError("CPU thread cap must be in 1..4")
            torch.set_num_threads(args.torch_threads)
        training, validation, manifest, input_timings, init_sha = load_inputs(
            args.data_root, args.split, args.cache_dir, args.initialization, budget,
            smoke=smoke, verified_input=not cpu)
        timings.update(input_timings)
        training_store = None
        if args.ram_training:
            from ram_training import preload_training, worker_count
            all_rows = (stratified_subset(training, 256) if args.ram_smoke_subset else
                        pd.concat([training, validation]).sort_values('image_id').reset_index(drop=True))
            cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
            budget.emit('ram_preload_started', images=len(all_rows), cores=cores)
            training_store = preload_training(all_rows, manifest_path=out / 'training_media_manifest.json',
                check_deadline=budget.check, decode_workers=min(4, cores),
                progress=lambda values: budget.emit('ram_preload_progress', **values))
            timings['ram_preload_seconds'] = training_store.preload_seconds
            timings['ram_bytes'] = training_store.pixels.nbytes
            args.workers = worker_count(cores)
            budget.emit('ram_preload_complete', seconds=training_store.preload_seconds,
                        bytes=training_store.pixels.nbytes, workers=args.workers,
                        torch_threads=torch.get_num_threads(), channels_last=args.channels_last)
        write_json(out / 'input_preflight.json', {'cache_path': str(Path(args.cache_dir).resolve()),
                   'initialization_path': str(Path(args.initialization).resolve()), 'initialization_sha256': init_sha,
                   'cache_key': manifest['key'], 'timings': timings, 'writes_to_input': False})
        if not cpu:
            timings['gpu_name'] = torch.cuda.get_device_name(0)
            budget.emit('accelerator_verified', model_device='cuda:0', gpu_name=timings['gpu_name'],
                        allocated_gpus=allocated, accounting_label='runner estimate')
        write_json(out / 'environment.json', {'python': sys.version, 'torch': str(torch.__version__),
                   'torchvision': torchvision.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
                   'pillow': PIL.__version__, 'device': 'cpu' if cpu else 'cuda', 'allocated_gpus': allocated})
        if args.preflight:
            began = time.perf_counter()
            model = build_model('convnext_tiny', initialization_path=args.initialization)
            del model
            timings['model_init_seconds'] = time.perf_counter() - began
            budget.emit('phase_completed', phase='persistent_backbone_model_load', seconds=timings['model_init_seconds'])
            status = 'cpu_preflight_complete'
        else:
            config = configuration(0 if args.smoke else args.workers, 16 if args.smoke else args.eval_batch_size,
                                   args.smoke, args.initialization)
            if args.ram_training:
                config.update(ram_training=True, workers=args.workers)
            if args.channels_last:
                config['channels_last'] = True
            if not cpu:
                config.update(device='cuda:0', mounted_training_input=str(Path(args.data_root).resolve()))
            if not cpu and not args.gpu_smoke:
                config['validation_policy'] = 'live_clean_ema_clean3_full_after'
            if args.gpu_smoke:
                config.update(name='convnext_tiny_gpu_smoke', max_epochs=1,
                              artifact_purpose='smoke_gpu', validation_policy='ema_full')
            if smoke:
                training = stratified_subset(training, 256)
                selected = stratified_subset(validation, 128)
                positions = {r['image_id']: i for i, r in enumerate(manifest['samples'])}
                config['validation_indices'] = [positions[i] for i in selected.image_id]
                write_json(out / 'smoke_samples.json', {'train_ids': training.image_id.tolist(),
                           'validation_ids': selected.image_id.tolist(), 'train_count': len(training), 'validation_count': len(selected),
                           'image_size': 224 if args.gpu_smoke else 32,
                           'initialization': 'persistent ImageNet' if args.gpu_smoke else 'random engineering smoke',
                           'competition_result': False})
            contract = {'scope': scope, 'configuration': config, 'cache_key': manifest['key'],
                        'split_sha256': fingerprint(args.split), 'initialization_sha256': init_sha,
                        'wall_cap_seconds': wall * 60,
                        'source_sha256': {p.name: fingerprint(p) for p in sorted(ROOT.glob('*.py'))}}
            path = out / 'run_contract.json'
            if path.exists() and (not args.resume or json.loads(path.read_text()) != contract):
                raise ValueError('Run contract changed or --resume missing')
            write_json(path, contract)
            while True:
                budget.check()
                epoch_started = time.perf_counter()
                state = experiment_epoch(config, training, args.cache_dir, manifest['key'], out / 'experiment', budget, training_store=training_store)
                measured_epoch_seconds = time.perf_counter() - epoch_started
                timings['epoch_end_to_end_seconds'] = measured_epoch_seconds
                last = state['history'][-1]
                timings.update({key: last[key] for key in ['train_batches', 'seconds_per_batch', 'steady_seconds_per_batch']})
                timings.update({key: last[key] for key in ['model_init_seconds', 'first_batch_seconds', 'train_seconds', 'validation_seconds']})
                report(out, status, budget.elapsed * allocated / 60, allocated, config['max_epochs'], quota, timings)
                if state['done']:
                    status = f'{scope}_complete' if args.gpu_smoke else 'cpu_smoke_complete' if args.smoke else 'completed_15_epochs'
                    break
                if last['epoch'] == 0 and not smoke:
                    forecast = step_one_forecast(last, measured_epoch_seconds, budget.elapsed, allocated)
                    timings.update(forecast)
                    write_json(forecast_path, forecast)
                    budget.emit('step_one_epoch_one_forecast', **forecast)
    except BudgetExceeded as exc:
        status, exit_code = str(exc), 124
        budget.emit('budget_stop', message=status, exit_code=exit_code)
    except KeyboardInterrupt:
        status, exit_code = 'interrupted; last completed epoch retained', 130
        budget.emit('interrupted', exit_code=exit_code)
    except Exception as exc:
        status, exit_code = f'failed: {type(exc).__name__}: {exc}', 1
        budget.emit('failure', message=status, exit_code=exit_code)
    finally:
        report(out, status, budget.elapsed * allocated / 60, allocated, 1 if smoke else 15, quota, timings)
        if exit_code == 0 and budget.remaining <= 0:
            status, exit_code = f'{scope}: STEP BUDGET exhausted while finalizing; exit124', 124
        budget.close()
        report(out, status, budget.elapsed * allocated / 60, allocated, 1 if smoke else 15, quota, timings)
        # Per-epoch files are already fsynced; an archive is optional and deliberately
        # not built here, so copying hundreds of MB cannot consume the step deadline.
        budget.emit('step_finished', status=status, exit_code=exit_code, runner_estimate_gpu_minutes=budget.elapsed * allocated / 60,
                    next_action='STOP; keep accelerator fixed; end session, record the Kaggle quota and wait for user')
        watchdog.cancel()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
