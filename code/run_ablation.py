"""Fold-zero ablations using the unchanged cached16-condition validation suite."""
import time
STARTED=time.monotonic()
import argparse,hashlib,json,os,sys,threading
from pathlib import Path
import pandas as pd
import torch
from budget import BudgetExceeded,write_json
from data import fingerprint,load_split
from ram_training import preload_training
from resource_guard import ResourceGuard
from run_stage_b import configuration,report
from step_budget import StepBudget
from train import experiment_epoch,training_weights
from validation_cache import verify_cache
ROOT=Path(__file__).resolve().parent


def recipe(kind,initialization,weights=None):
    config=configuration(initialization=initialization)
    config.update(name=kind+'_fold0_15',device='cuda:0',ram_training=True,channels_last=True,
                  validation_policy='live_clean_ema_clean3_full_after',artifact_purpose='ablation_validation')
    if kind=='layer_decay':config['layer_lr_decay']=.8
    elif kind=='noise':
        if weights is None:raise ValueError('Noise ablation needs training-only crossfit weights')
        config['sample_weights']=weights
    elif kind=='b0':config['architecture']='efficientnet_b0'
    else:raise ValueError('Unknown ablation')
    return config


def validated_noise_weights(path, training, split):
    from noise_weights import RULE, consistent_weights, restricted_losses
    path=Path(path)
    document=json.loads(path.read_text())
    provenance=document.get("provenance",{})
    if (document.get("version")!=1 or document.get("selection_rule")!=RULE
            or document.get("validation_labels_modified") is not False
            or document.get("labels_dropped_or_flipped") is not False
            or provenance.get("pool")!="original_fold0_training_only"
            or provenance.get("original_split_sha256")!=fingerprint(split)
            or provenance.get("excluded_validation_count")!=1400
            or fingerprint(path.parent/"sample_losses_and_weights.csv")!=document.get("csv_sha256")):
        raise ValueError("Noise-weight provenance or CSV checksum mismatch")
    losses,epochs,verified=restricted_losses(provenance["restricted_cv_root"],split,15)
    table,selected,summary=consistent_weights(losses,epochs)
    expected=dict(zip(table.image_id,table.weight.astype(float)))
    if (provenance!=verified or document.get("weights")!=expected
            or document.get("selected_ids")!=selected or document.get("summary")!=summary):
        raise ValueError("Noise weights differ from verified training-only cross-fit losses")
    training_weights({"sample_weights":expected},training)
    return expected


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['data-root','cache-dir','initialization','output']:
        parser.add_argument('--'+arg,required=True)
    parser.add_argument('--kind',choices=['layer_decay','noise','b0'],required=True)
    parser.add_argument('--split',default=str(ROOT/'splits/fold0.csv'))
    parser.add_argument('--weights')
    parser.add_argument('--session-start-unix',type=float)
    args=parser.parse_args(argv)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    startup=max(time.monotonic()-STARTED,time.time()-args.session_start_unix if args.session_start_unix is not None else 0)
    budget=StepBudget(out/'step_budget.json',8*3600,scope=args.kind,allocated_gpus=1,startup_seconds=startup)
    timer=threading.Timer(budget.remaining,lambda:os._exit(124));timer.daemon=True;timer.start()
    code,status=0,'running'
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Requires exactly one fixed GPU')
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not gpu.startswith('GPU-') or ',' in gpu:raise ValueError('Requires one fixed GPU UUID')
        budget=ResourceGuard(budget,gpu_uuid=gpu);budget.check()
        torch.cuda.set_per_process_memory_fraction(.75,0);torch.set_num_threads(4)
        training,validation=load_split(args.data_root,args.split)
        timings={};manifest=verify_cache(validation,args.cache_dir,check_deadline=budget.check,timings=timings,verified_input=True)
        budget.emit('cache_loaded',**timings)
        store=preload_training(pd.concat([training,validation]).sort_values('image_id').reset_index(drop=True),
                manifest_path=out/'training_media_manifest.json',check_deadline=budget.check,decode_workers=4)
        weights=None
        if args.kind=='noise':
            if not args.weights:raise ValueError('Missing noise weights')
            weights=validated_noise_weights(args.weights,training,args.split)
        config=recipe(args.kind,args.initialization,weights)
        contract={'configuration':config,'cache_key':manifest['key'],'split_sha256':fingerprint(args.split),
                  'initialization_sha256':fingerprint(args.initialization),'weights_sha256':fingerprint(args.weights) if args.weights else None,
                  'source_sha256':{p.name:fingerprint(p) for p in ROOT.glob('*.py')}}
        write_json(out/'run_contract.json',contract)
        while True:
            state=experiment_epoch(config,training,args.cache_dir,manifest['key'],out/'experiment',budget,training_store=store)
            report(out,'running',budget.elapsed/60,1,planned_epochs=15)
            if state['done']:break
        report(out,'completed_15_epochs',budget.elapsed/60,1,planned_epochs=15);status='completed'
    except BudgetExceeded as exc:code,status=124,str(exc)
    except KeyboardInterrupt:code,status=130,'interrupted'
    except Exception as exc:code,status=1,f'{type(exc).__name__}: {exc}'
    finally:
        write_json(out/'status.json',{'status':status,'exit_code':code,'wall_seconds':budget.elapsed})
        budget.emit('finished',status=status,exit_code=code);budget.close();timer.cancel()
    return code


if __name__=='__main__':raise SystemExit(main())
