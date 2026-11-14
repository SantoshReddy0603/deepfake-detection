"""Cross-fit a frozen weighted ConvNeXt/B0 ensemble without outer-label leakage."""
import time
STARTED=time.monotonic()
import argparse,hashlib,json,os,shutil,threading,uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from augment import CONDITIONS
from budget import BudgetExceeded,write_json
from compare_ensemble import compare
from data import fingerprint,load_split
from metrics import binary_metrics
from noise_weights import RULE,consistent_weights,weight_map_digest
from ram_training import preload_training
from resource_guard import ResourceGuard
from run_cv import group_folds,assemble_oof,f1_threshold,write_csv
from run_stage_b import configuration
from step_budget import StepBudget,fresh_session_budget
from train import experiment_epoch
from validation_cache import verify_cache
ROOT=Path(__file__).resolve().parent


def disjoint_pools(training,validation):
    if (training.image_id.duplicated().any() or validation.image_id.duplicated().any()
            or set(training.image_id)&set(validation.image_id) or set(training.group_id)&set(validation.group_id)):
        raise ValueError('Outer training and held-out IDs/groups must be disjoint')


def check_inner(outer_training,outer_validation,inner):
    disjoint_pools(outer_training,outer_validation)
    if (inner.image_id.duplicated().any() or set(inner.image_id)!=set(outer_training.image_id)
            or set(inner.fold)!={0,1,2} or inner.groupby('group_id').fold.nunique().ne(1).any()):
        raise ValueError('Inner folds must cover only the outer training groups')
    original=outer_training.set_index('image_id')
    aligned=original.loc[inner.image_id]
    if not np.array_equal(inner.label.to_numpy(),aligned.label.to_numpy()) or not np.array_equal(inner.group_id.to_numpy(),aligned.group_id.to_numpy()):raise ValueError('Inner labels/groups changed')


def clean_cache(rows,lookup,directory,budget):
    """Copy verified clean pixels once; this auxiliary cache never claims16 conditions."""
    directory=Path(directory)
    samples=[]
    for row in rows.itertuples():
        if row.image_id not in lookup:raise ValueError('Missing verified clean image')
        _,_,sample=lookup[row.image_id]
        if sample['label']!=int(row.label):raise ValueError('Clean cache label mismatch')
        samples.append({'image_id':row.image_id,'label':int(row.label),'sha256':sample['sha256']})
    if len({r['image_id'] for r in samples})!=len(samples):raise ValueError('Duplicate clean cache IDs')
    contract={'conditions':['clean'],'samples':samples,'size':224,'purpose':'outer_training_only_noise_loss','seed':42}
    key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    path=directory/'manifest.json'
    if path.exists():
        saved=json.loads(path.read_text())
        array=np.load(directory/'clean.npy',mmap_mode='r',allow_pickle=False)
        if saved.get('key')!=key or array.dtype!=np.uint8 or array.shape!=(len(rows),224,224,3) or fingerprint(directory/'clean.npy')!=saved['array_sha256']['clean']:raise ValueError('Changed auxiliary clean cache')
        return saved
    if directory.exists() and any(directory.iterdir()):raise ValueError('Incomplete cache; preserve and diagnose')
    directory.mkdir(parents=True,exist_ok=True)
    pixels=np.lib.format.open_memmap(directory/'clean.npy',mode='w+',dtype='uint8',shape=(len(rows),224,224,3))
    for i,row in enumerate(rows.itertuples()):
        budget.check();source,index,_=lookup[row.image_id];pixels[i]=source[index]
    pixels.flush();del pixels
    manifest={'key':key,**contract,'array_sha256':{'clean':fingerprint(directory/'clean.npy')}}
    write_json(path,manifest);return manifest


def clean_prediction(directory,epoch,expected,cache_key):
    path=Path(directory)/'experiment/predictions'/f'epoch_{epoch:02}_ema'
    manifest=json.loads((path/'manifest.json').read_text());detail=manifest.get('conditions',{}).get('clean',{})
    if (manifest.get('cache_key')!=cache_key or manifest.get('validation_indices') is not None
            or detail.get('file')!='clean.npz' or detail.get('count')!=expected
            or fingerprint(path/'clean.npz')!=detail.get('sha256')):raise ValueError('Changed clean cross-fit prediction')
    return path/'clean.npz'


def inner_weights(training,validation,inner,folders,cache_keys,epochs=15):
    check_inner(training,validation,inner)
    losses=inner[['image_id','label','fold']].sort_values('image_id').reset_index(drop=True)
    for epoch in range(epochs-2,epochs+1):
        paths={fold:clean_prediction(folders[fold],epoch,int((inner.fold==fold).sum()),cache_keys[fold]) for fold in range(3)}
        aligned=assemble_oof(inner,paths)
        if set(aligned.image_id)&set(validation.image_id):raise ValueError('Outer held-out image entered loss ranking')
        losses[f'loss_epoch_{epoch:02}']=aligned.set_index('image_id').loc[losses.image_id,'loss'].to_numpy()
    table,selected,summary=consistent_weights(losses,list(range(epochs-2,epochs+1)))
    weights=dict(zip(table.image_id,table.weight.astype(float)))
    if set(weights)!=set(training.image_id) or set(weights)&set(validation.image_id):raise ValueError('Invalid outer-training weight coverage')
    return weights,table,summary


def run_epochs(config,training,cache,manifest,directory,budget,store):
    while True:
        budget.check()
        state=experiment_epoch(config,training,cache,manifest['key'],Path(directory)/'experiment',budget,training_store=store)
        if state['done']:return state


def outer_assignments(rows,baseline,split):
    contract=json.loads((baseline/'run_contract.json').read_text());summary=json.loads((baseline/'cv_results.json').read_text())
    if contract.get('pool')!='all_official_training' or contract.get('epochs')!=15 or contract.get('folds')!=3 or contract.get('seed')!=42 or contract.get('split_sha256')!=fingerprint(split) or summary.get('status')!='completed' or summary.get('oof_count')!=7000:raise ValueError('Original full CV contract required')
    assignments=pd.read_csv(baseline/'fold_assignments.csv',dtype={'image_id':str,'group_id':str})
    columns=['image_id','label','group_id','fold']
    if set(assignments.columns)!=set(columns) or assignments.image_id.duplicated().any() or set(assignments.image_id)!=set(rows.image_id) or set(assignments.fold)!={0,1,2}:raise ValueError('Invalid original outer folds')
    if assignments.sort_values('image_id')[columns].to_dict('records')!=pd.DataFrame(contract['assignment']).sort_values('image_id')[columns].to_dict('records'):raise ValueError('Outer fold file differs from frozen contract')
    expected=rows.set_index('image_id').loc[assignments.image_id]
    if not np.array_equal(assignments.label.to_numpy(),expected.label.to_numpy()) or not np.array_equal(assignments.group_id.to_numpy(),expected.group_id.to_numpy()):raise ValueError('Outer metadata mismatch')
    merged=rows.merge(assignments[['image_id','fold']],on='image_id',validate='one_to_one').sort_values('image_id').reset_index(drop=True)
    for fold in range(3):
        fit,held=merged[merged.fold!=fold],merged[merged.fold==fold]
        disjoint_pools(fit,held)
        if fit.label.nunique()!=2 or held.label.nunique()!=2:raise ValueError('Both outer partitions require both classes')
    return merged


def average_oof(a,b):
    if (a.empty or a.image_id.duplicated().any() or b.image_id.duplicated().any()
            or any(not np.array_equal(a[key].to_numpy(),b[key].to_numpy()) for key in ['image_id','label','fold'])):
        raise ValueError('Member OOF alignment mismatch')
    values=np.c_[a.probability,b.probability]
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise ValueError('Invalid OOF probabilities')
    oof=a[['image_id','label','fold']].copy();oof['conv_probability']=a.probability;oof['b0_probability']=b.probability;oof['probability']=(a.probability+b.probability)/2
    clipped=np.clip(oof.probability.to_numpy(),1e-7,1-1e-7);oof['logit']=np.log(clipped)-np.log1p(-clipped);oof['loss']=np.logaddexp(0,oof.logit)-oof.label*oof.logit
    return oof


def new_session_budget(output,startup_seconds=0.):
    return fresh_session_budget(output,scope='ensemble_nested_calibration',startup_seconds=startup_seconds)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data-root','baseline-cv','conv-initialization','b0-initialization','candidate','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--original-split',default=str(ROOT/'splits/fold0.csv'));p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv);out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    import fcntl
    run_lock=(out/'.run.lock').open('a');fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget,prior_seconds=new_session_budget(out,time.monotonic()-STARTED)
    timer=threading.Timer(budget.remaining,lambda:os._exit(124));timer.daemon=True;timer.start()
    code,status=0,'running';fold_results=[]
    try:
        if shutil.disk_usage(out).free<30*1024**3:raise RuntimeError('Insufficient free disk for durable cross-fit checkpoints')
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not gpu.startswith('GPU-') or ',' in gpu:raise ValueError('One explicit GPU UUID required')
        budget=ResourceGuard(budget,gpu);budget.check()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('One fixed GPU required')
        torch.cuda.set_per_process_memory_fraction(.5,0);torch.set_num_threads(4)
        candidate=json.loads(Path(args.candidate).read_text())
        if (candidate.get('architecture')!='probability_ensemble' or candidate.get('aggregation')!='uniform_probability'
                or candidate.get('schedule_epochs')!=15 or candidate.get('conv_stop_epochs')!=5 or candidate.get('b0_stop_epochs')!=8
                or candidate.get('conv_noise_rule')!=RULE or candidate.get('evidence_no_test_used') is not True):raise ValueError('Frozen ensemble protocol differs from approved candidate')
        train,val=load_split(args.data_root,args.original_split);rows=pd.concat([train,val]).sort_values('image_id').reset_index(drop=True)
        baseline=Path(args.baseline_cv);rows=outer_assignments(rows,baseline,args.original_split)
        contract={'protocol':'training_only_nested_noise_fixed_ensemble','candidate_sha256':fingerprint(args.candidate),'outer_assignments':rows[['image_id','label','group_id','fold']].to_dict('records'),'schedule_epochs':15,'conv_stop':5,'b0_stop':8,'inner_epochs':15,'seed':42,'noise_rule':RULE,'aggregation':'uniform_probability','initialization_sha256':{'conv':fingerprint(args.conv_initialization),'b0':fingerprint(args.b0_initialization)},'baseline_contract_sha256':fingerprint(baseline/'run_contract.json'),'source_sha256':{q.name:fingerprint(q) for q in ROOT.glob('*.py')}}
        saved=out/'run_contract.json'
        if saved.exists() and (not args.resume or json.loads(saved.read_text())!=contract):raise ValueError('Frozen calibration protocol changed')
        write_json(saved,contract);write_csv(out/'fold_assignments.csv',rows[['image_id','label','group_id','fold']])
        store=preload_training(rows,manifest_path=out/'training_media_manifest.json',check_deadline=budget.check,decode_workers=4)
        lookup={};outer_caches={};manifests={};media_hashes=store.train_media_sha256
        for fold in range(3):
            held=rows[rows.fold==fold].reset_index(drop=True);cache=baseline/f'fold_{fold}/validation_cache'
            timing={};manifest=verify_cache(held,cache,verified_input=True,check_deadline=budget.check,timings=timing)
            array=np.load(cache/'clean.npy',mmap_mode='r',allow_pickle=False)
            for index,sample in enumerate(manifest['samples']):
                if sample['sha256']!=media_hashes[sample['image_id']]:raise ValueError('Cached pixels and current training images differ')
                if sample['image_id'] in lookup:raise ValueError('Duplicate global clean cache ID')
                lookup[sample['image_id']]=(array,index,sample)
            outer_caches[fold]=cache;manifests[fold]=manifest;budget.emit('outer_cache_verified',fold=fold,**timing)
        paths={'conv':{},'b0':{}}
        for outer in range(3):
            training=rows[rows.fold!=outer].reset_index(drop=True);validation=rows[rows.fold==outer].reset_index(drop=True)
            inner=group_folds(training,seed=42,folds=3);check_inner(training,validation,inner)
            folder=out/f'outer_{outer}';folder.mkdir(exist_ok=True);write_csv(folder/'inner_assignments.csv',inner[['image_id','label','group_id','fold']])
            folders={};keys={}
            for fold in range(3):
                fit=inner[inner.fold!=fold].reset_index(drop=True);held=inner[inner.fold==fold].reset_index(drop=True)
                disjoint_pools(fit,held)
                inner_dir=folder/f'inner_{fold}';inner_dir.mkdir(exist_ok=True)
                manifest=clean_cache(held,lookup,inner_dir/'validation_cache',budget);folders[fold]=inner_dir;keys[fold]=manifest['key']
                config=configuration(initialization=args.conv_initialization)
                config.update(name=f'outer_{outer}_inner_{fold}',ram_training=True,channels_last=True,device='cuda:0',validation_policy='ema_clean_only',artifact_purpose='training_only_noise_crossfit')
                run_epochs(config,fit,inner_dir/'validation_cache',manifest,inner_dir,budget,store)
                budget.emit('inner_complete',outer=outer,inner=fold)
            weights,table,summary=inner_weights(training,validation,inner,folders,keys)
            write_csv(folder/'sample_losses_and_weights.csv',table)
            write_json(folder/'noise_weights.json',{'weights':weights,'weight_map_sha256':weight_map_digest(weights),'summary':summary,'rule':RULE,'excluded_outer_validation_count':len(validation)})
            for name,architecture,stop,initialization in [('conv','convnext_tiny',5,args.conv_initialization),('b0','efficientnet_b0',8,args.b0_initialization)]:
                member=folder/name;member.mkdir(exist_ok=True)
                config=configuration(initialization=initialization)
                config.update(name=f'outer_{outer}_{name}',architecture=architecture,device='cuda:0',ram_training=True,channels_last=True,stop_epochs=stop,validation_policy='ema_clean_until_final',artifact_purpose='ensemble_oof_validation')
                if name=='conv':config['sample_weights']=weights
                state=run_epochs(config,training,outer_caches[outer],manifests[outer],member,budget,store)
                if state['history'][-1]['epoch']+1!=stop or not state['history'][-1]['ema']['selection_eligible']:raise ValueError('Frozen member stop/full16 validation missing')
                paths[name][outer]=member/'experiment/predictions'/f'epoch_{stop:02}_ema/clean.npz'
            first=paths['conv'][outer].parent;second=paths['b0'][outer].parent
            metrics=compare(first,second,expected_count=len(validation))
            write_json(folder/'paired_metrics.json',metrics)
            fold_results.append({'fold':outer,'train_count':len(training),'validation_count':len(validation),'weight_summary':summary,'metrics':metrics['metrics']})
            write_json(out/'fold_results.json',fold_results);budget.emit('outer_complete',fold=outer,mean_auc=metrics['metrics']['probability_average']['mean_auc'])
        a=assemble_oof(rows,paths['conv']);b=assemble_oof(rows,paths['b0'])
        oof=average_oof(a,b)
        write_csv(out/'oof_predictions.csv',oof);write_csv(out/'conv_oof_predictions.csv',a);write_csv(out/'b0_oof_predictions.csv',b)
        scores=[f['metrics']['probability_average']['mean_auc'] for f in fold_results]
        result={'status':'completed','oof_count':len(oof),'folds':fold_results,'mean_fold_selection_auc':float(np.mean(scores)),'std_fold_selection_auc':float(np.std(scores)),'label_threshold':f1_threshold(oof.label.to_numpy(),oof.probability.to_numpy()),'oof_metrics_at_half':binary_metrics(oof.label.to_numpy(),oof.probability.to_numpy()),'protocol':contract['protocol'],'schedule_epochs':15,'conv_stop':5,'b0_stop':8,'loss_clipping_epsilon':1e-7,'architecture_selection_unbiased':False,'outer_validation_used_for_weights':False,'wall_seconds':budget.elapsed,'prior_session_runner_seconds':prior_seconds,'cumulative_runner_seconds':prior_seconds+budget.elapsed}
        write_json(out/'calibration_results.json',result);status='completed'
    except BudgetExceeded as exc:code,status=124,str(exc)
    except KeyboardInterrupt:code,status=130,'interrupted'
    except Exception as exc:code,status=1,f'{type(exc).__name__}: {exc}'
    finally:
        write_json(out/'runner_status.json',{'status':status,'exit_code':code,'wall_seconds':budget.elapsed,'prior_session_runner_seconds':prior_seconds,'cumulative_runner_seconds':prior_seconds+budget.elapsed});budget.emit('calibration_finished',status=status,exit_code=code);budget.close();timer.cancel()
    return code


if __name__=='__main__':raise SystemExit(main())
