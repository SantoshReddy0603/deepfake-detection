"""Compare aligned saved validation predictions without GPU or test inputs."""
import argparse,json
from pathlib import Path
import numpy as np
from augment import CONDITIONS
from budget import write_json
from data import fingerprint
from metrics import binary_metrics


def saved_predictions(directory, expected_count=1400):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('selection_eligible') is not True or manifest.get('validation_indices') is not None or manifest.get('label_mapping')!={'0':'real','1':'fake'}:raise ValueError('Complete non-subset validation manifest required')
    if set(manifest['conditions'])!=set(CONDITIONS):raise ValueError('Full16 validation predictions required')
    result={}
    for condition,detail in manifest['conditions'].items():
        if Path(detail['file']).name!=detail['file']:raise ValueError('Unsafe prediction path')
        path=directory/detail['file']
        if fingerprint(path)!=detail['sha256']:raise ValueError('Prediction checksum mismatch')
        with np.load(path,allow_pickle=False) as data:
            ids=data['image_id'].astype(str);labels=data['label'];logits=data['logit'];prob=data['probability']
        if any(v.ndim!=1 or len(v)!=expected_count for v in (ids,labels,logits,prob)) or detail.get('count')!=expected_count:raise ValueError('Wrong validation prediction shape/count')
        if result:
            previous=next(iter(result.values()))
            if not np.array_equal(ids,previous[0]) or not np.array_equal(labels,previous[1]):raise ValueError('Within-suite ID/label alignment mismatch')
        if len(set(ids))!=len(ids) or not np.isfinite(logits).all() or not np.isfinite(prob).all():raise ValueError('Invalid prediction arrays')
        if not np.isin(labels,[0,1]).all() or not np.allclose(prob,1/(1+np.exp(-logits)),atol=1e-6):raise ValueError('Inconsistent labels/probabilities')
        result[condition]=(ids,labels,logits,prob)
    return manifest['cache_key'],result


def compare(first,second,expected_count=1400):
    key,a=saved_predictions(first,expected_count);key2,b=saved_predictions(second,expected_count)
    if key!=key2:raise ValueError('Different validation caches')
    scores={name:{} for name in ['first','second','probability_average','logit_average']}
    for condition in CONDITIONS:
        ids,y,la,pa=a[condition];ids2,y2,lb,pb=b[condition]
        if not np.array_equal(ids,ids2) or not np.array_equal(y,y2):raise ValueError('Prediction ID/label alignment mismatch')
        values=[pa,pb,(pa+pb)/2,1/(1+np.exp(-(la+lb)/2))]
        for name,prob in zip(scores,values):scores[name][condition]=binary_metrics(y,prob)['auc']
    result={name:{'mean_auc':float(np.mean(list(values.values()))),'minimum_auc':min(values.values()),'clean_auc':values['clean'],'conditions':values} for name,values in scores.items()}
    best=max(['first','second'],key=lambda n:(result[n]['mean_auc'],result[n]['minimum_auc']))
    for name in ['probability_average','logit_average']:
        result[name]['mean_gain']=result[name]['mean_auc']-result[best]['mean_auc']
        result[name]['quality_eligible']=bool(result[name]['mean_gain']>=.005 and result[name]['minimum_auc']>=result[best]['minimum_auc'])
    return {'better_member':best,'metrics':result,'runtime_gate':'Not measured by this offline comparison'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--first',required=True);p.add_argument('--second',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
    result=compare(a.first,a.second);write_json(a.output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
