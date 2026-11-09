




import argparse
import csv
import gzip
import hashlib
import json
import platform
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prepare import FEATURES, CIC_CLASSES, CAR_CLASSES, canon, integer, sha256_file, split_groups


def records(path, dataset, attack_type=None):
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        if dataset == 'ciciov2024':
            header = [x.strip() for x in next(reader)]
            required = FEATURES + ['label', 'category', 'specific_class']
            if len(set(header)) != len(header) or not set(required) <= set(header):
                raise ValueError(f'Invalid CIC decimal header in {path}')
            idx = [header.index(c) for c in required]
        for number, raw in enumerate(reader, 2 if dataset == 'ciciov2024' else 1):
            if not raw or all(not x.strip() for x in raw):
                continue
            try:
                timestamp = ''
                if dataset == 'ciciov2024':
                    if len(raw) != len(header):
                        raise ValueError('CSV row width differs from header')
                    row = [raw[i] for i in idx]
                    values = (integer(row[0],10,0x1FFFFFFF),) + tuple(integer(x,10,255) for x in row[1:9])
                    label, category, target = map(canon, row[9:])
                    if target in {'GAS','RPM','SPEED','STEERING_WHEEL'}:
                        target = 'SPOOFING_' + target
                    if target not in CIC_CLASSES or label not in {'BENIGN','ATTACK'}:
                        raise ValueError('unknown target token')
                    normal = target == 'BENIGN'
                    category_expected = 'BENIGN' if normal else ('DOS' if target=='DOS' else 'SPOOFING')
                    if category != category_expected or label != ('BENIGN' if normal else 'ATTACK'):
                        raise ValueError('inconsistent label hierarchy')
                else:
                    if attack_type not in CAR_CLASSES:
                        raise ValueError('invalid manifest attack type')
                    timestamp = float(raw[0])
                    if not np.isfinite(timestamp) or timestamp < 0:
                        raise ValueError('invalid timestamp')
                    can_id = integer(raw[1],16,0x1FFFFFFF)
                    dlc = integer(raw[2],10,8)
                    flag = raw[-1].strip().upper()
                    if flag not in {'T','R'}:
                        raise ValueError('invalid trailing flag')
                    payload = raw[3:-1]
                    if len(payload) not in {dlc,8}:
                        raise ValueError('invalid payload width')
                    active = tuple(integer(x,16,255) for x in payload[:dlc])
                    if any(x.strip() and integer(x,16,255) != 0 for x in payload[dlc:]):
                        raise ValueError('nonzero byte outside DLC')
                    if flag == 'T' and attack_type == 'BENIGN':
                        raise ValueError('attack flag in benign source')
                    values = (can_id,) + active + (0,)*(8-dlc) + (dlc,)
                    target = attack_type if flag=='T' else 'BENIGN'
                yield number, values, target, timestamp, None
            except (ValueError, IndexError, TypeError, OverflowError) as exc:
                yield number, None, None, None, str(exc)


def run(manifest_path, output, seed=42):
    manifest_path = Path(manifest_path).resolve()
    cfg = json.loads(manifest_path.read_text())
    dataset = cfg['dataset']
    if dataset not in {'ciciov2024','car_hacking'} or not cfg.get('files'):
        raise ValueError('Invalid manifest')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    groups, sources, paths, rejected = {}, [], [], 0
    with (output/'rejected_rows.csv').open('w',newline='') as reject_file:
        writer = csv.writer(reject_file); writer.writerow(['source','row','reason'])
        for i, entry in enumerate(cfg['files']):
            path = (manifest_path.parent/entry['path']).resolve()
            if not path.is_file() or path in paths:
                raise ValueError(f'Missing or duplicate file {path}')
            paths.append(path)
            attack = canon(entry.get('attack_type',''))
            counts = Counter(); bad=0
            for number, values, target, timestamp, error in records(path,dataset,attack):
                if error:
                    writer.writerow([path.name,number,error]); bad+=1; rejected+=1
                    continue
                counts[target]+=1
                if values not in groups:
                    groups[values] = Counter()
                groups[values][target] += 1
            sources.append({'source_id':i,'file':path.name,'bytes':path.stat().st_size,
                            'sha256':sha256_file(path),'valid_rows':sum(counts.values()),
                            'rejected_rows':bad,'class_counts':dict(counts)})
            print(f'Scanned {path.name}: {sum(counts.values()):,} valid; {bad:,} rejected',flush=True)
    if rejected:
        raise ValueError(f'{rejected} invalid rows; inspect rejected_rows.csv before proceeding')
    if not groups:
        raise ValueError('No valid data')
    
    keys = sorted(groups)
    group_ids = [hashlib.sha256(','.join(map(str,k)).encode()).hexdigest() for k in keys]
    if len(set(group_ids)) != len(keys):
        raise ValueError('Fingerprint collision')
    group_frame = pd.DataFrame([
        (group_ids[i], label, int(label!='BENIGN'))
        for i,k in enumerate(keys) for label in groups[k]
    ],columns=['feature_group','y_multiclass','y_binary'])
    rare_policy = cfg.get('rare_group_policy','error')
    if rare_policy not in {'error','train_test'}:
        raise ValueError('rare_group_policy must be error or train_test')
    assigned = split_groups(group_frame,seed,rare_policy).groupby('feature_group')['split'].first().to_dict()
    assignment = {k:(group_ids[i],assigned[group_ids[i]]) for i,k in enumerate(keys)}
    classes = sorted({label for c in groups.values() for label in c})
    label_map = {c:i for i,c in enumerate(classes)}
    features = FEATURES + (['DLC'] if dataset=='car_hacking' else [])
    counts_by_split = {p:Counter() for p in ['train','validation','test']}
    group_counts=Counter()
    dim=len(features); total_train=0
    sums=np.zeros(dim); squares=np.zeros((dim,dim))
    minima=np.full(dim,np.inf); maxima=np.full(dim,-np.inf)
    unique_rows=[]
    for k,counter in groups.items():
        gid,part=assignment[k]; group_counts[part]+=1
        counts_by_split[part].update(counter)
        for label,weight in counter.items():
            unique_rows.append((*k,label,weight,gid,part))
        if part=='train':
            weight=sum(counter.values()); x=np.array(k,dtype=float)
            total_train+=weight; sums+=weight*x; squares+=weight*np.outer(x,x)
            minima=np.minimum(minima,x); maxima=np.maximum(maxima,x)
    with gzip.open(output/'unique_inputs_with_counts.csv.gz','wt',newline='') as f:
        w=csv.writer(f);w.writerow(features+['y_multiclass','message_count','feature_group','split']);w.writerows(unique_rows)
    del unique_rows, group_frame, assigned, keys, group_ids
    
    hashes_before=[s['sha256'] for s in sources]
    with ExitStack() as stack:
        writers={}
        header=features+['y_binary','y_multiclass','y_multiclass_id','source_id','source_row','feature_group']
        if dataset=='car_hacking': header+=['timestamp']
        for part in counts_by_split:
            f=stack.enter_context(gzip.open(output/f'{part}.csv.gz','wt',newline='',compresslevel=3))
            writers[part]=csv.writer(f);writers[part].writerow(header)
        for i,(path,entry) in enumerate(zip(paths,cfg['files'])):
            for number, values, target, timestamp, error in records(path,dataset,canon(entry.get('attack_type',''))):
                if error:
                    raise ValueError('Input changed or unexpected invalid record in pass two')
                gid,part=assignment[values]
                row=list(values)+[int(target!='BENIGN'),target,label_map[target],i,number,gid]
                if dataset=='car_hacking': row+=[timestamp]
                writers[part].writerow(row)
            if sha256_file(path)!=hashes_before[i]:
                raise ValueError('Input file changed during processing')
            print(f'Exported {path.name}',flush=True)
    n=sum(sum(c.values()) for c in counts_by_split.values())
    means=sums/total_train
    covariance=(squares-np.outer(sums,sums)/total_train)/max(total_train-1,1)
    std=np.sqrt(np.maximum(np.diag(covariance),0))
    denom=np.outer(std,std)
    corr=np.divide(covariance,denom,out=np.zeros_like(covariance),where=denom>0)
    pd.DataFrame({'mean':means,'std':std,'min':minima,'max':maxima},index=features).to_csv(output/'train_feature_statistics.csv')
    pd.DataFrame(corr,index=features,columns=features).to_csv(output/'train_correlation.csv')
    count_frame=pd.DataFrame(counts_by_split).fillna(0).astype(int).reindex(classes)
    count_frame.to_csv(output/'class_distribution.csv')
    fig,ax=plt.subplots(figsize=(10,5));count_frame.plot.bar(ax=ax,rot=25)
    ax.set(title='Observed class counts in supplied files',ylabel='Messages',xlabel='Class')
    fig.tight_layout();fig.savefig(output/'class_distribution.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(corr,cmap='coolwarm',vmin=-1,vmax=1)
    ax.set_xticks(range(dim),features,rotation=90);ax.set_yticks(range(dim),features)
    ax.set_title('Training features only Pearson correlation');fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(output/'train_correlation.png',dpi=180);plt.close(fig)
    weights={c:total_train/(len(classes)*counts_by_split['train'][c]) for c in classes}
    (output/'training_class_weights.json').write_text(json.dumps(weights,indent=2))
    (output/'label_mapping.json').write_text(json.dumps({'binary':{'BENIGN':0,'ATTACK':1},'multiclass':label_map},indent=2))
    audit={'dataset':dataset,'run_scope':'all_supplied_files','rows_valid':n,'rows_rejected':rejected,
           'benchmark_completion_claimed':False,'sources':sources,'features':features,'seed':seed,
           'unique_input_groups':len(groups),'repeated_input_rows':n-len(groups),
           'ambiguous_input_groups':sum(len(c)>1 for c in groups.values()),
           'class_counts':dict(sum(counts_by_split.values(),Counter())),
           'strategy':'class-set-stratified unique-input groups; 70/15/15 approximate GROUP targets',
           'rare_group_policy':rare_policy,
           'missing_classes_by_split':{p:sorted(set(classes)-set(c)) for p,c in counts_by_split.items()},
           'splits':{p:{'rows':sum(c.values()),'row_fraction':sum(c.values())/n,'groups':group_counts[p],'class_counts':dict(c)} for p,c in counts_by_split.items()},
           'limitations':['Temporal or session independence not established.','Exact row ratios not guaranteed.','Source provenance and complete-release coverage require separate review.'],
           'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__}}
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    (output/'manifest_used.json').write_text(json.dumps(cfg,indent=2))
    (output/'SUCCESS.json').write_text(json.dumps({'status':'complete_for_supplied_files','rows_exported':n},indent=2))
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    try:
        print(json.dumps(run(a.manifest,a.output,a.seed),indent=2))
    except (ValueError,FileNotFoundError,KeyError) as exc:
        raise SystemExit(f'PREPARATION STOPPED: {exc}. Do not use a run without SUCCESS.json.')
