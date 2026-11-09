



import argparse,csv,gzip,hashlib,json,platform,subprocess
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prepare import FEATURES,sha256_file,split_groups


def run(raw,output,seed=42):
    raw=Path(raw).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Use an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    names=['DoS_dataset.csv','Fuzzy_dataset.csv','gear_dataset.csv','RPM_dataset.csv']
    before=[sha256_file(raw/n) for n in names]
    executable=output/'can_accel'
    subprocess.run(['g++','-O3','-std=c++17',str(Path(__file__).with_name('can_accel.cpp')),'-lz','-o',str(executable)],check=True)
    subprocess.run([str(executable),'scan',str(raw),str(output)],check=True)
    finalize(raw,output,before,seed,executable)


def finalize(raw,output,before,seed,executable):
    raw=Path(raw);output=Path(output)
    classes=['BENIGN','DOS','FUZZY','GEAR','RPM'];features=FEATURES+['DLC']
    names=['DoS_dataset.csv','Fuzzy_dataset.csv','gear_dataset.csv','RPM_dataset.csv']
    groups=pd.read_csv(output/'group_counts.csv')
    groups['feature_group']=[hashlib.sha256(','.join(map(str,v)).encode()).hexdigest() for v in groups[features].itertuples(index=False,name=None)]
    if groups.feature_group.duplicated().any():raise ValueError('Duplicate group or hash collision')
    labels=groups[classes].to_numpy();gids=groups.feature_group.to_numpy()
    rows=[(gids[i],c,int(c!='BENIGN')) for i in range(len(groups)) for j,c in enumerate(classes) if labels[i,j]>0]
    gf=pd.DataFrame(rows,columns=['feature_group','y_multiclass','y_binary'])
    assigned=split_groups(gf,seed,'train_test').groupby('feature_group')['split'].first()
    groups['split']=groups.feature_group.map(assigned)
    assignments=groups[features].copy();assignments['partition']=groups.split.map({'train':0,'validation':1,'test':2});assignments['feature_group']=groups.feature_group
    assignments.to_csv(output/'assignments.csv',index=False)
    
    count_by_split={part:{c:int(groups.loc[groups.split==part,c].sum()) for c in classes if groups.loc[groups.split==part,c].sum()>0} for part in ['train','validation','test']}
    total=int(groups[classes].to_numpy().sum())
    sources=pd.read_csv(output/'source_counts.csv').to_dict('records')
    for s,sha in zip(sources,before):
        s['sha256']=sha;s['bytes']=(raw/s['file']).stat().st_size
        s['valid_rows']=int(s.pop('rows'));s['rejected_rows']=0
        s['class_counts']={c:int(s.pop(c)) for c in classes if c in s}
        s['class_counts']={c:n for c,n in s['class_counts'].items() if n}
    rows=[]
    for i,row in enumerate(groups.itertuples(index=False,name=None)):
        vals=row[:10]
        for j,c in enumerate(classes):
            if labels[i,j]:rows.append((*vals,c,int(labels[i,j]),gids[i],row[-1]))
    with gzip.open(output/'unique_inputs_with_counts.csv.gz','wt',newline='') as f:
        w=csv.writer(f);w.writerow(features+['y_multiclass','message_count','feature_group','split']);w.writerows(rows)
    train=groups[groups.split=='train'];weights=train[classes].sum(axis=1).to_numpy(dtype=float);x=train[features].to_numpy(dtype=float);nt=weights.sum()
    mean=np.average(x,axis=0,weights=weights);center=x-mean
    cov=(center.T*weights)@center/max(nt-1,1);std=np.sqrt(np.maximum(np.diag(cov),0));den=np.outer(std,std)
    corr=np.divide(cov,den,out=np.zeros_like(cov),where=den>0)
    pd.DataFrame({'mean':mean,'std':std,'min':x.min(axis=0),'max':x.max(axis=0)},index=features).to_csv(output/'train_feature_statistics.csv')
    pd.DataFrame(corr,index=features,columns=features).to_csv(output/'train_correlation.csv')
    counts=pd.DataFrame(count_by_split).fillna(0).astype(int).reindex(classes);counts.to_csv(output/'class_distribution.csv')
    fig,ax=plt.subplots(figsize=(10,5));counts.plot.bar(ax=ax,rot=25);ax.set(title='Observed class counts in supplied files',ylabel='Messages',xlabel='Class');fig.tight_layout();fig.savefig(output/'class_distribution.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(corr,cmap='coolwarm',vmin=-1,vmax=1);ax.set_xticks(range(10),features,rotation=90);ax.set_yticks(range(10),features);ax.set_title('Training features only Pearson correlation');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(output/'train_correlation.png',dpi=180);plt.close(fig)
    audit={'dataset':'car_hacking','run_scope':'all_supplied_files','implementation':'native CSV parser with Python grouping and EDA','benchmark_completion_claimed':False,'rows_valid':total,'rows_rejected':0,'features':features,'seed':seed,'sources':sources,'unique_input_groups':len(groups),'repeated_input_rows':total-len(groups),'ambiguous_input_groups':int(((groups[classes]>0).sum(axis=1)>1).sum()),'class_counts':{c:int(groups[c].sum()) for c in classes},'rare_group_policy':'train_test','strategy':'class-set-stratified unique-input groups; 70/15/15 approximate GROUP targets','missing_classes_by_split':{p:sorted(set(classes)-set(c)) for p,c in count_by_split.items()},'splits':{p:{'rows':sum(c.values()),'row_fraction':sum(c.values())/total,'groups':int((groups.split==p).sum()),'class_counts':c} for p,c in count_by_split.items()},'limitations':['Temporal or session independence not established.','Exact row ratios not guaranteed.','Extra normal TXT capture excluded.'],'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'native_compiler':'g++ C++17 with zlib'}}
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    (output/'label_mapping.json').write_text(json.dumps({'binary':{'BENIGN':0,'ATTACK':1},'multiclass':{c:i for i,c in enumerate(classes)}},indent=2))
    (output/'training_class_weights.json').write_text(json.dumps({c:int(nt)/(len(classes)*count_by_split['train'][c]) for c in classes},indent=2))
    (output/'manifest_used.json').write_text(json.dumps({'dataset':'car_hacking','rare_group_policy':'train_test','files':[{'path':'../data/raw/car_hacking/'+n,'attack_type':c} for n,c in zip(names,classes[1:])]},indent=2))
    (output/'rejected_rows.csv').write_text('source,row,reason\n')
    subprocess.run([str(executable),'export',str(raw),str(output)],check=True)
    exported=pd.read_csv(output/'export_counts.csv')
    if any(int(r.rows)!=audit['splits'][r.split]['rows'] for r in exported.itertuples()):raise ValueError('Export counts mismatch')
    if before != [sha256_file(raw/n) for n in names]:raise ValueError('Input changed during processing')
    (output/'SUCCESS.json').write_text(json.dumps({'status':'complete_for_supplied_files','rows_exported':total},indent=2))
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args();run(a.raw,a.output,a.seed)
