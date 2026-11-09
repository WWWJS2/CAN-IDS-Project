


import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def summarize(folder):
    folder=Path(folder);audit=json.loads((folder/'audit.json').read_text())
    df=pd.read_csv(folder/'reference_group_counts.csv');features=audit['features'];classes=sorted(audit['class_counts'])
    parts=['train','validation','test'];columns={p:[c+'_'+p for c in classes] for p in parts}
    counts={p:{c:int(df[c+'_'+p].sum()) for c in classes} for p in parts}
    assert len(df)==audit['unique_input_groups']
    for c in classes:assert sum(counts[p][c] for p in parts)==audit['class_counts'][c]
    assert all(counts[p][c]>0 for p in parts for c in classes)
    membership=np.column_stack([(df[columns[p]].sum(axis=1)>0).to_numpy() for p in parts])
    weights=df[columns['train']].sum(axis=1).to_numpy(dtype=float);x=df[features].to_numpy(dtype=float);active=weights>0;nt=weights.sum();mean=np.average(x,axis=0,weights=weights);center=x-mean
    cov=(center.T*weights)@center/(nt-1);std=np.sqrt(np.maximum(np.diag(cov),0));den=np.outer(std,std);corr=np.divide(cov,den,out=np.zeros_like(cov),where=den>0)
    pd.DataFrame({'mean':mean,'std':std,'min':x[active].min(axis=0),'max':x[active].max(axis=0)},index=features).to_csv(folder/'reference_train_feature_statistics.csv')
    pd.DataFrame(corr,index=features,columns=features).to_csv(folder/'reference_train_correlation.csv')
    distribution=pd.DataFrame(counts);distribution.to_csv(folder/'reference_class_distribution.csv')
    result={'protocol':'seeded_within_capture_reference','seed':42,'assignment':'SplitMix64 of source identifier and original row number, modulo 10000, thresholds 7000 and 8500','splits':{p:{'rows':sum(c.values()),'class_counts':c,'missing_classes':[]} for p,c in counts.items()},'groups_shared_across_primary_subsets':int((membership.sum(axis=1)>1).sum()),'limitations':['Recurring predictor patterns are shared across subsets.','This is a within-dataset reference comparison, not independent-session, temporal, novel-pattern or cross-vehicle validation.'],'validation':'Native class totals reconcile with the independently verified full-message storage; native assignment matches the vectorized loader in automated tests.'}
    (folder/'reference_audit.json').write_text(json.dumps(result,indent=2))
    (folder/'reference_training_class_weights.json').write_text(json.dumps({c:int(nt)/(len(classes)*counts['train'][c]) for c in classes},indent=2))
    fig,ax=plt.subplots(figsize=(10,5));distribution.plot.bar(ax=ax,rot=25);ax.set(title='Seed 42 within-capture reference class counts',ylabel='Messages',xlabel='Class');fig.tight_layout();fig.savefig(folder/'reference_class_distribution.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(corr,cmap='coolwarm',vmin=-1,vmax=1);ax.set_xticks(range(len(features)),features,rotation=90);ax.set_yticks(range(len(features)),features);ax.set_title('Reference experiment training feature correlations');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(folder/'reference_train_correlation.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();summarize(a.folder)
