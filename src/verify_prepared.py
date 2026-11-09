
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from dataset_io import reference_assignment


def verify(folder):
    folder=Path(folder)
    assert (folder/'SUCCESS.json').is_file(), 'Run lacks success marker'
    audit=json.loads((folder/'audit.json').read_text())
    maps=json.loads((folder/'label_mapping.json').read_text())
    features=audit['features']
    known_groups={}; total=0; observed=Counter(); summaries={}
    ordered_counts={p:Counter() for p in ['train','validation','test']}
    ordered_groups={}; dim=len(features); nt=0
    sums=np.zeros(dim);squares=np.zeros((dim,dim));minima=np.full(dim,np.inf);maxima=np.full(dim,-np.inf)
    bitmaps={s['source_id']:np.zeros(s['valid_rows'],dtype=bool) for s in audit['sources']}
    offset=2 if audit['dataset']=='ciciov2024' else 1
    for part in ['train','validation','test']:
        counts=Counter(); n=0
        for chunk in pd.read_csv(folder/f'{part}.csv.gz',chunksize=200000):
            assert not chunk.isna().any().any(), 'Missing value in exported message'
            assert ((chunk.ID>=0)&(chunk.ID<=0x1FFFFFFF)).all()
            for col in features:
                assert (chunk[col]%1==0).all(), 'Noninteger feature'
                if col.startswith('DATA_'): assert chunk[col].between(0,255).all()
            if 'DLC' in features: assert chunk.DLC.between(0,8).all()
            assert chunk.y_multiclass.isin(maps['multiclass']).all()
            assert (chunk.y_multiclass.map(maps['multiclass'])==chunk.y_multiclass_id).all()
            assert ((chunk.y_multiclass!='BENIGN').astype(int)==chunk.y_binary).all()
            primary=reference_assignment(chunk,audit)
            for p in ordered_counts:
                selected=chunk.loc[primary==p]
                ordered_counts[p].update(selected.y_multiclass.value_counts().to_dict())
                bit={'train':1,'validation':2,'test':4}[p]
                for gid in selected.feature_group.unique():ordered_groups[gid]=ordered_groups.get(gid,0)|bit
                if p=='train' and len(selected):
                    x=selected[features].to_numpy(dtype=float);nt+=len(x);sums+=x.sum(axis=0);squares+=x.T@x
                    minima=np.minimum(minima,x.min(axis=0));maxima=np.maximum(maxima,x.max(axis=0))
            for source,block in chunk.groupby('source_id'):
                rows=block.source_row.to_numpy()-offset
                assert source in bitmaps and (rows>=0).all() and (rows<len(bitmaps[source])).all()
                assert len(np.unique(rows))==len(rows), 'Repeated provenance within chunk'
                assert not bitmaps[source][rows].any(), 'Repeated source message across chunks or subsets'
                bitmaps[source][rows]=True
            unique=chunk[features+['feature_group']].drop_duplicates()
            for row in unique.itertuples(index=False,name=None):
                vals=row[:-1]; group=row[-1]
                if group not in known_groups:
                    digest=hashlib.sha256(','.join(str(int(v)) for v in vals).encode()).hexdigest()
                    assert digest==group, 'Fingerprint does not match feature tuple'
                    known_groups[group]=(part,vals)
                else:
                    assert known_groups[group]==(part,vals), 'Cross partition input overlap or collision'
            counts.update(chunk.y_multiclass.value_counts().to_dict()); n+=len(chunk)
        assert n==audit['splits'][part]['rows']
        assert dict(counts)==audit['splits'][part]['class_counts']
        summaries[part]={'rows':n,'class_counts':dict(counts)}
        observed.update(counts);total+=n
    assert total==audit['rows_valid'] and dict(observed)==audit['class_counts']
    assert len(known_groups)==audit['unique_input_groups']
    assert all(bitmap.all() for bitmap in bitmaps.values()), 'Missing original source messages'
    result={'status':'PASS','dataset':audit['dataset'],'rows_verified':total,
            'unique_groups_verified':len(known_groups),'cross_partition_group_overlap':0,
            'all_source_messages_present_exactly_once':True,'splits':summaries,
            'missing_classes_by_split':audit.get('missing_classes_by_split',{})}
    (folder/'independent_verification.json').write_text(json.dumps(result,indent=2))
    classes=set(observed)
    primary_audit={'protocol':'seeded_within_capture_reference','seed':42,'scope':'within the acquired captures only',
                   'source_order':'seed 42 SplitMix64 of source identifier and original CSV row',
                   'ratios':'approximately 70/15/15 message assignment by fixed hash thresholds',
                   'splits':{p:{'rows':sum(c.values()),'class_counts':dict(c),'missing_classes':sorted(classes-set(c))} for p,c in ordered_counts.items()},
                   'groups_shared_across_primary_subsets':sum(mask not in {1,2,4} for mask in ordered_groups.values()),
                   'limitations':['Naturally recurring CAN inputs appear in different subsets.','Within-capture autocorrelation remains possible.','No unseen-pattern, independent-session or cross-vehicle generalization claim.']}
    assert sum(s['rows'] for s in primary_audit['splits'].values())==total
    assert all(not s['missing_classes'] for s in primary_audit['splits'].values()), 'Primary experiment lacks class support'
    (folder/'reference_audit.json').write_text(json.dumps(primary_audit,indent=2))
    counts=pd.DataFrame(ordered_counts).fillna(0).astype(int).sort_index();counts.to_csv(folder/'reference_class_distribution.csv')
    cov=(squares-np.outer(sums,sums)/nt)/max(nt-1,1);std=np.sqrt(np.maximum(np.diag(cov),0));den=np.outer(std,std)
    corr=np.divide(cov,den,out=np.zeros_like(cov),where=den>0)
    pd.DataFrame({'mean':sums/nt,'std':std,'min':minima,'max':maxima},index=features).to_csv(folder/'reference_train_feature_statistics.csv')
    pd.DataFrame(corr,index=features,columns=features).to_csv(folder/'reference_train_correlation.csv')
    weights={c:nt/(len(classes)*ordered_counts['train'][c]) for c in sorted(classes)}
    (folder/'reference_training_class_weights.json').write_text(json.dumps(weights,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5));counts.plot.bar(ax=ax,rot=25);ax.set(title='Seed 42 within-capture reference class counts',ylabel='Messages',xlabel='Class');fig.tight_layout();fig.savefig(folder/'reference_class_distribution.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(corr,cmap='coolwarm',vmin=-1,vmax=1);ax.set_xticks(range(dim),features,rotation=90);ax.set_yticks(range(dim),features);ax.set_title('Reference experiment training feature correlations');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(folder/'reference_train_correlation.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();verify(a.folder)
