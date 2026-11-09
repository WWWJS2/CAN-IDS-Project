
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURES = ['ID'] + [f'DATA_{i}' for i in range(8)]
CIC_CLASSES = {'BENIGN', 'DOS', 'SPOOFING_GAS', 'SPOOFING_RPM',
               'SPOOFING_SPEED', 'SPOOFING_STEERING_WHEEL'}
CAR_CLASSES = {'BENIGN', 'DOS', 'FUZZY', 'GEAR', 'RPM'}


@lru_cache(maxsize=4096)
def canon(value):
    return str(value).strip().upper().replace('-', '_').replace(' ', '_')


@lru_cache(maxsize=8192)
def integer(value, base, maximum):

    s = str(value).strip()
    if base == 16:
        n = int(s, 16)
    else:
        x = float(s)
        if not np.isfinite(x) or not x.is_integer():
            raise ValueError('not a finite integer')
        n = int(x)
    if not 0 <= n <= maximum:
        raise ValueError('outside valid range')
    return n


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(values):
    return hashlib.sha256(','.join(map(str, values)).encode()).hexdigest()


def load_cic(path, rejects, max_rows=None):

    frame = pd.read_csv(path, dtype=str, nrows=max_rows, keep_default_na=False)
    frame.columns = [c.strip().lstrip('\ufeff') for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError(f'{path}: duplicate column names')
    required = FEATURES + ['label', 'category', 'specific_class']
    if not set(required) <= set(frame.columns):
        raise ValueError(f'{path}: missing columns {sorted(set(required)-set(frame.columns))}')
    rows = []
    for line, values in enumerate(frame[required].itertuples(index=False, name=None), 2):
        try:
            xs = [integer(values[0], 10, 0x1FFFFFFF)]
            xs += [integer(v, 10, 255) for v in values[1:9]]
            label, category, specific = map(canon, values[9:])
            
            aliases = {x: 'SPOOFING_' + x for x in ['GAS', 'RPM', 'SPEED', 'STEERING_WHEEL']}
            specific = aliases.get(specific, specific)
            if label not in {'BENIGN', 'ATTACK'}:
                raise ValueError('unknown label; explicit mapping required')
            if specific not in CIC_CLASSES:
                raise ValueError('unknown specific_class; explicit mapping required')
            normal = specific == 'BENIGN'
            expected_category = 'BENIGN' if normal else ('DOS' if specific == 'DOS' else 'SPOOFING')
            if label != ('BENIGN' if normal else 'ATTACK') or category != expected_category:
                raise ValueError('inconsistent label hierarchy')
            rows.append(dict(zip(FEATURES, xs), y_binary=int(not normal),
                             y_multiclass=specific, source_row=line))
        except (ValueError, TypeError, OverflowError) as exc:
            rejects.append({'source': str(path), 'source_row': line, 'reason': str(exc)})
    return pd.DataFrame(rows), len(frame)


def load_car(path, attack_type, rejects, max_rows=None):





    attack_type = canon(attack_type)
    if attack_type not in CAR_CLASSES:
        raise ValueError('invalid manifest attack_type')
    rows, count = [], 0
    with open(path, newline='', encoding='utf-8-sig') as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or all(not x.strip() for x in row):
                continue
            if max_rows is not None and count >= max_rows:
                break
            count += 1
            try:
                timestamp = float(row[0])
                if not np.isfinite(timestamp) or timestamp < 0:
                    raise ValueError('invalid timestamp')
                can_id = integer(row[1], 16, 0x1FFFFFFF)
                dlc = integer(row[2], 10, 8)
                
                flag = row[-1].strip().upper()
                if flag not in {'T', 'R'}:
                    raise ValueError('missing or invalid trailing flag')
                payload = row[3:-1]
                if len(payload) not in {dlc, 8}:
                    raise ValueError('payload width does not match DLC or padded width')
                active = [integer(v, 16, 255) for v in payload[:dlc]]
                trailing = payload[dlc:]
                if any(v.strip() and integer(v, 16, 255) != 0 for v in trailing):
                    raise ValueError('nonzero payload outside DLC')
                if flag == 'T' and attack_type == 'BENIGN':
                    raise ValueError('injected flag in benign source')
                xs = [can_id] + active + [0] * (8 - dlc)
                rows.append(dict(zip(FEATURES, xs), DLC=dlc, timestamp=timestamp,
                                 y_binary=int(flag == 'T'),
                                 y_multiclass=attack_type if flag == 'T' else 'BENIGN',
                                 source_row=line))
            except (ValueError, TypeError, IndexError, OverflowError) as exc:
                rejects.append({'source': str(path), 'source_row': line, 'reason': str(exc)})
    return pd.DataFrame(rows), count


def split_groups(frame, seed, rare_policy='error'):






    group_classes = frame.groupby('feature_group', sort=True).y_multiclass.agg(
        lambda s: '|'.join(sorted(set(s))))
    rng = np.random.default_rng(seed)
    mapping = {}
    for signature in sorted(group_classes.unique()):
        groups = group_classes.index[group_classes == signature].to_numpy().copy()
        if len(groups) < 3 and rare_policy == 'error':
            raise ValueError(f'Cannot create three disjoint subsets for stratum {signature}: '
                             f'only {len(groups)} unique input groups. Review data; do not row-split.')
        rng.shuffle(groups)
        if len(groups) < 3:
            
            
            mapping[groups[0]] = 'train'
            if len(groups) == 2:
                mapping[groups[1]] = 'test'
            continue
        n_test = max(1, int(round(len(groups) * .15)))
        n_val = max(1, int(round(len(groups) * .15)))
        for i, group in enumerate(groups):
            mapping[group] = 'test' if i < n_test else ('validation' if i < n_test+n_val else 'train')
    result = frame.copy()
    result['split'] = result.feature_group.map(mapping)
    for target in ['y_binary', 'y_multiclass']:
        labels = set(result[target])
        for part in ['train', 'validation', 'test']:
            if (rare_policy == 'error' or part == 'train') and set(result.loc[result.split == part, target]) != labels:
                raise ValueError(f'{target}: a split lacks classes')
    if result.groupby('feature_group')['split'].nunique().max() != 1:
        raise AssertionError('duplicate input leakage')
    return result


def class_counts(frame):
    return {str(k): int(v) for k, v in frame.y_multiclass.value_counts().items()}


def prepare(manifest_path, output, seed=42, max_rows=None, allow_rejected=False):
    manifest_path = Path(manifest_path).resolve()
    cfg = json.loads(manifest_path.read_text(encoding='utf-8'))
    dataset = cfg['dataset']
    if dataset not in {'ciciov2024', 'car_hacking'}:
        raise ValueError('dataset must be ciciov2024 or car_hacking')
    if not cfg.get('files'):
        raise ValueError('manifest has no input files')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; use a fresh directory to preserve runs')
    output.mkdir(parents=True, exist_ok=True)
    rejects, frames, sources = [], [], []
    seen_paths = set()
    total = 0
    for entry in cfg['files']:
        path = (manifest_path.parent / entry['path']).resolve()
        if path in seen_paths:
            raise ValueError(f'duplicate input path: {path}')
        seen_paths.add(path)
        if not path.is_file():
            raise FileNotFoundError(f'Raw data not found: {path}. Download from the official source.')
        before = len(rejects)
        if dataset == 'ciciov2024':
            data, count = load_cic(path, rejects, max_rows)
        else:
            data, count = load_car(path, entry['attack_type'], rejects, max_rows)
        if not data.empty:
            data['source_file'] = path.name
            data['source_id'] = hashlib.sha256(str(path).encode()).hexdigest()[:16]
            frames.append(data)
        total += count
        sources.append({'file': path.name, 'source_id': hashlib.sha256(str(path).encode()).hexdigest()[:16],
                        'bytes': path.stat().st_size, 'sha256': sha256_file(path),
                        'rows_read': count, 'rows_rejected': len(rejects)-before})
    pd.DataFrame(rejects, columns=['source', 'source_row', 'reason']).to_csv(output/'rejected_rows.csv', index=False)
    if rejects and not allow_rejected:
        raise ValueError(f'{len(rejects)} invalid rows found; inspect rejected_rows.csv. '
                         'Use --allow-rejected only after reviewing the reasons.')
    if not frames:
        raise ValueError('no valid rows')
    frame = pd.concat(frames, ignore_index=True)
    features = FEATURES + (['DLC'] if dataset == 'car_hacking' else [])
    
    for col in features:
        frame[col] = frame[col].astype('int64')
    frame['feature_group'] = [fingerprint(row) for row in frame[features].itertuples(index=False, name=None)]
    frame['row_id'] = frame.source_id + ':' + frame.source_row.astype(str)
    if frame.row_id.duplicated().any():
        raise ValueError('row identity collision')
    result = split_groups(frame, seed)
    result[['row_id','source_id','source_file','source_row','feature_group','split']].to_csv(output/'split_manifest.csv', index=False)
    if dataset == 'car_hacking':
        result[['row_id','timestamp','source_id']].to_csv(output/'time_metadata.csv', index=False)
    summary = {'dataset': dataset, 'run_scope': 'partial_per_file' if max_rows else 'all_supplied_files',
               'benchmark_completion_claimed': False,
               'note': 'Coverage means supplied files only. Independently verify complete dataset acquisition.',
               'seed': seed, 'rows_read': total, 'rows_valid': len(result), 'rows_rejected': len(rejects),
               'features': features, 'unique_input_groups': int(result.feature_group.nunique()),
               'repeated_input_rows': int(result.feature_group.duplicated().sum()),
               'ambiguous_input_groups': int((result.groupby('feature_group').y_multiclass.nunique()>1).sum()),
               'class_counts': class_counts(result), 'sources': sources, 'splits': {},
               'strategy': 'class-set-stratified unique-input groups; 70/15/15 approximate GROUP targets',
               'limitations': ['No identical inputs shared across splits; temporal or session independence is NOT established.',
                               'No cross-vehicle generalization claim.',
                               'Group-weighted ratios may differ substantially from row-weighted ratios.'],
               'environment': {'python': platform.python_version(),
                               **{p: importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib']}}}
    label_map = {name: i for i, name in enumerate(sorted(result.y_multiclass.unique()))}
    for part in ['train', 'validation', 'test']:
        subset = result[result.split == part]
        
        subset[features].to_csv(output/f'X_{part}.csv', index=False)
        y = subset[['y_binary','y_multiclass']].copy()
        y['y_multiclass_id'] = y.y_multiclass.map(label_map)
        y.to_csv(output/f'y_{part}.csv', index=False)
        subset[['row_id']].to_csv(output/f'rows_{part}.csv', index=False)
        summary['splits'][part] = {'rows': len(subset), 'row_fraction': len(subset)/len(result),
                                   'groups': int(subset.feature_group.nunique()), 'class_counts': class_counts(subset)}
    result[result.split == 'train'][features].describe().to_csv(output/'train_feature_statistics.csv')
    counts = result.groupby(['split','y_multiclass']).size().unstack(fill_value=0)
    counts.to_csv(output/'class_distribution.csv')
    fig, ax = plt.subplots(figsize=(10,5))
    counts.reindex(['train','validation','test']).T.plot.bar(ax=ax, rot=25)
    ax.set(title='Observed class counts in supplied files', ylabel='Rows', xlabel='Class')
    fig.tight_layout(); fig.savefig(output/'class_distribution.png', dpi=180); plt.close(fig)
    train = result[result.split == 'train'][features]
    corr = train.corr().fillna(0)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), features, rotation=90)
    ax.set_yticks(range(len(features)), features)
    ax.set_title('Training features only Pearson correlation')
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(output/'train_correlation.png', dpi=180); plt.close(fig)
    
    weights = {}
    for target in ['y_binary','y_multiclass']:
        c = Counter(result.loc[result.split == 'train', target])
        weights[target] = {str(k): len(train)/(len(c)*v) for k,v in c.items()}
    (output/'training_class_weights.json').write_text(json.dumps(weights, indent=2))
    (output/'label_mapping.json').write_text(json.dumps({'binary': {'BENIGN':0,'ATTACK':1}, 'multiclass':label_map}, indent=2))
    (output/'audit.json').write_text(json.dumps(summary, indent=2))
    (output/'manifest_used.json').write_text(json.dumps(cfg, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-rows', type=int, help='DEBUG ONLY: first N rows per file; never full benchmark')
    p.add_argument('--allow-rejected', action='store_true', help='Explicitly allow audited invalid rows to be excluded')
    a = p.parse_args()
    if a.max_rows is not None and a.max_rows < 1:
        p.error('--max-rows must be positive')
    try:
        result = prepare(a.manifest, a.output, a.seed, a.max_rows, a.allow_rejected)
    except (ValueError, FileNotFoundError, KeyError, pd.errors.ParserError) as exc:
        print(f'PREPARATION STOPPED: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
