from __future__ import annotations
import io
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATASETS = {'ciciov2024', 'car_hacking'}
ALGORITHMS = {'rf', 'xgb'}
TASKS = {'binary', 'multiclass'}
MAX_BYTES = 12 * 1024 * 1024
MAX_ROWS = 50000

app = FastAPI(title='CAN Intrusion Detection API', version='1.0.0')
origins = [item.strip() for item in os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'POST'], allow_headers=['*'])


def check_selection(dataset: str, algorithm: str, task: str):
    if dataset not in DATASETS or algorithm not in ALGORITHMS or task not in TASKS:
        raise HTTPException(status_code=400, detail='Unsupported model selection')


@lru_cache(maxsize=8)
def model_bundle(dataset: str, algorithm: str, task: str):
    check_selection(dataset, algorithm, task)
    location = ROOT / 'models' / dataset
    metadata = json.loads((location / f'{algorithm}_{task}_metadata.json').read_text(encoding='utf-8'))
    if algorithm == 'rf':
        model = joblib.load(location / f'{algorithm}_{task}.joblib')
    else:
        model = XGBClassifier()
        model.load_model(str(location / f'{algorithm}_{task}.json'))
    return model, metadata


@app.get('/api/health')
def health():
    return {'status': 'ok', 'models': 8}


@app.get('/api/models')
def available_models():
    models = []
    for dataset in sorted(DATASETS):
        for algorithm in sorted(ALGORITHMS):
            for task in sorted(TASKS):
                path = ROOT / 'models' / dataset / f'{algorithm}_{task}_metadata.json'
                if not path.is_file():
                    continue
                metadata = json.loads(path.read_text(encoding='utf-8'))
                models.append({'dataset': dataset, 'algorithm': algorithm, 'task': task, 'features': metadata['features'], 'classes': metadata['class_names'], 'training_rows': metadata.get('actual_training_rows'), 'training_sampled': metadata.get('training_sampled')})
    return {'models': models}


@app.get('/api/metrics')
def metrics(dataset: str = Query('car_hacking'), algorithm: str = Query('rf'), task: str = Query('binary')):
    check_selection(dataset, algorithm, task)
    filename = ROOT / 'results' / 'model_training' / 'metrics' / f'{dataset}_{algorithm}_{task}_test_metrics.json'
    if not filename.exists():
        raise HTTPException(status_code=404, detail='Stored evaluation is unavailable')
    return {'dataset': dataset, 'algorithm': algorithm, 'task': task, 'source': 'stored_test_evaluation', 'metrics': json.loads(filename.read_text(encoding='utf-8'))}


def parse_frame(raw: bytes, features: list[str]):
    try:
        frame = pd.read_csv(io.BytesIO(raw), dtype=str, encoding='utf-8-sig', on_bad_lines='error')
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid CSV: {exc}') from exc
    frame.columns = [col.strip().upper() for col in frame.columns]
    if frame.columns.duplicated().any():
        raise HTTPException(status_code=422, detail='Duplicate CSV column names')
    if len(frame) == 0 or len(frame) > MAX_ROWS:
        raise HTTPException(status_code=422, detail=f'CSV must contain between 1 and {MAX_ROWS} rows')
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={'missing_columns': missing})
    values = {}
    for column in features:
        source = frame[column].str.strip()
        if column == 'ID':
            def parse_id(value):
                if not value:
                    raise ValueError('Empty ID')
                if value.lower().startswith('0x'):
                    return int(value, 16)
                return int(value, 10)
            try:
                parsed = source.map(parse_id)
            except (ValueError, TypeError) as exc:
                raise HTTPException(status_code=422, detail='ID must be decimal or 0x-prefixed hexadecimal') from exc
            if ((parsed < 0) | (parsed > 536870911)).any():
                raise HTTPException(status_code=422, detail='ID out of CAN identifier range')
            values[column] = parsed.astype(np.int64)
        else:
            parsed = pd.to_numeric(source, errors='coerce')
            limit = 8 if column == 'DLC' else 255
            if parsed.isna().any() or (~np.isfinite(parsed)).any() or ((parsed % 1) != 0).any() or (parsed < 0).any() or (parsed > limit).any():
                raise HTTPException(status_code=422, detail=f'{column} must contain integers from 0 to {limit}')
            values[column] = parsed.astype(np.int64)
    return pd.DataFrame(values, columns=features)


@app.post('/api/predict')
async def predict(file: UploadFile = File(...), dataset: str = Query('car_hacking'), algorithm: str = Query('rf'), task: str = Query('binary')):
    check_selection(dataset, algorithm, task)
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail='Only CSV uploads are supported')
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail='CSV exceeds 12 MB limit')
    if not raw:
        raise HTTPException(status_code=400, detail='Empty upload')
    try:
        model, metadata = model_bundle(dataset, algorithm, task)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'Model could not be loaded: {type(exc).__name__}') from exc
    frame = parse_frame(raw, metadata['features'])
    try:
        labels = model.predict(frame)
        confidence = model.predict_proba(frame).max(axis=1) if hasattr(model, 'predict_proba') else np.full(len(frame), np.nan)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Inference failed: {type(exc).__name__}') from exc
    class_names = metadata['class_names']
    counts = {name: 0 for name in class_names}
    rows = []
    for index, (label, probability) in enumerate(zip(labels, confidence)):
        label_id = int(label)
        if label_id < 0 or label_id >= len(class_names):
            raise HTTPException(status_code=500, detail='Unexpected model class')
        name = class_names[label_id]
        counts[name] += 1
        rows.append({'row': index + 1, 'id': int(frame.iloc[index]['ID']), 'label': name, 'is_attack': name != 'BENIGN', 'confidence': round(float(probability), 6) if np.isfinite(probability) else None})
    return {'dataset': dataset, 'algorithm': algorithm, 'task': task, 'filename': file.filename, 'total': len(frame), 'benign': counts.get('BENIGN', 0), 'attacks': len(frame) - counts.get('BENIGN', 0), 'classes': counts, 'predictions': rows, 'notice': 'Inference on an uploaded CSV; not a live vehicle connection. Confidence is a model score, not a calibrated probability.'}
