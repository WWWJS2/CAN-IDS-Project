from pathlib import Path
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)
root = Path(__file__).resolve().parents[1]


def test_health():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_models():
    response = client.get('/api/models')
    assert response.status_code == 200
    assert len(response.json()['models']) == 8


def test_metrics():
    response = client.get('/api/metrics?dataset=car_hacking&algorithm=rf&task=binary')
    assert response.status_code == 200
    assert response.json()['metrics']['accuracy'] > 0.99


def test_invalid_columns():
    response = client.post('/api/predict',params={'dataset':'car_hacking','algorithm':'rf','task':'binary'},files={'file':('test.csv', b'ID,DATA_0\n1,22\n','text/csv')})
    assert response.status_code == 422


def test_real_model_prediction():
    file = root / 'tests' / 'fixtures' / 'car_hacking_test_probe.csv'
    with file.open('rb') as stream:
        response = client.post('/api/predict',params={'dataset':'car_hacking','algorithm':'rf','task':'binary'},files={'file':('probe.csv',stream,'text/csv')})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['total'] > 0
    assert data['benign'] + data['attacks'] == data['total']
    assert len(data['predictions']) == data['total']


def test_xgboost_prediction():
    file = root / 'tests' / 'fixtures' / 'ciciov2024_test_probe.csv'
    with file.open('rb') as stream:
        response = client.post('/api/predict',params={'dataset':'ciciov2024','algorithm':'xgb','task':'multiclass'},files={'file':('probe.csv',stream,'text/csv')})
    assert response.status_code == 200, response.text
    assert response.json()['total'] > 0
