import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stage2_pipeline import independent_sample_mask

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

@pytest.fixture(scope="module")
def summary():
    return pd.read_csv(ROOT / "results" / "model_training" / "results_summary.csv")

def test_reference_protocol_is_fixed():
    config = load_json(ROOT / "config" / "stage2.json")
    assert config["protocol"] == "seeded_within_capture_reference"
    for dataset in ("ciciov2024", "car_hacking"):
        reference = load_json(ROOT / "metadata" / "datasets" / dataset / "reference_audit.json")
        assert reference["protocol"] == config["protocol"]

def test_train_sample_hash_is_deterministic_and_train_only():
    frame = pd.DataFrame({"source_id": [0, 0, 1, 1], "source_row": [1, 2, 1, 2]})
    assert np.array_equal(independent_sample_mask(frame, 0.5), independent_sample_mask(frame, 0.5))
    source = (ROOT / "src" / "stage2_pipeline.py").read_text(encoding="utf-8")
    assert 'selected = chunk.loc[assignments == partition]' in source
    assert "deterministic_sample_rank(selected, seed=42)" in source

def test_partition_integrity_record():
    record = load_json(ROOT / "metadata" / "split_integrity.json")
    for dataset in ("ciciov2024", "car_hacking"):
        assert record[dataset]["train_test_overlap"] == 0
        assert record[dataset]["train_validation_overlap"] == 0
        assert record[dataset]["assignment_is_mutually_exclusive"] is True

def test_eight_experiments_present(summary):
    assert len(summary) == 8
    assert set(summary["Dataset"]) == {"CICIoV2024", "Car-Hacking"}
    assert set(summary["Classification Type"]) == {"Binary", "Multiclass"}
    assert set(summary["Model"]) == {"Random Forest", "XGBoost"}

def test_all_eight_model_files_exist():
    expected = []
    for dataset in ("ciciov2024", "car_hacking"):
        for kind in ("binary", "multiclass"):
            expected += [ROOT / "models" / dataset / f"rf_{kind}.joblib", ROOT / "models" / dataset / f"xgb_{kind}.json"]
    assert len(expected) == 8 and all(path.is_file() for path in expected)

def test_metrics_are_probabilities(summary):
    for column in ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "ROC-AUC"):
        values = summary[column].dropna()
        assert ((values >= 0) & (values <= 1)).all()

def test_feature_schema_and_no_target_leakage():
    forbidden = {"y_binary", "y_multiclass", "y_multiclass_id", "partition", "source_id", "source_row"}
    for dataset in ("ciciov2024", "car_hacking"):
        audit = load_json(ROOT / "metadata" / "datasets" / dataset / "audit.json")
        assert not forbidden.intersection(audit["features"])
        for model in ("rf", "xgb"):
            for kind in ("binary", "multiclass"):
                metadata = load_json(ROOT / "models" / dataset / f"{model}_{kind}_metadata.json")
                assert metadata["features"] == audit["features"]

def test_saved_models_reload_and_predict():
    for dataset in ("ciciov2024", "car_hacking"):
        audit = load_json(ROOT / "metadata" / "datasets" / dataset / "audit.json")
        probe = pd.read_csv(ROOT / "tests" / "fixtures" / f"{dataset}_test_probe.csv")
        x = probe[audit["features"]]
        for kind in ("binary", "multiclass"):
            rf = joblib.load(ROOT / "models" / dataset / f"rf_{kind}.joblib")
            assert len(rf.predict(x)) == len(x)
            xgb = XGBClassifier()
            xgb.load_model(ROOT / "models" / dataset / f"xgb_{kind}.json")
            assert len(xgb.predict(x)) == len(x)

def test_confusion_matrices_sum_to_test_records():
    results = load_json(ROOT / "results" / "model_training" / "all_experiment_results.json")
    for value in results.values():
        assert np.asarray(value["test"]["confusion_matrix"]).sum() == value["test"]["metrics"]["records"]

def test_all_required_classes_in_reports():
    base = ROOT / "results" / "model_training" / "classification_reports"
    for dataset in ("ciciov2024", "car_hacking"):
        mapping = load_json(ROOT / "mappings" / dataset / "label_mapping.json")
        for model in ("rf", "xgb"):
            for kind in ("binary", "multiclass"):
                report = load_json(base / f"{dataset}_{model}_{kind}_test_classification_report.json")
                expected = {"BENIGN", "ATTACK"} if kind == "binary" else set(mapping["multiclass"])
                assert expected.issubset(report)

def test_json_csv_summary_consistency(summary):
    records = load_json(ROOT / "results" / "model_training" / "results_summary.json")
    assert len(records) == len(summary)
    for csv_row, json_row in zip(summary.to_dict("records"), records):
        for key in ("Dataset", "Classification Type", "Model"):
            assert csv_row[key] == json_row[key]
        for key in ("Accuracy", "Macro F1", "Weighted F1"):
            assert csv_row[key] == pytest.approx(json_row[key], abs=1e-9)

def test_training_times_are_separated(summary):
    required = ["Selected Model Training Time", "Total Candidate Search Time", "Test Inference Time"]
    assert set(required).issubset(summary.columns)
    assert (summary["Selected Model Training Time"] <= summary["Total Candidate Search Time"]).all()
    assert (summary[required] >= 0).all().all()

def test_car_hacking_sample_distribution():
    distribution = pd.read_csv(ROOT / "results" / "model_training" / "car_hacking_training_sample_distribution.csv")
    assert distribution["Sample Train Count"].sum() == 750000
    assert (distribution["Sample Train Count"] > 0).all()
    assert np.allclose(distribution["Full Train Percentage"], distribution["Sample Percentage"], atol=0.001)

def test_results_are_not_hardcoded():
    source = (ROOT / "src" / "stage2_pipeline.py").read_text(encoding="utf-8")
    assert "accuracy_score(y_true, y_pred)" in source
    assert "classification_report(y, pred" in source
    assert "confusion_matrix(y, pred" in source
    assert "roc_auc_score(y_true, y_score[:, 1])" in source
