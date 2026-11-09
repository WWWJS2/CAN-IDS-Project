from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from dataset_io import iter_partition, reference_assignment


DATASET_TITLES = {"ciciov2024": "CICIoV2024", "car_hacking": "Car-Hacking"}


@dataclass
class DatasetBundle:
    name: str
    features: list[str]
    mapping: dict
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    full_train_rows: int
    sampled: bool


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def deterministic_sample_rank(frame: pd.DataFrame, seed: int = 42) -> np.ndarray:
    source = frame["source_id"].to_numpy(dtype=np.uint64)
    row = frame["source_row"].to_numpy(dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = row ^ ((source + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(seed)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        bucket = x ^ (x >> np.uint64(31))
    return bucket


def independent_sample_mask(frame: pd.DataFrame, probability: float, seed: int = 42) -> np.ndarray:
    bucket = deterministic_sample_rank(frame, seed)
    threshold = np.uint64(min(2**64 - 1, round(probability * (2**64 - 1))))
    return bucket <= threshold


def proportional_allocation(counts: pd.Series, target: int) -> pd.Series:
    exact = counts.astype(float) / float(counts.sum()) * target
    allocation = np.floor(exact).astype(int)
    remainder = target - int(allocation.sum())
    if remainder:
        for cls in (exact - allocation).sort_values(ascending=False).index[:remainder]:
            allocation.loc[cls] += 1
    return allocation


def load_partition_frame(folder: Path, partition: str, features: list[str], chunk_size: int) -> pd.DataFrame:
    keep = features + ["y_binary", "y_multiclass", "y_multiclass_id", "source_id", "source_row"]
    chunks = []
    for chunk in iter_partition(folder, partition, protocol="seeded_within_capture_reference", chunksize=chunk_size):
        chunks.append(chunk[keep])
    frame = pd.concat(chunks, ignore_index=True)
    for col in features:
        frame[col] = pd.to_numeric(frame[col], downcast="unsigned")
    frame["y_binary"] = frame["y_binary"].astype(np.uint8)
    frame["y_multiclass_id"] = frame["y_multiclass_id"].astype(np.uint8)
    return frame


def load_train_frame(folder: Path, features: list[str], full_count: int, cap: int | None, chunk_size: int) -> tuple[pd.DataFrame, bool]:
    keep = features + ["y_binary", "y_multiclass", "y_multiclass_id", "source_id", "source_row"]
    probability = 1.0 if cap is None or full_count <= cap else cap / full_count
    chunks = []
    for chunk in iter_partition(folder, "train", protocol="seeded_within_capture_reference", chunksize=chunk_size):
        chunk = chunk[keep]
        if probability < 1.0:
            chunk = chunk.loc[independent_sample_mask(chunk, probability)]
        if not chunk.empty:
            chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True)
    if cap is not None and len(frame) > cap:
        counts = frame["y_multiclass_id"].value_counts().sort_index()
        allocation = (counts / counts.sum() * cap).astype(int)
        for cls in counts.index:
            allocation.loc[cls] = max(1, allocation.loc[cls])
        while allocation.sum() < cap:
            cls = ((counts / counts.sum() * cap) - allocation).idxmax()
            allocation.loc[cls] += 1
        pieces = [
            group.sample(n=min(len(group), int(allocation.loc[cls])), random_state=42)
            for cls, group in frame.groupby("y_multiclass_id", sort=True)
        ]
        frame = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    for col in features:
        frame[col] = pd.to_numeric(frame[col], downcast="unsigned")
    frame["y_binary"] = frame["y_binary"].astype(np.uint8)
    frame["y_multiclass_id"] = frame["y_multiclass_id"].astype(np.uint8)
    return frame, probability < 1.0


def load_dataset(root: Path, name: str, config: dict, logger: logging.Logger) -> DatasetBundle:
    folder = root / "runs" / name
    audit = read_json(folder / "audit.json")
    reference = read_json(folder / "reference_audit.json")
    mapping = read_json(folder / "label_mapping.json")
    if reference["protocol"] != config["protocol"]:
        raise ValueError(f"{name}: reference protocol mismatch")
    features = audit["features"]
    full_count = int(reference["splits"]["train"]["rows"])
    cap = config["training_caps"].get(name)
    logger.info("Loading %s reference partitions in one pass (train full=%d, cap=%s)", name, full_count, cap)
    keep = features + ["y_binary", "y_multiclass", "y_multiclass_id", "source_id", "source_row"]
    probability = 1.0 if cap is None or full_count <= cap else cap / full_count
    full_class_counts = pd.Series({
        int(mapping["multiclass"][label]): int(count)
        for label, count in reference["splits"]["train"]["class_counts"].items()
    }).sort_index()
    target_allocation = proportional_allocation(full_class_counts, cap) if probability < 1.0 else full_class_counts
    selection_probability = {
        int(cls): min(1.0, (int(target_allocation.loc[cls]) / int(count)) * 1.05)
        for cls, count in full_class_counts.items()
    }
    collected = {"train": [], "validation": [], "test": []}
    for storage in ("train", "validation", "test"):
        for chunk in pd.read_csv(folder / f"{storage}.csv.gz", usecols=keep, chunksize=config["chunk_size"]):
            assignments = reference_assignment(chunk, audit, seed=42)
            for partition in ("train", "validation", "test"):
                selected = chunk.loc[assignments == partition]
                if partition == "train" and probability < 1.0 and not selected.empty:
                    ranks = deterministic_sample_rank(selected, seed=42)
                    thresholds = selected["y_multiclass_id"].map(selection_probability).to_numpy(dtype=float)
                    mask = ranks.astype(np.float64) / float(2**64 - 1) <= thresholds
                    selected = selected.loc[mask].copy()
                    selected["__sample_rank"] = ranks[mask]
                if not selected.empty:
                    collected[partition].append(selected.copy())
    train = pd.concat(collected["train"], ignore_index=True)
    validation = pd.concat(collected["validation"], ignore_index=True)
    test = pd.concat(collected["test"], ignore_index=True)
    if probability < 1.0:
        pieces = []
        for cls, target_rows in target_allocation.items():
            group = train.loc[train["y_multiclass_id"] == cls]
            if len(group) < int(target_rows):
                raise AssertionError(f"{name}: insufficient deterministic candidates for class {cls}")
            pieces.append(group.nsmallest(int(target_rows), "__sample_rank"))
        train = pd.concat(pieces, ignore_index=True).sort_values("__sample_rank").reset_index(drop=True)
        train = train.drop(columns="__sample_rank")
    for frame in (train, validation, test):
        for col in features:
            frame[col] = pd.to_numeric(frame[col], downcast="unsigned")
        frame["y_binary"] = frame["y_binary"].astype(np.uint8)
        frame["y_multiclass_id"] = frame["y_multiclass_id"].astype(np.uint8)
    sampled = probability < 1.0
    logger.info("Loaded %s rows: train=%d validation=%d test=%d", name, len(train), len(validation), len(test))
    expected = reference["splits"]
    observed = {"train_full": full_count, "validation": len(validation), "test": len(test)}
    if observed["validation"] != expected["validation"]["rows"] or observed["test"] != expected["test"]["rows"]:
        raise AssertionError(f"{name}: partition counts do not match reference audit: {observed}")
    return DatasetBundle(name, features, mapping, train, validation, test, full_count, sampled)


def class_names(bundle: DatasetBundle, kind: str) -> tuple[list[int], list[str]]:
    if kind == "binary":
        return [0, 1], ["BENIGN", "ATTACK"]
    pairs = sorted(bundle.mapping["multiclass"].items(), key=lambda kv: kv[1])
    return [value for _, value in pairs], [name for name, _ in pairs]


def make_model(model_name: str, kind: str, params: dict, n_classes: int, n_jobs: int):
    common = dict(random_state=42, n_jobs=n_jobs)
    if model_name == "rf":
        return RandomForestClassifier(**common, **params)
    objective = "binary:logistic" if kind == "binary" else "multi:softprob"
    extra = {"objective": objective, "tree_method": "hist", "eval_metric": "logloss" if kind == "binary" else "mlogloss"}
    if kind == "multiclass":
        extra["num_class"] = n_classes
    return XGBClassifier(**common, **extra, **params)


def prediction_scores(model, x: pd.DataFrame, labels: list[int]) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.ndim == 1:
        return np.column_stack([1.0 - proba, proba])
    if proba.shape[1] != len(labels):
        result = np.zeros((len(x), len(labels)), dtype=np.float32)
        for source_idx, cls in enumerate(model.classes_):
            result[:, labels.index(int(cls))] = proba[:, source_idx]
        return result
    return proba


def calculate_metrics(y_true, y_pred, y_score, labels: list[int], kind: str) -> dict:
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "weighted_precision": weighted[0],
        "weighted_recall": weighted[1],
        "weighted_f1": weighted[2],
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if kind == "binary":
        tn, fp, fn, tp = cm.ravel()
        result.update({"true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp})
        result["roc_auc"] = roc_auc_score(y_true, y_score[:, 1])
    else:
        try:
            result["roc_auc"] = roc_auc_score(y_true, y_score, labels=labels, multi_class="ovr", average="macro")
            result["roc_auc_note"] = "Macro one-vs-rest ROC-AUC."
        except ValueError as exc:
            result["roc_auc"] = None
            result["roc_auc_note"] = str(exc)
    return result


def save_confusion_matrix(cm, names, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)), xticklabels=names, yticklabels=names,
           xlabel="Predicted class", ylabel="True class", title=title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > threshold else "black", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true, scores, labels, names, path: Path, title: str, kind: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    if kind == "binary":
        fpr, tpr, _ = roc_curve(y_true, scores[:, 1])
        auc = roc_auc_score(y_true, scores[:, 1])
        ax.plot(fpr, tpr, color="#1f4e79", lw=2, label=f"ROC-AUC = {auc:.4f}")
    else:
        binary = label_binarize(y_true, classes=labels)
        for idx, name in enumerate(names):
            if len(np.unique(binary[:, idx])) < 2:
                continue
            fpr, tpr, _ = roc_curve(binary[:, idx], scores[:, idx])
            auc = roc_auc_score(binary[:, idx], scores[:, idx])
            ax.plot(fpr, tpr, lw=1.4, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title, xlim=(0, 1), ylim=(0, 1.01))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def evaluate(model, frame: pd.DataFrame, bundle: DatasetBundle, kind: str, split: str, output_root: Path, stem: str, save_plots: bool):
    target = "y_binary" if kind == "binary" else "y_multiclass_id"
    labels, names = class_names(bundle, kind)
    x = frame[bundle.features]
    y = frame[target].to_numpy()
    started = time.perf_counter()
    pred = model.predict(x)
    score = prediction_scores(model, x, labels)
    inference = time.perf_counter() - started
    metrics = calculate_metrics(y, pred, score, labels, kind)
    metrics.update({"split": split, "records": len(frame), "inference_time_seconds": inference})
    report = classification_report(y, pred, labels=labels, target_names=names, output_dict=True, zero_division=0)
    report_text = classification_report(y, pred, labels=labels, target_names=names, digits=6, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    if save_plots:
        model_label = "Random Forest" if "_rf_" in stem else "XGBoost"
        save_confusion_matrix(cm, names, output_root / "confusion_matrices" / f"{stem}_confusion_matrix.png",
                              f"{DATASET_TITLES[bundle.name]} {model_label} {kind.title()} Confusion Matrix")
        save_roc_curve(y, score, labels, names, output_root / "roc_curves" / f"{stem}_roc_curve.png",
                       f"{DATASET_TITLES[bundle.name]} {model_label} {kind.title()} ROC", kind)
    return metrics, report, report_text, cm


def train_experiment(bundle: DatasetBundle, model_name: str, kind: str, config: dict, root: Path, output_root: Path, logger: logging.Logger):
    target = "y_binary" if kind == "binary" else "y_multiclass_id"
    labels, names = class_names(bundle, kind)
    candidates = config["random_forest_candidates"] if model_name == "rf" else config["xgboost_candidates"]
    x_train = bundle.train[bundle.features]
    y_train = bundle.train[target].to_numpy()
    x_validation = bundle.validation[bundle.features]
    y_validation = bundle.validation[target].to_numpy()
    weights = compute_sample_weight(class_weight="balanced", y=y_train) if model_name == "xgb" else None
    trials = []
    best = None
    best_score = -1.0
    total_training_time = 0.0
    for index, params in enumerate(candidates, start=1):
        logger.info("Training %s %s %s candidate %d: %s", bundle.name, model_name, kind, index, params)
        model = make_model(model_name, kind, params, len(labels), config["n_jobs"])
        started = time.perf_counter()
        if weights is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, sample_weight=weights)
        elapsed = time.perf_counter() - started
        total_training_time += elapsed
        validation_pred = model.predict(x_validation)
        score = precision_recall_fscore_support(y_validation, validation_pred, labels=labels, average="macro", zero_division=0)[2]
        trial = {"candidate": index, "parameters": params, "training_time_seconds": elapsed, "validation_macro_f1": score}
        trials.append(trial)
        logger.info("Candidate %d validation macro F1: %.6f", index, score)
        if score > best_score:
            best_score = score
            best = model
            best_params = params
            best_candidate = index
    stem = f"{bundle.name}_{model_name}_{kind}"
    model_dir = root / "models" / bundle.name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / (f"{model_name}_{kind}.joblib" if model_name == "rf" else f"{model_name}_{kind}.json")
    if model_name == "rf":
        joblib.dump(best, model_path, compress=3)
        loaded = joblib.load(model_path)
    else:
        best.save_model(model_path)
        loaded = make_model(model_name, kind, best_params, len(labels), config["n_jobs"])
        loaded.load_model(model_path)
    probe_count = min(1000, len(bundle.test))
    if len(loaded.predict(bundle.test[bundle.features].iloc[:probe_count])) != probe_count:
        raise AssertionError("Saved model reload prediction count mismatch")
    metadata = {
        "dataset": bundle.name,
        "model": model_name,
        "classification_type": kind,
        "protocol": config["protocol"],
        "random_state": 42,
        "features": bundle.features,
        "class_labels": labels,
        "class_names": names,
        "class_mapping": bundle.mapping,
        "selected_candidate": best_candidate,
        "selected_parameters": best_params,
        "validation_selection_macro_f1": best_score,
        "trials": trials,
        "full_reference_training_rows": bundle.full_train_rows,
        "actual_training_rows": len(bundle.train),
        "training_sampled": bundle.sampled,
        "training_class_counts": bundle.train[target].value_counts().sort_index().to_dict(),
        "total_candidate_training_time_seconds": total_training_time,
        "selected_model_training_time_seconds": trials[best_candidate - 1]["training_time_seconds"],
        "model_file": str(model_path.relative_to(root)),
    }
    write_json(model_dir / f"{model_name}_{kind}_metadata.json", metadata)
    results = {}
    for split, frame in (("validation", bundle.validation), ("test", bundle.test)):
        metrics, report, report_text, cm = evaluate(best, frame, bundle, kind, split, output_root, stem, save_plots=(split == "test"))
        metrics["selected_model_training_time_seconds"] = trials[best_candidate - 1]["training_time_seconds"]
        metrics["total_candidate_search_time_seconds"] = total_training_time
        metrics["actual_training_rows"] = len(bundle.train)
        metrics["full_reference_training_rows"] = bundle.full_train_rows
        metrics["training_sampled"] = bundle.sampled
        write_json(output_root / "metrics" / f"{stem}_{split}_metrics.json", metrics)
        write_json(output_root / "classification_reports" / f"{stem}_{split}_classification_report.json", report)
        (output_root / "classification_reports").mkdir(parents=True, exist_ok=True)
        (output_root / "classification_reports" / f"{stem}_{split}_classification_report.txt").write_text(report_text, encoding="utf-8")
        pd.DataFrame(report).transpose().to_csv(output_root / "classification_reports" / f"{stem}_{split}_classification_report.csv")
        results[split] = {"metrics": metrics, "classification_report": report, "confusion_matrix": cm.tolist()}
    write_json(output_root / "logs" / f"{stem}_trials.json", metadata)
    return metadata, results


def evaluate_existing_experiment(bundle: DatasetBundle, model_name: str, kind: str, config: dict, root: Path, output_root: Path, logger: logging.Logger):
    model_dir = root / "models" / bundle.name
    metadata_path = model_dir / f"{model_name}_{kind}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")
    metadata = read_json(metadata_path)
    if metadata["features"] != bundle.features or metadata["protocol"] != config["protocol"]:
        raise ValueError(f"{bundle.name} {model_name} {kind}: saved metadata does not match Stage 1 schema or protocol")
    if model_name == "rf":
        model = joblib.load(model_dir / f"{model_name}_{kind}.joblib")
    else:
        labels, _ = class_names(bundle, kind)
        model = make_model(model_name, kind, metadata["selected_parameters"], len(labels), config["n_jobs"])
        model.load_model(model_dir / f"{model_name}_{kind}.json")
    logger.info("Evaluating existing %s %s %s model", bundle.name, model_name, kind)
    stem = f"{bundle.name}_{model_name}_{kind}"
    results = {}
    for split, frame in (("validation", bundle.validation), ("test", bundle.test)):
        metrics, report, report_text, cm = evaluate(model, frame, bundle, kind, split, output_root, stem, save_plots=(split == "test"))
        metrics["selected_model_training_time_seconds"] = metadata.get(
            "selected_model_training_time_seconds",
            metadata["trials"][metadata["selected_candidate"] - 1]["training_time_seconds"],
        )
        metrics["total_candidate_search_time_seconds"] = metadata["total_candidate_training_time_seconds"]
        metrics["actual_training_rows"] = metadata["actual_training_rows"]
        metrics["full_reference_training_rows"] = metadata["full_reference_training_rows"]
        metrics["training_sampled"] = metadata["training_sampled"]
        write_json(output_root / "metrics" / f"{stem}_{split}_metrics.json", metrics)
        write_json(output_root / "classification_reports" / f"{stem}_{split}_classification_report.json", report)
        (output_root / "classification_reports" / f"{stem}_{split}_classification_report.txt").write_text(report_text, encoding="utf-8")
        pd.DataFrame(report).transpose().to_csv(output_root / "classification_reports" / f"{stem}_{split}_classification_report.csv")
        results[split] = {"metrics": metrics, "classification_report": report, "confusion_matrix": cm.tolist()}
    return metadata, results


def plot_comparisons(summary: pd.DataFrame, output_root: Path):
    table = summary.copy()
    labels = table["Dataset"] + "\n" + table["Classification Type"] + "\n" + table["Model"]
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    ax.bar(x, table["Test Errors"], color=["#1f4e79" if model == "Random Forest" else "#70ad47" for model in table["Model"]])
    ax.set(ylabel="Misclassified test records", title="Model Error Count Comparison", xticks=x, xticklabels=labels)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_root / "tables" / "model_error_count_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    ax.bar(x, table["Total Candidate Search Time"], color=["#5b9bd5" if model == "Random Forest" else "#a5a5a5" for model in table["Model"]])
    ax.set(ylabel="Total candidate training time (seconds)", title="Training Time Comparison", xticks=x, xticklabels=labels)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_root / "tables" / "training_time_comparison_chart.png", bbox_inches="tight")
    plt.close(fig)


def environment_information() -> dict:
    memory = psutil.virtual_memory()
    cpu_name = platform.processor()
    if platform.system() == "Windows":
        try:
            detected = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, check=True, timeout=15,
            ).stdout.strip()
            if detected:
                cpu_name = detected
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "operating_system": platform.platform(),
        "cpu": cpu_name,
        "logical_cpu_count": os.cpu_count(),
        "available_ram_gb_at_start": round(memory.available / (1024 ** 3), 3),
        "total_ram_gb": round(memory.total / (1024 ** 3), 3),
        "gpu_used": False,
        "xgboost_tree_method": "hist (CPU)",
    }


def run(root: Path, skip_training: bool = False, datasets: tuple[str, ...] = ("ciciov2024", "car_hacking")):
    config = read_json(root / "config" / "stage2.json")
    output_root = root / "results" / "model_training"
    for folder in ("metrics", "classification_reports", "confusion_matrices", "roc_curves", "tables", "logs"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_root / "logs" / "stage2.log", encoding="utf-8", mode="a")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    env = environment_information()
    write_json(output_root / "environment.json", env)
    results_path = output_root / "all_experiment_results.json"
    sampling_path = output_root / "sampling.json"
    all_results = read_json(results_path) if set(datasets) != {"ciciov2024", "car_hacking"} and results_path.exists() else {}
    sampling_records = read_json(sampling_path) if set(datasets) != {"ciciov2024", "car_hacking"} and sampling_path.exists() else {}
    for dataset_name in datasets:
        bundle = load_dataset(root, dataset_name, config, logger)
        sampling_records[dataset_name] = {
            "full_reference_training_rows": bundle.full_train_rows,
            "actual_training_rows": len(bundle.train),
            "sampled": bundle.sampled,
            "multiclass_counts": bundle.train["y_multiclass_id"].value_counts().sort_index().to_dict(),
            "binary_counts": bundle.train["y_binary"].value_counts().sort_index().to_dict(),
        }
        if dataset_name == "car_hacking":
            full_counts = pd.Series({
                int(bundle.mapping["multiclass"][label]): int(count)
                for label, count in read_json(root / "runs" / dataset_name / "reference_audit.json")["splits"]["train"]["class_counts"].items()
            }).sort_index()
            sample_counts = bundle.train["y_multiclass_id"].value_counts().sort_index()
            id_to_name = {value: key for key, value in bundle.mapping["multiclass"].items()}
            distribution = pd.DataFrame({
                "Class": [id_to_name[int(cls)] for cls in full_counts.index],
                "Full Train Count": full_counts.values,
                "Sample Train Count": [int(sample_counts.loc[cls]) for cls in full_counts.index],
                "Full Train Percentage": (full_counts / full_counts.sum() * 100).values,
                "Sample Percentage": ([sample_counts.loc[cls] / sample_counts.sum() * 100 for cls in full_counts.index]),
            })
            distribution.to_csv(output_root / "car_hacking_training_sample_distribution.csv", index=False, float_format="%.8f")
        for kind in ("binary", "multiclass"):
            for model_name in ("rf", "xgb"):
                if skip_training:
                    metadata, result = evaluate_existing_experiment(bundle, model_name, kind, config, root, output_root, logger)
                else:
                    metadata, result = train_experiment(bundle, model_name, kind, config, root, output_root, logger)
                key = f"{dataset_name}_{model_name}_{kind}"
                all_results[key] = {"metadata": metadata, **result}
        del bundle
    summary_rows = []
    for key in sorted(all_results):
        entry = all_results[key]
        metadata = entry["metadata"]
        metrics = entry["test"]["metrics"]
        cm = np.asarray(entry["test"]["confusion_matrix"])
        summary_rows.append({
            "Dataset": DATASET_TITLES[metadata["dataset"]],
            "Classification Type": metadata["classification_type"].title(),
            "Model": "Random Forest" if metadata["model"] == "rf" else "XGBoost",
            "Accuracy": metrics["accuracy"],
            "Macro Precision": metrics["macro_precision"],
            "Macro Recall": metrics["macro_recall"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "ROC-AUC": metrics["roc_auc"],
            "Selected Model Training Time": metadata.get("selected_model_training_time_seconds", metadata["trials"][metadata["selected_candidate"] - 1]["training_time_seconds"]),
            "Total Candidate Search Time": metadata["total_candidate_training_time_seconds"],
            "Test Inference Time": metrics["inference_time_seconds"],
            "Training Records": metadata["actual_training_rows"],
            "Validation Records": entry["validation"]["metrics"]["records"],
            "Test Records": metrics["records"],
            "Test Errors": int(cm.sum() - np.trace(cm)),
            "Training Sampled": metadata["training_sampled"],
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "results_summary.csv", index=False, float_format="%.10f")
    write_json(output_root / "results_summary.json", summary_rows)
    write_json(output_root / "sampling.json", sampling_records)
    write_json(output_root / "all_experiment_results.json", all_results)
    plot_comparisons(summary, output_root)
    logger.info("Stage 2 completed: %d experiments", len(summary))
    return summary

