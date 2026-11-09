from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from lime.lime_tabular import LimeTabularExplainer

try:
    from dataset_io import iter_partition
except ModuleNotFoundError:      from .dataset_io import iter_partition


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def load_model(root: Path, dataset: str, model_name: str, task: str):
    stem = f"{model_name}_{task}"
    metadata = read_json(root / "models" / dataset / f"{stem}_metadata.json")
    if model_name == "rf":
        model = joblib.load(root / "models" / dataset / f"{stem}.joblib")
    else:
        model = xgboost.XGBClassifier()
        model.load_model(root / "models" / dataset / f"{stem}.json")
    return model, metadata


def target_column(task: str):
    return "y_binary" if task == "binary" else "y_multiclass_id"


def stable_row_ids(frame: pd.DataFrame):
    if "row_id" in frame:
        return frame["row_id"].astype(str)
    if {"source_id", "source_row"}.issubset(frame.columns):
        return frame["source_id"].astype(str) + ":" + frame["source_row"].astype(str)
    return pd.Series(range(len(frame)), index=frame.index).astype(str)


def collect_balanced(folder: Path, partition: str, task: str, class_labels, per_class: int, seed: int):
    target = target_column(task)
    rng = np.random.default_rng(seed)
    collected = {int(label): [] for label in class_labels}
    for chunk in iter_partition(folder, partition, chunksize=200_000):
        for label in class_labels:
            need = per_class - len(collected[int(label)])
            if need <= 0:
                continue
            candidates = chunk.loc[chunk[target] == int(label)]
            if not candidates.empty:
                take = candidates.sample(min(need, len(candidates)), random_state=int(rng.integers(0, 2**31 - 1)))
                collected[int(label)].extend(take.to_dict("records"))
        if all(len(rows) >= per_class for rows in collected.values()):
            break
    missing = {label: per_class - len(rows) for label, rows in collected.items() if len(rows) < per_class}
    if missing:
        raise RuntimeError(f"Insufficient {partition} support for {task}: {missing}")
    frame = pd.DataFrame([row for label in class_labels for row in collected[int(label)]])
    return frame.reset_index(drop=True)


def find_error_cases(folder: Path, task: str, model, features, maximum: int):
    if maximum <= 0:
        return pd.DataFrame()
    target = target_column(task)
    found = []
    for chunk in iter_partition(folder, "test", chunksize=200_000):
        prediction = model.predict(chunk[features])
        wrong = chunk.loc[prediction.astype(int) != chunk[target].to_numpy(dtype=int)]
        if not wrong.empty:
            found.extend(wrong.head(maximum - len(found)).to_dict("records"))
        if len(found) >= maximum:
            break
    return pd.DataFrame(found)


def shap_target_vectors(explainer, x: pd.DataFrame, targets: np.ndarray):
    raw = explainer.shap_values(x)
    if isinstance(raw, list):
        return np.vstack([np.asarray(raw[int(target)])[i] for i, target in enumerate(targets)])
    values = np.asarray(raw)
    if values.ndim == 3:
        return np.vstack([values[i, :, int(target)] for i, target in enumerate(targets)])
    if values.ndim == 2:
        rows = values.copy()
        if len(np.unique(targets)) > 1:
            rows[targets == 0] *= -1.0
        return rows
    raise RuntimeError(f"Unsupported SHAP value shape {values.shape}")


def lime_vector(explanation, target: int, feature_count: int):
    vector = np.zeros(feature_count, dtype=float)
    for index, weight in explanation.local_exp[int(target)]:
        vector[int(index)] = float(weight)
    return vector


def cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 1.0


def compactness(vector, mass=0.80):
    values = np.sort(np.abs(vector))[::-1]
    total = float(values.sum())
    if total == 0:
        return 0
    return int(np.searchsorted(np.cumsum(values), mass * total) + 1)


def faithfulness(predict_proba, row, target, vector, baseline, top_k):
    ranking = np.argsort(np.abs(vector))[::-1]
    original = float(predict_proba(row.reshape(1, -1))[0, int(target)])
    drops = []
    changed = row.copy()
    for rank, index in enumerate(ranking[:top_k], 1):
        changed[index] = baseline[index]
        probability = float(predict_proba(changed.reshape(1, -1))[0, int(target)])
        drops.append({"k": rank, "probability_drop": original - probability})
    return float(np.mean([item["probability_drop"] for item in drops])) if drops else 0.0, drops


def nearest_pairs(x: np.ndarray, predictions: np.ndarray, scale: np.ndarray):
    pairs = []
    scaled = x / np.where(scale == 0, 1, scale)
    for i in range(len(x)):
        candidates = [j for j in range(len(x)) if j != i and predictions[j] == predictions[i]]
        if not candidates:
            continue
        j = min(candidates, key=lambda candidate: float(np.linalg.norm(scaled[i] - scaled[candidate])))
        if i < j:
            pairs.append((i, j))
    return sorted(set(pairs))


def plot_global(values, features, title, output):
    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.asarray(features)[order], importance[order], color="#1f4e79")
    ax.set_xlabel("Mean absolute attribution")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_local(shap_vector, lime_values, features, title, output):
    score = np.maximum(np.abs(shap_vector), np.abs(lime_values))
    order = np.argsort(score)[-8:]
    positions = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(positions - 0.18, shap_vector[order], height=0.34, label="SHAP", color="#1f4e79")
    ax.barh(positions + 0.18, lime_values[order], height=0.34, label="LIME", color="#c55a11")
    ax.set_yticks(positions, np.asarray(features)[order])
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Native attribution weight")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_experiment(root: Path, dataset: str, model_name: str, task: str, protocol: dict):
    model, metadata = load_model(root, dataset, model_name, task)
    features = metadata["features"]
    class_labels = [int(value) for value in metadata["class_labels"]]
    class_names = metadata["class_names"]
    folder = root / "runs" / dataset
    seed = int(protocol["random_seed"])
    background_frame = collect_balanced(folder, "train", task, class_labels, int(protocol["background_per_class"]), seed)
    sample_frame = collect_balanced(folder, "test", task, class_labels, int(protocol["samples_per_class"]), seed + 1)
    error_frame = find_error_cases(folder, task, model, features, int(protocol["maximum_error_cases_per_experiment"]))
    if not error_frame.empty:
        existing = set(stable_row_ids(sample_frame))
        error_frame = error_frame.loc[~stable_row_ids(error_frame).isin(existing)]
        sample_frame = pd.concat([sample_frame, error_frame], ignore_index=True)

    x_background = background_frame[features].astype(float)
    x_sample = sample_frame[features].astype(float)
    predict_proba = lambda values: model.predict_proba(pd.DataFrame(values, columns=features))
    predictions = model.predict(x_sample).astype(int)
    probabilities = model.predict_proba(x_sample)
    true_values = sample_frame[target_column(task)].to_numpy(dtype=int)
    targets = predictions.copy()
    baseline = x_background.median().to_numpy(dtype=float)
    scale = x_background.std(ddof=1).replace(0, 1).to_numpy(dtype=float)

    start = time.perf_counter()
    shap_explainer = shap.TreeExplainer(model)
    shap_init = time.perf_counter() - start
    start = time.perf_counter()
    shap_vectors = shap_target_vectors(shap_explainer, x_sample, targets)
    shap_total = time.perf_counter() - start

    start = time.perf_counter()
    lime_explainer = LimeTabularExplainer(
        x_background.to_numpy(), feature_names=features, class_names=class_names,
        mode="classification", discretize_continuous=False, random_state=seed,
    )
    lime_init = time.perf_counter() - start
    lime_vectors = []
    lime_times = []
    for i, target in enumerate(targets):
        start = time.perf_counter()
        explanation = lime_explainer.explain_instance(
            x_sample.iloc[i].to_numpy(), predict_proba, labels=[int(target)],
            num_features=len(features), num_samples=int(protocol["lime_num_samples"]),
        )
        lime_times.append(time.perf_counter() - start)
        lime_vectors.append(lime_vector(explanation, int(target), len(features)))
    lime_vectors = np.vstack(lime_vectors)

    output = root / "results" / "xai" / f"{dataset}_{model_name}_{task}"
    output.mkdir(parents=True, exist_ok=True)
    sample_ids = stable_row_ids(sample_frame).tolist()
    records = []
    curves = []
    threshold = float(protocol["sparsity_relative_threshold"])
    for i, row_id in enumerate(sample_ids):
        for method, vector, elapsed in (
            ("SHAP", shap_vectors[i], shap_total / len(sample_frame)),
            ("LIME", lime_vectors[i], lime_times[i]),
        ):
            faithful, curve = faithfulness(predict_proba, x_sample.iloc[i].to_numpy(), targets[i], vector, baseline, int(protocol["faithfulness_top_k"]))
            curves.extend({"row_id": row_id, "method": method, **point} for point in curve)
            maximum = float(np.max(np.abs(vector)))
            nonzero = int(np.count_nonzero(np.abs(vector) > threshold * maximum)) if maximum else 0
            records.append({
                "row_id": row_id, "method": method, "true_class": int(true_values[i]),
                "predicted_class": int(predictions[i]), "target_class": int(targets[i]),
                "is_error": bool(true_values[i] != predictions[i]),
                "faithfulness_aopc_top_k": faithful,
                "sparsity_active_features": nonzero,
                "sparsity_active_fraction": nonzero / len(features),
                "compactness_features_for_80pct_mass": compactness(vector, float(protocol["compactness_mass_threshold"])),
                "explanation_time_seconds": elapsed,
            })

    repetitions = int(protocol["stability_repetitions"])
    stability = []
    for method in ("SHAP", "LIME"):
        for i, row_id in enumerate(sample_ids):
            repeated = []
            for repetition in range(repetitions):
                if method == "SHAP":
                    repeated.append(shap_target_vectors(shap_explainer, x_sample.iloc[[i]], targets[i:i+1])[0])
                else:
                    repeated_explainer = LimeTabularExplainer(
                        x_background.to_numpy(), feature_names=features, class_names=class_names,
                        mode="classification", discretize_continuous=False, random_state=seed + repetition,
                    )
                    exp = repeated_explainer.explain_instance(
                        x_sample.iloc[i].to_numpy(), predict_proba, labels=[int(targets[i])],
                        num_features=len(features), num_samples=int(protocol["lime_num_samples"]),
                    )
                    repeated.append(lime_vector(exp, int(targets[i]), len(features)))
            similarities = [cosine(repeated[a], repeated[b]) for a in range(repetitions) for b in range(a + 1, repetitions)]
            stability.append({"row_id": row_id, "method": method, "same_input_repeat_cosine": float(np.mean(similarities))})

    consistency = []
    pairs = nearest_pairs(x_sample.to_numpy(), predictions, scale)
    for method, vectors in (("SHAP", shap_vectors), ("LIME", lime_vectors)):
        for i, j in pairs:
            consistency.append({
                "method": method, "row_id_a": sample_ids[i], "row_id_b": sample_ids[j],
                "standardized_input_distance": float(np.linalg.norm((x_sample.iloc[i].to_numpy() - x_sample.iloc[j].to_numpy()) / scale)),
                "near_input_explanation_cosine": cosine(vectors[i], vectors[j]),
            })

    attribution_rows = []
    for i, row_id in enumerate(sample_ids):
        for method, vector in (("SHAP", shap_vectors[i]), ("LIME", lime_vectors[i])):
            attribution_rows.extend({"row_id": row_id, "method": method, "feature": feature, "attribution": float(value), "target_class": int(targets[i])} for feature, value in zip(features, vector))
    pd.DataFrame(attribution_rows).to_csv(output / "attributions.csv", index=False)
    pd.DataFrame(records).to_csv(output / "sample_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "faithfulness_curves.csv", index=False)
    pd.DataFrame(stability).to_csv(output / "stability.csv", index=False)
    pd.DataFrame(consistency).to_csv(output / "consistency.csv", index=False)
    plot_global(shap_vectors, features, f"{dataset} {model_name.upper()} {task} SHAP", output / "shap_global.png")
    plot_global(lime_vectors, features, f"{dataset} {model_name.upper()} {task} LIME", output / "lime_global.png")
    example_indices = [0]
    attack = next((i for i, target in enumerate(true_values) if target != 0), None)
    error = next((i for i, value in enumerate(true_values != predictions) if value), None)
    for index in (attack, error):
        if index is not None and index not in example_indices:
            example_indices.append(index)
    for index in example_indices:
        kind = "error" if true_values[index] != predictions[index] else ("attack" if true_values[index] != 0 else "benign")
        plot_local(shap_vectors[index], lime_vectors[index], features,
                   f"{dataset} {model_name.upper()} {task}: {kind} row {sample_ids[index]}",
                   output / f"local_comparison_{kind}_{index}.png")

    timing = {
        "shap_initialization_seconds": shap_init,
        "shap_mean_explanation_seconds": shap_total / len(sample_frame),
        "lime_initialization_seconds": lime_init,
        "lime_mean_explanation_seconds": float(np.mean(lime_times)),
        "lime_explanation_seconds_std": float(np.std(lime_times, ddof=1)) if len(lime_times) > 1 else 0.0,
    }
    write_json(output / "timing.json", timing)
    write_json(output / "sample_protocol.json", {
        "dataset": dataset, "model": model_name, "classification": task,
        "features": features, "class_names": class_names, "sample_row_ids": sample_ids,
        "true_classes": true_values.tolist(), "predicted_classes": predictions.tolist(),
        "target_classes": targets.tolist(), "background_row_ids": stable_row_ids(background_frame).tolist(),
        "protocol": protocol,
    })
    summary = pd.DataFrame(records).groupby("method").agg(
        n=("row_id", "count"), faithfulness_mean=("faithfulness_aopc_top_k", "mean"),
        faithfulness_std=("faithfulness_aopc_top_k", "std"), sparsity_fraction_mean=("sparsity_active_fraction", "mean"),
        compactness_mean=("compactness_features_for_80pct_mass", "mean"), time_mean=("explanation_time_seconds", "mean"),
    ).reset_index()
    stability_frame = pd.DataFrame(stability).groupby("method")["same_input_repeat_cosine"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "stability_mean", "std": "stability_std"})
    consistency_frame = pd.DataFrame(consistency)
    if not consistency_frame.empty:
        consistency_frame = consistency_frame.groupby("method")["near_input_explanation_cosine"].agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "consistency_mean", "std": "consistency_std", "count": "consistency_pairs"})
        summary = summary.merge(consistency_frame, on="method", how="left")
    summary = summary.merge(stability_frame, on="method", how="left")
    summary.insert(0, "dataset", dataset)
    summary.insert(1, "model", model_name)
    summary.insert(2, "classification", task)
    summary.to_csv(output / "summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run reproducible SHAP and LIME comparisons for the eight saved IDS models.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    protocol = read_json(root / "config" / "xai_protocol.json")
    summaries = []
    for dataset in ("ciciov2024", "car_hacking"):
        for task in ("binary", "multiclass"):
            for model_name in ("rf", "xgb"):
                print(f"Running {dataset} {model_name} {task}", flush=True)
                summaries.append(run_experiment(root, dataset, model_name, task, protocol))
    combined = pd.concat(summaries, ignore_index=True)
    result_root = root / "results" / "xai"
    combined.to_csv(result_root / "comparison_summary.csv", index=False)
    write_json(result_root / "environment.json", {
        "python": platform.python_version(), "operating_system": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "shap": shap.__version__,
        "protocol_file": "config/xai_protocol.json",
    })
    print(combined.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
