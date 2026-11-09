from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
DATASETS = ("ciciov2024", "car_hacking")
MODELS = ("rf_binary", "rf_multiclass", "xgb_binary", "xgb_multiclass")


def check_package():
    missing = []
    for dataset in DATASETS:
        for model in MODELS:
            suffix = ".joblib" if model.startswith("rf_") else ".json"
            for file in (ROOT / "models" / dataset / (model + suffix), ROOT / "models" / dataset / (model + "_metadata.json")):
                if not file.is_file():
                    missing.append(str(file.relative_to(ROOT)))
        for file in (ROOT / "metadata" / "datasets" / dataset / "audit.json", ROOT / "metadata" / "datasets" / dataset / "reference_audit.json"):
            if not file.is_file():
                missing.append(str(file.relative_to(ROOT)))
    for file in (ROOT / "results" / "model_training" / "results_summary.csv", ROOT / "results" / "xai" / "comparison_summary.csv"):
        if not file.is_file():
            missing.append(str(file.relative_to(ROOT)))
    if missing:
        print("MISSING PACKAGE FILES:")
        print("\n".join(missing))
        return 1
    print("PASS: eight saved model files, metadata, dataset audits and summary results are present.")
    return 0


def check_raw():
    absent = []
    mismatch = []
    verified = 0
    for dataset in DATASETS:
        audit = json.loads((ROOT / "metadata" / "datasets" / dataset / "audit.json").read_text(encoding="utf-8"))
        for entry in audit["sources"]:
            source = ROOT / "data" / "raw" / dataset / entry["file"]
            if not source.exists():
                absent.append(str(source.relative_to(ROOT)))
                continue
            digest = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
            if digest != entry["sha256"]:
                mismatch.append(str(source.relative_to(ROOT)))
            else:
                verified += 1
    if absent:
        print(f"RAW DATA NOT INCLUDED: {len(absent)} source CSV files are absent (expected for the distribution zip).")
        for file in absent: print("  " + file)
    if mismatch:
        print("RAW DATA HASH MISMATCH:", ", ".join(mismatch))
        return 1
    print(f"SHA256 verified for {verified} available source files.")
    return 0


if __name__ == "__main__":
    result = check_package()
    if "--check-raw" in sys.argv:
        result |= check_raw()
    raise SystemExit(result)
