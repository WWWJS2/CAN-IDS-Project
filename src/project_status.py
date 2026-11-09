import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["ciciov2024", "car_hacking"]
PARTITIONS = ["train", "validation", "test"]

print("Explainable AI Driven Intrusion Detection Systems for In-Vehicle Networks")
print("Practical Project Status")
print()

for dataset in DATASETS:
    folder = ROOT / "results" / dataset
    success_file = folder / "SUCCESS.json"
    audit_file = folder / "audit.json"
    reference_file = folder / "reference_audit.json"
    if not success_file.is_file() or not audit_file.is_file() or not reference_file.is_file():
        raise SystemExit(f"Project results are incomplete for {dataset}")
    success = json.loads(success_file.read_text(encoding="utf-8"))
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    print(dataset.upper())
    print(f"Status: {success.get('status', 'complete')}")
    print(f"Validated messages: {audit['rows_valid']:,}")
    print(f"Rejected messages: {audit['rows_rejected']:,}")
    print(f"Unique input groups: {audit['unique_input_groups']:,}")
    print("Classes:")
    for label, count in audit["class_counts"].items():
        print(f"  {label}: {count:,}")
    print("Reference partitions:")
    for partition in PARTITIONS:
        split = reference["splits"][partition]
        print(f"  {partition}: {split['rows']:,}")
    print()

print("Project source code, validation evidence, and completed run results are available.")
