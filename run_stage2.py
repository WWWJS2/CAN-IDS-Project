from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stage2_pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the eight Stage 2 CAN IDS models.")
    parser.add_argument("--skip-training", action="store_true", help="Evaluate saved models after a completed run.")
    parser.add_argument("--dataset", choices=("all", "ciciov2024", "car_hacking"), default="all",
                        help="Run both datasets or only one dataset. A single-dataset run preserves the other results.")
    args = parser.parse_args()
    required = [
        ROOT / "runs" / dataset / name
        for dataset in ("ciciov2024", "car_hacking")
        for name in ("audit.json", "reference_audit.json", "label_mapping.json", "SUCCESS.json", "train.csv.gz", "validation.csv.gz", "test.csv.gz")
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing Stage 1 files:\n" + "\n".join(missing))
    datasets = ("ciciov2024", "car_hacking") if args.dataset == "all" else (args.dataset,)
    run(ROOT, skip_training=args.skip_training, datasets=datasets)


if __name__ == "__main__":
    main()
