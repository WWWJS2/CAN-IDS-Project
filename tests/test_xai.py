import json
from pathlib import Path

import numpy as np

from src.xai_pipeline import compactness, cosine


ROOT = Path(__file__).resolve().parents[1]


def test_xai_protocol_is_complete():
    protocol = json.loads((ROOT / "config" / "xai_protocol.json").read_text())
    assert protocol["random_seed"] == 42
    assert protocol["samples_per_class"] > 0
    assert protocol["target_policy"] == "predicted_class"


def test_similarity_and_compactness_helpers():
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert compactness(np.array([0.8, 0.1, 0.1]), 0.80) == 1
