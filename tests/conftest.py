"""Shared fixtures: a tiny synthetic Cleveland-shaped frame for fast tests.

Tests must not depend on network access, so we synthesize a small frame with
the exact 13 feature columns plus the raw ``num`` and binary ``target``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import FEATURE_COLUMNS


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    data = {
        "age": rng.integers(35, 75, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 350, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 4, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    }
    df = pd.DataFrame(data)[FEATURE_COLUMNS]
    # A learnable but noisy target so models can fit without being trivial.
    logit = (
        0.04 * (df["age"] - 55)
        + 0.6 * df["exang"]
        + 0.5 * (df["cp"] >= 3)
        - 0.02 * (df["thalach"] - 150)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["target"] = (prob > 0.5).astype(int)
    df["num"] = df["target"]
    return df
