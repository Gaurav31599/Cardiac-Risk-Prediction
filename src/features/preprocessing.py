"""Scaling, encoding, and stratified train/test splitting.

The 13 Cleveland features mix continuous measurements (age, cholesterol, ...)
with low-cardinality categoricals (chest pain type, slope, thal, ...). We scale
the numeric block and one-hot encode the categorical block inside a single
``ColumnTransformer`` so the exact same transform is reused at predict time.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import FEATURE_COLUMNS

# Continuous measurements get standardized.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Low-cardinality clinical codes get one-hot encoded.
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(FEATURE_COLUMNS)


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Return a ColumnTransformer: optionally standardize numerics, one-hot categoricals.

    ``scale_numeric`` defaults to True (the original behavior, used by the
    scaling-sensitive models — logistic regression, SVM, KNN). Tree-based models
    and Naive Bayes pass ``scale_numeric=False`` since they are scale-invariant;
    categoricals are one-hot encoded either way.
    """
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def split_data(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the feature columns.

    Stratifying on the target preserves the class ratio in both folds, which
    matters here because the dataset is only mildly balanced.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[target_col].copy()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
