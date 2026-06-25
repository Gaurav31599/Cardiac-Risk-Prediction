"""Tests for preprocessing and the stratified split."""

from __future__ import annotations

import numpy as np

from src.data.load_data import FEATURE_COLUMNS
from src.features.preprocessing import build_preprocessor, split_data


def test_split_is_stratified_and_disjoint(tiny_df):
    X_train, X_test, y_train, y_test = split_data(tiny_df, test_size=0.25)

    # Sizes add up and indices are disjoint.
    assert len(X_train) + len(X_test) == len(tiny_df)
    assert set(X_train.index).isdisjoint(set(X_test.index))

    # Stratification keeps the positive rate close between folds.
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.2

    # Only the feature columns are carried into X.
    assert list(X_train.columns) == FEATURE_COLUMNS


def test_preprocessor_outputs_finite_numeric_matrix(tiny_df):
    pre = build_preprocessor()
    X = tiny_df[FEATURE_COLUMNS]
    transformed = pre.fit_transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    # One-hot encoding expands beyond the 13 raw columns.
    assert transformed.shape[0] == len(tiny_df)
    assert transformed.shape[1] >= len(FEATURE_COLUMNS)
    assert np.isfinite(transformed).all()


def test_numeric_features_are_standardized(tiny_df):
    pre = build_preprocessor()
    pre.fit(tiny_df[FEATURE_COLUMNS])
    names = list(pre.get_feature_names_out())
    # Standardized numeric columns are prefixed "num__".
    assert any(name.startswith("num__") for name in names)
