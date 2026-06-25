"""Smoke tests: fit/predict on a tiny fixture with correct shapes."""

from __future__ import annotations

import numpy as np
import pytest

from src.clustering import cluster_patients, crosstab_against_target
from src.evaluation.metrics import summarize_models
from src.features.preprocessing import split_data
from src.models import MODEL_BUILDERS


@pytest.mark.parametrize("name", list(MODEL_BUILDERS))
def test_model_fits_and_predicts_correct_shapes(name, tiny_df):
    X_train, X_test, y_train, y_test = split_data(tiny_df)
    model = MODEL_BUILDERS[name]()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_summarize_models_table(tiny_df):
    X_train, X_test, y_train, y_test = split_data(tiny_df)
    fitted = {n: b().fit(X_train, y_train) for n, b in MODEL_BUILDERS.items()}
    table = summarize_models(fitted, X_test, y_test)
    assert set(table.index) == set(MODEL_BUILDERS)
    assert "roc_auc" in table.columns


def test_clustering_labels_and_crosstab(tiny_df):
    clusters = cluster_patients(tiny_df, n_clusters=3)
    assert len(clusters) == len(tiny_df)
    assert clusters.nunique() <= 3

    ct = crosstab_against_target(clusters, tiny_df["target"])
    # Row-normalized, so each cluster row sums to 1.
    assert np.allclose(ct.sum(axis=1).values, 1.0)
