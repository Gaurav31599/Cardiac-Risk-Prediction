"""Tests for the evaluation metrics."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    classification_metrics,
    confusion,
    cross_validate_auc,
)
from src.features.preprocessing import split_data
from src.models import build_logreg


def test_classification_metrics_perfect_prediction():
    y_true = np.array([0, 1, 0, 1])
    y_pred = y_true.copy()
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    m = classification_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["roc_auc"] == 1.0


def test_classification_metrics_without_proba_omits_auc():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    m = classification_metrics(y_true, y_pred)
    assert "roc_auc" not in m
    assert 0.0 <= m["recall"] <= 1.0


def test_confusion_matrix_shape():
    cm = confusion(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    assert cm.shape == (2, 2)
    assert cm.sum() == 4


def test_cross_validate_auc_returns_mean_and_std(tiny_df):
    X_train, _, y_train, _ = split_data(tiny_df)
    out = cross_validate_auc(build_logreg(), X_train, y_train, n_splits=3)
    assert "cv_auc_mean" in out and "cv_auc_std" in out
    assert 0.0 <= out["cv_auc_mean"] <= 1.0
