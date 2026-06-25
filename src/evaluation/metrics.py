"""Evaluation: ROC-AUC, precision/recall, confusion matrix, cross-validation.

Accuracy alone is misleading under class imbalance, so the headline metrics
here are ROC-AUC and precision/recall. Model selection uses stratified
cross-validation rather than a single split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Return the headline classification metrics as a flat dict."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix [[TN, FP], [FN, TP]]."""
    return confusion_matrix(y_true, y_pred)


def cross_validate_auc(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Stratified k-fold ROC-AUC for model selection."""
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc")
    return {"cv_auc_mean": float(scores.mean()), "cv_auc_std": float(scores.std())}


def summarize_models(
    fitted: dict[str, BaseEstimator],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Build a side-by-side comparison table across fitted models."""
    rows = []
    for name, model in fitted.items():
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        row = {"model": name, **classification_metrics(y_test, y_pred, y_proba)}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").sort_values(
        "roc_auc", ascending=False
    )
