"""Tests for the MLflow algorithm sweep (fold reuse + ordering contracts)."""

from __future__ import annotations

from src.train import build_sweep_estimators, get_cv, run_sweep


def test_same_fold_indices_reused_across_every_model(tiny_df):
    """Every algorithm must be evaluated on the identical CV folds."""
    _, fold_indices = run_sweep(
        tiny_df, log_to_mlflow=False, return_fold_indices=True
    )
    names = list(fold_indices)
    assert len(names) >= 2

    reference = fold_indices[names[0]]
    for name in names[1:]:
        assert fold_indices[name] == reference, f"{name} used different folds"


def test_shared_cv_matches_sweep_folds(tiny_df):
    """The folds used in the sweep match the shared get_cv() splitter."""
    X = tiny_df[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]
    y = tiny_df["target"]
    expected = [tuple(int(i) for i in test) for _, test in get_cv().split(X, y)]

    _, fold_indices = run_sweep(
        tiny_df, log_to_mlflow=False, return_fold_indices=True
    )
    assert next(iter(fold_indices.values())) == expected


def test_metrics_sorted_descending_by_roc_auc(tiny_df):
    result = run_sweep(tiny_df, log_to_mlflow=False)
    aucs = list(result["roc_auc_mean"])
    assert aucs == sorted(aucs, reverse=True)


def test_sweep_covers_all_available_algorithms(tiny_df):
    result = run_sweep(tiny_df, log_to_mlflow=False)
    # All estimators that built successfully appear exactly once.
    assert set(result["model"]) == set(build_sweep_estimators())
    assert result["model"].is_unique
