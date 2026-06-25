"""End-to-end training entrypoint with an MLflow-tracked algorithm sweep.

Chains: load data -> preprocess -> classical-algorithm sweep -> MLflow logging.
Nine classical classifiers are compared under one shared StratifiedKFold so the
comparison is fair (identical folds, identical metrics). Each algorithm is a
child run nested under a single parent "algorithm-sweep" run. After the sweep we
write ``results/metrics.csv`` (sorted by ROC-AUC) and run SHAP on the winner only.

    python src/train.py     # run the sweep
    mlflow ui               # then open the printed local URL to compare runs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support both ``python src/train.py`` and ``python -m src.train`` by making the
# repo root importable before reaching into src/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import re  # noqa: E402
import warnings  # noqa: E402

import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from src.data.load_data import FEATURE_COLUMNS, class_balance, load_heart_disease  # noqa: E402
from src.features.preprocessing import build_preprocessor  # noqa: E402

SEED = 42
N_SPLITS = 5

# Metric names -> scorer. precision/recall/f1 silence the zero-division warning.
SCORERS = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def get_cv() -> StratifiedKFold:
    """The single shared CV splitter reused by every algorithm in the sweep."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def _raw_estimators() -> dict[str, tuple[object, bool]]:
    """Map algorithm name -> (estimator, needs_scaling).

    XGBoost and LightGBM are imported lazily so the sweep still runs (with a
    warning) if they are not installed.
    """
    estimators: dict[str, tuple[object, bool]] = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=SEED
            ),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEED,
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEED),
            False,
        ),
    }

    try:
        from xgboost import XGBClassifier

        estimators["XGBoost"] = (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=-1,
                random_state=SEED,
            ),
            False,
        )
    except ImportError:
        warnings.warn("xgboost not installed — skipping XGBoost.", stacklevel=2)

    try:
        from lightgbm import LGBMClassifier

        estimators["LightGBM"] = (
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                verbose=-1,
                n_jobs=-1,
                random_state=SEED,
            ),
            False,
        )
    except ImportError:
        warnings.warn("lightgbm not installed — skipping LightGBM.", stacklevel=2)

    estimators["SVM (RBF)"] = (
        SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
        True,
    )
    estimators["KNN"] = (KNeighborsClassifier(), True)
    estimators["Gaussian Naive Bayes"] = (GaussianNB(), False)
    return estimators


def build_sweep_estimators() -> dict[str, tuple[Pipeline, bool]]:
    """Wrap each estimator in a preprocessing Pipeline.

    Scaling-sensitive models (LogReg, SVM, KNN) standardize numerics; tree-based
    models and Naive Bayes skip scaling. Categoricals are one-hot encoded in all
    cases. Steps are named "preprocess"/"model" for SHAP compatibility.
    """
    built: dict[str, tuple[Pipeline, bool]] = {}
    for name, (estimator, needs_scaling) in _raw_estimators().items():
        pipe = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(scale_numeric=needs_scaling)),
                ("model", estimator),
            ]
        )
        built[name] = (pipe, needs_scaling)
    return built


def _preprocessing_params(needs_scaling: bool) -> dict[str, object]:
    return {
        "scaler_used": needs_scaling,
        "encoding": "onehot",
        "split_ratio": f"{N_SPLITS}-fold stratified CV",
        "random_seed": SEED,
    }


def run_sweep(
    df: pd.DataFrame,
    *,
    log_to_mlflow: bool = True,
    return_fold_indices: bool = False,
):
    """Cross-validate every algorithm under one shared CV; optionally log to MLflow.

    Returns the metrics DataFrame (sorted by ``roc_auc_mean`` desc). When
    ``return_fold_indices`` is True, also returns ``{model_name: [test_idx,...]}``
    so callers/tests can confirm the folds were identical across models.
    """
    X = df[FEATURE_COLUMNS]
    y = df["target"]
    cv = get_cv()
    estimators = build_sweep_estimators()

    rows: list[dict[str, object]] = []
    fold_indices: dict[str, list[tuple[int, ...]]] = {}

    mlflow = None
    if log_to_mlflow:
        import mlflow as _mlflow
        import mlflow.sklearn  # noqa: F401

        mlflow = _mlflow
        mlflow.set_experiment("cardiac-risk-classification")
        mlflow.start_run(run_name="algorithm-sweep")

    try:
        for name, (pipe, needs_scaling) in estimators.items():
            cv_res = cross_validate(
                pipe,
                X,
                y,
                cv=cv,
                scoring=SCORERS,
                return_indices=True,
                n_jobs=None,
            )
            fold_indices[name] = [
                tuple(int(i) for i in test) for test in cv_res["indices"]["test"]
            ]

            metrics: dict[str, float] = {}
            for metric in SCORERS:
                scores = cv_res[f"test_{metric}"]
                metrics[f"{metric}_mean"] = float(scores.mean())
                metrics[f"{metric}_std"] = float(scores.std())

            rows.append({"model": name, **metrics})

            if log_to_mlflow:
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.log_param("model_name", name)
                    model_params = pipe.named_steps["model"].get_params()
                    for key, value in model_params.items():
                        mlflow.log_param(f"model__{key}", value)
                    for key, value in _preprocessing_params(needs_scaling).items():
                        mlflow.log_param(key, value)
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)
                    fitted = clone(pipe).fit(X, y)
                    # cloudpickle handles the XGBoost/LightGBM sklearn wrappers,
                    # which the default skops serializer rejects as untrusted.
                    mlflow.sklearn.log_model(
                        fitted,
                        name="model",
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                    )
    finally:
        if log_to_mlflow:
            mlflow.end_run()

    metrics_df = (
        pd.DataFrame(rows)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )

    if return_fold_indices:
        return metrics_df, fold_indices
    return metrics_df


def model_slug(name: str) -> str:
    """Turn a display name ("SVM (RBF)") into a filename slug ("svm_rbf")."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


def save_all_models(df: pd.DataFrame, models_dir: Path | None = None) -> dict[str, Path]:
    """Fit every sweep pipeline on the full data and persist as models/<slug>.pkl.

    The saved object is the full fitted Pipeline (preprocess + model), so the API
    can call ``predict_proba`` directly without re-deriving preprocessing.
    """
    import joblib

    models_dir = models_dir or (_ROOT / "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    X, y = df[FEATURE_COLUMNS], df["target"]
    for name, (pipe, _scaling) in build_sweep_estimators().items():
        fitted = clone(pipe).fit(X, y)
        path = models_dir / f"{model_slug(name)}.pkl"
        joblib.dump(fitted, path)
        saved[name] = path
    return saved


def _run_shap_on_winner(winner: str, df: pd.DataFrame) -> None:
    """Run the existing SHAP module on the winning model only."""
    try:
        from src.explainability.shap_analysis import save_shap_summary
    except ImportError:
        print("\n(shap/matplotlib not installed — skipping SHAP plot)")
        return

    pipe = build_sweep_estimators()[winner][0]
    pipe.fit(df[FEATURE_COLUMNS], df["target"])
    try:
        path = save_shap_summary(pipe, df[FEATURE_COLUMNS])
        print(f"\nSHAP summary for winner ({winner}) saved to {path}")
    except Exception as exc:  # noqa: BLE001 - non-tree models lack TreeExplainer
        print(
            f"\nSHAP skipped: TreeExplainer does not support the winning model "
            f"({winner}). [{type(exc).__name__}]"
        )


def main() -> None:
    df = load_heart_disease()
    print(f"Loaded {len(df)} patients.")
    print("Class balance (0 = no disease, 1 = disease):")
    print(class_balance(df).round(3).to_string(), "\n")

    metrics_df = run_sweep(df, log_to_mlflow=True)

    out_path = _ROOT / "results" / "metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_path, index=False)

    print("Algorithm sweep (sorted by ROC-AUC, mean across folds):")
    display_cols = ["model", "roc_auc_mean", "f1_mean", "recall_mean", "precision_mean"]
    print(metrics_df[display_cols].round(3).to_string(index=False))
    print(f"\nFull metrics written to {out_path}")

    saved = save_all_models(df)
    print(f"\nSaved {len(saved)} fitted models to {(_ROOT / 'models')}")

    winner = str(metrics_df.iloc[0]["model"])
    _run_shap_on_winner(winner, df)


if __name__ == "__main__":
    main()
