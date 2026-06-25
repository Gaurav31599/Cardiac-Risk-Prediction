"""SHAP feature-importance analysis for the final tree model.

SHAP values explain *which clinical features drive each prediction* — the
single highest-value addition for an explainable-healthcare-ML narrative. We
fit the gradient-boosting pipeline, then explain the trained tree on the
transformed feature space, recovering readable feature names from the
ColumnTransformer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Recover output feature names from the fitted preprocessor."""
    preprocess = pipeline.named_steps["preprocess"]
    return list(preprocess.get_feature_names_out())


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for the model step on transformed features.

    Returns ``(shap_values, X_transformed_df)``. Requires ``shap`` installed.
    """
    import shap

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    X_trans = preprocess.transform(X)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    names = _transformed_feature_names(pipeline)
    X_trans_df = pd.DataFrame(X_trans, columns=names, index=X.index)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans_df)
    except Exception:  # noqa: BLE001 - fall back for any non-tree model
        # Model-agnostic fallback for non-tree winners (LogReg, SVM, KNN, NB):
        # explain the positive-class probability on a small background sample so
        # the SHAP summary still renders regardless of which model wins.
        background = shap.utils.sample(X_trans_df, min(50, len(X_trans_df)), random_state=42)
        explainer = shap.Explainer(
            lambda d: model.predict_proba(d)[:, 1], background
        )
        explained = X_trans_df.iloc[: min(100, len(X_trans_df))]
        shap_values = explainer(explained).values
        return shap_values, explained

    # Some explainers return a list (per class) — take the positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    # Newer SHAP may return a 3D array (samples, features, classes).
    if getattr(shap_values, "ndim", 2) == 3:
        shap_values = shap_values[..., -1]

    return shap_values, X_trans_df


def save_shap_summary(
    pipeline: Pipeline,
    X: pd.DataFrame,
    out_path: str | Path = "reports/shap_summary.png",
) -> Path:
    """Compute SHAP values and write a summary beeswarm plot to disk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap_values, X_trans_df = compute_shap_values(pipeline, X)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_trans_df, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
