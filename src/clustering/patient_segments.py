"""Exploratory KMeans patient segmentation.

This is a *complementary, unsupervised* view of the population — "what natural
risk profiles exist in these patients?" — and is deliberately kept separate
from the supervised classifier. It is NOT a substitute for prediction, and the
clusters are only described as aligning with the diagnostic label when an
explicit cross-tabulation shows real overlap (see ``crosstab_against_target``).
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans

from src.data.load_data import FEATURE_COLUMNS
from src.features.preprocessing import build_preprocessor


def cluster_patients(
    df: pd.DataFrame,
    *,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.Series:
    """Assign each patient a KMeans cluster label on the scaled features.

    Returns a Series of integer cluster ids aligned to ``df.index``.
    """
    X = df[FEATURE_COLUMNS]
    # Reuse the same preprocessing so clustering sees scaled/encoded features,
    # not raw magnitudes (otherwise cholesterol would dominate the distance).
    X_transformed = build_preprocessor().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_transformed)
    return pd.Series(labels, index=df.index, name="cluster")


def crosstab_against_target(
    clusters: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    """Cross-tabulate cluster membership against the diagnostic label.

    Use this to *check* whether clusters relate to disease status before making
    any such claim, rather than assuming they do.
    """
    return pd.crosstab(clusters, target, normalize="index").rename_axis(
        index="cluster", columns="target"
    )


if __name__ == "__main__":  # pragma: no cover - manual exploration
    from src.data.load_data import load_heart_disease

    data = load_heart_disease()
    clusters = cluster_patients(data)
    print("Cluster sizes:\n", clusters.value_counts().sort_index().to_string())
    print("\nCluster vs. target (row-normalized):")
    print(crosstab_against_target(clusters, data["target"]).round(2).to_string())
