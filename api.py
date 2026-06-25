"""FastAPI backend for the Cardiac Risk frontend.

Endpoints
---------
GET  /health   -> connectivity + which models are loaded
POST /predict  -> probability, prediction, and (tree models) per-feature SHAP

Models are loaded from ``models/<slug>.pkl`` (written by ``src/train.py`` via
``save_all_models``). Each pickle is a fitted sklearn Pipeline (preprocess +
model); standalone estimators / (scaler, model) pairs are also tolerated.

Run:
    pip install fastapi uvicorn shap joblib
    uvicorn api:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

# The 13 Cleveland features, in the order the models were trained on.
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

MODELS_DIR = BASE_DIR / "models"

app = FastAPI(title="Cardiac Risk API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_CACHE: dict[str, object] = {}


def _available_slugs() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in MODELS_DIR.glob("*.pkl"))


def _load_model(slug: str):
    """Load a model by slug, caching the deserialized object."""
    if slug in _CACHE:
        return _CACHE[slug]
    path = MODELS_DIR / f"{slug}.pkl"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{slug}' not found. Available: {_available_slugs()}",
        )
    _CACHE[slug] = joblib.load(path)
    return _CACHE[slug]


def _unwrap(obj):
    """Return (preprocess_or_None, estimator) from a Pipeline / pair / estimator."""
    # sklearn Pipeline with named steps "preprocess"/"model".
    if hasattr(obj, "named_steps"):
        steps = obj.named_steps
        pre = steps.get("preprocess")
        model = steps.get("model", obj[-1])
        return pre, model
    # (scaler, model) tuple/list.
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return obj[0], obj[1]
    # Bare estimator.
    return None, obj


def _proba(obj, X: pd.DataFrame) -> float:
    """Positive-class probability, tolerant of Pipelines vs (scaler, model)."""
    if hasattr(obj, "predict_proba") and hasattr(obj, "named_steps"):
        return float(obj.predict_proba(X)[0, 1])
    pre, model = _unwrap(obj)
    Xt = pre.transform(X) if pre is not None else X
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xt)[0, 1])
    # decision_function fallback squashed to (0, 1).
    score = float(np.ravel(model.decision_function(Xt))[0])
    return float(1.0 / (1.0 + np.exp(-score)))


def _source_feature(transformed_name: str) -> str:
    """Map a ColumnTransformer output name back to its source feature.

    "num__age" -> "age"; "cat__cp_2" -> "cp".
    """
    name = transformed_name.split("__", 1)[-1]
    for col in FEATURE_COLUMNS:
        if name == col or name.startswith(f"{col}_"):
            return col
    return name


def _tree_shap(obj, X: pd.DataFrame) -> dict[str, float] | None:
    """Per-source-feature SHAP for tree models; None if unsupported."""
    try:
        import shap
    except ImportError:
        return None

    pre, model = _unwrap(obj)
    try:
        Xt = pre.transform(X) if pre is not None else X.values
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        names = (
            list(pre.get_feature_names_out())
            if pre is not None
            else list(FEATURE_COLUMNS)
        )
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(Xt)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., -1]
        row = np.ravel(values[0]) if values.ndim == 2 else np.ravel(values)
    except Exception:  # noqa: BLE001 - non-tree models simply get no SHAP
        return None

    # Aggregate one-hot contributions back onto the 13 source features.
    agg: dict[str, float] = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    for name, val in zip(names, row, strict=False):
        agg[_source_feature(name)] += float(val)
    return agg


class PredictRequest(BaseModel):
    model: str = Field(..., description="model slug, e.g. 'random_forest'")
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([[getattr(self, c) for c in FEATURE_COLUMNS]],
                            columns=FEATURE_COLUMNS)


@app.get("/health")
def health() -> dict:
    slugs = _available_slugs()
    return {
        "status": "ok",
        "models": slugs,
        "n_models": len(slugs),
        "features": FEATURE_COLUMNS,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    obj = _load_model(req.model)
    X = req.to_frame()
    probability = _proba(obj, X)
    shap_values = _tree_shap(obj, X)
    return {
        "model": req.model,
        "probability": probability,
        "prediction": int(probability >= 0.5),
        "shap": shap_values,           # dict of 13 values, or null for non-tree
        "shap_supported": shap_values is not None,
    }


# ── Static frontend ──────────────────────────────────────────────────────────
# Serve the design and its runtime from this same service, so the deployed app
# is a single origin (frontend + API together — no CORS, no mixed content).
# Declared AFTER the API routes so /health and /predict always take precedence.


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.get("/support.js")
def support_js() -> FileResponse:
    return FileResponse(BASE_DIR / "support.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
