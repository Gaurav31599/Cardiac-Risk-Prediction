"""Lightweight Streamlit risk-input demo.

Mirrors the engine/app split used elsewhere in the portfolio: all modeling
logic lives in ``src/``; this file is a thin UI shell that loads the data,
trains the gradient-boosting pipeline once, and scores a single user-entered
patient. It is a demonstration only — NOT a diagnostic tool.

    streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file with app/ on sys.path, not the project root, so make
# the repo root importable before reaching into src/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from src.data.load_data import FEATURE_COLUMNS, load_heart_disease
from src.features.preprocessing import split_data
from src.models import build_gradient_boosting


@st.cache_resource
def _train_model():
    df = load_heart_disease()
    X_train, _, y_train, _ = split_data(df)
    model = build_gradient_boosting()
    model.fit(X_train, y_train)
    return model


def main() -> None:
    st.title("Cardiac Risk Classification — Demo")
    st.warning(
        "This is a portfolio demonstration of a healthcare ML pipeline. "
        "It is **not** a diagnostic tool and must not be used for real "
        "clinical decisions."
    )

    model = _train_model()

    st.subheader("Enter patient features")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 90, 54)
        sex = st.selectbox("Sex", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        cp = st.selectbox("Chest pain type (1-4)", [1, 2, 3, 4], index=3)
        trestbps = st.slider("Resting BP (mm Hg)", 80, 200, 130)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 246)
        fbs = st.selectbox("Fasting blood sugar > 120", [0, 1])
        restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    with col2:
        thalach = st.slider("Max heart rate", 60, 220, 150)
        exang = st.selectbox("Exercise-induced angina", [0, 1])
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
        slope = st.selectbox("ST slope (1-3)", [1, 2, 3])
        ca = st.selectbox("Major vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (3=normal, 6=fixed, 7=reversible)", [3, 6, 7])

    row = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
          oldpeak, slope, ca, thal]],
        columns=FEATURE_COLUMNS,
    )

    if st.button("Estimate risk"):
        proba = float(model.predict_proba(row)[0, 1])
        st.metric("Model-estimated disease probability", f"{proba:.0%}")
        st.caption(
            "Probability from the gradient-boosting model on the UCI Cleveland "
            "dataset. For demonstration of model behavior only."
        )


if __name__ == "__main__":
    main()
