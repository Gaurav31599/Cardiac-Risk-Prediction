# Pipeline Guide — what this project does and why

A walkthrough of every stage of the cardiac-risk pipeline: cleaning, feature
engineering, modelling, train/test discipline, model choice, and the
statistical claims the pipeline can and cannot support. Read this alongside the
[README](../README.md).

---

## 1. End-to-end flow

```
Raw data ──► Cleaning ──► Feature engineering ──► Stratified split
                                                        │
                                          ┌─────────────┴─────────────┐
                                       Train                        Test
                                          │                           │
                                   Train + 5-fold CV  ──────────►  Evaluation
                                  (LogReg · RF · GBM)            (ROC-AUC, P/R)
                                          │                           │
                                          └──────► Final model ──► SHAP explain
                                                                      │
                                                                   Outputs
                                                       (comparison table, SHAP plot, demo)

   Cleaned data ──► KMeans patient segments (exploratory, kept separate from the classifier)
```

A rendered version of this flow is shown in the chat alongside this guide. Each
stage below maps to a module under `src/`.

---

## 2. Data cleaning

Module: [`src/data/load_data.py`](../src/data/load_data.py)

| Step | What happens | Why |
|------|--------------|-----|
| Schema validation | `_validate_schema` asserts all 13 features + `num` are present and non-empty | Fail loudly on a malformed download rather than silently training on garbage |
| Missing-value markers | The raw file uses `?`; read as `NaN` via `na_values="?"` | Cleveland encodes missingness as a literal character |
| Imputation | `ca` and `thal` (the only columns with NaNs) get **median** imputation | Low-cardinality, few missing; median is robust and leak-free at the column level |
| Type coercion | All columns forced numeric; `num` cast to int | Downstream models need numeric arrays |
| Target derivation | `target = (num > 0)` for the binary cut | `num` is a 0–4 severity score; the conventional task is "disease present / absent" |
| Caching | First successful download is written to `data/processed.cleveland.csv` | Reproducible, offline-capable runs and tests |

**Cleaning honesty:** only ~6 rows carry missing values in Cleveland. We impute
rather than drop to preserve the already-small sample, and we impute with a
statistic (median) that does not depend on the target — so no target leakage.

---

## 3. Feature engineering

Module: [`src/features/preprocessing.py`](../src/features/preprocessing.py)

The 13 features are not homogeneous, so they are split into two blocks and
transformed differently inside a single `ColumnTransformer`:

- **Numeric** (`age, trestbps, chol, thalach, oldpeak`) → `StandardScaler`
  (zero mean, unit variance). Scaling matters for the logistic-regression
  baseline and for distance-based KMeans; trees are scale-invariant but are
  unharmed by it.
- **Categorical** (`sex, cp, fbs, restecg, exang, slope, ca, thal`) →
  `OneHotEncoder(handle_unknown="ignore", drop="if_binary")`. These are clinical
  *codes*, not magnitudes — `cp=4` is not "twice" `cp=2`, so one-hot avoids
  imposing a false ordinal scale. `handle_unknown="ignore"` keeps prediction
  robust if a rare category appears only in the test fold.

**Why a `ColumnTransformer` inside a `Pipeline`:** the exact same fitted
transform (scaler means, encoder categories) is reused at predict time. The
scaler is fit on **training data only** — it never sees the test fold — which is
the mechanism that prevents preprocessing leakage.

The binary `target` itself is the engineered label (see cleaning above).

---

## 4. Train/test splitting

Module: [`src/features/preprocessing.py`](../src/features/preprocessing.py) → `split_data`

- **Stratified** 80/20 split (`stratify=y`): the ~54/46 class ratio is preserved
  in both folds. Without stratification, a small test set can drift to an
  unrepresentative balance and make metrics noisy or optimistic.
- **Fixed `random_state=42`**: reproducible splits across runs.
- The test set is touched **once**, for final reporting — never for model
  selection (that's what cross-validation on the *training* set is for).

This two-level discipline — CV inside training for selection, a held-out test
set for the final number — is what lets the reported metrics stand as an
honest out-of-sample estimate rather than a fit-on-the-same-data illusion.

---

## 5. Model choice

Modules: [`src/models/`](../src/models)

| Model | Role | Rationale |
|-------|------|-----------|
| Logistic Regression | Baseline | Linear, fast, well-calibrated reference the ensembles must beat |
| Random Forest | Non-linear ensemble | Captures feature interactions, robust with little tuning |
| Gradient Boosting (XGBoost) | **Primary / final** | Best precision–recall trade-off on tabular data of this size |

All three carry `class_weight="balanced"` (or boosting equivalents) so the
minority class is not ignored. **No CNN/RNN:** the data is 303×13 tabular rows —
deep learning has no structural advantage here and would signal modelling for
impressiveness over fit. Choosing classical models *is* the engineering
judgement being demonstrated.

The gradient-boosting step prefers XGBoost and **falls back** to scikit-learn's
`HistGradientBoostingClassifier` if XGBoost is absent — both are tree ensembles,
so the SHAP analysis works on either.

---

## 6. Statistical / methodological claims this pipeline supports

What the pipeline **does** demonstrate, with the mechanism that backs each claim:

1. **Out-of-sample generalisation** — a held-out stratified test set, untouched
   during selection, gives an honest estimate of unseen-patient performance.
2. **Selection stability** — 5-fold stratified **cross-validation** reports a
   mean **and standard deviation** of ROC-AUC (`cross_validate_auc`), so model
   ranking reflects variance across folds, not a single lucky split. Example:
   LogReg `0.902 ± 0.015` vs RF `0.893 ± 0.032` — the overlapping bands say the
   two are statistically hard to separate on this sample.
3. **Imbalance-aware evaluation** — because classes are uneven, the headline
   metrics are **ROC-AUC and precision/recall**, not accuracy. Accuracy is
   reported but explicitly framed as secondary (`classification_metrics`).
4. **Threshold-independent ranking quality** — ROC-AUC measures how well the
   model *orders* patients by risk, independent of the 0.5 cut-off.
5. **Confusion-matrix error structure** — false-negative vs false-positive
   counts (`confusion`) expose the clinically relevant asymmetry (a missed case
   is worse than a false alarm).
6. **Explainability / attribution** — SHAP (`shap_analysis.py`) attributes each
   prediction to features; the top drivers (`ca`, `cp`, sex, age, `thal`) align
   with established cardiac risk factors — a sanity check that the model learned
   signal, not noise.
7. **Unsupervised structure (exploratory)** — KMeans surfaces natural patient
   segments, and `crosstab_against_target` *tests* whether those segments relate
   to the diagnosis rather than assuming it.

What the pipeline **does not** claim:

- It is **not** a hypothesis test of disease causation — no p-values on feature
  effects are presented as causal, and SHAP attributions are associational.
- It does **not** prove clinical validity. Single hospital, ~300 patients,
  1980s data, demographic skew → no external validity. See the README limitations.
- The KMeans clusters are **not** the classifier and are not claimed to equal the
  labels unless the cross-tab shows real overlap.
- **Not a diagnostic tool.** Full stop.

> Suggested rigorous extensions (not yet implemented): DeLong's test to compare
> two ROC curves statistically; bootstrap confidence intervals on test AUC;
> a calibration curve / Brier score for probability reliability. These would
> turn the descriptive comparison above into formal significance statements.

---

## 7. Reproducing every stage

```bash
pip install -r requirements.txt
python -m src.train        # runs stages 2–6 and writes reports/shap_summary.png
pytest                     # asserts the contracts described above hold
ruff check .               # lint
```
