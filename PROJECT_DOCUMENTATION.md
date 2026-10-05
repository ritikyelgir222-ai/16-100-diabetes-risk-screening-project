# Project Documentation: Diabetes Risk Screening
### Technical Documentation & Handover — Phase 12 of the SDLC

This document is the technical documentation and user manual a real handover
package would include. Every number below came from actually running the
code in this repository — none are illustrative. **This project is a
screening decision-support tool, not a diagnostic system** — this
distinction is enforced throughout, including in the live API response.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Flag higher-risk patients for confirmatory diabetes testing and preventive follow-up |
| **Client context** | Primary-care / preventive-health clinic |
| **Data source** | Pima Indians Diabetes Database (NIDDK), 768 real records |
| **Final model** | Random Forest Classifier, 20 features |
| **Test AUC** | 0.8421 |
| **Business framing** | Catch cases within a clinic's realistic confirmatory-testing capacity (top 20% highest-risk) |

---

## 2. Architecture

```
data/diabetes.csv
        │
        ▼
data_loader.py ──────► loads raw CSV, confirms zero-as-missing pattern
        │
        ▼
   split.py ───────────► splits RAW data (Outcome-stratified) BEFORE
        │                any cleaning -- leakage-safe ordering
        ▼
clean_and_engineer.py ─► 0 -> NaN, missingness flags, TRAIN-ONLY median
        │                imputation (reused for val/test/serving),
        │                clinical-threshold engineered features
        ▼
train_model.py ───────► baseline, Random Forest, XGBoost; PROGRAMMATIC
        │                selection (with a rounding-bug fix -- see
        │                Section 5); business validation
        ▼
outputs/diabetes_risk_model.joblib, feature_columns.joblib, train_medians.joblib
        │
        ├──────────────► app.py ─── FastAPI /screen endpoint, disclaimer
        │                           built into every response
        │
        └──────────────► monitor.py ─ PSI drift check + retrain trigger
```

**Key structural difference from every other project in this series**:
the split happens on raw data, BEFORE cleaning, because this project's
cleaning step (median imputation) is itself a statistic computed from
data — imputing with a median calculated across train+val+test combined
would leak val/test information into training.

---

## 3. Data Dictionary & Cleaning Decisions

| Column | Description | Decision |
|---|---|---|
| `Pregnancies` | Number of pregnancies | Kept as-is |
| `Glucose` | Plasma glucose (OGTT), mg/dL | 0 → NaN → train-median imputed; `Glucose_was_missing` flag added |
| `BloodPressure` | Diastolic BP, mm Hg | Same treatment |
| `SkinThickness` | Triceps skin fold, mm | Same treatment (29.6% missing) |
| `Insulin` | 2-hour serum insulin | Same treatment (48.7% missing) |
| `BMI` | Body mass index | Same treatment |
| `DiabetesPedigreeFunction` | Genetic diabetes-risk score | Kept as-is |
| `Age` | Age in years | Kept as-is |
| `Outcome` | 0/1 diabetes diagnosis | **Target** |

**Engineered features**: `Glucose_category` (Normal / Prediabetic_range
/ Diabetic_range, real clinical OGTT thresholds), `BMI_category` (WHO
clinical bins), and 5 `*_was_missing` flags.

---

## 4. Key EDA Findings

| Finding | Detail |
|---|---|
| **The zero-as-missing pattern** | Glucose 0.7% zeros, BloodPressure 4.6%, SkinThickness 29.6%, Insulin 48.7%, BMI 1.4% — physiologically impossible values used as a "not recorded" placeholder |
| Missingness vs. outcome | Insulin missing in 51.5% of diabetic cases vs. 47.2% of non-diabetic; SkinThickness 32.8% vs. 27.8% — a modest pattern, motivating the missingness flags rather than plain imputation alone |
| Glucose by outcome (zeros excluded) | Non-diabetic: 111.1 mg/dL avg · Diabetic: 145.2 mg/dL — the clearest single split, matching clinical expectation |
| BMI by outcome | Non-diabetic: 30.9 · Diabetic: 35.4 |
| Age by outcome | Non-diabetic: 31.2 · Diabetic: 37.1 |
| Pregnancies by outcome | Non-diabetic: 3.30 · Diabetic: 4.87 |
| Correlation with Outcome | Glucose 0.495 (strongest) → BMI 0.314 → Insulin 0.303 → SkinThickness 0.259 → Age 0.238 → Pregnancies 0.222 → DiabetesPedigreeFunction 0.174 → BloodPressure 0.171 |

Every correlation direction matches established clinical literature —
useful confirmation the data (despite its missingness quirks) is
behaving sensibly before any modeling.

---

## 5. Modeling Results

### The bug this project caught

Random Forest and XGBoost tied **exactly** on validation recall-at-top-
20% (both: 19 of 41 positives caught = 0.46341463...). An earlier
version of the selection code compared `experiment_log`'s **rounded**
value (`round(0.46341463, 4)` = `0.4634`) against XGBoost's
**unrounded** raw float (`0.46341463...`) directly. Since `0.4634 <
0.46341463`, the comparison silently resolved in XGBoost's favor — not
because XGBoost actually won, but because one side of the comparison had
been rounded and the other hadn't. Caught by verifying both models' raw
scores independently outside the script and finding them identical.
**Fixed** by always comparing raw, unrounded floats for model selection,
rounding only for display/logging (see the WHY comment in
`train_model.py`). This is a subtle, easy-to-make bug worth watching for
in any pipeline that logs rounded metrics and later reuses them for a
decision.

### Experiment log (validation set)

| Model | Val AUC | Val Recall @ Top 20% |
|---|---|---|
| Logistic Regression (baseline) | 0.8267 | 46.34% |
| **Random Forest** | **0.8364** | 46.34% |
| XGBoost | 0.8241 | 46.34% |

All three models land within a few points of AUC, and two of the three
tie exactly on the business metric — Random Forest was selected as the
final model both because it wins the tie-break (`>=` in the selection
code) and because it has the highest secondary AUC of the three.

### Held-out test set

| Metric | Value |
|---|---|
| Test AUC | 0.8421 |
| Test recall @ top 20% | 42.50% |
| N test patients | 116 |
| N test diabetes-positive | 40 |

### What drives the model (feature importance)

Top 5 by importance:

1. `Glucose` (highest — matches its 0.495 EDA correlation, the strongest single signal)
2. `BMI`
3. `Age`
4. `DiabetesPedigreeFunction`
5. `Insulin`

Consistent with both the EDA correlation ranking and established
clinical risk factors for type 2 diabetes — a good sign the model
learned genuine signal rather than an artifact of the encoding.

### Business validation

**No dollar-value figure is calculated for this project — a deliberate
choice, not an oversight.** Translating "more diabetes cases caught
early" into a cost figure would need real healthcare-cost and
intervention-effectiveness data this public research dataset doesn't
have, and a fabricated dollar figure for a health outcome carries a
different, more serious risk of being misleading than a retail or HR
estimate would. The honest, directly-measured business metric is the
catch rate itself: at a clinic's realistic follow-up capacity (the
highest-risk 20% of screened patients), the model catches **42.5%** of
actual diabetes-positive cases in this test cohort.

---

## 6. API Reference (Phase 10)

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Test form, disclaimer shown prominently |
| `/docs` | GET | Swagger UI |
| `/health` | GET | Health check |
| `/screen` | POST | Screen a single patient |

**Response (verified, two contrasting real profiles) — every response
includes the disclaimer field:**
```json
{
  "risk_score": 0.8718,
  "risk_tier": "High",
  "top_factors": ["Glucose", "Age", "BMI"],
  "disclaimer": "This is a screening decision-support tool, not a medical diagnosis. A 'High' flag means confirmatory clinical testing is recommended, not that the patient has diabetes."
}
```
A low-risk profile (young, normal BMI, low glucose) scored **0.02** on
the same model.

---

## 7. Monitoring & Maintenance Plan (Phase 13)

- **Feature drift (PSI)** on `Glucose`, `BMI`, `Age`, `Insulin`: Glucose
  and Age stable (PSI 0.054, 0.051); BMI and Insulin show "moderate"
  drift (0.157, 0.120) even on a same-snapshot stratified split —
  plausibly a small-sample artifact (115 test rows, quantile-binning
  noise from repeated imputed median values) rather than a genuine
  signal, worth re-examining with more data.
- **Performance decay check**: no retrain triggered on this run.

**The more important monitoring point for this specific project**: this
model is trained on ONE population (women 21+, Pima heritage, near
Phoenix). Deploying it unchanged on a materially different patient
population — different demographics, different regional health
patterns — is a more serious risk than ordinary feature drift, and
cannot be detected by any PSI check run against this dataset alone.
Stated directly in `monitor.py`'s output, not left as an assumption.

---

## 8. Known Limitations (stated for the handover record)

1. **A single, specific population** — see Section 7.
2. **Substantial imputation** — 48.7% of Insulin and 29.6% of
   SkinThickness values are imputed medians, not genuine measurements.
3. **The rounding bug (Section 5)** is worth internalizing as a general
   lesson: never compare a rounded metric against an unrounded one for
   an automated decision.
4. **No dollar-value business case** — a deliberate choice for a health
   outcome; see Section 5.
5. **Screening tool only** — never a diagnostic replacement; enforced in
   the API response itself.
6. **Modest test-set catch rate** (42.5% at top-20% capacity) — an
   honest reflection of a genuinely hard screening problem with only 8
   clinical measurements, not dressed up.

---

## 9. File Map (for quick reference)

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw data, confirm zero-as-missing pattern |
| `eda.py` | 6 | Missingness characterization, clinical-cut EDA |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Leakage-safe imputation, missingness flags, feature engineering |
| `split.py` | 7 | Raw-data split (before cleaning), stratified |
| `train_model.py` | 8-9 | Model training, **the rounding-bug fix**, business validation |
| `app.py` | 10 | FastAPI screening service with disclaimer |
| `monitor.py` | 13 | Drift detection, retrain trigger |
| `README.md` | 12 | Setup and run instructions |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation and handover |
