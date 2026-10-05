# Diabetes Risk Screening — Full Project Code (Real Dataset)

Companion code to Day 16 of the 100-day LinkedIn series. Uses the real
Pima Indians Diabetes Database, run end to end. **This is a screening
decision-support tool, not a diagnostic one** — see the disclaimer baked
directly into the API response, not just the docs.

## Data source

**Pima Indians Diabetes Database** (NIDDK) — 768 real clinical records,
8 measurements + a binary diagnosis outcome. Already included at
`data/diabetes.csv` (no header row).

## The bug this project caught (worth reading before trusting the model choice)

Random Forest and XGBoost tied EXACTLY on the primary business metric —
both caught 19 of 41 positive cases in the top-20%-risk group on
validation (0.46341463...). The first version of the model-selection
code compared experiment_log's **rounded** value (0.4634) against
XGBoost's **unrounded** value (0.46341463...) — a genuine tie silently
resolved in XGBoost's favor purely from a floating-point rounding
artifact, not a real performance difference. Fixed by always comparing
raw, unrounded floats for selection and rounding only for display — see
the WHY comment in `train_model.py`. After the fix, Random Forest is
correctly selected (it also has a secondary edge: higher validation AUC,
0.8364 vs. 0.8241).

## The data quality finding

Five columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
`BMI`) use **0 as a placeholder for "not recorded"** — a living person
cannot have 0 blood pressure or 0 BMI. `Insulin` is missing this way in
**48.7%** of rows, `SkinThickness` in **29.6%**. Handled with
train-set-only median imputation plus explicit missingness flags (in
case the missingness itself carries signal) — see `clean_and_engineer.py`.

## Why every script is commented the way it is

- **The train/val/test split happens on RAW data, before imputation** —
  unlike every other project in this series, where cleaning happens
  once before splitting. Imputing with a median computed across
  train+val+test combined would leak information across the split; the
  median is computed from training data only and explicitly reused for
  val/test/serving (see `split.py` and `clean_and_engineer.py`).
- **`Glucose_category` uses real clinical OGTT thresholds**, not
  arbitrary bins — the same "use the real industry standard" principle
  as the insurance-cost project's BMI categories.
- **The business metric is recall at the top 20% highest-risk patients**
  (a clinic's realistic confirmatory-testing capacity), not raw
  accuracy — and **no dollar-value business case is calculated**, since
  fabricating a health-outcome dollar figure from a public research
  dataset would be more misleading than illustrative (unlike a retail or
  HR dollar estimate).

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py         # Phase 5 — loads the raw CSV, confirms the zero-as-missing pattern
python eda.py                  # Phase 6 — glucose/BMI/age cuts, missingness-vs-outcome check + outputs/eda_summary.png
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — leakage-safe imputation, missingness flags, engineered features
python train_model.py          # Phase 8-9 — baseline + final model (data-driven selection), business validation
python monitor.py              # Phase 13 — drift check + retrain-trigger simulation
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/screen \
  -H "Content-Type: application/json" \
  -d '{
        "Pregnancies": 6, "Glucose": 195, "BloodPressure": 80,
        "SkinThickness": 35, "Insulin": 200, "BMI": 38.5,
        "DiabetesPedigreeFunction": 1.2, "Age": 52
      }'
```

Expected: a risk score around **0.87** and `"risk_tier": "High"` — this
profile (high glucose, elevated BMI, older age) matches the pattern EDA
identifies as highest-risk. The response always includes the
screening-not-diagnosis disclaimer.

## File map

| File | SDLC Phase | What it does |
|---|---|---|
| `data_loader.py` | 5 | Loads the real dataset, confirms the zero-as-missing pattern |
| `eda.py` | 6 | Missingness characterization, glucose/BMI/age cuts vs. outcome |
| `clean_and_engineer.py` | 5 (fixes) + 7 | **Leakage-safe median imputation**, missingness flags, clinical-threshold engineered features |
| `split.py` | 7 | Raw-data split (before imputation), stratified by Outcome |
| `train_model.py` | 8-9 | Baseline → Random Forest → XGBoost, **the rounding-bug fix**, business validation |
| `app.py` | 10 | FastAPI screening service — disclaimer built into every response |
| `monitor.py` | 13 | PSI drift check + retrain trigger |

## Outputs produced (in `outputs/`)

- `engineered_data.csv`, `eda_summary.png`, `experiment_log.csv`
- `business_validation.json` — AUC, top-20%-recall, no dollar figure (by design)
- `feature_importance.csv`
- `diabetes_risk_model.joblib`, `feature_columns.joblib`, `train_medians.joblib`, `model_card.json`

## Known limitations (stated honestly, not hidden)

- **A single population** (women 21+, Pima heritage, near Phoenix, AZ,
  collected some years ago) — does not generalize to other demographics
  without revalidation. Stated directly in `monitor.py`'s output, not
  left implicit.
- **Nearly half of Insulin values, ~30% of SkinThickness values are
  imputed**, not genuinely measured — model reliability for patients
  missing those fields specifically is weaker than for fully-measured
  patients.
- **BMI and Insulin show "moderate" PSI drift even on a same-snapshot
  split** — plausibly a small-sample-size artifact (115 test rows,
  quantile-binning noise from repeated imputed median values), not
  necessarily meaningful; worth re-checking with more data before
  treating as a real signal.
- **This is a screening tool only.** Every API response includes an
  explicit disclaimer; a "High" flag means "recommend confirmatory
  testing," never "diagnosed."
