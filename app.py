"""
Phase 10: MLOps & Deployment
--------------------------------
A minimal FastAPI service that scores a single patient's diabetes risk.
Reuses clean_and_engineer.py (with the SAVED training-set imputation
medians -- see outputs/train_medians.joblib) so training and serving
logic never drift apart.

*** THIS IS A SCREENING / DECISION-SUPPORT TOOL, NOT A DIAGNOSTIC ONE. ***
A high-risk flag means "recommend confirmatory clinical testing," not
"this patient has diabetes." This distinction is enforced in the
response itself (see the `disclaimer` field), not left to documentation
alone.

Run with:  uvicorn app:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from clean_and_engineer import clean_data, engineer_features

app = FastAPI(title="Diabetes Risk Screening API (decision support only)", version="1.0")

MODEL = joblib.load("outputs/diabetes_risk_model.joblib")
FEATURE_COLUMNS = joblib.load("outputs/feature_columns.joblib")
TRAIN_MEDIANS = joblib.load("outputs/train_medians.joblib")

DISCLAIMER = (
    "This is a screening decision-support tool, not a medical diagnosis. "
    "A 'High' flag means confirmatory clinical testing is recommended, "
    "not that the patient has diabetes."
)


class PatientRecord(BaseModel):
    Pregnancies: int
    Glucose: float  # 0 will be treated as "not recorded" and imputed, matching training-time handling
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


class ScreeningResponse(BaseModel):
    risk_score: float
    risk_tier: str
    top_factors: list[str]
    disclaimer: str


def tier_from_score(score: float) -> str:
    # WHY THESE CUTOFFS: calibrated to roughly the top-20% operating
    # point used in evaluation (High ~ top 20% highest risk), a business
    # judgment call matching the follow-up capacity reasoning in
    # train_model.py, not a clinically-validated diagnostic threshold.
    if score >= 0.5:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
    <html>
    <head><title>Diabetes Risk Screening</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 40px auto;">
        <h2>Diabetes Risk Screening — Test Form</h2>
        <p style="background:#fff3cd; padding:10px; border-radius:6px;">{DISCLAIMER}</p>
        <p>Enter 0 for any measurement not available — it will be imputed
           the same way training data was. Full API docs at <a href="/docs">/docs</a>.</p>
        <form id="screenForm" style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px 20px;">
            <label>Pregnancies<br><input name="Pregnancies" type="number" value="2" required></label>
            <label>Glucose (mg/dL)<br><input name="Glucose" type="number" value="150" required></label>
            <label>Blood pressure (mm Hg)<br><input name="BloodPressure" type="number" value="70" required></label>
            <label>Skin thickness (mm)<br><input name="SkinThickness" type="number" value="0"></label>
            <label>Insulin (mu U/mL)<br><input name="Insulin" type="number" value="0"></label>
            <label>BMI<br><input name="BMI" type="number" step="0.1" value="34" required></label>
            <label>Diabetes pedigree function<br><input name="DiabetesPedigreeFunction" type="number" step="0.01" value="0.5" required></label>
            <label>Age<br><input name="Age" type="number" value="45" required></label>
            <button type="submit" style="grid-column: 1 / -1; margin-top: 10px;">Screen</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("screenForm").addEventListener("submit", async function(e) {{
            e.preventDefault();
            const form = new FormData(e.target);
            const payload = {{}};
            for (const [k, v] of form.entries()) {{
                payload[k] = k === "Pregnancies" || k === "Age" ? parseInt(v, 10) : parseFloat(v);
            }}
            const res = await fetch("/screen", {{
                method: "POST",
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify(payload)
            }});
            if (!res.ok) {{
                document.getElementById("result").innerText = "Error " + res.status + ": " + await res.text();
                return;
            }}
            const data = await res.json();
            document.getElementById("result").innerText =
                "Risk score: " + data.risk_score + " | Tier: " + data.risk_tier +
                " | Top factors: " + data.top_factors.join(", ");
        }});
        </script>
    </body>
    </html>
    """


@app.post("/screen", response_model=ScreeningResponse)
def screen_patient(record: PatientRecord):
    raw_row = record.model_dump()
    raw_row["Outcome"] = 0  # placeholder; unused for scoring but clean_data/engineer_features expect the column
    raw_df = pd.DataFrame([raw_row])

    cleaned, _ = clean_data(raw_df, train_medians=TRAIN_MEDIANS)
    engineered = engineer_features(cleaned)
    X = engineered.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    proba = float(MODEL.predict_proba(X)[0, 1])
    tier = tier_from_score(proba)

    if hasattr(MODEL, "feature_importances_"):
        importances = pd.Series(MODEL.feature_importances_, index=FEATURE_COLUMNS)
        top_factors = importances.sort_values(ascending=False).head(3).index.tolist()
    else:
        top_factors = ["Glucose", "BMI", "Age"]

    return ScreeningResponse(
        risk_score=round(proba, 4),
        risk_tier=tier,
        top_factors=top_factors,
        disclaimer=DISCLAIMER,
    )
