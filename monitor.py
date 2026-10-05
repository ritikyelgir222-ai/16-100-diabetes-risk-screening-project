"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY PSI (Population Stability Index): the standard, interview-
recognizable metric for feature drift, with established severity
thresholds (< 0.1 stable, 0.1-0.25 moderate, > 0.25 significant).

WHY THIS SCRIPT SIMULATES A "NEW BATCH": this dataset is a single
snapshot (no timestamp) from one population, the same limitation as the
churn/attrition/insurance/salary projects in this series. The held-out
test set is reused as a stand-in, confirming the monitoring code works,
not that real-world drift is being caught.

WHAT MATTERS MOST FOR THIS PROJECT SPECIFICALLY: a diabetes screening
model trained on one population (women 21+, Pima heritage, near
Phoenix) should NOT be deployed unchanged on a materially different
population without revalidation -- this is a more serious form of
"drift" than ordinary feature drift, and no PSI check on this dataset
alone can catch it. Stated directly, not left implicit.
"""

import json

import numpy as np
import joblib

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data
from train_model import recall_at_top_quintile

RECALL_DROP_THRESHOLD = 0.05
DRIFT_FEATURES = ["Glucose", "BMI", "Age", "Insulin"]


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/diabetes_risk_model.joblib")
    feature_cols = joblib.load("outputs/feature_columns.joblib")
    train_medians = joblib.load("outputs/train_medians.joblib")

    raw = load_raw_data()
    df_train, df_val, df_test = split_data(raw)
    train_clean, _ = clean_data(df_train)
    test_clean, _ = clean_data(df_test, train_medians=train_medians)
    train_eng = engineer_features(train_clean)
    test_eng = engineer_features(test_clean)

    X_train = train_eng.reindex(columns=feature_cols, fill_value=0)
    X_test = test_eng.reindex(columns=feature_cols, fill_value=0)
    y_test = test_eng["Outcome"]

    print("=== Feature Drift (PSI): train distribution vs. held-out batch ===")
    print("PSI < 0.1: stable | 0.1-0.25: moderate | > 0.25: significant drift\n")
    for feat in DRIFT_FEATURES:
        psi = population_stability_index(X_train[feat], X_test[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")

    scores = model.predict_proba(X_test)[:, 1]
    recall = recall_at_top_quintile(y_test, scores)

    print(f"\n=== Performance on held-out batch ===")
    print(f"Recall at top 20%: {recall:.4f}")

    with open("outputs/business_validation.json") as f:
        baseline_recall = json.load(f)["test_recall_at_top20pct"]

    drop = baseline_recall - recall
    if drop > RECALL_DROP_THRESHOLD:
        print(f"\n⚠️  RETRAIN TRIGGERED: recall dropped by {drop:.3f} (threshold: {RECALL_DROP_THRESHOLD})")
    else:
        print(f"\n✅ No retrain needed (recall change: {drop:.3f}, threshold: {RECALL_DROP_THRESHOLD})")

    print("\nNOTE: this dataset represents ONE population (women 21+, Pima heritage,")
    print("near Phoenix, AZ). Deploying this model on a materially different patient")
    print("population without revalidation is a more serious risk than ordinary")
    print("feature drift, and cannot be caught by a PSI check against this dataset alone.")


if __name__ == "__main__":
    run_monitoring_check()
