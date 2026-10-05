"""
Phase 8: Model Development & Phase 9: Evaluation & Business Validation
---------------------------------------------------------------------------
Every modeling choice below has a WHY comment.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data


def recall_at_top_quintile(y_true, y_score, quintile=0.20):
    """
    BUSINESS-RELEVANT METRIC, not just a technical one.
    WHY THIS METRIC: a primary-care clinic can only realistically bring
    in a limited share of screened patients for confirmatory testing
    (an oral glucose tolerance test, a follow-up consultation) each
    cycle -- not everyone flagged. Asking "if we can only follow up with
    the highest-risk 20% of patients, what fraction of the people who
    actually have diabetes did we correctly flag in that group?" is
    exactly the capacity-constrained question a clinic operations team
    has to answer, the same reasoning used for top-decile recall in the
    churn and attrition projects, adapted to a slightly larger
    follow-up capacity (20%) appropriate for a screening/outreach
    program rather than a retention offer.
    """
    y_true = np.asarray(y_true)
    n_top = max(1, int(len(y_score) * quintile))
    top_idx = np.argsort(y_score)[-n_top:]
    caught = y_true[top_idx].sum()
    total_positive = y_true.sum()
    return caught / total_positive if total_positive > 0 else 0.0


def train_and_evaluate():
    raw = load_raw_data()
    df_train, df_val, df_test = split_data(raw)

    # WHY CLEAN EACH SPLIT SEPARATELY, REUSING TRAIN-ONLY MEDIANS: see
    # split.py's module docstring -- this is the leakage-safe order.
    train_clean, train_medians = clean_data(df_train)
    val_clean, _ = clean_data(df_val, train_medians=train_medians)
    test_clean, _ = clean_data(df_test, train_medians=train_medians)

    train_eng = engineer_features(train_clean)
    val_eng = engineer_features(val_clean)
    test_eng = engineer_features(test_clean)

    feature_cols = get_feature_columns(train_eng)
    # WHY REINDEX val/test AGAINST TRAIN'S COLUMNS: a small split can
    # occasionally miss a rare Glucose/BMI category entirely, which
    # would produce a different set of one-hot columns -- reindexing
    # against train's column list (filling any gap with 0) guarantees
    # all three splits have identical, aligned columns for the model.
    X_train = train_eng.reindex(columns=feature_cols, fill_value=0)
    X_val = val_eng.reindex(columns=feature_cols, fill_value=0)
    X_test = test_eng.reindex(columns=feature_cols, fill_value=0)
    y_train, y_val, y_test = train_eng["Outcome"], val_eng["Outcome"], test_eng["Outcome"]

    experiment_log = []

    # -----------------------------------------------------------------
    # Baseline: Logistic Regression
    # WHY: simple, fast, fully explainable -- a clinician can read each
    # feature's coefficient directly. Established floor before a more
    # complex model, per this series' Phase 8 guidance.
    # WHY class_weight="balanced": Outcome is imbalanced (34.9%
    # positive).
    # WHY SCALE FEATURES HERE (not for the tree models below): logistic
    # regression's coefficients are sensitive to feature scale (Glucose
    # ranges ~0-200, DiabetesPedigreeFunction ~0-2.5).
    # -----------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    logreg_val_scores = logreg.predict_proba(X_val_scaled)[:, 1]
    experiment_log.append({
        "model": "logistic_regression (baseline)",
        "val_auc": round(roc_auc_score(y_val, logreg_val_scores), 4),
        "val_recall_at_top20pct": round(recall_at_top_quintile(y_val, logreg_val_scores), 4),
    })

    # -----------------------------------------------------------------
    # Candidate: Random Forest
    # WHY TRIED: captures interactions (e.g. Glucose x BMI combined risk)
    # a linear model can't represent directly.
    # -----------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val_scores = rf.predict_proba(X_val)[:, 1]
    experiment_log.append({
        "model": "random_forest",
        "val_auc": round(roc_auc_score(y_val, rf_val_scores), 4),
        "val_recall_at_top20pct": round(recall_at_top_quintile(y_val, rf_val_scores), 4),
    })

    # -----------------------------------------------------------------
    # Final candidate: XGBoost
    # SELECTION CRITERION: chosen by val_recall_at_top20pct (the
    # business-relevant, capacity-constrained metric) -- selected
    # PROGRAMMATICALLY against Random Forest, not assumed by default
    # (consistent with this series since Day 10).
    # -----------------------------------------------------------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_val_scores = xgb.predict_proba(X_val)[:, 1]
    xgb_val_auc = roc_auc_score(y_val, xgb_val_scores)
    xgb_val_recall = recall_at_top_quintile(y_val, xgb_val_scores)
    experiment_log.append({
        "model": "xgboost",
        "val_auc": round(xgb_val_auc, 4),
        "val_recall_at_top20pct": round(xgb_val_recall, 4),
    })

    print("=== Experiment Log (validation set) ===")
    log_df = pd.DataFrame(experiment_log)
    print(log_df.to_string(index=False))
    log_df.to_csv("outputs/experiment_log.csv", index=False)

    # WHY WE RE-COMPUTE rf's raw recall HERE rather than pulling it from
    # experiment_log: an earlier version of this script compared
    # experiment_log's ROUNDED value (round(...,4)) against xgb's
    # unrounded value directly -- on this exact dataset, Random Forest
    # and XGBoost caught the identical 19 of 41 positives in the top 20%
    # (0.46341463...), a genuine tie. Comparing a rounded 0.4634 against
    # the unrounded 0.46341463 made the tie silently resolve in XGBoost's
    # favor due to a floating-point rounding artifact, not a real
    # performance difference -- caught by checking the raw, unrounded
    # values match when the rounded ones did. Selection now always
    # compares raw floats; rounding is applied only for display/logging.
    rf_val_recall_raw = recall_at_top_quintile(y_val, rf_val_scores)
    if rf_val_recall_raw >= xgb_val_recall:
        final_model, final_name = rf, "random_forest"
        final_val_recall = rf_val_recall_raw
    else:
        final_model, final_name = xgb, "xgboost"
        final_val_recall = xgb_val_recall
    print(f"\n=== Selected final model: {final_name} (higher val_recall_at_top20pct) ===")

    # -----------------------------------------------------------------
    # Phase 9: Final evaluation on the held-out TEST set
    # -----------------------------------------------------------------
    test_scores = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_scores)
    test_recall_top20 = recall_at_top_quintile(y_test, test_scores)

    threshold = np.percentile(test_scores, 80)
    test_preds = (test_scores >= threshold).astype(int)
    test_precision = precision_score(y_test, test_preds)
    test_recall_at_threshold = recall_score(y_test, test_preds)

    # -----------------------------------------------------------------
    # Business validation
    # WHY WE STOP AT A CATCH-RATE FIGURE, NOT A DOLLAR ESTIMATE (unlike
    # some other projects in this series): this is a health-screening
    # tool, and translating "more diabetes cases caught early" into a
    # dollar figure would require real healthcare-cost and
    # intervention-effectiveness data this public research dataset
    # doesn't have -- and, more importantly, a fabricated dollar figure
    # for a health outcome risks being genuinely misleading in a way a
    # retail or HR dollar estimate wouldn't be. The catch-rate itself is
    # the honest, real, directly measured business metric here.
    # -----------------------------------------------------------------
    n_test_positive = int(y_test.sum())
    caught_in_top20 = int(round(test_recall_top20 * n_test_positive))

    business_summary = {
        "final_model": final_name,
        "test_auc": round(test_auc, 4),
        "test_recall_at_top20pct": round(test_recall_top20, 4),
        "test_precision_at_top20pct_threshold": round(test_precision, 4),
        "test_recall_at_top20pct_threshold": round(test_recall_at_threshold, 4),
        "n_test_patients": len(y_test),
        "n_test_diabetes_positive": n_test_positive,
        "estimated_cases_caught_in_top20pct_screening": caught_in_top20,
        "note": (
            "This is a screening/decision-support tool, NOT a diagnostic "
            "replacement -- a flagged patient needs confirmatory clinical "
            "testing, not an automated diagnosis. No dollar-value business "
            "case is calculated; the catch-rate at a realistic follow-up "
            "capacity (top 20% highest-risk) is the honest, directly-measured "
            "business metric for a screening tool like this."
        ),
    }

    print("\n=== Business Validation Summary (Phase 9) ===")
    for k, v in business_summary.items():
        print(f"{k}: {v}")
    with open("outputs/business_validation.json", "w") as f:
        json.dump(business_summary, f, indent=2)

    if hasattr(final_model, "feature_importances_"):
        importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\n=== Top 10 Feature Importances (final model) ===")
        print(importances.head(10).round(4).to_string())
        importances.to_csv("outputs/feature_importance.csv", header=["importance"])

    joblib.dump(final_model, "outputs/diabetes_risk_model.joblib")
    joblib.dump(feature_cols, "outputs/feature_columns.joblib")
    joblib.dump(train_medians, "outputs/train_medians.joblib")

    with open("outputs/model_card.json", "w") as f:
        json.dump({
            "model_type": "RandomForestClassifier" if final_name == "random_forest" else "XGBClassifier",
            "data_source": "Pima Indians Diabetes Database (NIDDK)",
            "n_features": len(feature_cols),
            "features": feature_cols,
            "training_rows": len(X_train),
            "validation_recall_at_top20pct": round(final_val_recall, 4),
            "test_auc": round(test_auc, 4),
            "test_recall_at_top20pct": round(test_recall_top20, 4),
            "intended_use": "SCREENING decision support only -- flags patients for confirmatory clinical testing and preventive follow-up. NOT a diagnostic tool and must not be presented to a patient as a diagnosis.",
            "known_limitations": (
                "Small dataset (768 rows), a single population (women aged 21+ "
                "near Phoenix, Arizona, of Pima Indian heritage) -- does not "
                "generalize to other demographics without revalidation. Nearly "
                "half of Insulin values and ~30% of SkinThickness values were "
                "imputed (median, from training-set statistics only), not "
                "genuinely measured -- imputation quality directly affects "
                "model reliability for patients with missing values on those "
                "two fields especially."
            ),
        }, f, indent=2)

    print("\nSaved model -> outputs/diabetes_risk_model.joblib")
    print("Saved model card -> outputs/model_card.json")


if __name__ == "__main__":
    train_and_evaluate()
