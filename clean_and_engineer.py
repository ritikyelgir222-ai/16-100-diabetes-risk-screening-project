"""
Phases 5 (data quality fixes) & 7 (feature engineering)
------------------------------------------------------------
Every transformation and engineered feature below has a one-line WHY
comment next to it. FINAL FEATURE LIST is built at the bottom as
FEATURE_COLUMNS.
"""

import numpy as np
import pandas as pd

ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def clean_data(raw_df: pd.DataFrame, train_medians: dict = None):
    """
    WHY train_medians IS A PARAMETER (not just computed fresh every
    call): the correct median to impute with must come from the
    TRAINING set only -- computing it fresh on new data (including at
    serving time, where there's just 1 row) would either leak
    information across the train/test split or be impossible on a
    single row. When train_medians is None (the normal case when
    processing the full training set for the first time), it's computed
    from the input and returned so the caller can persist it and reuse
    it consistently for validation, test, and live serving data -- the
    same "single source of truth, computed once, reused everywhere"
    principle used for the LotFrontage neighborhood medians in the
    house-price project.
    """
    df = raw_df.copy()

    # -----------------------------------------------------------------
    # Fix 1: 0 -> NaN in the 5 columns where 0 is not physiologically
    # possible (verified in eda.py -- zeros range from 0.7% of rows for
    # Glucose to 48.7% for Insulin, clustering in ways inconsistent with
    # a living person's actual measurements).
    # -----------------------------------------------------------------
    for col in ZERO_AS_MISSING_COLUMNS:
        df[col] = df[col].replace(0, np.nan)

    # -----------------------------------------------------------------
    # Fix 2: missingness indicator flags, BEFORE imputing
    # WHY: EDA found missingness rates for Insulin and SkinThickness
    # differ slightly by Outcome (Insulin: 51.5% missing among diabetic
    # cases vs. 47.2% among non-diabetic; SkinThickness: 32.8% vs.
    # 27.8%) -- a modest but real pattern. Rather than just filling the
    # gap and losing that signal entirely, an explicit "was this
    # measurement missing" flag lets the model use it if it's
    # informative, without forcing the assumption either way.
    # -----------------------------------------------------------------
    for col in ZERO_AS_MISSING_COLUMNS:
        df[f"{col}_was_missing"] = df[col].isnull().astype(int)

    # -----------------------------------------------------------------
    # Fix 3: median imputation, using TRAINING-SET medians only
    # WHY MEDIAN (not mean): all 5 columns have outliers/skew (e.g.
    # Insulin's valid range is heavily right-skewed) -- median is more
    # robust to that than mean, which a few extreme values would pull
    # upward.
    # -----------------------------------------------------------------
    if train_medians is None:
        train_medians = {col: df[col].median() for col in ZERO_AS_MISSING_COLUMNS}

    for col in ZERO_AS_MISSING_COLUMNS:
        df[col] = df[col].fillna(train_medians[col])

    return df, train_medians


def engineer_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()

    # -----------------------------------------------------------------
    # Engineered feature: Glucose_category
    # BUSINESS LOGIC: these are the actual clinical OGTT (oral glucose
    # tolerance test) thresholds used to classify normal / prediabetic /
    # diabetic-range glucose -- not arbitrary bins fit to this dataset,
    # the same "use the real industry-standard thresholds" principle as
    # the BMI categories in the insurance-cost project.
    # -----------------------------------------------------------------
    df["Glucose_category"] = pd.cut(
        df["Glucose"], bins=[0, 140, 199, 1000],
        labels=["Normal", "Prediabetic_range", "Diabetic_range"],
    )

    # -----------------------------------------------------------------
    # Engineered feature: BMI_category (WHO clinical bins, same as the
    # insurance-cost project)
    # -----------------------------------------------------------------
    df["BMI_category"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    df = pd.get_dummies(df, columns=["Glucose_category", "BMI_category"], prefix=["GlucoseCat", "BMICat"])
    dummy_cols = [c for c in df.columns if c.startswith("GlucoseCat_") or c.startswith("BMICat_")]
    df[dummy_cols] = df[dummy_cols].astype(int)

    return df


NUMERIC_MODEL_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age",
    "Glucose_was_missing", "BloodPressure_was_missing", "SkinThickness_was_missing",
    "Insulin_was_missing", "BMI_was_missing",
]


def get_feature_columns(engineered_df: pd.DataFrame) -> list:
    dummy_cols = [c for c in engineered_df.columns if c.startswith("GlucoseCat_") or c.startswith("BMICat_")]
    return NUMERIC_MODEL_FEATURES + dummy_cols


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned, train_medians = clean_data(raw)
    print("Training medians used for imputation:", train_medians)

    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)
    print(f"\nFinal feature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  - {c}")

    engineered.to_csv("outputs/engineered_data.csv", index=False)
    print("\nSaved -> outputs/engineered_data.csv")
