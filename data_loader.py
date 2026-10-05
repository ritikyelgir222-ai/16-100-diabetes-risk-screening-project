"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the Pima Indians Diabetes Database, originally
collected by the National Institute of Diabetes and Digestive and Kidney
Diseases and one of the most widely used real medical datasets in
applied ML education:
    https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

768 records of women aged 21+ from a population near Phoenix, Arizona,
with 8 clinical measurements and a binary diabetes diagnosis outcome
(268 positive, 500 negative).

If you're following along fresh: the link above is a stable, commonly-
used mirror of the original UCI dataset; download it and place it at
`data/diabetes.csv` (no header row — column names are assigned in code).

WHY THIS DATASET
-----------------
- It's real clinical measurement data (not synthetic), with a genuine,
  well-documented data quality issue built in: several columns use 0 as
  a placeholder for "not recorded" rather than a true physiological
  zero (see clean_and_engineer.py) — a realistic, checkable case study
  for Phase 5, not a contrived example.
- It supports a genuinely useful clinical framing: an early SCREENING
  tool to flag higher-risk patients for confirmatory testing and
  preventive follow-up — not a diagnostic replacement. This distinction
  matters throughout the project (see the disclaimers in app.py and
  PROJECT_DOCUMENTATION.md).

IMPORTANT FRAMING NOTE: this project (and its API) is a decision-support
screening aid for a clinical team, not a diagnostic tool, and should
never be presented to an end patient as a diagnosis. This is stated
explicitly, not left implicit, throughout the project.
"""

import pandas as pd

COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
RAW_DATA_PATH = "data/diabetes.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMN_NAMES)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {RAW_DATA_PATH}")
    print(f"Diabetes-positive rate: {df['Outcome'].mean():.2%}")
    print(f"\nMissing values (pandas nulls): {df.isnull().sum().sum()}  |  Duplicate rows: {df.duplicated().sum()}")
    print("\nZero values in columns where 0 is not physiologically possible:")
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        n_zero = (df[col] == 0).sum()
        print(f"  {col}: {n_zero} rows ({n_zero / len(df):.1%})")
