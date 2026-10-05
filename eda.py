"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THESE SPECIFIC CUTS: before trusting any clinical measurement in
this dataset, the zero-as-missing issue has to be characterized (is it
random, or does it cluster in ways that matter?), and then the actual
risk factors need checking against real medical literature expectations
(glucose, BMI, and age are the textbook drivers of type 2 diabetes risk)
rather than assumed.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from data_loader import load_raw_data

ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def run_eda(df: pd.DataFrame, out_dir: str = "outputs"):
    df = df.copy()

    print("=== THE DATA QUALITY FINDING: zeros as missing-value placeholders ===")
    for col in ZERO_AS_MISSING_COLUMNS:
        n_zero = (df[col] == 0).sum()
        print(f"  {col}: {n_zero} zeros ({n_zero / len(df):.1%})")
    print("-> A living person cannot have 0 blood pressure, 0 BMI, or 0 glucose --")
    print("   these are missing measurements recorded as 0, not true physiological")
    print("   values. Insulin (48.7%) and SkinThickness (29.6%) are missing often")
    print("   enough that dropping those rows would discard nearly half the dataset;")
    print("   handled with imputation + missingness flags in clean_and_engineer.py.")

    print("\n=== Does missingness cluster with the outcome? ===")
    for col in ["Insulin", "SkinThickness"]:
        missing = df[col] == 0
        print(f"  {col} missing rate by Outcome: {df.groupby('Outcome').apply(lambda g: (g[col]==0).mean(), include_groups=False).round(3).to_dict()}")
    # WHY THIS MATTERS: if missingness itself correlates with the
    # outcome (e.g. sicker patients get more or fewer tests recorded),
    # a missingness FLAG could carry real signal beyond just "fill the
    # gap" -- worth checking rather than assuming missingness is random.

    # Replace zeros with NaN just for this EDA pass (not saved/reused --
    # the actual imputation logic lives in clean_and_engineer.py)
    df_valid = df.copy()
    for col in ZERO_AS_MISSING_COLUMNS:
        df_valid[col] = df_valid[col].replace(0, np.nan)

    print("\n=== Average Glucose by Outcome (zeros excluded) ===")
    print(df_valid.groupby("Outcome")["Glucose"].mean().round(1))
    # WHY GLUCOSE FIRST: elevated fasting/OGTT glucose is the direct
    # clinical basis for a diabetes diagnosis -- if this dataset didn't
    # show a strong split here, something would be wrong with the data
    # itself, not just the model.

    print("\n=== Average BMI by Outcome (zeros excluded) ===")
    print(df_valid.groupby("Outcome")["BMI"].mean().round(1))

    print("\n=== Average Age by Outcome ===")
    print(df.groupby("Outcome")["Age"].mean().round(1))

    print("\n=== Average Pregnancies by Outcome ===")
    print(df.groupby("Outcome")["Pregnancies"].mean().round(2))
    # WHY: higher pregnancy count is a documented type-2/gestational-
    # diabetes risk correlate -- checked directly rather than assumed.

    print("\n=== Correlation with Outcome (zeros excluded, valid values only) ===")
    corr = df_valid.corr(numeric_only=True)["Outcome"].sort_values(ascending=False)
    print(corr.round(3))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sns.boxplot(data=df_valid, x="Outcome", y="Glucose", ax=axes[0], hue="Outcome", legend=False, palette=["#55A868", "#C44E52"])
    axes[0].set_title("Glucose by outcome (zeros excluded)")

    sns.boxplot(data=df_valid, x="Outcome", y="BMI", ax=axes[1], hue="Outcome", legend=False, palette=["#55A868", "#C44E52"])
    axes[1].set_title("BMI by outcome (zeros excluded)")

    sns.boxplot(data=df, x="Outcome", y="Age", ax=axes[2], hue="Outcome", legend=False, palette=["#55A868", "#C44E52"])
    axes[2].set_title("Age by outcome")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
