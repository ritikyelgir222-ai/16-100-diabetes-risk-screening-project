"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY THIS SPLIT HAPPENS ON THE RAW DATAFRAME, BEFORE CLEANING (unlike
every other project in this series, where cleaning happens once before
the split): this dataset's cleaning step involves MEDIAN IMPUTATION
(clean_and_engineer.py), and a median is a statistic computed FROM data
-- imputing missing Glucose/BMI/etc. values using a median calculated
across the entire dataset (train + val + test combined) would leak
information from the val/test sets into training, the same class of
mistake as scaling before splitting. Splitting the raw data first, then
computing imputation medians from the training split ONLY, and reusing
those exact medians for val/test/serving, is the leakage-safe order of
operations -- a common mistake in tutorials that skip this.

WHY STRATIFIED (not random): Outcome is imbalanced (34.9% positive) --
the same reasoning as the churn and attrition projects.
"""

from sklearn.model_selection import train_test_split


def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    df_temp, df_test = train_test_split(
        df, test_size=test_size, stratify=df["Outcome"], random_state=random_state
    )
    val_relative_size = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_temp, test_size=val_relative_size, stratify=df_temp["Outcome"], random_state=random_state
    )
    return df_train, df_val, df_test
