import numpy as np
import pandas as pd


def detect_outliers_zscore(df, column="Close", threshold=3):
    """
    Flags outliers using z-score on daily returns (not raw price - price
    trends upward over time, so z-scoring raw price would misfire; z-scoring
    returns catches abnormal single-day moves instead, which is what actually
    matters for a trading signal).
    Returns the dataframe with an added 'is_outlier' boolean column.
    """
    returns = df[column].pct_change()
    z_scores = (returns - returns.mean()) / returns.std()
    df = df.copy()
    df["is_outlier"] = z_scores.abs() > threshold
    n_outliers = df["is_outlier"].sum()
    print(f"Flagged {n_outliers} outliers out of {len(df)} rows ({n_outliers/len(df)*100:.2f}%)")
    return df


def remove_outliers(df):
    """Removes rows flagged as outliers. Use with caution - see docs/outliers.md."""
    before = len(df)
    df_clean = df[~df["is_outlier"]].drop(columns=["is_outlier"])
    print(f"Removed {before - len(df_clean)} outlier rows")
    return df_clean