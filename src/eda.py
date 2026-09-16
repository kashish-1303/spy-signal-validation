import pandas as pd


def eda_summary(df, column="Close"):
    """
    Returns key statistical summaries for a given column: central tendency,
    spread, distribution shape, and data quality checks.
    """
    returns = df[column].pct_change().dropna()

    summary = {
        "mean_price": df[column].mean(),
        "median_price": df[column].median(),
        "std_price": df[column].std(),
        "min_price": df[column].min(),
        "max_price": df[column].max(),
        "missing_values": df[column].isnull().sum(),
        "mean_daily_return": returns.mean(),
        "std_daily_return": returns.std(),
        "skewness_returns": returns.skew(),
    }

    for key, val in summary.items():
        print(f"{key}: {val:.6f}" if isinstance(val, float) else f"{key}: {val}")

    return summary