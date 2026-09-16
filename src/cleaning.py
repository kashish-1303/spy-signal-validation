import pandas as pd


def ensure_datetime_index(df, date_col="Date"):
    """Ensures the dataframe has a proper DatetimeIndex, handling both
    a Date column or an unnamed index column from a CSV reload."""
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    elif df.index.name != date_col:
        df.index = pd.to_datetime(df.index)
        df.index.name = date_col
    return df


def remove_duplicates(df):
    """Drops duplicate rows (by index/date)."""
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    print(f"Removed {before - len(df)} duplicate rows")
    return df


def handle_missing_values(df):
    """Forward-fills missing price data (standard for financial time series -
    assumes last known price holds until a new one arrives)."""
    missing_before = df.isnull().sum().sum()
    df = df.ffill()
    print(f"Filled {missing_before} missing values via forward-fill")
    return df


def sort_by_date(df):
    """Ensures chronological order - critical for any time-series work later."""
    return df.sort_index()


def preprocess_pipeline(df):
    """Runs the full cleaning sequence in order."""
    df = ensure_datetime_index(df)
    df = remove_duplicates(df)
    df = sort_by_date(df)
    df = handle_missing_values(df)
    return df