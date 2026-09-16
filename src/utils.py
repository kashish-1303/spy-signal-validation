def clean_column_names(df):
    """
    Standardizes dataframe column names: lowercase, no spaces, no extra whitespace.
    """
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

import os
import pandas as pd
from pathlib import Path


def get_data_dir():
    """Reads DATA_DIR from .env, defaults to './data' if not set."""
    return Path(os.getenv("DATA_DIR", "./data"))


def save_raw_data(df, filename):
    path = get_data_dir() / "raw" / filename
    df.to_csv(path, index=True)  # changed from False to True
    print(f"Saved {len(df)} rows to {path}")
    return path


def load_raw_data(filename):
    path = get_data_dir() / "raw" / filename
    df = pd.read_csv(path, index_col=0)
    print(f"Loaded {len(df)} rows from {path}")
    return df