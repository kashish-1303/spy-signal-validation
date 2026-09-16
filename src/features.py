import pandas as pd
import numpy as np


def add_moving_averages(df, windows=[5, 10, 20]):
    """Moving averages smooth out noise and reveal trend direction -
    price relative to its own MA is a classic momentum/trend signal."""
    for w in windows:
        df[f"ma_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_momentum(df, window=10):
    """Momentum = how much price has moved over the recent window.
    Positive momentum has historically shown some persistence
    (trend continuation) over short horizons."""
    df["momentum"] = df["Close"] - df["Close"].shift(window)
    return df


def add_volatility(df, window=20):
    """Rolling volatility of returns - regime context. The same price move
    means something different in a calm market vs. a turbulent one."""
    returns = df["Close"].pct_change()
    df["volatility"] = returns.rolling(window=window).std()
    return df


def add_rsi(df, window=14):
    """Relative Strength Index - measures whether the asset is
    overbought/oversold relative to its recent price action, a standard
    technical indicator for potential reversals."""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_target(df):
    """The label to predict: 1 if next day's close is higher than today's,
    else 0. Uses .shift(-1) to look FORWARD - this is intentional and the
    only place we peek ahead, since it's literally the thing being predicted."""
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def build_features(df):
    """Runs the full feature engineering sequence in order."""
    df = df.copy()
    df = add_moving_averages(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_rsi(df)
    df = add_target(df)
    df = df.dropna()  # drop rows with NaN from rolling windows / shift
    return df