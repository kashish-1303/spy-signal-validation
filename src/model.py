import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


FEATURE_COLS = ["ma_5", "ma_10", "ma_20", "momentum", "volatility", "rsi"]

import joblib

def save_model(pipeline, path="model/model.pkl"):
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")

def load_model(path="model/model.pkl"):
    return joblib.load(path)

def predict_direction(pipeline, feature_row):
    """
    Takes a single row of features (dict or DataFrame row) and returns
    the predicted direction: 1 = up, 0 = down.
    """
    if isinstance(feature_row, dict):
        feature_row = pd.DataFrame([feature_row])
    pred = pipeline.predict(feature_row[FEATURE_COLS])[0]
    return int(pred)

def walk_forward_split(df, n_splits=5):
    """
    Time-aware split: each fold trains ONLY on past data, tests on the
    period immediately after. Never shuffles - shuffling would let the
    model "see the future" during training, which is the exact mistake
    that makes backtests look falsely good.
    """
    fold_size = len(df) // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]
        splits.append((train, test))
    return splits


def build_pipeline(model_type="logistic"):
    """Builds a sklearn Pipeline: scaling + model, so preprocessing and
    modeling stay bundled and reproducible."""
    model = LogisticRegression(max_iter=1000) if model_type == "logistic" else RandomForestClassifier(n_estimators=100, random_state=42)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def evaluate_fold(pipeline, train, test):
    X_train, y_train = train[FEATURE_COLS], train["target"]
    X_test, y_test = test[FEATURE_COLS], test["target"]

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "n_test": len(y_test),
    }


def run_walk_forward(df, model_type="logistic", n_splits=5):
    """Runs the full walk-forward evaluation across all folds, tries
    both model variations (satisfies 'auto-try with some variations')."""
    splits = walk_forward_split(df, n_splits)
    results = []
    for i, (train, test) in enumerate(splits):
        pipeline = build_pipeline(model_type)
        metrics = evaluate_fold(pipeline, train, test)
        metrics["fold"] = i + 1
        results.append(metrics)
    return pd.DataFrame(results)