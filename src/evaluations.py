import numpy as np
import pandas as pd


def bootstrap_accuracy_ci(y_true, y_pred, n_iterations=1000, ci=0.95, seed=42):
    """
    Bootstrap confidence interval for accuracy: resamples the test set with
    replacement many times, recomputing accuracy each time, to see how much
    the metric could plausibly vary by chance alone.
    """
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    accuracies = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)
        acc = (y_true[idx] == y_pred[idx]).mean()
        accuracies.append(acc)

    lower = np.percentile(accuracies, (1 - ci) / 2 * 100)
    upper = np.percentile(accuracies, (1 + ci) / 2 * 100)
    return np.mean(accuracies), lower, upper


def majority_baseline_ci(y_true, majority_class, n_iterations=1000, ci=0.95, seed=42):
    """Same bootstrap logic, applied to the naive majority-class baseline,
    so it can be plotted side-by-side against the model's real CI."""
    y_pred_baseline = np.full(len(y_true), majority_class)
    return bootstrap_accuracy_ci(y_true, y_pred_baseline, n_iterations, ci, seed)


def compare_scenarios(featured_df, model_fn, feature_sets, target_col="target"):
    """
    Scenario/sensitivity analysis: reruns the model under different feature
    set choices, to see how much conclusions depend on that assumption.
    """
    results = {}
    for name, cols in feature_sets.items():
        acc = model_fn(featured_df, cols, target_col)
        results[name] = acc
    return results