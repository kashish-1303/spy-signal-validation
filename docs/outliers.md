# Outlier Assumptions

**Definition:** A day is flagged as an outlier if its return is more than 3 standard
deviations from the mean daily return (z-score threshold).

**Why z-score on returns, not raw price:** SPY's price trends upward over decades,
so a simple price-based outlier check would flag legitimate long-term growth as
"outliers." Z-scoring daily *returns* instead isolates abnormal single-day moves.

**Risk to results:** Outlier days (e.g., 2020 COVID crash, other volatility spikes)
are often the most informative days for a trading signal, not noise to discard.
For this reason, outliers are flagged but NOT automatically removed from the
modeling dataset by default - `remove_outliers()` exists but is used selectively,
only for sensitivity comparisons, not as a default cleaning step.