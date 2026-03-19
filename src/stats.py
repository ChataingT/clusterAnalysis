"""
Statistical utilities for the clusterAnalysis pipeline.

All functions:
- validate inputs (log warnings on NaN, small N)
- return named tuples or dicts for clarity
- are designed to be called in tight loops over many clusters/metrics
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr

logger = logging.getLogger(__name__)

MIN_GROUP_SIZE = 3  # minimum samples per group for meaningful statistics


# ── Result types ─────────────────────────────────────────────────────────────

class MannWhitneyResult(NamedTuple):
    U: float
    p_value: float
    cohens_d: float
    direction: str    # "higher_in_group_a" | "higher_in_group_b" | "equal"
    n_a: int
    n_b: int


class SpearmanResult(NamedTuple):
    rho: float
    p_value: float
    ci_low: float
    ci_high: float
    n: int


# ── Mann-Whitney U ────────────────────────────────────────────────────────────

def mann_whitney_with_effect(
    group_a: np.ndarray | pd.Series,
    group_b: np.ndarray | pd.Series,
    label_a: str = "A",
    label_b: str = "B",
) -> MannWhitneyResult:
    """
    Mann-Whitney U test with Cohen's d effect size.

    Parameters
    ----------
    group_a, group_b : array-like
        Observed values per group (e.g., cluster prevalence for ASD vs TD).
    label_a, label_b : str
        Labels used only in log messages.

    Returns
    -------
    MannWhitneyResult
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    # Drop NaN
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < MIN_GROUP_SIZE:
        logger.debug(
            "Mann-Whitney: group '%s' has only %d samples (< %d) — result unreliable",
            label_a, len(a), MIN_GROUP_SIZE
        )
    if len(b) < MIN_GROUP_SIZE:
        logger.debug(
            "Mann-Whitney: group '%s' has only %d samples (< %d) — result unreliable",
            label_b, len(b), MIN_GROUP_SIZE
        )

    if len(a) == 0 or len(b) == 0:
        return MannWhitneyResult(
            U=float("nan"), p_value=float("nan"), cohens_d=float("nan"),
            direction="undefined", n_a=len(a), n_b=len(b)
        )

    U, p = mannwhitneyu(a, b, alternative="two-sided")

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
        / (len(a) + len(b) - 2)
    ) if (len(a) + len(b)) > 2 else float("nan")

    if pooled_std > 0:
        d = (a.mean() - b.mean()) / pooled_std
    else:
        d = 0.0

    if a.mean() > b.mean():
        direction = f"higher_in_{label_a}"
    elif b.mean() > a.mean():
        direction = f"higher_in_{label_b}"
    else:
        direction = "equal"

    return MannWhitneyResult(U=U, p_value=p, cohens_d=d, direction=direction,
                              n_a=len(a), n_b=len(b))


# ── Spearman correlation ──────────────────────────────────────────────────────

def spearman_with_ci(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> SpearmanResult:
    """
    Spearman correlation with bootstrap 95% confidence interval.

    Parameters
    ----------
    x, y : array-like
        Paired observations. NaN pairs are dropped.
    n_bootstrap : int
        Number of bootstrap resamples for CI.

    Returns
    -------
    SpearmanResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Drop NaN pairs
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    n = len(x)
    if n < MIN_GROUP_SIZE:
        logger.debug(
            "Spearman: only %d valid pairs (< %d) — result unreliable", n, MIN_GROUP_SIZE
        )
        return SpearmanResult(rho=float("nan"), p_value=float("nan"),
                               ci_low=float("nan"), ci_high=float("nan"), n=n)

    rho, p = spearmanr(x, y)

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(x[idx], y[idx])
        if np.isfinite(r):
            boot_rhos.append(r)

    if len(boot_rhos) >= 10:
        ci_low = float(np.percentile(boot_rhos, 2.5))
        ci_high = float(np.percentile(boot_rhos, 97.5))
    else:
        ci_low = ci_high = float("nan")

    return SpearmanResult(rho=float(rho), p_value=float(p),
                           ci_low=ci_low, ci_high=ci_high, n=n)


# ── FDR correction ────────────────────────────────────────────────────────────

def fdr_correct(
    p_values: np.ndarray | pd.Series,
    method: str = "bh",
) -> np.ndarray:
    """
    Apply FDR (multiple-testing) correction to an array of p-values.

    Parameters
    ----------
    p_values : array-like
        Raw p-values. NaN values are preserved as NaN in output.
    method : str
        "bh" (Benjamini-Hochberg) or "bonferroni".

    Returns
    -------
    np.ndarray of corrected p-values, same shape as input.
    """
    p = np.asarray(p_values, dtype=float)
    corrected = np.full_like(p, fill_value=float("nan"))

    valid_mask = np.isfinite(p)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        logger.warning("FDR correction: all p-values are NaN")
        return corrected

    valid_p = p[valid_mask]

    if method == "bh":
        # Benjamini-Hochberg procedure
        n = len(valid_p)
        order = np.argsort(valid_p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n + 1)
        corrected_valid = np.minimum(1.0, valid_p * n / ranks)
        # Enforce monotonicity: p_fdr[i] <= p_fdr[j] if rank[i] < rank[j]
        corrected_valid = _bh_monotone(corrected_valid, order)
    elif method == "bonferroni":
        corrected_valid = np.minimum(1.0, valid_p * n_valid)
    else:
        raise ValueError(f"Unknown FDR method: '{method}'. Use 'bh' or 'bonferroni'.")

    corrected[valid_mask] = corrected_valid
    return corrected


def _bh_monotone(corrected: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Enforce BH monotonicity by cumulative minimum from largest rank."""
    out = corrected.copy()
    for i in range(len(order) - 2, -1, -1):
        out[order[i]] = min(out[order[i]], out[order[i + 1]])
    return out


# ── Summary helpers ───────────────────────────────────────────────────────────

def add_significance_flags(
    df: pd.DataFrame,
    p_col: str = "p_fdr",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Add a 'significant' boolean column and a 'sig_label' string column.

    sig_label uses standard asterisk notation:
      p < 0.001 → "***"
      p < 0.01  → "**"
      p < 0.05  → "*"
      otherwise → "ns"
    """
    df = df.copy()
    p = df[p_col].to_numpy(dtype=float)
    df["significant"] = p < alpha

    labels = np.full(len(p), "ns", dtype=object)
    labels[p < 0.05] = "*"
    labels[p < 0.01] = "**"
    labels[p < 0.001] = "***"
    df["sig_label"] = labels
    return df


def log_test_summary(
    results_df: pd.DataFrame,
    test_name: str,
    p_col: str = "p_fdr",
    alpha: float = 0.05,
) -> None:
    """Log how many tests survived FDR correction."""
    n_total = len(results_df)
    n_sig = (results_df[p_col] < alpha).sum() if p_col in results_df.columns else 0
    logger.info(
        "%s: %d tests run, %d significant after FDR (alpha=%.2f)",
        test_name, n_total, n_sig, alpha
    )
