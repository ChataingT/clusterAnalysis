"""
Clinical analysis for the clusterAnalysis pipeline.

Uses the full dataset (all records with clusters + clinical data) to:
1. Binary comparison (ASD vs TD): Mann-Whitney U per cluster + FDR correction
2. Continuous correlations (ADOS, MSEL, Vineland...): Spearman per cluster × metric + FDR

Input: prevalence matrix (N_subjects × N_clusters) + clinical_df.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .stats import (
    MannWhitneyResult,
    SpearmanResult,
    add_significance_flags,
    fdr_correct,
    log_test_summary,
    mann_whitney_with_effect,
    spearman_with_ci,
)

logger = logging.getLogger(__name__)


# ── Binary analysis (ASD vs TD) ──────────────────────────────────────────────

def run_binary_analysis(
    prevalence_matrix: pd.DataFrame,
    clinical_df: pd.DataFrame,
    group_column: str = "diagnosis",
    groups: tuple[str, str] = ("ASD", "TD"),
    alpha: float = 0.05,
    fdr_method: str = "bh",
) -> pd.DataFrame:
    """
    Mann-Whitney U test for each cluster between two diagnosis groups.

    Parameters
    ----------
    prevalence_matrix : pd.DataFrame
        Shape (N_subjects, N_clusters), index = uuid.
    clinical_df : pd.DataFrame
        Index = uuid, must contain `group_column`.
    group_column : str
        Column to split into two groups (e.g. "diagnosis").
    groups : tuple[str, str]
        Labels for group A and group B (e.g. ("ASD", "TD")).
    alpha : float
        Significance threshold (after FDR correction).
    fdr_method : str
        "bh" (Benjamini-Hochberg) or "bonferroni".

    Returns
    -------
    pd.DataFrame with columns:
        cluster_id, U, p_raw, p_fdr, cohens_d, direction,
        n_a, n_b, significant, sig_label
    """
    if group_column not in clinical_df.columns:
        raise ValueError(f"Column '{group_column}' not in clinical data")

    # Align prevalence matrix with clinical data
    common_uuids = prevalence_matrix.index.intersection(clinical_df.index)
    if len(common_uuids) == 0:
        raise ValueError("No overlapping subjects between prevalence matrix and clinical data")
    if len(common_uuids) < len(prevalence_matrix):
        logger.warning(
            "Binary analysis: %d / %d subjects have clinical data",
            len(common_uuids), len(prevalence_matrix)
        )

    prev = prevalence_matrix.loc[common_uuids]
    clin = clinical_df.loc[common_uuids, group_column]

    group_a_mask = clin == groups[0]
    group_b_mask = clin == groups[1]

    n_a = int(group_a_mask.sum())
    n_b = int(group_b_mask.sum())
    logger.info(
        "Binary analysis (%s vs %s): n_%s=%d, n_%s=%d, testing %d clusters",
        groups[0], groups[1], groups[0], n_a, groups[1], n_b, prev.shape[1]
    )

    rows = []
    for cluster_id in prev.columns:
        vals_a = prev.loc[group_a_mask, cluster_id].values
        vals_b = prev.loc[group_b_mask, cluster_id].values

        res: MannWhitneyResult = mann_whitney_with_effect(
            vals_a, vals_b,
            label_a=groups[0], label_b=groups[1]
        )
        rows.append({
            "cluster_id": cluster_id,
            "U": res.U,
            "p_raw": res.p_value,
            "cohens_d": res.cohens_d,
            "direction": res.direction,
            "n_a": res.n_a,
            "n_b": res.n_b,
        })

    results = pd.DataFrame(rows)

    # FDR correction
    results["p_fdr"] = fdr_correct(results["p_raw"].values, method=fdr_method)
    results = add_significance_flags(results, p_col="p_fdr", alpha=alpha)

    log_test_summary(results, f"Mann-Whitney ({groups[0]} vs {groups[1]})", alpha=alpha)
    return results.sort_values("p_fdr")


# ── Continuous correlations ───────────────────────────────────────────────────

def run_continuous_correlations(
    prevalence_matrix: pd.DataFrame,
    clinical_df: pd.DataFrame,
    metrics: list[str],
    alpha: float = 0.05,
    fdr_method: str = "bh",
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """
    Spearman correlation for each cluster × clinical metric combination.

    Parameters
    ----------
    prevalence_matrix : pd.DataFrame
        Shape (N_subjects, N_clusters), index = uuid.
    clinical_df : pd.DataFrame
        Index = uuid.
    metrics : list[str]
        Clinical column names to correlate against cluster prevalences.
    alpha, fdr_method : float, str
        FDR correction parameters.
    n_bootstrap : int
        Bootstrap samples for confidence intervals.

    Returns
    -------
    pd.DataFrame with columns:
        cluster_id, metric, rho, p_raw, p_fdr, ci_low, ci_high, n, significant, sig_label
    """
    # Validate metrics
    available = [m for m in metrics if m in clinical_df.columns]
    missing = [m for m in metrics if m not in clinical_df.columns]
    if missing:
        logger.warning(
            "Continuous correlations: %d metrics not found in clinical data: %s",
            len(missing), missing
        )
    if not available:
        raise ValueError("None of the requested clinical metrics are available in the data")

    common_uuids = prevalence_matrix.index.intersection(clinical_df.index)
    if len(common_uuids) == 0:
        raise ValueError("No overlapping subjects between prevalence matrix and clinical data")

    prev = prevalence_matrix.loc[common_uuids]
    clin = clinical_df.loc[common_uuids, available]

    logger.info(
        "Continuous correlations: %d subjects, %d clusters × %d metrics = %d tests",
        len(common_uuids), prev.shape[1], len(available), prev.shape[1] * len(available)
    )

    rows = []
    for metric in available:
        y = clin[metric].values.astype(float)
        for cluster_id in prev.columns:
            x = prev[cluster_id].values.astype(float)
            res: SpearmanResult = spearman_with_ci(x, y, n_bootstrap=n_bootstrap)
            rows.append({
                "cluster_id": cluster_id,
                "metric": metric,
                "rho": res.rho,
                "p_raw": res.p_value,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "n": res.n,
            })

    results = pd.DataFrame(rows)

    # FDR over all (cluster, metric) pairs simultaneously
    results["p_fdr"] = fdr_correct(results["p_raw"].values, method=fdr_method)
    results = add_significance_flags(results, p_col="p_fdr", alpha=alpha)

    log_test_summary(results, "Spearman (continuous)", alpha=alpha)
    return results.sort_values("p_fdr")


# ── Full clinical analysis orchestrator ──────────────────────────────────────

def run_clinical_analysis(
    prevalence_matrix: pd.DataFrame,
    clinical_df: pd.DataFrame,
    binary_groups: list[str],
    continuous_metrics: list[str],
    alpha: float = 0.05,
    fdr_method: str = "bh",
    n_bootstrap: int = 500,
) -> dict[str, pd.DataFrame]:
    """
    Run all clinical analyses.

    Returns
    -------
    dict with keys "binary_{group}" and "continuous" containing result DataFrames.
    """
    results: dict[str, pd.DataFrame] = {}

    for group_col in binary_groups:
        if group_col not in clinical_df.columns:
            logger.warning("Binary group column '%s' not found — skipping", group_col)
            continue
        unique_vals = clinical_df[group_col].dropna().unique()
        if len(unique_vals) != 2:
            logger.warning(
                "Column '%s' has %d unique values (expected 2): %s — skipping binary analysis",
                group_col, len(unique_vals), list(unique_vals)
            )
            continue
        groups = tuple(sorted(unique_vals))
        logger.info("Binary analysis: '%s' → groups %s", group_col, groups)
        results[f"binary_{group_col}"] = run_binary_analysis(
            prevalence_matrix, clinical_df,
            group_column=group_col,
            groups=groups,
            alpha=alpha,
            fdr_method=fdr_method,
        )

    if continuous_metrics:
        results["continuous"] = run_continuous_correlations(
            prevalence_matrix, clinical_df,
            metrics=continuous_metrics,
            alpha=alpha,
            fdr_method=fdr_method,
            n_bootstrap=n_bootstrap,
        )

    return results
