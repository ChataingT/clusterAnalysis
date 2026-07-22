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
from scipy.stats import spearmanr

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


def _build_scope_masks(
    diagnosis_series: pd.Series,
    groups: tuple[str, str],
) -> dict[str, pd.Series]:
    """Return row masks for all-subject and diagnosis-specific scopes."""
    masks: dict[str, pd.Series] = {
        "all": pd.Series(True, index=diagnosis_series.index),
    }
    masks[groups[0]] = diagnosis_series == groups[0]
    masks[groups[1]] = diagnosis_series == groups[1]
    return masks


def _spearman_pairwise(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Spearman rho on finite pairs only; returns (rho, n_valid_pairs)."""
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid.sum())
    if n_valid < 3:
        return float("nan"), n_valid
    rho, _ = spearmanr(x[valid], y[valid])
    return float(rho), n_valid


# ── Binary analysis (ASD vs TD) ──────────────────────────────────────────────

def run_binary_analysis(
    prevalence_matrix: pd.DataFrame,
    clinical_df: pd.DataFrame,
    group_column: str = "diagnosis",
    groups: tuple[str, str] = ("ASD", "TD"),
    alpha: float = 0.05,
    fdr_method: str = "bh",
    seed: int | None = None,
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
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        cluster_id, U, p_raw, p_fdr, cohens_d, rank_biserial_r, direction,
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
            "rank_biserial_r": res.rank_biserial_r,
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
    diagnosis_column: str = "diagnosis",
    diagnosis_groups: tuple[str, str] = ("ASD", "TD"),
    alpha: float = 0.05,
    fdr_method: str = "bh",
    n_bootstrap: int = 500,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Spearman correlation for each scope × cluster × clinical metric combination.

    Parameters
    ----------
    prevalence_matrix : pd.DataFrame
        Shape (N_subjects, N_clusters), index = uuid.
    clinical_df : pd.DataFrame
        Index = uuid.
    metrics : list[str]
        Clinical column names to correlate against cluster prevalences.
    diagnosis_column : str
        Binary diagnosis column used to define subgroup scopes.
    diagnosis_groups : tuple[str, str]
        Labels for subgroup scopes (e.g. ("ASD", "TD")).
    alpha, fdr_method : float, str
        FDR correction parameters.
    n_bootstrap : int
        Bootstrap samples for confidence intervals.

    Returns
    -------
    pd.DataFrame with columns:
        scope, cluster_id, metric, rho, p_raw, p_fdr, ci_low, ci_high,
        n, n_subjects_scope, significant, sig_label
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
    if diagnosis_column not in clinical_df.columns:
        raise ValueError(f"Diagnosis column '{diagnosis_column}' not in clinical data")

    prev = prevalence_matrix.loc[common_uuids]
    clin = clinical_df.loc[common_uuids, available]
    diagnosis = clinical_df.loc[common_uuids, diagnosis_column]
    scope_masks = _build_scope_masks(diagnosis, diagnosis_groups)

    logger.info(
        "Continuous correlations: %d subjects, %d clusters × %d metrics × %d scopes = %d tests",
        len(common_uuids), prev.shape[1], len(available), len(scope_masks),
        prev.shape[1] * len(available) * len(scope_masks)
    )

    rows = []
    for scope_name, mask in scope_masks.items():
        prev_scope = prev.loc[mask]
        clin_scope = clin.loc[mask]
        n_subjects_scope = int(mask.sum())
        if n_subjects_scope == 0:
            logger.warning("Continuous correlations [%s]: no matching subjects, skipping", scope_name)
            continue
        logger.info(
            "Continuous correlations [%s]: n_subjects=%d, %d clusters × %d metrics = %d tests",
            scope_name,
            n_subjects_scope,
            prev_scope.shape[1],
            len(available),
            prev_scope.shape[1] * len(available),
        )

        for metric in available:
            y = clin_scope[metric].values.astype(float)
            for cluster_id in prev_scope.columns:
                x = prev_scope[cluster_id].values.astype(float)

                # check if y or y are constant
                if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                    logger.warning(
                        "Continuous correlations [%s, cluster %s, metric %s]: constant values, skipping",
                        scope_name, cluster_id, metric
                    )
                    rows.append({
                        "scope": scope_name,
                        "cluster_id": cluster_id,
                        "metric": metric,
                        "rho": float("nan"),
                        "p_raw": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n": 0,
                        "n_subjects_scope": n_subjects_scope,
                    })
                    continue

                res: SpearmanResult = spearman_with_ci(
                    x,
                    y,
                    n_bootstrap=n_bootstrap,
                    seed=42 if seed is None else int(seed),
                )
                rows.append({
                    "scope": scope_name,
                    "cluster_id": cluster_id,
                    "metric": metric,
                    "rho": res.rho,
                    "p_raw": res.p_value,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "n": res.n,
                    "n_subjects_scope": n_subjects_scope,
                })

    results = pd.DataFrame(rows)

    # FDR is applied independently per scope over all (cluster, metric) pairs.
    results["p_fdr"] = np.nan
    for scope_name, scope_df in results.groupby("scope", sort=False):
        idx = scope_df.index
        results.loc[idx, "p_fdr"] = fdr_correct(
            results.loc[idx, "p_raw"].values,
            method=fdr_method,
        )
    results = add_significance_flags(results, p_col="p_fdr", alpha=alpha)

    for scope_name in results["scope"].unique():
        scope_df = results[results["scope"] == scope_name]
        log_test_summary(scope_df, f"Spearman (continuous, scope={scope_name})", alpha=alpha)

    return results.sort_values(["scope", "p_fdr", "metric", "cluster_id"])


def run_continuous_prevalence_permutation(
    prevalence_matrix: pd.DataFrame,
    clinical_df: pd.DataFrame,
    continuous_results: pd.DataFrame,
    metrics: list[str],
    diagnosis_column: str = "diagnosis",
    diagnosis_groups: tuple[str, str] = ("ASD", "TD"),
    alpha: float = 0.05,
    fdr_method: str = "bh",
    n_permutations: int = 1000,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Permutation test for prevalence-correlation-strength association.

    For each scope and clinical metric, tests whether clusters with larger mean
    prevalence tend to have larger absolute cluster-wise correlations:

        Spearman( mean_prevalence_per_cluster, |rho_cluster_metric| )

    Null is built by shuffling metric values across subjects within each scope,
    recomputing per-cluster rho values, then recomputing the across-cluster
    Spearman statistic.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    if diagnosis_column not in clinical_df.columns:
        raise ValueError(f"Diagnosis column '{diagnosis_column}' not in clinical data")

    available = [m for m in metrics if m in clinical_df.columns]
    if not available:
        raise ValueError("None of the requested clinical metrics are available in the data")

    common_uuids = prevalence_matrix.index.intersection(clinical_df.index)
    if len(common_uuids) == 0:
        raise ValueError("No overlapping subjects between prevalence matrix and clinical data")

    prev_all = prevalence_matrix.loc[common_uuids]
    clin_all = clinical_df.loc[common_uuids, available + [diagnosis_column]]
    scope_masks = _build_scope_masks(clin_all[diagnosis_column], diagnosis_groups)

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for scope_name, mask in scope_masks.items():
        prev_scope = prev_all.loc[mask]
        clin_scope = clin_all.loc[mask, available]
        if prev_scope.empty or clin_scope.empty:
            logger.warning("Continuous permutation [%s]: no subjects, skipping", scope_name)
            continue

        mean_prevalence = prev_scope.mean(axis=0).astype(float)
        n_subjects_scope = int(mask.sum())

        for metric in available:
            result_sub = continuous_results[
                (continuous_results["scope"] == scope_name)
                & (continuous_results["metric"] == metric)
            ]
            observed_abs_rho = result_sub.set_index("cluster_id")["rho"].abs()

            observed_df = pd.DataFrame({
                "mean_prevalence": mean_prevalence.reindex(observed_abs_rho.index),
                "abs_rho": observed_abs_rho,
            }).dropna()

            if len(observed_df) < 3:
                rows.append({
                    "scope": scope_name,
                    "metric": metric,
                    "stat_rho_prevalence_absrho": float("nan"),
                    "p_raw": float("nan"),
                    "null_mean": float("nan"),
                    "null_std": float("nan"),
                    "z_like": float("nan"),
                    "n_clusters_valid": int(len(observed_df)),
                    "n_subjects_scope": n_subjects_scope,
                    "n_permutations": int(n_permutations),
                })
                continue

            observed_stat, _ = _spearman_pairwise(
                observed_df["mean_prevalence"].to_numpy(dtype=float),
                observed_df["abs_rho"].to_numpy(dtype=float),
            )

            y = clin_scope[metric].to_numpy(dtype=float)
            null_stats: list[float] = []
            cluster_ids = observed_df.index.tolist()

            for _ in range(n_permutations):
                y_perm = rng.permutation(y)
                abs_rho_perm: list[float] = []
                mean_prev_perm: list[float] = []

                for cluster_id in cluster_ids:
                    x = prev_scope[cluster_id].to_numpy(dtype=float)
                    rho_perm, _n_perm = _spearman_pairwise(x, y_perm)
                    if np.isfinite(rho_perm):
                        abs_rho_perm.append(abs(rho_perm))
                        mean_prev_perm.append(float(mean_prevalence.loc[cluster_id]))

                if len(abs_rho_perm) < 3:
                    continue

                stat_perm, _ = _spearman_pairwise(
                    np.asarray(mean_prev_perm, dtype=float),
                    np.asarray(abs_rho_perm, dtype=float),
                )
                if np.isfinite(stat_perm):
                    null_stats.append(float(stat_perm))

            null_arr = np.asarray(null_stats, dtype=float)
            if np.isfinite(observed_stat) and len(null_arr) > 0:
                p_raw = float((np.sum(np.abs(null_arr) >= abs(observed_stat)) + 1) / (len(null_arr) + 1))
                null_mean = float(np.mean(null_arr))
                null_std = float(np.std(null_arr, ddof=1)) if len(null_arr) > 1 else float("nan")
                if np.isfinite(null_std) and null_std > 0:
                    z_like = float((observed_stat - null_mean) / null_std)
                else:
                    z_like = float("nan")
            else:
                p_raw = float("nan")
                null_mean = float("nan")
                null_std = float("nan")
                z_like = float("nan")

            rows.append({
                "scope": scope_name,
                "metric": metric,
                "stat_rho_prevalence_absrho": observed_stat,
                "p_raw": p_raw,
                "null_mean": null_mean,
                "null_std": null_std,
                "z_like": z_like,
                "n_clusters_valid": int(len(observed_df)),
                "n_subjects_scope": n_subjects_scope,
                "n_permutations": int(n_permutations),
            })

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results["p_fdr"] = np.nan
    for scope_name, scope_df in results.groupby("scope", sort=False):
        idx = scope_df.index
        results.loc[idx, "p_fdr"] = fdr_correct(
            results.loc[idx, "p_raw"].values,
            method=fdr_method,
        )
    results = add_significance_flags(results, p_col="p_fdr", alpha=alpha)

    for scope_name in results["scope"].unique():
        scope_df = results[results["scope"] == scope_name]
        log_test_summary(
            scope_df,
            f"Permutation prevalence~|rho| (scope={scope_name})",
            alpha=alpha,
        )

    return results.sort_values(["scope", "p_fdr", "metric"])


# ── Full clinical analysis orchestrator ──────────────────────────────────────

def run_clinical_analysis(
    prevalence_matrix: pd.DataFrame,
    clinical_df: pd.DataFrame,
    binary_groups: list[str],
    continuous_metrics: list[str],
    alpha: float = 0.05,
    fdr_method: str = "bh",
    n_bootstrap: int = 500,
    n_permutations: int = 1000,
    seed: int | None = None,
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
            seed=seed,
        )

    if continuous_metrics:
        results["continuous"] = run_continuous_correlations(
            prevalence_matrix, clinical_df,
            metrics=continuous_metrics,
            alpha=alpha,
            fdr_method=fdr_method,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        results["continuous_permutation"] = run_continuous_prevalence_permutation(
            prevalence_matrix=prevalence_matrix,
            clinical_df=clinical_df,
            continuous_results=results["continuous"],
            metrics=continuous_metrics,
            alpha=alpha,
            fdr_method=fdr_method,
            n_permutations=n_permutations,
            seed=seed,
        )

    return results
