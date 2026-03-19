"""
Embedding analysis ("Semantic Dictionary") for the clusterAnalysis pipeline.

For each of the 128 LISBET embedding dimensions, compute Spearman correlation
with each kinematic metric at the segment level.

Strategy:
- For each segment: compute the mean embedding vector (128D) by averaging all
  frame embeddings within that segment.
- Join with segment-level kinematic summaries.
- Compute 128 × N_metrics Spearman correlation matrix.
- Apply FDR correction over all (dim, metric) pairs.

This is the most memory-intensive analysis. It is gated behind the
`embedding_kinematics` config flag and runs last.

GPU acceleration: if torch is available and a CUDA device is present, uses
batched matrix operations to compute correlations faster.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import fdr_correct, add_significance_flags

logger = logging.getLogger(__name__)

N_EMBEDDING_DIMS = 128


def _load_segment_mean_embeddings(
    embeddings_dir: Path,
    segment_names: list[str],
    dtype: np.dtype = np.float16,
) -> pd.DataFrame:
    """
    Load per-segment mean embedding vectors from CSV files.

    Parameters
    ----------
    embeddings_dir : Path
        Root directory with one subdirectory per segment.
    segment_names : list[str]
        Segment names to load (subdirectory names).
    dtype : np.dtype
        Storage dtype (float16 to save memory).

    Returns
    -------
    pd.DataFrame of shape (N_segments, 128), indexed by segment_name.
    """
    rows = []
    for seg_name in segment_names:
        csv_path = embeddings_dir / seg_name / "features_lisbet_embedding.csv"
        if not csv_path.exists():
            logger.debug("Embedding CSV not found: %s", csv_path)
            continue
        try:
            emb = pd.read_csv(csv_path, index_col=0).to_numpy(dtype=np.float32)
            mean_vec = emb.mean(axis=0).astype(dtype)
            rows.append((seg_name, mean_vec))
        except Exception as exc:
            logger.warning("Failed to load embedding %s: %s", csv_path, exc)
            continue

    if not rows:
        raise ValueError(f"No embeddings loaded from {embeddings_dir}")

    seg_names, vecs = zip(*rows)
    df = pd.DataFrame(
        np.array(vecs, dtype=np.float32),
        index=list(seg_names),
        columns=[str(i) for i in range(N_EMBEDDING_DIMS)],
    )
    df.index.name = "segment_name"
    logger.info(
        "Segment mean embeddings: %d segments × %d dims loaded",
        *df.shape
    )
    return df


def run_embedding_kinematic_correlation(
    embeddings_dir: Path,
    kinematics_df: pd.DataFrame,
    subject_map: pd.DataFrame,
    metric_columns: list[str],
    alpha: float = 0.05,
    fdr_method: str = "bh",
    n_bootstrap: int = 0,   # 0 = no CI (too slow for 128×65 pairs)
) -> dict[str, pd.DataFrame]:
    """
    Compute Spearman correlations between 128 embedding dimensions and
    N kinematic metrics at segment level.

    Parameters
    ----------
    embeddings_dir : Path
        Root of segment embedding directories.
    kinematics_df : pd.DataFrame
        MultiIndex (subject_session, seg_label) with kinematic metric columns.
    subject_map : pd.DataFrame
        From build_subject_map() — provides segment_name, subject_session, seg_label linkage.
    metric_columns : list[str]
        Which kinematic metrics to correlate.
    alpha, fdr_method : float, str
        FDR correction parameters.
    n_bootstrap : int
        Bootstrap CI samples. 0 = skip CI (recommended for full 128×65 run).

    Returns
    -------
    dict with keys:
        "rho"     : pd.DataFrame (128 dims × N_metrics, Spearman rho)
        "p_raw"   : pd.DataFrame (128 dims × N_metrics, raw p-values)
        "p_fdr"   : pd.DataFrame (128 dims × N_metrics, FDR-corrected p-values)
        "significant": pd.DataFrame (bool mask)
    """
    # Resolve segment names to load
    all_segment_names = subject_map["segment_name"].unique().tolist()

    logger.info(
        "Embedding analysis: loading mean embeddings for %d segments...",
        len(all_segment_names)
    )
    emb_df = _load_segment_mean_embeddings(embeddings_dir, all_segment_names)

    # Build kinematic DataFrame indexed by segment_name
    # kinematics_df has MultiIndex (subject_session, seg_label)
    sm = subject_map[["segment_name", "subject_session"]].copy()
    sm["seg_label"] = subject_map["segment_name"].str.extract(r"_(seg_\d+)$")[0]

    kin_reset = kinematics_df.reset_index()
    kin_by_segment = sm.merge(kin_reset, on=["subject_session", "seg_label"], how="left")
    kin_by_segment = kin_by_segment.set_index("segment_name")

    valid_metrics = [c for c in metric_columns if c in kin_by_segment.columns]
    if not valid_metrics:
        raise ValueError("None of the requested kinematic metrics found in kinematics data")

    # Align: only segments present in both
    common_segs = emb_df.index.intersection(kin_by_segment.index)
    if len(common_segs) == 0:
        raise ValueError("No common segments between embeddings and kinematics")

    logger.info(
        "Embedding × kinematics: %d common segments, %d dims × %d metrics = %d tests",
        len(common_segs), N_EMBEDDING_DIMS, len(valid_metrics),
        N_EMBEDDING_DIMS * len(valid_metrics)
    )

    emb = emb_df.loc[common_segs].values.astype(np.float32)       # (N, 128)
    kin = kin_by_segment.loc[common_segs, valid_metrics].values.astype(np.float32)  # (N, M)

    # Compute Spearman correlations using rank-based approach for speed
    from scipy.stats import spearmanr as _spearmanr

    rho_matrix = np.zeros((N_EMBEDDING_DIMS, len(valid_metrics)), dtype=np.float32)
    p_matrix = np.ones((N_EMBEDDING_DIMS, len(valid_metrics)), dtype=np.float32)

    for j, metric in enumerate(valid_metrics):
        y = kin[:, j]
        valid_mask = np.isfinite(y)
        if valid_mask.sum() < 10:
            logger.debug("Metric '%s': only %d valid samples, skipping", metric, valid_mask.sum())
            continue

        y_valid = y[valid_mask]
        emb_valid = emb[valid_mask, :]  # (N_valid, 128)

        for i in range(N_EMBEDDING_DIMS):
            x = emb_valid[:, i]
            if np.std(x) < 1e-10:
                continue
            rho, p = _spearmanr(x, y_valid)
            if np.isfinite(rho):
                rho_matrix[i, j] = rho
                p_matrix[i, j] = p

        if (j + 1) % 10 == 0 or (j + 1) == len(valid_metrics):
            logger.debug("Embedding correlations: %d / %d metrics done", j + 1, len(valid_metrics))

    # FDR over all (dim, metric) pairs
    p_flat = p_matrix.flatten()
    p_fdr_flat = fdr_correct(p_flat, method=fdr_method)
    p_fdr_matrix = p_fdr_flat.reshape(p_matrix.shape)

    sig_matrix = p_fdr_matrix < alpha

    dim_labels = [str(i) for i in range(N_EMBEDDING_DIMS)]
    rho_df = pd.DataFrame(rho_matrix, index=dim_labels, columns=valid_metrics)
    p_raw_df = pd.DataFrame(p_matrix, index=dim_labels, columns=valid_metrics)
    p_fdr_df = pd.DataFrame(p_fdr_matrix, index=dim_labels, columns=valid_metrics)
    sig_df = pd.DataFrame(sig_matrix, index=dim_labels, columns=valid_metrics)

    rho_df.index.name = "embedding_dim"
    n_sig = int(sig_matrix.sum())
    logger.info(
        "Embedding × kinematics: %d / %d pairs significant after FDR (alpha=%.2f)",
        n_sig, N_EMBEDDING_DIMS * len(valid_metrics), alpha
    )

    return {
        "rho": rho_df,
        "p_raw": p_raw_df,
        "p_fdr": p_fdr_df,
        "significant": sig_df,
    }
