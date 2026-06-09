"""
Embedding analysis ("Semantic Dictionary") for the clusterAnalysis pipeline.

For each of the 128 LISBET embedding dimensions, compute Spearman correlation
with each kinematic metric at the frame level.

Strategy:
- For each segment, load the per-frame embedding rows and per-frame kinematic
    metrics.
- Align rows by frame index within each segment.
- Concatenate all matched frame rows across segments / records.
- Compute 128 × N_metrics Spearman correlation matrix.
- Apply FDR correction over all (dim, metric) pairs.

Segment boundaries are treated only as record partitions; they are not the
statistical unit of the embedding-vs-kinematics analysis.

This is the most memory-intensive analysis. It is gated behind the
`embedding_kinematics` config flag and runs last.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .kinematic_frame_analysis import _load_segment_frame_metrics
from .stats import fdr_correct, add_significance_flags

logger = logging.getLogger(__name__)

N_EMBEDDING_DIMS = 128


def _load_segment_frame_embeddings(seg_dir: Path) -> pd.DataFrame | None:
    """Load per-frame embedding rows for a single segment."""
    csv_path = seg_dir / "features_lisbet_embedding.csv"
    if not csv_path.exists():
        logger.debug("Embedding CSV not found: %s", csv_path)
        return None

    try:
        emb = pd.read_csv(csv_path, index_col=0)
    except Exception as exc:
        logger.warning("Failed to load embedding %s: %s", csv_path, exc)
        return None

    if emb.empty:
        return None

    frame_index = pd.Index(pd.to_numeric(emb.index, errors="coerce"), name="frame")
    if frame_index.isna().all():
        emb = emb.reset_index(drop=True)
        emb.index = pd.RangeIndex(len(emb), name="frame")
    else:
        valid_mask = ~frame_index.isna()
        emb = emb.loc[valid_mask].copy()
        emb.index = frame_index[valid_mask].astype(int)

    emb = emb.apply(pd.to_numeric, errors="coerce")
    return emb


def run_embedding_kinematic_correlation(
    embeddings_dir: Path,
    pose_records_dir: Path,
    segment_names: list[str],
    metric_columns: list[str] | None = None,
    max_common_frames: int | None = None,
    sampling_seed: int = 42,
    use_normalized: bool = True,
    alpha: float = 0.05,
    fdr_method: str = "bh",
    n_bootstrap: int = 0,   # 0 = no CI (too slow for 128×65 pairs)
) -> dict[str, pd.DataFrame]:
    """
    Compute Spearman correlations between embedding dimensions and
    kinematic metrics at the frame level.

    Parameters
    ----------
    embeddings_dir : Path
        Root of segment embedding directories.
    pose_records_dir : Path
        Root of pose record directories.
    segment_names : list[str]
        Segment names to load (subdirectory names).
    metric_columns : list[str] | None
        Which kinematic metrics to correlate. None = load all available metrics.
    max_common_frames : int | None
        Optional cap on the number of common frame rows used for correlation.
        If None, all common frame rows are used.
    sampling_seed : int
        Seed used when sampling common frames.
    use_normalized : bool
        If True, load metrics_normalised.csv; otherwise metrics_summary.csv.
    alpha, fdr_method : float, str
        FDR correction parameters.
    n_bootstrap : int
        Bootstrap CI samples. 0 = skip CI (recommended for full 128×65 run).

    Returns
    -------
    dict with keys:
        "rho"     : pd.DataFrame (dims × N_metrics, Spearman rho)
        "p_raw"   : pd.DataFrame (dims × N_metrics, raw p-values)
        "p_fdr"   : pd.DataFrame (dims × N_metrics, FDR-corrected p-values)
        "significant": pd.DataFrame (bool mask)
    """
    if not segment_names:
        raise ValueError("No segment names provided for embedding correlation")

    pose_records_dir = Path(pose_records_dir)
    embeddings_dir = Path(embeddings_dir)

    emb_blocks: list[np.ndarray] = []
    kin_blocks: list[np.ndarray] = []
    discovered_metrics: list[str] | None = list(metric_columns) if metric_columns else None
    n_segments_loaded = 0
    n_frames_total = 0

    logger.info(
        "Embedding analysis: loading frame rows for %d segments...",
        len(segment_names)
    )

    for seg_name in segment_names:
        parts = seg_name.rsplit("_seg_", maxsplit=1)
        if len(parts) != 2:
            logger.debug("Skipping unparseable segment name: %s", seg_name)
            continue

        seg_root, seg_num = parts
        emb_dir = embeddings_dir / seg_name
        seg_dir = (
            pose_records_dir
            / f"results_skeleton_{seg_root}"
            / "segments"
            / f"seg_{seg_num}"
        )

        emb_df = _load_segment_frame_embeddings(emb_dir)
        if emb_df is None or emb_df.empty:
            continue

        kin_df = _load_segment_frame_metrics(
            seg_dir,
            use_normalized=use_normalized,
            metric_cols=metric_columns,
        )
        if kin_df is None or kin_df.empty or "frame" not in kin_df.columns:
            continue

        kin_df = kin_df.copy()
        kin_df["frame"] = pd.to_numeric(kin_df["frame"], errors="coerce")
        kin_df = kin_df.dropna(subset=["frame"])
        if kin_df.empty:
            continue
        kin_df["frame"] = kin_df["frame"].astype(int)
        kin_df = kin_df.set_index("frame")

        if discovered_metrics is None:
            discovered_metrics = list(kin_df.columns)
            logger.info(
                "Embedding analysis: discovered %d frame-level kinematic metrics from %s",
                len(discovered_metrics), seg_dir,
            )
        else:
            kin_df = kin_df[[c for c in discovered_metrics if c in kin_df.columns]]

        if not discovered_metrics or kin_df.empty:
            continue

        common_frames = emb_df.index.intersection(kin_df.index)
        if len(common_frames) == 0:
            if len(emb_df) != len(kin_df):
                logger.debug(
                    "No shared frame index for %s and lengths differ (%d vs %d)",
                    seg_name, len(emb_df), len(kin_df),
                )
                continue
            emb_aligned = emb_df.reset_index(drop=True)
            kin_aligned = kin_df.reset_index(drop=True)
        else:
            emb_aligned = emb_df.loc[common_frames]
            kin_aligned = kin_df.loc[common_frames]

        if emb_aligned.empty or kin_aligned.empty:
            continue

        emb_blocks.append(emb_aligned.to_numpy(dtype=np.float32, copy=False))
        kin_blocks.append(kin_aligned.to_numpy(dtype=np.float32, copy=False))
        n_segments_loaded += 1
        n_frames_total += len(emb_aligned)

    if not emb_blocks or not kin_blocks or not discovered_metrics:
        raise ValueError("No common frame-level embedding/kinematic rows could be loaded")

    emb = np.concatenate(emb_blocks, axis=0)
    kin = np.concatenate(kin_blocks, axis=0)
    valid_metrics = discovered_metrics

    n_embedding_dims = emb.shape[1]
    if n_embedding_dims != N_EMBEDDING_DIMS:
        logger.warning(
            "Embedding analysis: expected %d dims but loaded %d",
            N_EMBEDDING_DIMS, n_embedding_dims,
        )

    if max_common_frames is not None and max_common_frames > 0 and len(emb) > max_common_frames:
        rng = np.random.default_rng(sampling_seed)
        sampled_idx = rng.choice(len(emb), size=max_common_frames, replace=False)
        emb = emb[sampled_idx]
        kin = kin[sampled_idx]
        logger.info(
            "Embedding × kinematics: sampled %d / %d common frame rows",
            len(emb), n_frames_total,
        )

    logger.info(
        "Embedding × kinematics: %d segments, %d frame rows, %d dims × %d metrics = %d tests",
        n_segments_loaded, len(emb), n_embedding_dims, len(valid_metrics),
        n_embedding_dims * len(valid_metrics)
    )

    # Compute Spearman correlations using rank-based approach for speed
    from scipy.stats import spearmanr as _spearmanr

    rho_matrix = np.zeros((n_embedding_dims, len(valid_metrics)), dtype=np.float32)
    p_matrix = np.ones((n_embedding_dims, len(valid_metrics)), dtype=np.float32)

    for j, metric in enumerate(valid_metrics):
        y = kin[:, j]
        valid_mask = np.isfinite(y)
        if valid_mask.sum() < 10:
            logger.debug("Metric '%s': only %d valid samples, skipping", metric, valid_mask.sum())
            continue

        y_valid = y[valid_mask]
        emb_valid = emb[valid_mask, :]  # (N_valid, 128)

        for i in range(n_embedding_dims):
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

    dim_labels = [str(i) for i in range(n_embedding_dims)]
    rho_df = pd.DataFrame(rho_matrix, index=dim_labels, columns=valid_metrics)
    p_raw_df = pd.DataFrame(p_matrix, index=dim_labels, columns=valid_metrics)
    p_fdr_df = pd.DataFrame(p_fdr_matrix, index=dim_labels, columns=valid_metrics)
    sig_df = pd.DataFrame(sig_matrix, index=dim_labels, columns=valid_metrics)

    rho_df.index.name = "embedding_dim"
    n_sig = int(sig_matrix.sum())
    logger.info(
        "Embedding × kinematics: %d / %d pairs significant after FDR (alpha=%.2f)",
        n_sig, n_embedding_dims * len(valid_metrics), alpha
    )

    return {
        "rho": rho_df,
        "p_raw": p_raw_df,
        "p_fdr": p_fdr_df,
        "significant": sig_df,
    }
