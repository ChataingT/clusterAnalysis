"""
Kinematic analysis for the clusterAnalysis pipeline.

Two analyses:

1. Per-cluster kinematic profiles
   For each cluster, compute the weighted mean ± std of all kinematic metrics
   across segments where that cluster is dominant (from cross_video_train.csv).
   Uses the full dataset (all records with kinematics + clusters).

2. V-subset vs Non-V consistency check
   Repeat the same analysis separately for V-records and non-V records.
   Compare kinematic profiles to assess whether clusters have consistent
   kinematic signatures regardless of annotation status.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _build_segment_cluster_kin(
    cross_video_df: pd.DataFrame,
    kinematics_df: pd.DataFrame,
    subject_map: pd.DataFrame,
    min_frames: int = 100,
) -> pd.DataFrame:
    """
    Join per-segment dominant cluster with per-segment kinematic metrics.

    Parameters
    ----------
    cross_video_df : pd.DataFrame
        Output of cross_video_train.csv: columns include segment (int index),
        most_common_cluster, most_common_count, n_frames.
    kinematics_df : pd.DataFrame
        MultiIndex (subject_session, seg_label), columns = kinematic metrics.
    subject_map : pd.DataFrame
        Must have 'segment_name', 'uuid', 'subject_session', 'dominant_cluster', 'n_frames'.
    min_frames : int
        Minimum n_frames for a segment to be included.

    Returns
    -------
    pd.DataFrame: one row per segment, with columns:
        segment_name, subject_session, cluster_id, n_frames, is_vrecord,
        + all kinematic metric columns
    """
    # Merge subject_map with kinematics_df
    # kinematics_df index level "seg_label" is like "seg_001"
    # subject_map has "segment_name" like "7797_T1a_ADOS1_seg_001"

    # Extract seg_label from segment_name
    subject_map = subject_map.copy()
    subject_map["seg_label"] = subject_map["segment_name"].str.extract(r"_(seg_\d+)$")[0]

    # Reset kinematics_df index for merging
    kin_reset = kinematics_df.reset_index()  # columns: subject_session, seg_label, ...

    merged = subject_map.merge(
        kin_reset,
        on=["subject_session", "seg_label"],
        how="left",
    )

    missing_kin = merged[kin_reset.columns[2:]].isna().all(axis=1).sum()
    if missing_kin > 0:
        logger.warning(
            "%d / %d segments have no kinematic data after merge",
            missing_kin, len(merged)
        )

    # Filter by minimum frames
    merged = merged[merged["n_frames"] >= min_frames]
    logger.debug(
        "Segment-cluster-kinematics: %d segments after min_frames=%d filter",
        len(merged), min_frames
    )

    return merged


def compute_cluster_kinematic_profiles(
    segment_cluster_kin: pd.DataFrame,
    metric_columns: list[str],
    min_frames_per_cluster: int = 100,
) -> pd.DataFrame:
    """
    Compute weighted mean ± std of kinematic metrics per cluster.

    Parameters
    ----------
    segment_cluster_kin : pd.DataFrame
        Output of _build_segment_cluster_kin().
    metric_columns : list[str]
        Kinematic metric column names to include.
    min_frames_per_cluster : int
        Clusters with fewer total frames are flagged in the output.

    Returns
    -------
    pd.DataFrame indexed by cluster_id with columns:
        {metric}__mean, {metric}__std  for each metric, plus n_segments, total_frames, flagged.
    """
    valid_metrics = [c for c in metric_columns if c in segment_cluster_kin.columns]
    if not valid_metrics:
        raise ValueError("None of the requested metric columns found in segment data")

    rows = []
    for cluster_id, grp in segment_cluster_kin.groupby("dominant_cluster"):
        weights = grp["n_frames"].values.astype(float)
        total_frames = weights.sum()

        row: dict = {
            "cluster_id": cluster_id,
            "n_segments": len(grp),
            "total_frames": int(total_frames),
            "flagged": total_frames < min_frames_per_cluster,
        }

        for metric in valid_metrics:
            vals = grp[metric].values.astype(float)
            valid_mask = np.isfinite(vals)
            if valid_mask.sum() == 0:
                row[f"{metric}__mean"] = float("nan")
                row[f"{metric}__std"] = float("nan")
                continue

            w = weights[valid_mask]
            v = vals[valid_mask]
            wmean = np.average(v, weights=w)
            wvar = np.average((v - wmean) ** 2, weights=w)
            row[f"{metric}__mean"] = float(wmean)
            row[f"{metric}__std"] = float(np.sqrt(wvar))

        rows.append(row)

    if not rows:
        logger.warning("No cluster kinematic profiles computed")
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("cluster_id").sort_index()

    flagged = result["flagged"].sum()
    if flagged > 0:
        logger.warning(
            "%d / %d clusters flagged as low-frame-count (< %d frames)",
            flagged, len(result), min_frames_per_cluster
        )

    logger.info(
        "Cluster kinematic profiles: %d clusters × %d metrics",
        len(result), len(valid_metrics)
    )
    return result


def run_kinematic_analysis(
    subject_map: pd.DataFrame,
    kinematics_df: pd.DataFrame,
    metric_columns: list[str],
    v_uuids: set[str],
    min_frames_per_cluster: int = 100,
    min_frames_per_segment: int = 100,
) -> dict[str, pd.DataFrame]:
    """
    Run kinematic profiles for full dataset and V vs non-V subsets.

    Parameters
    ----------
    subject_map : pd.DataFrame
        Output of build_subject_map() with dominant_cluster, uuid, subject_session.
    kinematics_df : pd.DataFrame
        Output of load_kinematics_summary().
    metric_columns : list[str]
        Which kinematic metrics to analyze.
    v_uuids : set[str]
        Set of uuid strings for V-records (annotated subjects).
    min_frames_per_cluster : int
    min_frames_per_segment : int

    Returns
    -------
    dict with keys "global", "vsubset", "nonv"
    """
    seg_kin = _build_segment_cluster_kin(
        cross_video_df=None,      # unused — subject_map already has dominant_cluster
        kinematics_df=kinematics_df,
        subject_map=subject_map,
        min_frames=min_frames_per_segment,
    )

    if len(seg_kin) == 0:
        logger.warning("No segment-kinematics data available for kinematic analysis")
        return {}

    results: dict[str, pd.DataFrame] = {}

    # Global profiles
    logger.info("Computing kinematic profiles for all %d segments", len(seg_kin))
    results["global"] = compute_cluster_kinematic_profiles(
        seg_kin, metric_columns, min_frames_per_cluster
    )

    # V-subset
    v_mask = seg_kin["uuid"].isin(v_uuids)
    n_v = v_mask.sum()
    if n_v > 0:
        logger.info("Computing kinematic profiles for V-subset (%d segments)", n_v)
        results["vsubset"] = compute_cluster_kinematic_profiles(
            seg_kin[v_mask], metric_columns, min_frames_per_cluster
        )
    else:
        logger.warning("No V-record segments found for V-subset kinematic profile")

    # Non-V subset
    nonv_mask = ~v_mask
    n_nonv = nonv_mask.sum()
    if n_nonv > 0:
        logger.info("Computing kinematic profiles for non-V subset (%d segments)", n_nonv)
        results["nonv"] = compute_cluster_kinematic_profiles(
            seg_kin[nonv_mask], metric_columns, min_frames_per_cluster
        )

    # Consistency comparison: for clusters in both subsets, compute correlation of profiles
    if "vsubset" in results and "nonv" in results:
        common_clusters = results["vsubset"].index.intersection(results["nonv"].index)
        n_common = len(common_clusters)
        if n_common > 0:
            mean_cols = [c for c in results["global"].columns if c.endswith("__mean")]
            v_vals = results["vsubset"].loc[common_clusters, mean_cols].values.flatten()
            nonv_vals = results["nonv"].loc[common_clusters, mean_cols].values.flatten()
            valid = np.isfinite(v_vals) & np.isfinite(nonv_vals)
            if valid.sum() > 10:
                from scipy.stats import spearmanr
                rho, p = spearmanr(v_vals[valid], nonv_vals[valid])
                logger.info(
                    "V vs non-V kinematic consistency: Spearman rho=%.3f (p=%.3e) "
                    "across %d clusters × %d metrics",
                    rho, p, n_common, len(mean_cols)
                )
                if rho < 0.7:
                    logger.warning(
                        "Low V vs non-V kinematic consistency (rho=%.3f) — "
                        "cluster profiles may differ between annotated and unannotated records",
                        rho
                    )
        else:
            logger.warning("No clusters in common between V-subset and non-V subset profiles")

    return results
