"""
Data loading utilities for the clusterAnalysis pipeline.

Each loader function:
- logs what it loaded (shape, N missing, N skipped)
- warns on data quality issues
- raises informative errors on hard failures
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
METRICS_SUMMARY_FILENAME = "metrics_summary.csv"
POSE_RECORD_PREFIX = "results_skeleton_"


def _extract_segment_prefix(results_path: object) -> str:
    """
    Extract the segment-name prefix from a results_path value.

    E.g. ".../results_skeleton_7797_T1a_ADOS1.json" → "7797_T1a_ADOS1"
         ".../results_skeleton_V012.json"            → "V012"
    Returns "" for missing / unparseable values.
    """
    if not isinstance(results_path, str) or not results_path.strip():
        return ""
    try:
        stem = Path(results_path).stem  # drop .json
        if stem.startswith(POSE_RECORD_PREFIX):
            return stem[len(POSE_RECORD_PREFIX):]
        return stem
    except Exception:
        return ""


# ── Cluster mapping ──────────────────────────────────────────────────────────

def load_cluster_mapping(path: str | Path) -> pd.DataFrame:
    """
    Load the frame-level cluster assignment file.

    Expected columns: index, segment_name, segment_id, cluster_id.

    Returns
    -------
    pd.DataFrame with columns [index, segment_name, segment_id, cluster_id].
    cluster_id is stored as int16 to save memory.

    Raises
    ------
    FileNotFoundError, ValueError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cluster mapping not found: {path}")

    df = pd.read_csv(path)

    required = {"index", "segment_name", "segment_id", "cluster_id"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Cluster mapping missing columns: {missing_cols}")

    nan_count = df["cluster_id"].isna().sum()
    if nan_count > 0:
        logger.warning("Cluster mapping: %d rows with NaN cluster_id (%.2f%%)",
                       nan_count, 100 * nan_count / len(df))

    df["cluster_id"] = df["cluster_id"].astype("int16")
    df["segment_id"] = df["segment_id"].astype("int32")
    df["index"] = df["index"].astype("int32")

    n_segments = df["segment_name"].nunique()
    n_clusters = df["cluster_id"].nunique()
    logger.info(
        "Cluster mapping loaded: %d frames, %d segments, %d clusters from %s",
        len(df), n_segments, n_clusters, path
    )
    return df


# ── Clinical metrics ─────────────────────────────────────────────────────────

def load_clinical(path: str | Path) -> pd.DataFrame:
    """
    Load the clinical metrics CSV.

    Expected to contain columns: uuid, code, diagnosis, and various ADOS/MSEL/Vineland columns.
    Indexed by 'uuid'.

    Returns
    -------
    pd.DataFrame indexed by uuid (string).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {path}")

    df = pd.read_csv(path)

    if "uuid" not in df.columns:
        raise ValueError("Clinical CSV must have a 'uuid' column")
    if "code" not in df.columns:
        logger.warning("Clinical CSV has no 'code' column — annotation join will fail")

    # Keep uuid as-is (it may be "7797_Visite1_Recherche" or similar full strings).
    # Do NOT try to normalize — the join is done via results_path, not uuid.
    df = df.set_index("uuid")

    # Add segment_prefix: the part of results_path that matches segment_name prefixes.
    # E.g. ".../results_skeleton_7797_T1a_ADOS1.json" → "7797_T1a_ADOS1"
    if "results_path" in df.columns:
        df["segment_prefix"] = df["results_path"].apply(_extract_segment_prefix)
        n_with_prefix = (df["segment_prefix"] != "").sum()
        logger.info(
            "Clinical data: %d / %d records have a parseable segment_prefix from results_path",
            n_with_prefix, len(df)
        )
    else:
        logger.warning("Clinical CSV has no 'results_path' column — segment_prefix join will fail")

    # Log missing values per column
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        logger.warning(
            "Clinical CSV missing values:\n%s",
            missing.to_string()
        )

    n_asd = (df.get("diagnosis", pd.Series()) == "ASD").sum()
    n_td = (df.get("diagnosis", pd.Series()) == "TD").sum()
    logger.info(
        "Clinical data loaded: %d subjects (%d ASD, %d TD) from %s",
        len(df), n_asd, n_td, path
    )
    return df


# ── Annotations ──────────────────────────────────────────────────────────────

def load_annotations(path: str | Path) -> pd.DataFrame:
    """
    Load expert behavioral annotations.

    Expected columns: code, behavior, behavioral_category, modifier_1,
                      start, status, stop, duration.

    Sanity checks:
    - Rows where start > stop are dropped with a warning.
    - Rows with NaN start are dropped with a warning.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {path}")

    df = pd.read_csv(path)

    required = {"code", "behavior", "start", "status"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Annotations CSV missing columns: {missing_cols}")

    n_initial = len(df)

    # Drop rows with NaN start
    nan_start = df["start"].isna().sum()
    if nan_start > 0:
        logger.warning("Dropping %d annotation rows with NaN 'start'", nan_start)
        df = df.dropna(subset=["start"])

    # Drop rows where start > stop (only for START status events with stop)
    if "stop" in df.columns:
        invalid_mask = (
            df["status"].str.upper() == "START"
        ) & df["stop"].notna() & (df["start"] > df["stop"])
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            logger.warning(
                "Dropping %d annotation rows where start > stop", n_invalid
            )
            df = df[~invalid_mask]

    n_dropped = n_initial - len(df)
    if n_dropped > 0:
        logger.warning("Annotations: %d rows dropped (total %d → %d)", n_dropped, n_initial, len(df))

    n_codes = df["code"].nunique()
    n_behaviors = df["behavior"].nunique()
    n_per_code = df.groupby("code").size()
    logger.info(
        "Annotations loaded: %d events, %d unique codes (V-records), %d unique behaviors from %s",
        len(df), n_codes, n_behaviors, path
    )
    logger.debug("Events per code:\n%s", n_per_code.to_string())
    return df


# ── Kinematics summary ───────────────────────────────────────────────────────

def load_kinematics_summary(
    pose_records_dir: str | Path,
    subject_session_ids: list[str],
    use_normalized: bool = True,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load per-segment kinematic summaries for a list of subject-session IDs.

    Each subject-session directory is expected at:
        <pose_records_dir>/results_skeleton_<subject_session_id>/segments/seg_XXX/metrics_summary.csv

    Parameters
    ----------
    pose_records_dir : Path
        Root directory containing pose record folders.
    subject_session_ids : list[str]
        IDs like "7797_T1a_ADOS1" (without "results_skeleton_" prefix).
    use_normalized : bool
        If True, use norm_mean/norm_std columns; otherwise use raw_mean/raw_std.
    metrics : list[str] | None
        Subset of metrics to load. None = load all.

    Returns
    -------
    pd.DataFrame with MultiIndex (subject_session, seg_label) and one column per metric.
    """
    pose_records_dir = Path(pose_records_dir)
    prefix = "norm" if use_normalized else "raw"
    mean_col = f"{prefix}_mean"
    std_col = f"{prefix}_std"

    records = []
    skipped = []

    for ss_id in subject_session_ids:
        record_dir = pose_records_dir / f"{POSE_RECORD_PREFIX}{ss_id}"
        if not record_dir.is_dir():
            logger.warning("Pose record directory not found: %s", record_dir)
            skipped.append(ss_id)
            continue

        seg_dirs = sorted((record_dir / "segments").glob("seg_*"))
        if not seg_dirs:
            logger.warning("No segment subdirectories found under %s", record_dir / "segments")
            skipped.append(ss_id)
            continue

        for seg_dir in seg_dirs:
            csv_path = seg_dir / METRICS_SUMMARY_FILENAME
            if not csv_path.exists():
                logger.debug("Missing %s in %s", METRICS_SUMMARY_FILENAME, seg_dir)
                continue

            try:
                summary = pd.read_csv(csv_path, index_col=0)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", csv_path, exc)
                continue

            if mean_col not in summary.columns:
                logger.warning(
                    "Column '%s' not in %s — available: %s",
                    mean_col, csv_path, list(summary.columns)
                )
                continue

            row = summary[mean_col].rename(lambda m: f"{m}__mean")
            if std_col in summary.columns:
                row_std = summary[std_col].rename(lambda m: f"{m}__std")
                row = pd.concat([row, row_std])

            if metrics is not None:
                keep = [c for c in row.index if any(c.startswith(m) for m in metrics)]
                row = row[keep]

            row.name = (ss_id, seg_dir.name)
            records.append(row)

    if not records:
        raise ValueError(
            f"No kinematics data loaded for any of {len(subject_session_ids)} subject-sessions"
        )

    df = pd.DataFrame(records)
    df.index = pd.MultiIndex.from_tuples(df.index, names=["subject_session", "seg_label"])

    nan_frac = df.isna().mean()
    high_nan = nan_frac[nan_frac > 0.5]
    if len(high_nan) > 0:
        logger.warning(
            "Kinematic metrics with >50%% NaN across segments:\n%s",
            high_nan.to_string()
        )

    if skipped:
        logger.warning(
            "Kinematics: skipped %d / %d subject-sessions (not found or empty): %s",
            len(skipped), len(subject_session_ids), skipped[:10]
        )

    logger.info(
        "Kinematics loaded: %d segments from %d subject-sessions, %d metrics (%s)",
        len(df), df.index.get_level_values("subject_session").nunique(),
        len(df.columns), "normalized" if use_normalized else "raw_2d"
    )
    return df
