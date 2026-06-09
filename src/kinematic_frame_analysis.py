"""
Frame-level kinematic × cluster analysis.

For each frame in the cluster mapping, loads the corresponding per-frame
kinematic metrics from `metrics_normalised.csv` (or `metrics_summary.csv`
for raw values) and aggregates per cluster.

Unlike the segment-level kinematic analysis (kinematic_analysis.py) which
uses segment-aggregate statistics, this module aligns individual frames
with their cluster assignments, giving a more accurate per-cluster kinematic
profile.

Outputs
-------
- cluster_kinematic_frame_profiles.csv
    Per-cluster robust profile statistics for all kinematic metrics, with frame counts.
- kinematic_frame_kruskal.csv
    Kruskal-Wallis H-statistic, p-value, and effect size per metric (clusters as groups).
    FDR-corrected p-values (BH method) included.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import pickle

from .stats import fdr_correct

logger = logging.getLogger(__name__)

POSE_RECORD_PREFIX = "results_skeleton_"
NORM_METRICS_FILENAME = "metrics_normalised.csv"
RAW_METRICS_FILENAME = "metrics_summary.csv"


def _tukey_filter(values: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Filter values using Tukey fences with configurable factor k."""
    if values.size == 0:
        return values

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if not np.isfinite(iqr):
        return np.array([], dtype=float)
    if iqr == 0:
        # Keep constant-valued distributions unchanged.
        return values

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    ret = values[(values >= lower) & (values <= upper)]
    # percent_filtered = 100.0 * (1.0 - len(ret) / len(values))
    # logger.info("Tukey filtered %0.2f%% of values", percent_filtered)
    return ret


def _compute_metric_distribution_stats(
    values: np.ndarray,
    tukey_fence_k: float = 3.0,
) -> dict[str, float]:
    """
    Compute robust distribution statistics for one metric.

    Skewness and kurtosis are computed on Tukey-filtered values.
    If filtered variance is zero, skewness/kurtosis return NaN.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "median": float("nan"),
            "p95": float("nan"),
            "p05": float("nan"),
            "skew": float("nan"),
            "kurtosis": float("nan"),
        }



    filtered = _tukey_filter(finite, k=tukey_fence_k)
    if filtered.size < 3 or float(np.var(filtered)) == 0.0:
        return {
            "median": float("nan"),
            "p95": float("nan"),
            "p05": float("nan"),
            "skew": float("nan"),
            "kurtosis": float("nan"),
        }
    else:
        median = float(np.median(finite))
        p95 = float(np.percentile(finite, 95))
        p05 = float(np.percentile(finite, 5))
        skew = float(stats.skew(filtered, bias=False))
        kurtosis = float(stats.kurtosis(filtered, fisher=True, bias=False))

    return {
        "median": median,
        "p95": p95,
        "p05": p05,
        "skew": skew,
        "kurtosis": kurtosis,
    }


def _load_segment_frame_metrics(
    seg_dir: Path,
    use_normalized: bool = True,
    metric_cols: list[str] | None = None,
) -> pd.DataFrame | None:
    """
    Load per-frame kinematic metrics for a single segment.

    Returns a DataFrame with columns ['frame', <metric_cols...>] or None on failure.
    """
    if use_normalized:
        csv_path = seg_dir / NORM_METRICS_FILENAME
    else:
        csv_path = seg_dir / RAW_METRICS_FILENAME

    if not csv_path.exists():
        # Fallback: try the other file
        alt = seg_dir / (RAW_METRICS_FILENAME if use_normalized else NORM_METRICS_FILENAME)
        if alt.exists():
            logger.debug("Primary metrics file not found in %s, using fallback %s", seg_dir, alt.name)
            csv_path = alt
        else:
            return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", csv_path, exc)
        return None

    # Detect frame index column
    frame_col = None
    for candidate in ("frame", "index", "frame_index", "Frame"):
        if candidate in df.columns:
            frame_col = candidate
            break
    if frame_col is None:
        # Assume the first column is the frame index
        frame_col = df.columns[0]

    df = df.rename(columns={frame_col: "frame"})

    # Keep only requested metrics (plus frame)
    if metric_cols:
        keep = ["frame"] + [c for c in metric_cols if c in df.columns]
        df = df[keep]

    return df.reset_index(drop=True)


def run_kinematic_frame_analysis(
    cluster_mapping: pd.DataFrame,
    pose_records_dir: str | Path,
    subject_session_ids: list[str],
    use_normalized: bool = True,
    metric_cols: list[str] | None = None,
    min_frames_per_cluster: int = 100,
    min_valid_ratio_per_metric: float = 0.6,
    tukey_fence_k: float = 3.0,
    kruskal_max_samples_per_cluster: int = 5000,
    fdr_method: str = "bh",
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """
    Compute frame-level per-cluster kinematic profiles.

    Iterates over segments, loads per-frame metrics, joins with cluster
    assignments, and accumulates per-cluster statistics. Memory-efficient:
    processes one segment at a time using running sum / sum-of-squares.

    Parameters
    ----------
    cluster_mapping : pd.DataFrame
        Must have columns: segment_name, index (relative frame), cluster_id.
    pose_records_dir : Path
        Root directory of pose record folders.
    subject_session_ids : list[str]
        Subject-session IDs like "7797_T1a_ADOS1".
    use_normalized : bool
        If True, load metrics_normalised.csv; else metrics_summary.csv.
    metric_cols : list[str] | None
        Subset of metric columns to load. None = load all.
    min_frames_per_cluster : int
        Clusters with fewer frames are flagged (but still included).
    min_valid_ratio_per_metric : float
        Minimum valid frame ratio required to keep a metric in a cluster.
        Metrics below this threshold are marked excluded and all stats are NaN.
    tukey_fence_k : float
        Tukey fence factor used for robust skewness/kurtosis estimation.
    kruskal_max_samples_per_cluster : int
        Maximum number of frame samples retained per cluster for Kruskal-Wallis.
    fdr_method : str
        FDR correction method for Kruskal-Wallis p-values.
    alpha : float
        Significance threshold.

    Returns
    -------
    dict with keys:
        "profiles" : pd.DataFrame (cluster_id × metrics, robust summary stats)
        "kruskal"  : pd.DataFrame (metric × H/p_raw/epsilon_sq/p_fdr/significant)
    """
    pose_records_dir = Path(pose_records_dir)

    # Build fast lookup: (segment_name, rel_frame) → cluster_id
    logger.info("Building frame→cluster lookup for frame-level kinematic analysis...")
    frame_to_cluster: dict[tuple[str, int], int] = {
        (row.segment_name, int(row.index)): int(row.cluster_id)
        for row in cluster_mapping.itertuples()
    }
    logger.info("Frame→cluster lookup: %d entries", len(frame_to_cluster))

    # Running accumulators: cluster_id → aggregate arrays per metric.
    # For distributional stats / Kruskal-Wallis, keep a bounded reservoir sample.
    MAX_SAMPLES_PER_CLUSTER = int(kruskal_max_samples_per_cluster)
    rng = np.random.default_rng(42)

    cluster_sum: dict[int, np.ndarray] = {}
    cluster_sum_sq: dict[int, np.ndarray] = {}
    cluster_valid_count: dict[int, np.ndarray] = {}
    cluster_count: dict[int, int] = {}
    cluster_samples: dict[int, list[np.ndarray]] = {}  # for Kruskal-Wallis
    cluster_seen: dict[int, int] = {}

    discovered_metric_cols: list[str] | None = None
    n_segments_loaded = 0
    n_segments_skipped = 0

    for ss_id in subject_session_ids:
        record_dir = pose_records_dir / f"{POSE_RECORD_PREFIX}{ss_id}"
        seg_parent = record_dir / "segments"

        if not seg_parent.is_dir():
            continue

        for seg_dir in sorted(seg_parent.glob("seg_*")):
            seg_label = seg_dir.name  # "seg_001"
            seg_num = seg_label.replace("seg_", "")
            segment_name = f"{ss_id}_seg_{seg_num}"

            # Check if this segment has any cluster assignments
            seg_frames = cluster_mapping[cluster_mapping["segment_name"] == segment_name]
            if seg_frames.empty:
                continue

            metrics_df = _load_segment_frame_metrics(seg_dir, use_normalized, metric_cols)
            if metrics_df is None or metrics_df.empty:
                n_segments_skipped += 1
                continue

            # Discover metric columns on first loaded segment
            if discovered_metric_cols is None:
                discovered_metric_cols = [c for c in metrics_df.columns if c != "frame"]
                logger.info(
                    "Frame-level kinematic metrics: %d columns discovered from %s",
                    len(discovered_metric_cols), seg_dir
                )

            n_metrics = len(discovered_metric_cols)

            # Join metrics with cluster assignments via frame index
            metrics_df = metrics_df.set_index("frame")
            # Keep only numeric columns
            metrics_df = metrics_df[[c for c in discovered_metric_cols if c in metrics_df.columns]]
            metrics_df = metrics_df.apply(pd.to_numeric, errors="coerce")

            # For each frame in this segment that has a cluster assignment:
            seg_cluster_map = seg_frames.set_index("index")["cluster_id"].to_dict()

            for rel_frame, cluster_id in seg_cluster_map.items():
                rel_frame = int(rel_frame)
                cluster_id = int(cluster_id)

                if rel_frame not in metrics_df.index:
                    continue

                row_vals = metrics_df.loc[rel_frame].to_numpy(dtype=np.float32)
                if np.all(np.isnan(row_vals)):
                    continue

                if cluster_id not in cluster_sum:
                    cluster_sum[cluster_id] = np.zeros(n_metrics, dtype=np.float64)
                    cluster_sum_sq[cluster_id] = np.zeros(n_metrics, dtype=np.float64)
                    cluster_valid_count[cluster_id] = np.zeros(n_metrics, dtype=np.int64)
                    cluster_count[cluster_id] = 0
                    cluster_samples[cluster_id] = []
                    cluster_seen[cluster_id] = 0

                valid_mask = ~np.isnan(row_vals)
                vals_filled = np.where(valid_mask, row_vals, 0.0)
                cluster_sum[cluster_id] += vals_filled
                cluster_sum_sq[cluster_id] += vals_filled ** 2
                cluster_valid_count[cluster_id] += valid_mask.astype(np.int64)
                cluster_count[cluster_id] += 1

                # Reservoir sampling to avoid early-frame bias.
                cluster_seen[cluster_id] += 1
                seen = cluster_seen[cluster_id]
                if len(cluster_samples[cluster_id]) < MAX_SAMPLES_PER_CLUSTER:
                    cluster_samples[cluster_id].append(row_vals.copy())
                else:
                    replace_idx = int(rng.integers(0, seen))
                    if replace_idx < MAX_SAMPLES_PER_CLUSTER:
                        cluster_samples[cluster_id][replace_idx] = row_vals.copy()

            n_segments_loaded += 1

    logger.info(
        "Frame-level kinematic analysis: %d segments loaded, %d skipped (no metrics file)",
        n_segments_loaded, n_segments_skipped
    )

    if not cluster_sum or discovered_metric_cols is None:
        logger.warning("No frame-level kinematic data collected.")
        return {"profiles": pd.DataFrame(), "kruskal": pd.DataFrame()}

    n_metrics = len(discovered_metric_cols)

    # save kinemtics selected for debugging
    with open("debug_selected_kinematic_metrics.pkl", "wb") as f:
        pickle.dump(discovered_metric_cols, f)


    # ── Build profile DataFrame ─────────────────────────────────────────────
    profile_rows = []
    for cluster_id in sorted(cluster_sum.keys()):
        n = cluster_count[cluster_id]
        if n == 0:
            continue
        s = cluster_sum[cluster_id]
        s2 = cluster_sum_sq[cluster_id]
        valid_n = cluster_valid_count[cluster_id].astype(np.float64)

        mean = np.full(n_metrics, np.nan, dtype=np.float64)
        std = np.full(n_metrics, np.nan, dtype=np.float64)
        valid_nonzero = valid_n > 0
        mean[valid_nonzero] = s[valid_nonzero] / valid_n[valid_nonzero]
        # Variance via E[X^2] - E[X]^2 on valid observations only
        var = np.full(n_metrics, np.nan, dtype=np.float64)
        var[valid_nonzero] = np.maximum(
            s2[valid_nonzero] / valid_n[valid_nonzero] - mean[valid_nonzero] ** 2,
            0.0,
        )
        std[valid_nonzero] = np.sqrt(var[valid_nonzero])

        sampled_arr = np.stack(cluster_samples[cluster_id], axis=0) if cluster_samples[cluster_id] else np.empty((0, n_metrics), dtype=np.float32)

        row: dict = {"cluster_id": cluster_id, "n_frames": n,
                     "flagged_low_frames": n < min_frames_per_cluster}
        for j, col in enumerate(discovered_metric_cols):
            valid_ratio = float(valid_n[j] / n) if n > 0 else 0.0
            excluded = valid_ratio < min_valid_ratio_per_metric

            row[f"{col}__valid_ratio"] = valid_ratio
            row[f"{col}__excluded_missing"] = excluded

            if excluded:
                row[f"{col}__mean"] = float("nan")
                row[f"{col}__std"] = float("nan")
                row[f"{col}__median"] = float("nan")
                row[f"{col}__p95"] = float("nan")
                row[f"{col}__p05"] = float("nan")
                row[f"{col}__skew"] = float("nan")
                row[f"{col}__kurtosis"] = float("nan")
                continue

            dist_stats = _compute_metric_distribution_stats(
                sampled_arr[:, j] if sampled_arr.shape[0] > 0 else np.array([], dtype=float),
                tukey_fence_k=tukey_fence_k,
            )
            row[f"{col}__mean"] = float(mean[j])
            row[f"{col}__std"] = float(std[j])
            row[f"{col}__median"] = dist_stats["median"]
            row[f"{col}__p95"] = dist_stats["p95"]
            row[f"{col}__p05"] = dist_stats["p05"]
            row[f"{col}__skew"] = dist_stats["skew"]
            row[f"{col}__kurtosis"] = dist_stats["kurtosis"]
        profile_rows.append(row)

    profiles_df = pd.DataFrame(profile_rows).set_index("cluster_id")

    n_flagged = profiles_df["flagged_low_frames"].sum()
    if n_flagged > 0:
        logger.warning(
            "%d clusters have fewer than %d frames in frame-level analysis (flagged)",
            n_flagged, min_frames_per_cluster
        )

    logger.info(
        "Frame-level kinematic profiles: %d clusters × %d metrics",
        len(profiles_df), n_metrics
    )

    # ── Kruskal-Wallis per metric ───────────────────────────────────────────
    logger.info("Running Kruskal-Wallis tests for %d metrics across %d clusters...",
                n_metrics, len(cluster_samples))

    kruskal_rows = []
    for j, metric in enumerate(discovered_metric_cols):
        groups = []
        for cluster_id in sorted(cluster_samples.keys()):
            samples = cluster_samples[cluster_id]
            if not samples:
                continue
            arr = np.stack(samples, axis=0)[:, j]
            arr = arr[~np.isnan(arr)]
            if len(arr) >= 3:
                groups.append(arr)

        if len(groups) < 2:
            kruskal_rows.append({
                "metric": metric, "H": float("nan"), "p_raw": float("nan"),
                "epsilon_sq": float("nan"),
                "p_fdr": float("nan"), "significant": False,
            })
            continue

        try:
            H, p = stats.kruskal(*groups)
        except Exception as exc:
            logger.debug("Kruskal-Wallis failed for %s: %s", metric, exc)
            H, p = float("nan"), float("nan")

        # Kruskal effect size (epsilon-squared): (H - k + 1) / (N - k)
        # with k groups and N total observations across groups.
        k_groups = len(groups)
        n_total = int(sum(len(g) for g in groups))
        if np.isfinite(H) and n_total > k_groups:
            epsilon_sq = float(max(0.0, (H - k_groups + 1.0) / (n_total - k_groups)))
        else:
            epsilon_sq = float("nan")

        kruskal_rows.append({"metric": metric, "H": H, "p_raw": p, "epsilon_sq": epsilon_sq})

    kruskal_df = pd.DataFrame(kruskal_rows)

    # FDR correction on valid p-values
    if not kruskal_df.empty:
        valid_mask = kruskal_df["p_raw"].notna()
        p_vals = kruskal_df.loc[valid_mask, "p_raw"].to_numpy()
        if len(p_vals) > 0:
            p_fdr = fdr_correct(p_vals, method=fdr_method)
            kruskal_df.loc[valid_mask, "p_fdr"] = p_fdr
            kruskal_df["significant"] = kruskal_df["p_fdr"] < alpha
        else:
            kruskal_df["p_fdr"] = float("nan")
            kruskal_df["significant"] = False

    n_sig = int(kruskal_df["significant"].sum()) if "significant" in kruskal_df.columns else 0
    logger.info(
        "Kruskal-Wallis: %d / %d metrics significant after FDR correction (alpha=%.2f)",
        n_sig, len(kruskal_df), alpha
    )

    return {
        "profiles": profiles_df,
        "kruskal": kruskal_df.set_index("metric") if not kruskal_df.empty else kruskal_df,
    }
