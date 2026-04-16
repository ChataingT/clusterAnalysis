"""
Significance testing for the clusterAnalysis pipeline.

Two reusable tests:

1. Event-level permutation test for enrichment (annotation-cluster overlap)
    -------------------------------------------------------------------------
    H0: annotation labels are not temporally aligned with movement clusters.
    The unit of observation is the annotation EVENT — not the frame — because
    frames within an event are autocorrelated (a 5-second episode at 20 fps
    yields 100 dependent frames).

     Algorithm:
     - Pre-compute per-event cluster count vectors (n_events × n_clusters).
     - For each permutation: circularly shift labels within each subject-session
         timeline, preserving event durations and behavior sequence structure while
         breaking alignment to the observed movement timeline.
   - p-value: fraction of permutations where |enrichment_perm| ≥ |enrichment_obs|
     (two-sided). Effect size: log2(enrichment).
   - FDR correction: BH per annotation level (L1/L2/L3 separately).

2. Bootstrap CI for centroid distances
   ------------------------------------
   For each behavior, resample annotation events with replacement and
   recompute the centroid → distance to each cluster centroid.

   Reports:
   - 95% CI per (behavior, cluster) distance pair.
   - Nearest-cluster stability: fraction of bootstrap samples where the
     same cluster is the nearest neighbor (reliability metric for labeling).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .linking import annotation_to_frames, parse_subject_session
from .stats import fdr_correct, add_significance_flags

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class PermutationEnrichmentResult:
    """Results of an event-level permutation test for enrichment significance."""
    enrichment: pd.DataFrame         # behaviors × clusters (observed enrichment)
    log2_effect: pd.DataFrame        # log2(enrichment), behaviors × clusters
    p_raw: pd.DataFrame              # behaviors × clusters
    p_fdr: pd.DataFrame              # behaviors × clusters (BH per level)
    significant: pd.DataFrame        # bool, behaviors × clusters
    sig_label: pd.DataFrame          # */**/*** /ns, behaviors × clusters
    null_mean: pd.DataFrame          # mean null enrichment per cell
    null_std: pd.DataFrame           # std of null distribution per cell
    n_permutations: int
    n_events_per_label: pd.Series    # label → number of annotation events
    long_format: pd.DataFrame        # tidy long-format summary of all statistics


@dataclass
class BootstrapCentroidResult:
    """Results of a bootstrap confidence-interval test for centroid distances."""
    observed_distance: pd.DataFrame          # behaviors × clusters
    ci_low: pd.DataFrame                     # behaviors × clusters (alpha/2 percentile)
    ci_high: pd.DataFrame                    # behaviors × clusters (1-alpha/2 percentile)
    ci_width: pd.DataFrame                   # behaviors × clusters (ci_high - ci_low)
    nearest_cluster_stability: pd.DataFrame  # per behavior: nearest_cluster_id,
                                             # fraction_same_nearest, observed_distance
    n_bootstrap: int
    n_events_per_label: pd.Series            # label → number of annotation events
    long_format: pd.DataFrame                # tidy long-format summary
    distance_metric: str = "euclidean"       # metric used for centroid distances


# ── Pre-computation: per-event cluster count matrix ───────────────────────────

def build_event_cluster_matrix(
    cluster_mapping: pd.DataFrame,
    annotations_df: pd.DataFrame,
    segment_registry: dict,
    clinical_df: pd.DataFrame,
    fps: int = 20,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Pre-compute a per-event cluster count matrix for significance testing.

    This is the core pre-computation step for the permutation test. For each
    annotation event, we count how many of its frames fall in each cluster.
    The result is a dense matrix (n_events × n_clusters) that can be
    reused across all permutations without re-accessing the cluster mapping.

    Parameters
    ----------
    cluster_mapping : pd.DataFrame
        Frame-level cluster assignments (columns: segment_name, index, cluster_id).
    annotations_df : pd.DataFrame
        Annotation events. Must have 'behavior' and 'code' columns.
        The 'behavior' column is used as the label (rename before calling
        if using composite L2/L3 labels).
    segment_registry : dict
        Output of build_segment_registry().
    clinical_df : pd.DataFrame
        Used to map V-record code → subject_session prefix.
    fps : int
        Frames per second.

    Returns
    -------
    event_cluster_matrix : np.ndarray, shape (n_events, n_clusters)
        Row i = frame counts per cluster for event i.
    labels : np.ndarray of str, shape (n_events,)
        Behavior label for each event row.
    event_meta : pd.DataFrame
        Metadata per event row: code, label, subject_session,
        session_event_order, n_frames_matched.
    """
    # Build code → subject_session map
    code_to_ss: dict[str, str] = {}
    if "code" in clinical_df.columns:
        for uuid_str, row in clinical_df.iterrows():
            code = row.get("code")
            if not isinstance(code, str):
                continue
            session_segs = [s for s in segment_registry if s.startswith(str(uuid_str))]
            if not session_segs:
                session_segs = [s for s in segment_registry if s.startswith(code)]
            if session_segs:
                uuid, session = parse_subject_session(session_segs[0])
                code_to_ss[code] = f"{uuid}_{session}" if session else uuid

    # Build frame→cluster lookup
    logger.debug("Building frame→cluster lookup for event matrix...")
    frame_to_cluster: dict[tuple[str, int], int] = (
        cluster_mapping
        .set_index(["segment_name", "index"])["cluster_id"]
        .to_dict()
    )

    # Discover all cluster IDs (sorted for stable column order)
    all_cluster_ids = sorted(cluster_mapping["cluster_id"].dropna().unique().astype(int))
    cluster_to_col = {c: i for i, c in enumerate(all_cluster_ids)}
    n_clusters = len(all_cluster_ids)

    v_codes = sorted(annotations_df["code"].unique())
    logger.info(
        "Building event-cluster matrix: %d V-records, %d clusters",
        len(v_codes), n_clusters
    )

    rows: list[np.ndarray] = []
    labels_list: list[str] = []
    meta_rows: list[dict] = []
    n_skipped_no_ss = 0
    n_skipped_no_frames = 0

    for code in v_codes:
        ss = code_to_ss.get(code)
        if ss is None:
            n_skipped_no_ss += 1
            continue

        events = annotations_df[annotations_df["code"] == code].copy()
        if "start" in events.columns:
            sort_cols = ["start"]
            if "stop" in events.columns:
                sort_cols.append("stop")
            events = events.sort_values(sort_cols, kind="stable")

        for event_order, (_, ev) in enumerate(events.iterrows()):
            label = str(ev.get("behavior", "unknown"))

            frame_list = annotation_to_frames(
                ev, segment_registry, fps, subject_session_filter=ss
            )
            if not frame_list:
                n_skipped_no_frames += 1
                continue

            count_vec = np.zeros(n_clusters, dtype=np.float32)
            n_matched = 0
            for seg_name, rel_frame in frame_list:
                cid = frame_to_cluster.get((seg_name, rel_frame))
                if cid is not None:
                    count_vec[cluster_to_col[int(cid)]] += 1
                    n_matched += 1

            if n_matched == 0:
                n_skipped_no_frames += 1
                continue

            rows.append(count_vec)
            labels_list.append(label)
            meta_rows.append({
                "code": code,
                "label": label,
                "subject_session": ss,
                "session_event_order": int(event_order),
                "n_frames_matched": n_matched,
            })

    if n_skipped_no_ss > 0:
        logger.warning(
            "%d V-record codes had no subject-session match and were skipped",
            n_skipped_no_ss
        )
    if n_skipped_no_frames > 0:
        logger.warning(
            "%d annotation events had no matched frames and were skipped",
            n_skipped_no_frames
        )

    if not rows:
        logger.error("No events collected — event-cluster matrix is empty")
        return np.zeros((0, n_clusters)), np.array([]), pd.DataFrame(meta_rows)

    event_cluster_matrix = np.stack(rows, axis=0)  # (n_events, n_clusters)
    labels = np.array(labels_list, dtype=object)
    event_meta = pd.DataFrame(meta_rows)

    # Log per-label event counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info(
        "Event-cluster matrix built: %d events, %d labels, %d clusters",
        len(rows), len(unique_labels), n_clusters
    )
    for lbl, cnt in sorted(zip(unique_labels, counts), key=lambda x: -x[1]):
        logger.info("  %-40s  %3d events", lbl, cnt)

    return event_cluster_matrix, labels, event_meta


def _compute_enrichment_matrix(
    event_matrix: np.ndarray,
    labels: np.ndarray,
    global_cluster_frac: np.ndarray,
    unique_labels: list[str],
    min_events: int = 1,
) -> np.ndarray:
    """
    Compute the enrichment matrix (n_behaviors × n_clusters) from an event matrix.

    Parameters
    ----------
    event_matrix : np.ndarray, shape (n_events, n_clusters)
    labels : np.ndarray of str, shape (n_events,)
    global_cluster_frac : np.ndarray, shape (n_clusters,)
        Global cluster prevalence among V-record frames (the expected distribution).
    unique_labels : list[str]
        Ordered list of behavior labels (row order for output).
    min_events : int
        Rows with fewer events are filled with nan.

    Returns
    -------
    np.ndarray, shape (n_behaviors, n_clusters)
    """
    n_behaviors = len(unique_labels)
    n_clusters = event_matrix.shape[1]
    result = np.full((n_behaviors, n_clusters), np.nan, dtype=np.float32)

    for b_idx, label in enumerate(unique_labels):
        mask = labels == label
        n_ev = mask.sum()
        if n_ev < min_events:
            continue
        behavior_counts = event_matrix[mask].sum(axis=0)  # (n_clusters,)
        total = behavior_counts.sum()
        if total == 0:
            continue
        observed_frac = behavior_counts / total
        # enrichment = observed / expected; avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            enrichment = np.where(
                global_cluster_frac > 0,
                observed_frac / global_cluster_frac,
                np.nan,
            )
        result[b_idx] = enrichment

    return result


# ── Permutation test for enrichment ───────────────────────────────────────────

def permutation_test_enrichment(
    event_cluster_matrix: np.ndarray,
    labels: np.ndarray,
    event_meta: pd.DataFrame,
    global_cluster_counts: pd.Series,
    n_permutations: int = 1000,
    seed: int = 42,
    fdr_method: str = "bh",
    alpha: float = 0.05,
    min_events: int = 3,
) -> PermutationEnrichmentResult:
    """
    Event-level permutation test for enrichment significance.

    H0: annotation labels are not temporally aligned with movement clusters.
    Null draws are generated via circular shifts within each subject-session
    event timeline.
    The test statistic is the enrichment ratio per (behavior, cluster) pair.
    p-values are two-sided: the fraction of permutations where the absolute
    enrichment deviation from 1 equals or exceeds the observed deviation.

    FDR correction is applied across all (behavior × cluster) pairs jointly
    within this call; call separately per annotation level (L1/L2/L3).

    Parameters
    ----------
    event_cluster_matrix : np.ndarray, shape (n_events, n_clusters)
        Output of build_event_cluster_matrix() — frame counts per cluster per event.
    labels : np.ndarray, shape (n_events,)
        Behavior label for each event.
    event_meta : pd.DataFrame
        Metadata from build_event_cluster_matrix(). Must include
        `subject_session` and `session_event_order` columns.
    global_cluster_counts : pd.Series
        cluster_id → total frame count in V-records (the expected denominator).
        Index must align with the cluster column order in event_cluster_matrix.
    n_permutations : int
        Number of label shuffles. 1000 gives p-values down to ~0.001;
        use 5000 for higher precision.
    seed : int
        Random seed for reproducibility.
    fdr_method : str
        "bh" (Benjamini-Hochberg) or "bonferroni".
    alpha : float
        Significance threshold applied after FDR correction.
    min_events : int
        Behaviors with fewer events are excluded from the test.

    Returns
    -------
    PermutationEnrichmentResult
    """
    if len(labels) != event_cluster_matrix.shape[0]:
        raise ValueError(
            "labels length must match event_cluster_matrix rows "
            f"({len(labels)} != {event_cluster_matrix.shape[0]})"
        )
    if len(event_meta) != event_cluster_matrix.shape[0]:
        raise ValueError(
            "event_meta rows must match event_cluster_matrix rows "
            f"({len(event_meta)} != {event_cluster_matrix.shape[0]})"
        )

    required_meta_cols = {"subject_session", "session_event_order"}
    missing_cols = required_meta_cols - set(event_meta.columns)
    if missing_cols:
        raise ValueError(
            f"event_meta is missing required columns: {sorted(missing_cols)}"
        )

    rng = np.random.default_rng(seed)

    # Build deterministic session-wise event groups used by circular shifts.
    session_groups: list[np.ndarray] = []
    ss_series = event_meta["subject_session"].astype(str)
    order_series = pd.to_numeric(
        event_meta["session_event_order"], errors="coerce"
    ).fillna(0).astype(int)
    for ss in sorted(ss_series.unique()):
        idx = np.flatnonzero((ss_series == ss).to_numpy())
        if idx.size == 0:
            continue
        ord_idx = idx[np.argsort(order_series.to_numpy()[idx], kind="stable")]
        session_groups.append(ord_idx)

    n_shiftable_sessions = sum(g.size > 1 for g in session_groups)
    if n_shiftable_sessions == 0:
        raise ValueError(
            "Circular-shift permutation requires at least one subject-session "
            "with >= 2 matched events"
        )

    # Map global cluster counts to the column order of event_cluster_matrix
    n_clusters = event_cluster_matrix.shape[1]
    if len(global_cluster_counts) != n_clusters:
        logger.warning(
            "global_cluster_counts has %d entries but event_cluster_matrix has %d columns; "
            "reindexing — ensure cluster IDs are aligned",
            len(global_cluster_counts), n_clusters
        )
    global_total = global_cluster_counts.sum()
    global_frac = (global_cluster_counts / global_total).to_numpy(dtype=np.float32)

    # Determine unique labels (those passing min_events filter)
    unique_labels_all, event_counts = np.unique(labels, return_counts=True)
    valid_mask = event_counts >= min_events
    skipped = unique_labels_all[~valid_mask]
    if len(skipped):
        logger.info(
            "Permutation test: skipping %d labels with < %d events: %s",
            len(skipped), min_events, list(skipped)
        )
    unique_labels = list(unique_labels_all[valid_mask])
    n_events_per_label = pd.Series(
        dict(zip(unique_labels_all, event_counts)), name="n_events"
    )

    if not unique_labels:
        logger.error("No behaviors pass the min_events=%d threshold", min_events)
        empty = pd.DataFrame()
        return PermutationEnrichmentResult(
            enrichment=empty, log2_effect=empty, p_raw=empty, p_fdr=empty,
            significant=empty, sig_label=empty, null_mean=empty, null_std=empty,
            n_permutations=n_permutations,
            n_events_per_label=n_events_per_label,
            long_format=pd.DataFrame(),
        )

    n_behaviors = len(unique_labels)
    logger.info(
        "Permutation test (session circular-shift null): %d behaviors × %d clusters, "
        "%d permutations, %d subject-sessions (%d shiftable), seed=%d",
        n_behaviors, n_clusters, n_permutations,
        len(session_groups), n_shiftable_sessions, seed
    )

    # Observed enrichment
    observed = _compute_enrichment_matrix(
        event_cluster_matrix, labels, global_frac, unique_labels, min_events
    )  # (n_behaviors, n_clusters)

    # Deviation from 1.0 (expected under H0) for two-sided test
    obs_dev = np.abs(observed - 1.0)

    # Accumulate null distribution
    null_sum = np.zeros_like(observed, dtype=np.float64)
    null_sum_sq = np.zeros_like(observed, dtype=np.float64)
    exceed_count = np.zeros_like(observed, dtype=np.float64)

    def _circular_shift_labels() -> np.ndarray:
        perm = labels.copy()
        for g in session_groups:
            if g.size <= 1:
                continue
            shift = int(rng.integers(1, g.size))
            perm[g] = np.roll(labels[g], shift)
        return perm

    for i in range(n_permutations):
        perm_labels = _circular_shift_labels()
        perm_enrich = _compute_enrichment_matrix(
            event_cluster_matrix, perm_labels, global_frac, unique_labels, min_events
        )
        perm_dev = np.abs(perm_enrich - 1.0)

        # Count permutations where null deviation ≥ observed deviation
        exceed_count += (perm_dev >= obs_dev).astype(np.float64)

        # Accumulate for null mean/std
        valid_perm = np.isfinite(perm_enrich)
        null_sum = np.where(valid_perm, null_sum + perm_enrich, null_sum)
        null_sum_sq = np.where(valid_perm, null_sum_sq + perm_enrich ** 2, null_sum_sq)

        if (i + 1) % 250 == 0:
            logger.debug("Permutation progress: %d / %d", i + 1, n_permutations)

    logger.info("Permutation test complete (%d permutations)", n_permutations)

    # p-values: fraction of permutations exceeding or equaling observed
    p_raw_arr = exceed_count / n_permutations
    # Minimum non-zero p-value is 1/n_permutations
    p_raw_arr = np.clip(p_raw_arr, 1.0 / n_permutations, 1.0)

    # Null statistics
    null_mean_arr = null_sum / n_permutations
    null_std_arr = np.sqrt(
        np.maximum(null_sum_sq / n_permutations - (null_sum / n_permutations) ** 2, 0)
    )

    # log2 effect size
    log2_eff = np.where(
        np.isfinite(observed) & (observed > 0),
        np.log2(np.maximum(observed, 1e-6)),
        np.nan,
    ).astype(np.float32)

    # FDR correction over all (behavior × cluster) pairs (flatten → correct → reshape)
    flat_p = p_raw_arr.flatten()
    finite_mask = np.isfinite(flat_p)
    flat_p_fdr = np.full_like(flat_p, np.nan)
    if finite_mask.sum() > 0:
        flat_p_fdr[finite_mask] = fdr_correct(flat_p[finite_mask], method=fdr_method)
    p_fdr_arr = flat_p_fdr.reshape(p_raw_arr.shape)

    # Build DataFrames
    idx = pd.Index(unique_labels, name="behavior")
    cols = pd.Index(global_cluster_counts.index.tolist(), name="cluster_id")

    def _df(arr):
        return pd.DataFrame(arr, index=idx, columns=cols)

    enrichment_df = _df(observed)
    log2_df = _df(log2_eff)
    p_raw_df = _df(p_raw_arr.astype(np.float32))
    p_fdr_df = _df(p_fdr_arr.astype(np.float32))
    null_mean_df = _df(null_mean_arr.astype(np.float32))
    null_std_df = _df(null_std_arr.astype(np.float32))

    significant_df = p_fdr_df < alpha
    sig_label_arr = np.full(p_fdr_arr.shape, "ns", dtype=object)
    sig_label_arr[p_fdr_arr < 0.05] = "*"
    sig_label_arr[p_fdr_arr < 0.01] = "**"
    sig_label_arr[p_fdr_arr < 0.001] = "***"
    sig_label_df = _df(sig_label_arr)

    # Long-format tidy summary
    long_rows = []
    for b_idx, behavior in enumerate(unique_labels):
        for c_idx, cid in enumerate(cols):
            enr = float(observed[b_idx, c_idx])
            l2e = float(log2_eff[b_idx, c_idx])
            pr = float(p_raw_arr[b_idx, c_idx])
            pf = float(p_fdr_arr[b_idx, c_idx])
            long_rows.append({
                "behavior": behavior,
                "cluster_id": cid,
                "enrichment": enr,
                "log2_enrichment": l2e,
                "p_raw": pr,
                "p_fdr": pf,
                "significant": bool(pf < alpha) if np.isfinite(pf) else False,
                "sig_label": sig_label_arr[b_idx, c_idx],
                "n_events_behavior": int(n_events_per_label.get(behavior, 0)),
                "null_mean": float(null_mean_arr[b_idx, c_idx]),
                "null_std": float(null_std_arr[b_idx, c_idx]),
            })
    long_df = pd.DataFrame(long_rows)

    # Log summary
    n_total = np.isfinite(p_fdr_arr).sum()
    n_sig = int((p_fdr_arr < alpha).sum())
    logger.info(
        "Enrichment permutation test: %d (behavior, cluster) pairs tested, "
        "%d significant after FDR-%s (alpha=%.2f)",
        n_total, n_sig, fdr_method.upper(), alpha
    )
    # Per-behavior breakdown
    for behavior in unique_labels:
        row_fdr = p_fdr_df.loc[behavior]
        n_sig_b = int((row_fdr < alpha).sum())
        n_test_b = int(row_fdr.notna().sum())
        logger.info("  %-40s  %3d / %3d clusters significant", behavior, n_sig_b, n_test_b)

    return PermutationEnrichmentResult(
        enrichment=enrichment_df,
        log2_effect=log2_df,
        p_raw=p_raw_df,
        p_fdr=p_fdr_df,
        significant=significant_df,
        sig_label=sig_label_df,
        null_mean=null_mean_df,
        null_std=null_std_df,
        n_permutations=n_permutations,
        n_events_per_label=n_events_per_label,
        long_format=long_df,
    )


# ── Bootstrap CI for centroid distances ───────────────────────────────────────

def _vec_to_mat_distances(
    vec: np.ndarray,
    mat: np.ndarray,
    metric: str = "euclidean",
    cov_inv: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute distances from a single vector to each row of a matrix.

    Parameters
    ----------
    vec : np.ndarray, shape (d,)
    mat : np.ndarray, shape (n, d)
    metric : "euclidean" | "cosine" | "mahalanobis"
    cov_inv : np.ndarray, shape (d, d) — required for mahalanobis only.

    Returns
    -------
    np.ndarray, shape (n,), float64
    """
    vec = vec.astype(np.float64)
    mat = mat.astype(np.float64)

    if metric == "euclidean":
        diffs = mat - vec[None, :]
        return np.linalg.norm(diffs, axis=1)

    elif metric == "cosine":
        norm_a = float(np.linalg.norm(vec))
        norms_b = np.linalg.norm(mat, axis=1)
        if norm_a < 1e-12:
            return np.full(len(mat), np.nan)
        cos_sim = (mat @ vec) / (norms_b * norm_a + 1e-12)
        return 1.0 - np.clip(cos_sim, -1.0, 1.0)

    elif metric == "mahalanobis":
        if cov_inv is None:
            raise ValueError("cov_inv must be provided for Mahalanobis distance")
        VI = cov_inv.astype(np.float64)
        diffs = mat - vec[None, :]
        # Vectorized: row-wise sqrt(d @ VI @ d) via einsum trick
        # maha_sq[i] = diffs[i] @ VI @ diffs[i] = (diffs @ VI * diffs).sum(axis=1)
        maha_sq = (diffs @ VI * diffs).sum(axis=1)
        return np.sqrt(np.maximum(maha_sq, 0.0))

    else:
        raise ValueError(
            f"Unknown distance_metric '{metric}'. "
            "Choose from: euclidean, cosine, mahalanobis"
        )


def bootstrap_centroid_distances(
    event_means_by_behavior: dict[str, list[np.ndarray]],
    cluster_centroids: pd.DataFrame,
    n_bootstrap: int = 500,
    seed: int = 42,
    alpha_ci: float = 0.95,
    min_events: int = 3,
    distance_metric: str = "euclidean",
    cov_inv: np.ndarray | None = None,
) -> BootstrapCentroidResult:
    """
    Bootstrap confidence intervals on annotation centroid → cluster distances.

    For each behavior, we have a list of per-event mean embeddings (128D).
    The observed centroid = mean of all event means.  For each bootstrap
    resample (sampling events with replacement) we recompute the centroid
    and its distance to every cluster centroid.

    This quantifies the precision of the centroid estimate, not whether the
    distance is "significant" in an inferential sense. A narrow CI indicates
    the centroid is well-determined by the available events; a wide CI
    indicates instability (too few events or high within-behavior variance).

    Nearest-cluster stability: the fraction of bootstrap samples for which
    the same cluster is the nearest neighbor as in the observed data. This
    is an intuitive reliability metric for the cluster labeling.

    Parameters
    ----------
    event_means_by_behavior : dict[str, list[np.ndarray]]
        behavior → list of per-event mean embedding vectors (shape (128,) each).
        Obtained from the 'event_embedding_means' key of run_annotation_centroids().
    cluster_centroids : pd.DataFrame
        Shape (n_clusters, 128), index = cluster_id.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.
    alpha_ci : float
        CI level (e.g. 0.95 for 95% CI).
    min_events : int
        Behaviors with fewer events are skipped.
    distance_metric : str
        "euclidean" | "cosine" | "mahalanobis".
    cov_inv : np.ndarray | None
        Inverse covariance matrix (128 × 128), required for Mahalanobis.
        Pass the value returned by run_annotation_centroids_multilevel().

    Returns
    -------
    BootstrapCentroidResult
    """
    if distance_metric == "mahalanobis" and cov_inv is None:
        raise ValueError(
            "cov_inv must be provided when distance_metric='mahalanobis'"
        )

    rng = np.random.default_rng(seed)
    half_alpha = (1.0 - alpha_ci) / 2.0

    behaviors = sorted(event_means_by_behavior.keys())
    cluster_ids = cluster_centroids.index.tolist()
    cluster_mat = cluster_centroids.to_numpy(dtype=np.float64)  # (n_clusters, 128)
    n_clusters = len(cluster_ids)

    logger.info(
        "Bootstrap centroid distances: %d behaviors, %d clusters, "
        "%d resamples, metric=%s (seed=%d)",
        len(behaviors), n_clusters, n_bootstrap, distance_metric, seed
    )

    n_events_per_label_dict: dict[str, int] = {}
    obs_dist_rows: dict[str, np.ndarray] = {}
    ci_low_rows: dict[str, np.ndarray] = {}
    ci_high_rows: dict[str, np.ndarray] = {}
    stability_rows: list[dict] = []

    skipped = []
    for behavior in behaviors:
        ev_means = event_means_by_behavior[behavior]
        n_ev = len(ev_means)
        n_events_per_label_dict[behavior] = n_ev

        if n_ev < min_events:
            skipped.append(behavior)
            continue

        means_mat = np.array(ev_means, dtype=np.float64)  # (n_ev, 128)

        # Observed centroid and distances
        centroid_obs = means_mat.mean(axis=0)  # (128,)
        dist_obs = _vec_to_mat_distances(centroid_obs, cluster_mat, distance_metric, cov_inv)
        obs_dist_rows[behavior] = dist_obs

        nearest_obs = int(np.argmin(dist_obs))
        nearest_same_count = 0

        # Bootstrap
        boot_dists = np.zeros((n_bootstrap, n_clusters), dtype=np.float64)
        for j in range(n_bootstrap):
            idx = rng.integers(0, n_ev, size=n_ev)
            boot_centroid = means_mat[idx].mean(axis=0)
            boot_dists[j] = _vec_to_mat_distances(boot_centroid, cluster_mat, distance_metric, cov_inv)
            if int(np.argmin(boot_dists[j])) == nearest_obs:
                nearest_same_count += 1

        ci_low_rows[behavior] = np.percentile(boot_dists, 100 * half_alpha, axis=0)
        ci_high_rows[behavior] = np.percentile(boot_dists, 100 * (1.0 - half_alpha), axis=0)
        stability = nearest_same_count / n_bootstrap

        stability_rows.append({
            "behavior": behavior,
            "nearest_cluster_id": cluster_ids[nearest_obs],
            "observed_distance": float(dist_obs[nearest_obs]),
            "fraction_bootstrap_same_nearest": float(stability),
            "n_events": n_ev,
            "n_bootstrap": n_bootstrap,
        })

        logger.info(
            "Bootstrap centroid '%s': %d events, nearest_cluster=%s "
            "(distance=%.3f, stability=%.1f%%)",
            behavior, n_ev, cluster_ids[nearest_obs],
            float(dist_obs[nearest_obs]), stability * 100
        )

    if skipped:
        logger.info(
            "Skipped %d behaviors with < %d events: %s",
            len(skipped), min_events, skipped
        )

    if not obs_dist_rows:
        logger.error("No behaviors passed the min_events=%d threshold for bootstrap", min_events)
        empty = pd.DataFrame()
        return BootstrapCentroidResult(
            observed_distance=empty, ci_low=empty, ci_high=empty, ci_width=empty,
            nearest_cluster_stability=pd.DataFrame(stability_rows),
            n_bootstrap=n_bootstrap,
            n_events_per_label=pd.Series(n_events_per_label_dict, name="n_events"),
            long_format=pd.DataFrame(),
            distance_metric=distance_metric,
        )

    valid_behaviors = sorted(obs_dist_rows.keys())
    idx = pd.Index(valid_behaviors, name="behavior")
    cols = pd.Index(cluster_ids, name="cluster_id")

    obs_df = pd.DataFrame(
        [obs_dist_rows[b] for b in valid_behaviors], index=idx, columns=cols
    )
    ci_low_df = pd.DataFrame(
        [ci_low_rows[b] for b in valid_behaviors], index=idx, columns=cols
    )
    ci_high_df = pd.DataFrame(
        [ci_high_rows[b] for b in valid_behaviors], index=idx, columns=cols
    )
    ci_width_df = ci_high_df - ci_low_df

    # Long-format summary
    long_rows = []
    for behavior in valid_behaviors:
        n_ev = n_events_per_label_dict[behavior]
        for c_idx, cid in enumerate(cluster_ids):
            long_rows.append({
                "behavior": behavior,
                "cluster_id": cid,
                "observed_distance": float(obs_dist_rows[behavior][c_idx]),
                "ci_low": float(ci_low_rows[behavior][c_idx]),
                "ci_high": float(ci_high_rows[behavior][c_idx]),
                "ci_width": float(ci_high_rows[behavior][c_idx] - ci_low_rows[behavior][c_idx]),
                "n_events": n_ev,
            })
    long_df = pd.DataFrame(long_rows)

    n_events_per_label = pd.Series(n_events_per_label_dict, name="n_events")
    nearest_stability_df = pd.DataFrame(stability_rows).set_index("behavior") \
        if stability_rows else pd.DataFrame()

    logger.info(
        "Bootstrap centroid distances complete: %d behaviors, %d clusters, "
        "%d resamples (%.0f%% CI, metric=%s, seed=%d)",
        len(valid_behaviors), n_clusters, n_bootstrap, alpha_ci * 100, distance_metric, seed
    )

    return BootstrapCentroidResult(
        observed_distance=obs_df,
        ci_low=ci_low_df,
        ci_high=ci_high_df,
        ci_width=ci_width_df,
        nearest_cluster_stability=nearest_stability_df,
        n_bootstrap=n_bootstrap,
        n_events_per_label=n_events_per_label,
        long_format=long_df,
        distance_metric=distance_metric,
    )
