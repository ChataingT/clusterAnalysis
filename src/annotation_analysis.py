"""
Annotation analysis for the clusterAnalysis pipeline.

Two analyses (V-subset only):

1. Cluster distribution during annotations
   For each behavior type, compute the histogram of cluster IDs across all frames
   where that behavior is annotated. Produces a contingency table and an enrichment
   score (observed / expected, where expected = global cluster prevalence in V-records).

2. Annotation centroids + cluster labeling
   For each behavior type, compute the mean 128-D LISBET embedding across all
   annotated frames ("Annotation Centroid"). Then label each global cluster by its
   nearest annotation centroid in embedding space.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .linking import annotation_to_frames, parse_subject_session

logger = logging.getLogger(__name__)


# ── Cluster distribution during annotations ──────────────────────────────────

def run_annotation_overlap(
    cluster_mapping: pd.DataFrame,
    annotations_df: pd.DataFrame,
    segment_registry: dict,
    clinical_df: pd.DataFrame,
    fps: int = 20,
    min_frames: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Compute cluster distribution during each behavior annotation.

    Parameters
    ----------
    cluster_mapping : pd.DataFrame
        Output of load_cluster_mapping().
    annotations_df : pd.DataFrame
        Output of load_annotations().
    segment_registry : dict
        Output of build_segment_registry().
    clinical_df : pd.DataFrame
        Output of load_clinical() — used to map code → uuid → subject_session.
    fps : int
    min_frames : int
        Minimum annotated frames required for a behavior to be included.

    Returns
    -------
    dict with keys:
        "contingency"  : pd.DataFrame (behavior × cluster, frame counts)
        "enrichment"   : pd.DataFrame (behavior × cluster, observed/expected)
        "per_record"   : pd.DataFrame (per-V-record annotation matching stats)
    """
    # Build code → subject_session map from clinical_df
    # Segments may be named by numeric uuid ("7797_T1a_ADOS1_seg_001")
    # OR by V-code ("V017_seg_001") — try both.
    code_to_ss: dict[str, str] = {}
    if "code" in clinical_df.columns:
        for uuid_str, row in clinical_df.iterrows():
            code = row.get("code")
            if not isinstance(code, str):
                continue
            # First try numeric uuid prefix, then V-code prefix
            session_segs = [s for s in segment_registry if s.startswith(str(uuid_str))]
            if not session_segs:
                session_segs = [s for s in segment_registry if s.startswith(code)]
            if session_segs:
                uuid, session = parse_subject_session(session_segs[0])
                code_to_ss[code] = f"{uuid}_{session}" if session else uuid

    # Build a fast lookup: (segment_name, frame_idx) → cluster_id
    logger.info("Building frame→cluster lookup index (may take a moment)...")
    frame_to_cluster = (
        cluster_mapping
        .set_index(["segment_name", "index"])["cluster_id"]
        .to_dict()
    )

    # Identify V-records
    v_codes = sorted(annotations_df["code"].unique())
    logger.info("Processing annotations for %d V-records", len(v_codes))

    # Accumulate (behavior, cluster_id) counts
    behavior_cluster_counts: dict[str, dict[int, int]] = {}
    per_record_rows = []

    for code in v_codes:
        ss = code_to_ss.get(code)
        if ss is None:
            logger.warning(
                "No subject-session found for code '%s' — skipping annotation analysis",
                code
            )
            continue

        events = annotations_df[annotations_df["code"] == code]
        n_events = len(events)
        n_matched = 0
        n_unmatched = 0
        n_frames_total = 0

        for _, ev in events.iterrows():
            frame_list = annotation_to_frames(ev, segment_registry, fps,
                                              subject_session_filter=ss)
            if not frame_list:
                n_unmatched += 1
                continue
            n_matched += 1

            behavior = str(ev.get("behavior", "unknown"))
            if behavior not in behavior_cluster_counts:
                behavior_cluster_counts[behavior] = {}

            for seg_name, rel_frame in frame_list:
                cluster_id = frame_to_cluster.get((seg_name, rel_frame))
                if cluster_id is None:
                    continue
                c = int(cluster_id)
                behavior_cluster_counts[behavior][c] = (
                    behavior_cluster_counts[behavior].get(c, 0) + 1
                )
                n_frames_total += 1

        per_record_rows.append({
            "code": code,
            "subject_session": ss,
            "n_events": n_events,
            "n_matched": n_matched,
            "n_unmatched": n_unmatched,
            "n_frames_covered": n_frames_total,
        })

    if not behavior_cluster_counts:
        logger.warning("No annotation-cluster overlap data collected. Check V-record mapping.")
        empty = pd.DataFrame()
        return {"contingency": empty, "enrichment": empty,
                "per_record": pd.DataFrame(per_record_rows)}

    # Build contingency table
    all_clusters = sorted(set(c for d in behavior_cluster_counts.values() for c in d))
    all_behaviors = sorted(behavior_cluster_counts.keys())

    contingency = pd.DataFrame(
        index=all_behaviors,
        columns=all_clusters,
        data=0,
        dtype=np.int32,
    )
    for behavior, counts in behavior_cluster_counts.items():
        for cluster_id, count in counts.items():
            contingency.loc[behavior, cluster_id] = count

    # Drop behaviors with too few total frames
    row_totals = contingency.sum(axis=1)
    dropped = row_totals[row_totals < min_frames].index.tolist()
    if dropped:
        logger.warning(
            "Behaviors with < %d annotated frames (excluded): %s", min_frames, dropped
        )
        contingency = contingency.loc[row_totals >= min_frames]

    # Compute enrichment: observed / expected
    # Expected = global cluster prevalence among V-records
    v_subject_sessions = set(r["subject_session"] for r in per_record_rows if r["n_matched"] > 0)

    # Get V-record cluster frame counts
    v_seg_names = {s for s in cluster_mapping["segment_name"].unique()
                   if any(s.startswith(ss) for ss in v_subject_sessions)}
    v_frames = cluster_mapping[cluster_mapping["segment_name"].isin(v_seg_names)]

    global_counts = v_frames["cluster_id"].value_counts()
    global_frac = (global_counts / global_counts.sum()).reindex(all_clusters, fill_value=0)

    enrichment = contingency.div(contingency.sum(axis=1), axis=0)  # row-normalize
    for cluster_id in all_clusters:
        expected = global_frac.get(cluster_id, 0)
        if expected > 0:
            enrichment[cluster_id] = enrichment[cluster_id] / expected
        else:
            enrichment[cluster_id] = float("nan")

    per_record_df = pd.DataFrame(per_record_rows)

    logger.info(
        "Annotation overlap: %d behaviors × %d clusters",
        *contingency.shape
    )

    return {
        "contingency": contingency,
        "enrichment": enrichment,
        "per_record": per_record_df,
    }


# ── Annotation centroids + cluster labeling ──────────────────────────────────

# ── Distance helpers ──────────────────────────────────────────────────────────

def _compute_pairwise_distances(
    centroids_a: dict[str, np.ndarray],
    centroids_b: dict[int, np.ndarray],
    distance_metric: str = "euclidean",
    cov_inv: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compute a (behavior × cluster) distance matrix.

    Parameters
    ----------
    centroids_a : dict mapping label → centroid vector (128D)
    centroids_b : dict mapping cluster_id (int) → centroid vector (128D)
    distance_metric : "euclidean" | "cosine" | "mahalanobis"
    cov_inv : np.ndarray, shape (128, 128) | None
        Inverse covariance matrix required for Mahalanobis. Must be provided
        when distance_metric == "mahalanobis".

    Returns
    -------
    pd.DataFrame of shape (n_behaviors, n_clusters), float32.
    """
    behaviors = sorted(centroids_a.keys())
    cluster_ids = sorted(centroids_b.keys())
    dist_data = np.zeros((len(behaviors), len(cluster_ids)), dtype=np.float32)

    if distance_metric == "euclidean":
        for i, b in enumerate(behaviors):
            a_vec = centroids_a[b].astype(np.float64)
            for j, cid in enumerate(cluster_ids):
                diff = a_vec - centroids_b[cid].astype(np.float64)
                dist_data[i, j] = float(np.linalg.norm(diff))

    elif distance_metric == "cosine":
        for i, b in enumerate(behaviors):
            a_vec = centroids_a[b].astype(np.float64)
            norm_a = np.linalg.norm(a_vec)
            for j, cid in enumerate(cluster_ids):
                b_vec = centroids_b[cid].astype(np.float64)
                norm_b = np.linalg.norm(b_vec)
                if norm_a < 1e-12 or norm_b < 1e-12:
                    dist_data[i, j] = float("nan")
                else:
                    cos_sim = np.dot(a_vec, b_vec) / (norm_a * norm_b)
                    # Clip to [-1, 1] to guard against floating-point noise
                    dist_data[i, j] = float(1.0 - np.clip(cos_sim, -1.0, 1.0))

    elif distance_metric == "mahalanobis":
        if cov_inv is None:
            raise ValueError("cov_inv must be provided for Mahalanobis distance")
        VI = cov_inv.astype(np.float64)
        for i, b in enumerate(behaviors):
            a_vec = centroids_a[b].astype(np.float64)
            for j, cid in enumerate(cluster_ids):
                diff = a_vec - centroids_b[cid].astype(np.float64)
                maha_sq = float(diff @ VI @ diff)
                dist_data[i, j] = float(np.sqrt(max(maha_sq, 0.0)))

    else:
        raise ValueError(
            f"Unknown distance_metric '{distance_metric}'. "
            "Choose from: euclidean, cosine, mahalanobis"
        )

    return pd.DataFrame(dist_data, index=pd.Index(behaviors, name="behavior"),
                        columns=pd.Index(cluster_ids, name="cluster_id"))


def _estimate_covariance_inverse(
    cov_sum: np.ndarray,
    cov_sum_sq: np.ndarray,
    cov_count: int,
    regularization_frac: float = 0.01,
) -> np.ndarray:
    """
    Estimate the inverse of the global covariance from running accumulators.

    Uses Tikhonov regularisation: cov_reg = cov + (alpha * trace(cov) / d) * I.
    This ensures invertibility even in high-dimensional spaces.

    Parameters
    ----------
    cov_sum : np.ndarray (128,) — sum of all frame vectors
    cov_sum_sq : np.ndarray (128, 128) — sum of outer products: Σ v @ v.T
    cov_count : int — number of frames accumulated
    regularization_frac : float — fraction of mean variance added to diagonal

    Returns
    -------
    np.ndarray (128, 128), the regularised precision matrix (cov^-1).
    """
    d = cov_sum.shape[0]
    mean_vec = cov_sum / cov_count
    cov = cov_sum_sq / cov_count - np.outer(mean_vec, mean_vec)

    # Symmetrise (guard against floating-point asymmetry)
    cov = (cov + cov.T) / 2.0

    # Tikhonov regularisation: add epsilon * I
    epsilon = regularization_frac * float(np.trace(cov)) / d
    epsilon = max(epsilon, 1e-6)  # absolute floor
    cov_reg = cov + epsilon * np.eye(d, dtype=np.float64)

    logger.info(
        "Covariance estimated from %d frames, regularization ε=%.6f "
        "(trace=%.4f, d=%d)",
        cov_count, epsilon, float(np.trace(cov)), d
    )

    cov_inv = np.linalg.pinv(cov_reg)
    return cov_inv.astype(np.float64)


# ── V-record embedding preloader ─────────────────────────────────────────────

def _preload_v_record_embeddings(
    segment_registry: dict,
    code_to_ss: dict[str, str],
    embeddings_dir: "Path",
    v_codes: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load all embedding CSVs for V-record subject-sessions once.

    Returns
    -------
    dict: subject_session → {segment_name → np.ndarray (N_frames × 128, float16)}
    """
    ss_to_embeddings: dict[str, dict[str, np.ndarray]] = {}

    for code in v_codes:
        ss = code_to_ss.get(code)
        if ss is None or ss in ss_to_embeddings:
            continue

        seg_dirs = [
            embeddings_dir / seg_name
            for seg_name in segment_registry
            if seg_name.startswith(ss)
        ]
        if not seg_dirs:
            logger.warning("No embedding segments found for %s", ss)
            continue

        seg_embeddings: dict[str, np.ndarray] = {}
        for seg_dir in seg_dirs:
            csv_path = seg_dir / "features_lisbet_embedding.csv"
            if not csv_path.exists():
                continue
            try:
                emb = pd.read_csv(csv_path, index_col=0).to_numpy(dtype=np.float16)
                seg_embeddings[seg_dir.name] = emb
            except Exception as exc:
                logger.warning("Failed to load embedding %s: %s", csv_path, exc)

        if seg_embeddings:
            ss_to_embeddings[ss] = seg_embeddings
        else:
            logger.warning("No embeddings loaded for %s", ss)

    logger.info(
        "Preloaded embeddings for %d / %d V-record subject-sessions",
        len(ss_to_embeddings), len(set(code_to_ss.values()))
    )
    return ss_to_embeddings


# ── Core centroid computation ─────────────────────────────────────────────────

def _compute_annotation_centroids_for_level(
    annotations_df: pd.DataFrame,
    segment_registry: dict,
    code_to_ss: dict[str, str],
    frame_to_cluster: dict,
    ss_to_embeddings: dict[str, dict[str, np.ndarray]],
    fps: int,
    label_col: str,
    distance_metric: str,
    cov_inv: np.ndarray | None,
) -> dict[str, object]:
    """
    Internal: compute centroids for one annotation level using preloaded embeddings.

    Parameters
    ----------
    annotations_df : DataFrame with `label_col` as the behavior label column.
    label_col : which column to read labels from ("behavior" for L1, "label" for L2/L3).
    distance_metric : "euclidean" | "cosine" | "mahalanobis"
    cov_inv : required for mahalanobis, ignored otherwise.

    Returns dict with same keys as run_annotation_centroids().
    """
    v_codes = sorted(annotations_df["code"].unique())
    n_labels = annotations_df[label_col].nunique()
    logger.info(
        "Computing annotation centroids: %d labels, %d V-records, metric=%s",
        n_labels, len(v_codes), distance_metric
    )

    behavior_embeddings: dict[str, list[np.ndarray]] = {}
    event_embedding_means: dict[str, list[np.ndarray]] = {}
    cluster_sum: dict[int, np.ndarray] = {}
    cluster_count: dict[int, int] = {}

    for code in v_codes:
        ss = code_to_ss.get(code)
        if ss is None:
            continue
        seg_embeddings = ss_to_embeddings.get(ss)
        if not seg_embeddings:
            continue

        events = annotations_df[annotations_df["code"] == code]
        for _, ev in events.iterrows():
            frame_list = annotation_to_frames(ev, segment_registry, fps,
                                              subject_session_filter=ss)
            if not frame_list:
                continue

            label = str(ev.get(label_col, "unknown"))
            if label not in behavior_embeddings:
                behavior_embeddings[label] = []
                event_embedding_means[label] = []

            event_vecs: list[np.ndarray] = []
            for seg_name, rel_frame in frame_list:
                emb_arr = seg_embeddings.get(seg_name)
                if emb_arr is None or rel_frame >= len(emb_arr):
                    continue
                vec = emb_arr[rel_frame].astype(np.float32)
                event_vecs.append(vec)
                behavior_embeddings[label].append(vec)

                cluster_id = frame_to_cluster.get((seg_name, rel_frame))
                if cluster_id is not None:
                    cid = int(cluster_id)
                    if cid not in cluster_sum:
                        cluster_sum[cid] = np.zeros(128, dtype=np.float64)
                        cluster_count[cid] = 0
                    cluster_sum[cid] += vec
                    cluster_count[cid] += 1

            if event_vecs:
                event_embedding_means[label].append(
                    np.mean(event_vecs, axis=0).astype(np.float32)
                )

    # Annotation centroids (event-weighted: mean of per-event means)
    annotation_centroids: dict[str, np.ndarray] = {}
    for label, event_means in event_embedding_means.items():
        if not event_means:
            continue
        means_mat = np.array(event_means, dtype=np.float32)
        annotation_centroids[label] = means_mat.mean(axis=0).astype(np.float32)
        n_frames = len(behavior_embeddings.get(label, []))
        logger.info(
            "Centroid '%s': %d events (%d frames)",
            label, len(event_means), n_frames
        )

    if not annotation_centroids:
        logger.warning("No annotation centroids computed")
        return {}

    # Cluster centroids (from V-record frames)
    cluster_centroids_dict = {
        cid: (cluster_sum[cid] / cluster_count[cid]).astype(np.float32)
        for cid in cluster_sum
    }
    cluster_centroids_df = pd.DataFrame(cluster_centroids_dict).T
    cluster_centroids_df.index.name = "cluster_id"

    # Distance matrix
    distance_matrix = _compute_pairwise_distances(
        annotation_centroids, cluster_centroids_dict,
        distance_metric=distance_metric, cov_inv=cov_inv,
    )

    # Cluster labeling: nearest annotation centroid per cluster
    cluster_ids = sorted(cluster_centroids_dict.keys())
    label_rows = []
    for cid in cluster_ids:
        if cid not in distance_matrix.columns:
            continue
        col = distance_matrix[cid]
        nearest_behavior = col.idxmin()
        min_dist = float(col.min())
        label_rows.append({
            "cluster_id": cid,
            "nearest_behavior": nearest_behavior,
            "min_distance": min_dist,
        })
    cluster_labels_df = pd.DataFrame(label_rows).set_index("cluster_id")

    logger.info(
        "Annotation centroids: %d labels, %d clusters labeled (metric=%s)",
        len(annotation_centroids), len(cluster_ids), distance_metric
    )

    return {
        "annotation_centroids": annotation_centroids,
        "cluster_centroids": cluster_centroids_df,
        "distance_matrix": distance_matrix,
        "cluster_behavior_labels": cluster_labels_df,
        "event_embedding_means": event_embedding_means,
        "distance_metric": distance_metric,
    }


def run_annotation_centroids(
    cluster_mapping: pd.DataFrame,
    annotations_df: pd.DataFrame,
    segment_registry: dict,
    clinical_df: pd.DataFrame,
    embeddings_dir: str | Path,
    fps: int = 20,
    distance_metric: str = "euclidean",
) -> dict[str, object]:
    """
    Compute mean 128-D embedding per behavior ("Annotation Centroid") and
    label each global cluster by its nearest annotation centroid.

    Parameters
    ----------
    distance_metric : str
        "euclidean" (default), "cosine", or "mahalanobis".
        For mahalanobis, the global covariance is estimated from all V-record
        frames accumulated during the embedding loading pass.

    Returns
    -------
    dict with keys:
        "annotation_centroids"   : dict[behavior → np.ndarray shape (128,)]
        "cluster_centroids"      : pd.DataFrame (cluster_id × 128 dims)
        "distance_matrix"        : pd.DataFrame (behavior × cluster, distances)
        "cluster_behavior_labels": pd.DataFrame (cluster_id, nearest_behavior, min_distance)
        "event_embedding_means"  : dict[behavior → list of np.ndarray shape (128,)]
        "distance_metric"        : str — the metric that was used
        "cov_inv"                : np.ndarray (128, 128) | None — precision matrix
                                   (set when distance_metric == "mahalanobis")
    """
    from pathlib import Path as _Path
    embeddings_dir = _Path(embeddings_dir)

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

    frame_to_cluster = (
        cluster_mapping
        .set_index(["segment_name", "index"])["cluster_id"]
        .to_dict()
    )

    v_codes = sorted(annotations_df["code"].unique())

    # Preload all V-record embeddings once
    ss_to_embeddings = _preload_v_record_embeddings(
        segment_registry, code_to_ss, embeddings_dir, v_codes
    )

    # Estimate covariance for Mahalanobis (extra pass over preloaded embeddings)
    cov_inv: np.ndarray | None = None
    if distance_metric == "mahalanobis":
        cov_inv = _estimate_cov_inv_from_preloaded(ss_to_embeddings)

    result = _compute_annotation_centroids_for_level(
        annotations_df=annotations_df,
        segment_registry=segment_registry,
        code_to_ss=code_to_ss,
        frame_to_cluster=frame_to_cluster,
        ss_to_embeddings=ss_to_embeddings,
        fps=fps,
        label_col="behavior",
        distance_metric=distance_metric,
        cov_inv=cov_inv,
    )
    result["cov_inv"] = cov_inv
    return result


def _estimate_cov_inv_from_preloaded(
    ss_to_embeddings: dict[str, dict[str, np.ndarray]],
    regularization_frac: float = 0.01,
) -> np.ndarray:
    """Estimate precision matrix from all preloaded V-record frame embeddings."""
    dim = 128
    cov_sum = np.zeros(dim, dtype=np.float64)
    cov_sum_sq = np.zeros((dim, dim), dtype=np.float64)
    cov_count = 0

    for seg_embeddings in ss_to_embeddings.values():
        for emb_arr in seg_embeddings.values():
            frames = emb_arr.astype(np.float64)  # (N_frames, 128)
            cov_sum += frames.sum(axis=0)
            cov_sum_sq += frames.T @ frames      # (128, 128)
            cov_count += len(frames)

    if cov_count < dim + 1:
        logger.warning(
            "Too few frames (%d) to estimate a reliable 128D covariance. "
            "Falling back to identity (diagonal) for Mahalanobis.",
            cov_count
        )
        return np.eye(dim, dtype=np.float64)

    return _estimate_covariance_inverse(cov_sum, cov_sum_sq, cov_count, regularization_frac)


# ── Multi-level centroid analysis ─────────────────────────────────────────────

def run_annotation_centroids_multilevel(
    cluster_mapping: pd.DataFrame,
    annotations_df: pd.DataFrame,
    segment_registry: dict,
    clinical_df: pd.DataFrame,
    embeddings_dir: str | Path,
    fps: int = 20,
    distance_metric: str = "euclidean",
) -> dict[str, dict[str, object]]:
    """
    Run annotation centroid analysis at all three label hierarchy levels (L1/L2/L3).

    Embeddings are loaded **once** and reused across all three levels, making
    this substantially faster than calling run_annotation_centroids three times.
    The global covariance (for Mahalanobis) is also estimated once.

    Parameters
    ----------
    distance_metric : str
        "euclidean" | "cosine" | "mahalanobis"

    Returns
    -------
    dict with keys "L1", "L2", "L3", each containing the same dict as
    run_annotation_centroids() would return for that level.
    """
    from pathlib import Path as _Path
    embeddings_dir = _Path(embeddings_dir)

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

    frame_to_cluster = (
        cluster_mapping
        .set_index(["segment_name", "index"])["cluster_id"]
        .to_dict()
    )

    v_codes = sorted(annotations_df["code"].unique())

    # ── Load embeddings once ─────────────────────────────────────────────────
    logger.info(
        "run_annotation_centroids_multilevel: preloading embeddings for "
        "%d V-record codes (metric=%s)...", len(v_codes), distance_metric
    )
    ss_to_embeddings = _preload_v_record_embeddings(
        segment_registry, code_to_ss, embeddings_dir, v_codes
    )

    # ── Estimate covariance once (for Mahalanobis) ───────────────────────────
    cov_inv: np.ndarray | None = None
    if distance_metric == "mahalanobis":
        cov_inv = _estimate_cov_inv_from_preloaded(ss_to_embeddings)

    # ── Build multilevel label DataFrames ────────────────────────────────────
    level_dfs = build_multilevel_annotations(annotations_df)
    # level_dfs["L1"] has "label" == behavior name
    # level_dfs["L2"/"L3"] have "label" == composite string

    results: dict[str, object] = {}
    for level, ann_level_df in level_dfs.items():
        if ann_level_df.empty:
            logger.warning("Level %s: no annotations, skipping centroid analysis", level)
            results[level] = {}
            continue

        logger.info(
            "── Computing annotation centroids at level %s (%d unique labels) ──",
            level, ann_level_df["label"].nunique()
        )
        level_result = _compute_annotation_centroids_for_level(
            annotations_df=ann_level_df,
            segment_registry=segment_registry,
            code_to_ss=code_to_ss,
            frame_to_cluster=frame_to_cluster,
            ss_to_embeddings=ss_to_embeddings,
            fps=fps,
            label_col="label",       # multilevel DataFrames use "label" column
            distance_metric=distance_metric,
            cov_inv=cov_inv,
        )
        results[level] = level_result

    # Expose cov_inv at the top level for bootstrap_centroid_distances()
    results["cov_inv"] = cov_inv

    return results


def save_annotation_centroids(
    centroids: dict[str, np.ndarray],
    output_dir: Path,
    suffix: str = "",
) -> None:
    """Save annotation centroids dict as a pickle file.

    Parameters
    ----------
    suffix : str
        Optional suffix appended to the filename before the extension,
        e.g. ``"_L1"`` → ``annotation_centroids_L1.pkl``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / f"annotation_centroids{suffix}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(centroids, f)
    logger.info("Annotation centroids saved to %s", pkl_path)


# ── Multi-level annotation hierarchy ─────────────────────────────────────────

def build_multilevel_annotations(
    annotations_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build annotation DataFrames with composite labels at three granularity levels.

    Level 1 (L1): behavior only
    Level 2 (L2): behavior + behavioral_category (concatenated)
    Level 3 (L3): behavior + behavioral_category + modifier_1 (concatenated)

    The composite label is built from available non-null, non-empty fields,
    separated by " | ".

    Parameters
    ----------
    annotations_df : pd.DataFrame
        Output of load_annotations(). Must have at least 'behavior'.
        Optionally 'behavioral_category' and 'modifier_1'.

    Returns
    -------
    dict with keys "L1", "L2", "L3", each a copy of annotations_df with a
    'label' column added (the composite label for that level).
    """
    result = {}
    df = annotations_df.copy()

    def _clean(val: object) -> str:
        s = str(val).strip() if not pd.isna(val) else ""
        return s if s.lower() not in ("nan", "none", "") else ""

    # L1: behavior only
    df_l1 = df.copy()
    df_l1["label"] = df_l1["behavior"].apply(_clean)
    df_l1 = df_l1[df_l1["label"] != ""]
    result["L1"] = df_l1
    logger.info("L1 annotation labels: %d unique behaviors", df_l1["label"].nunique())

    # L2: behavior + category
    df_l2 = df.copy()
    has_cat = "behavioral_category" in df_l2.columns
    if has_cat:
        df_l2["label"] = df_l2.apply(
            lambda r: " | ".join(
                p for p in [_clean(r["behavior"]), _clean(r.get("behavioral_category", ""))]
                if p
            ),
            axis=1,
        )
    else:
        logger.warning("No 'behavioral_category' column — L2 will equal L1")
        df_l2["label"] = df_l2["behavior"].apply(_clean)
    df_l2 = df_l2[df_l2["label"] != ""]
    result["L2"] = df_l2
    logger.info("L2 annotation labels: %d unique composites", df_l2["label"].nunique())

    # L3: behavior + category + modifier_1
    df_l3 = df.copy()
    has_mod = "modifier_1" in df_l3.columns
    if has_cat and has_mod:
        df_l3["label"] = df_l3.apply(
            lambda r: " | ".join(
                p for p in [
                    _clean(r["behavior"]),
                    _clean(r.get("behavioral_category", "")),
                    _clean(r.get("modifier_1", "")),
                ]
                if p
            ),
            axis=1,
        )
    elif has_cat:
        logger.warning("No 'modifier_1' column — L3 will equal L2")
        df_l3["label"] = df_l3.apply(
            lambda r: " | ".join(
                p for p in [_clean(r["behavior"]), _clean(r.get("behavioral_category", ""))]
                if p
            ),
            axis=1,
        )
    else:
        df_l3["label"] = df_l3["behavior"].apply(_clean)
    df_l3 = df_l3[df_l3["label"] != ""]
    result["L3"] = df_l3
    logger.info("L3 annotation labels: %d unique composites", df_l3["label"].nunique())

    return result


def run_annotation_overlap_multilevel(
    cluster_mapping: pd.DataFrame,
    annotations_df: pd.DataFrame,
    segment_registry: dict,
    clinical_df: pd.DataFrame,
    fps: int = 20,
    min_frames_level1: int = 10,
    min_frames_level2: int = 5,
    min_frames_level3: int = 3,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Run annotation-cluster overlap analysis at three label hierarchy levels.

    Thin wrapper around run_annotation_overlap that:
    1. Builds composite labels via build_multilevel_annotations()
    2. Runs run_annotation_overlap for each level using the 'label' column
       instead of the 'behavior' column, with per-level min_frames thresholds

    Parameters
    ----------
    cluster_mapping, segment_registry, clinical_df, fps
        As in run_annotation_overlap.
    min_frames_level1/2/3 : int
        Minimum annotated frames for inclusion at each level.

    Returns
    -------
    dict with keys "L1", "L2", "L3", each a dict with keys
    "contingency", "enrichment", "per_record".
    """
    level_dfs = build_multilevel_annotations(annotations_df)
    min_frames_map = {
        "L1": min_frames_level1,
        "L2": min_frames_level2,
        "L3": min_frames_level3,
    }

    results: dict[str, dict[str, pd.DataFrame]] = {}
    for level, df in level_dfs.items():
        logger.info("Running annotation overlap at level %s (%d events)...", level, len(df))
        # Temporarily rename 'label' → 'behavior' so run_annotation_overlap works unchanged
        df_renamed = df.copy()
        df_renamed["behavior"] = df_renamed["label"]
        try:
            level_result = run_annotation_overlap(
                cluster_mapping=cluster_mapping,
                annotations_df=df_renamed,
                segment_registry=segment_registry,
                clinical_df=clinical_df,
                fps=fps,
                min_frames=min_frames_map[level],
            )
        except Exception as exc:
            logger.error("Annotation overlap at level %s failed: %s", level, exc)
            level_result = {}
        results[level] = level_result

    return results


# ── Frame-level annotation × kinematic analysis ───────────────────────────────

def run_annotation_kinematics(
    annotations_df: pd.DataFrame,
    segment_registry: dict,
    clinical_df: pd.DataFrame,
    pose_records_dir: str | Path,
    fps: int = 20,
    use_normalized: bool = True,
    min_frames_level1: int = 10,
    min_frames_level2: int = 5,
    min_frames_level3: int = 3,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Compute frame-level kinematic profiles per annotation label.

    For each annotated frame, loads the corresponding per-frame kinematic
    metrics and computes mean ± std per label. Also computes profiles for
    non-annotated frames (background) for comparison.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        Output of load_annotations().
    segment_registry : dict
        Output of build_segment_registry().
    clinical_df : pd.DataFrame
        Output of load_clinical().
    pose_records_dir : Path
        Root of pose record directories.
    fps : int
        Frames per second.
    use_normalized : bool
        Load metrics_normalised.csv (True) or metrics_summary.csv (False).
    min_frames_level1/2/3 : int
        Minimum annotated frames to include a label at each level.

    Returns
    -------
    dict with keys "L1", "L2", "L3", each a dict with keys:
        "profiles"    : pd.DataFrame (label × metric_mean/std + n_frames)
        "background"  : pd.DataFrame (1 row, mean ± std of non-annotated frames)
    """
    from pathlib import Path as _Path

    pose_records_dir = _Path(pose_records_dir)
    POSE_REC_PREFIX = "results_skeleton_"
    METRICS_FILE = "metrics_normalised.csv" if use_normalized else "metrics_summary.csv"
    ALT_METRICS_FILE = "metrics_summary.csv" if use_normalized else "metrics_normalised.csv"

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
                from .linking import parse_subject_session
                u, s = parse_subject_session(session_segs[0])
                code_to_ss[code] = f"{u}_{s}" if s else u

    level_dfs = build_multilevel_annotations(annotations_df)
    min_frames_map = {
        "L1": min_frames_level1,
        "L2": min_frames_level2,
        "L3": min_frames_level3,
    }

    output: dict[str, dict[str, pd.DataFrame]] = {}

    for level, ann_df in level_dfs.items():
        logger.info("Computing annotation kinematics at level %s...", level)
        min_frames = min_frames_map[level]

        # label → {metric_name: [values]}
        label_metric_vals: dict[str, dict[str, list[float]]] = {}
        background_metric_vals: dict[str, list[float]] = {}
        discovered_cols: list[str] | None = None

        v_codes = sorted(ann_df["code"].unique())
        for code in v_codes:
            ss = code_to_ss.get(code)
            if ss is None:
                continue

            # Load kinematics for all segments of this subject-session
            seg_metrics: dict[str, pd.DataFrame] = {}  # seg_name → df
            for seg_name, info in segment_registry.items():
                if not seg_name.startswith(ss):
                    continue
                # Find segment directory
                parts = seg_name.rsplit("_seg_", maxsplit=1)
                if len(parts) != 2:
                    continue
                seg_dir = (
                    pose_records_dir / f"{POSE_REC_PREFIX}{parts[0]}"
                    / "segments" / f"seg_{parts[1]}"
                )
                csv_path = seg_dir / METRICS_FILE
                if not csv_path.exists():
                    csv_path = seg_dir / ALT_METRICS_FILE
                if not csv_path.exists():
                    continue
                try:
                    df = pd.read_csv(csv_path)
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", csv_path, exc)
                    continue

                # Detect frame column
                for fc in ("frame", "index", "frame_index", "Frame"):
                    if fc in df.columns:
                        df = df.rename(columns={fc: "frame"})
                        break
                else:
                    df = df.rename(columns={df.columns[0]: "frame"})

                df = df.set_index("frame")
                metric_cols = [c for c in df.columns if c not in ("frame",)]
                df = df[metric_cols].apply(pd.to_numeric, errors="coerce")

                if discovered_cols is None:
                    discovered_cols = metric_cols
                    background_metric_vals = {c: [] for c in discovered_cols}

                seg_metrics[seg_name] = df

            if not seg_metrics:
                continue

            # Track annotated (seg, frame) pairs to identify background
            annotated_frames: set[tuple[str, int]] = set()

            events = ann_df[ann_df["code"] == code]
            for _, ev in events.iterrows():
                from .linking import annotation_to_frames
                frame_list = annotation_to_frames(ev, segment_registry, fps,
                                                  subject_session_filter=ss)
                if not frame_list:
                    continue

                label = str(ev.get("label", "unknown"))
                if label not in label_metric_vals:
                    label_metric_vals[label] = {c: [] for c in (discovered_cols or [])}

                for seg_name, rel_frame in frame_list:
                    annotated_frames.add((seg_name, rel_frame))
                    mdf = seg_metrics.get(seg_name)
                    if mdf is None or rel_frame not in mdf.index:
                        continue

                    row_vals = mdf.loc[rel_frame]
                    for col in (discovered_cols or []):
                        if col in row_vals and not pd.isna(row_vals[col]):
                            if col not in label_metric_vals[label]:
                                label_metric_vals[label][col] = []
                            label_metric_vals[label][col].append(float(row_vals[col]))

            # Collect background (non-annotated) frames
            for seg_name, mdf in seg_metrics.items():
                for rel_frame in mdf.index:
                    if (seg_name, rel_frame) in annotated_frames:
                        continue
                    row_vals = mdf.loc[rel_frame]
                    for col in (discovered_cols or []):
                        if col in row_vals and not pd.isna(row_vals[col]):
                            if col not in background_metric_vals:
                                background_metric_vals[col] = []
                            background_metric_vals[col].append(float(row_vals[col]))

        # Build profile DataFrame
        if not label_metric_vals or discovered_cols is None:
            logger.warning("No annotation kinematics data at level %s", level)
            output[level] = {"profiles": pd.DataFrame(), "background": pd.DataFrame()}
            continue

        profile_rows = []
        for label, metric_dict in label_metric_vals.items():
            n_frames = max((len(v) for v in metric_dict.values()), default=0)
            if n_frames < min_frames:
                logger.debug("Level %s: label '%s' has %d frames < %d, excluded",
                             level, label, n_frames, min_frames)
                continue
            row: dict = {"label": label, "n_frames": n_frames}
            for col in discovered_cols:
                vals = metric_dict.get(col, [])
                row[f"{col}__mean"] = float(np.mean(vals)) if vals else float("nan")
                row[f"{col}__std"] = float(np.std(vals)) if vals else float("nan")
            profile_rows.append(row)

        profiles_df = pd.DataFrame(profile_rows).set_index("label") if profile_rows else pd.DataFrame()

        # Background profile (1 row)
        bg_row: dict = {"label": "_background_"}
        n_bg = max((len(v) for v in background_metric_vals.values()), default=0)
        bg_row["n_frames"] = n_bg
        for col in discovered_cols:
            vals = background_metric_vals.get(col, [])
            bg_row[f"{col}__mean"] = float(np.mean(vals)) if vals else float("nan")
            bg_row[f"{col}__std"] = float(np.std(vals)) if vals else float("nan")
        bg_df = pd.DataFrame([bg_row]).set_index("label")

        logger.info(
            "Level %s annotation kinematics: %d labels × %d metrics "
            "(background: %d frames)",
            level, len(profiles_df), len(discovered_cols), n_bg
        )
        output[level] = {"profiles": profiles_df, "background": bg_df}

    return output


# ── Global cluster centroids (all subjects) ───────────────────────────────────

def compute_global_cluster_centroids(
    cluster_mapping: pd.DataFrame,
    embeddings_dir: str | Path,
    segment_registry: dict,
    subject_session_ids: list[str],
) -> pd.DataFrame:
    """
    Compute mean 128-D LISBET embedding per cluster using all subjects
    (not just V-records).

    More accurate than V-only centroids from run_annotation_centroids.

    Parameters
    ----------
    cluster_mapping : pd.DataFrame
        Must have columns: segment_name, index (relative frame), cluster_id.
    embeddings_dir : Path
        Root of embedding directories (each subdir = one segment).
    segment_registry : dict
        Output of build_segment_registry() — used to filter segments.
    subject_session_ids : list[str]
        All subject-session IDs (used to scope which segments to load).

    Returns
    -------
    pd.DataFrame of shape (N_clusters, 128), index = cluster_id.
    """
    from pathlib import Path as _Path

    embeddings_dir = _Path(embeddings_dir)
    EMB_FILE = "features_lisbet_embedding.csv"

    cluster_sum: dict[int, np.ndarray] = {}
    cluster_count: dict[int, int] = {}

    # Build lookup: (segment_name, rel_frame) → cluster_id
    frame_to_cluster: dict[tuple[str, int], int] = {
        (row.segment_name, int(row.index)): int(row.cluster_id)
        for row in cluster_mapping.itertuples()
    }

    n_loaded = 0
    for seg_name in sorted(segment_registry.keys()):
        emb_path = embeddings_dir / seg_name / EMB_FILE
        if not emb_path.exists():
            continue
        try:
            emb = pd.read_csv(emb_path, index_col=0).to_numpy(dtype=np.float16)
        except Exception as exc:
            logger.warning("Failed to load embedding %s: %s", emb_path, exc)
            continue

        for rel_frame in range(len(emb)):
            cluster_id = frame_to_cluster.get((seg_name, rel_frame))
            if cluster_id is None:
                continue
            vec = emb[rel_frame].astype(np.float64)
            if np.any(np.isnan(vec)):
                continue
            if cluster_id not in cluster_sum:
                cluster_sum[cluster_id] = np.zeros(emb.shape[1], dtype=np.float64)
                cluster_count[cluster_id] = 0
            cluster_sum[cluster_id] += vec
            cluster_count[cluster_id] += 1

        n_loaded += 1

    if not cluster_sum:
        logger.warning("No embeddings loaded for global cluster centroids.")
        return pd.DataFrame()

    logger.info(
        "Global cluster centroids: computed from %d segments, %d clusters",
        n_loaded, len(cluster_sum)
    )

    rows = {
        cid: (cluster_sum[cid] / cluster_count[cid]).astype(np.float32)
        for cid in cluster_sum
    }
    df = pd.DataFrame(rows).T
    df.index.name = "cluster_id"
    return df
