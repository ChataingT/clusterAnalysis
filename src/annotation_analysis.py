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

def run_annotation_centroids(
    cluster_mapping: pd.DataFrame,
    annotations_df: pd.DataFrame,
    segment_registry: dict,
    clinical_df: pd.DataFrame,
    embeddings_dir: str | Path,
    fps: int = 20,
) -> dict[str, object]:
    """
    Compute mean 128-D embedding per behavior ("Annotation Centroid") and
    label each global cluster by its nearest annotation centroid.

    Uses existing data.py from behavior_clustering to load embeddings.

    Returns
    -------
    dict with keys:
        "annotation_centroids"   : dict[behavior → np.ndarray shape (128,)]
        "cluster_centroids"      : pd.DataFrame (cluster_id × 128 dims)
        "distance_matrix"        : pd.DataFrame (behavior × cluster, Euclidean distances)
        "cluster_behavior_labels": pd.DataFrame (cluster_id, nearest_behavior, min_distance)
    """
    from pathlib import Path as _Path

    embeddings_dir = _Path(embeddings_dir)

    # Build code → subject_session map (same fallback logic as run_annotation_overlap)
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
    frame_to_cluster = (
        cluster_mapping
        .set_index(["segment_name", "index"])["cluster_id"]
        .to_dict()
    )

    v_codes = sorted(annotations_df["code"].unique())
    logger.info(
        "Computing annotation centroids for %d behaviors across %d V-records",
        annotations_df["behavior"].nunique(), len(v_codes)
    )

    # Accumulate embeddings per behavior and per cluster
    # behavior → list of embedding vectors
    behavior_embeddings: dict[str, list[np.ndarray]] = {}
    # cluster_id → list of embedding vectors (for computing cluster centroids)
    cluster_sum: dict[int, np.ndarray] = {}
    cluster_count: dict[int, int] = {}

    for code in v_codes:
        ss = code_to_ss.get(code)
        if ss is None:
            continue

        # Load embeddings for all segments of this subject-session
        seg_dirs_for_ss = [
            embeddings_dir / seg_name
            for seg_name in segment_registry
            if seg_name.startswith(ss)
        ]
        if not seg_dirs_for_ss:
            logger.warning("No embedding segments found for %s", ss)
            continue

        # Load each segment embedding CSV
        seg_embeddings: dict[str, np.ndarray] = {}
        for seg_dir in seg_dirs_for_ss:
            csv_path = seg_dir / "features_lisbet_embedding.csv"
            if not csv_path.exists():
                continue
            try:
                emb = pd.read_csv(csv_path, index_col=0).to_numpy(dtype=np.float16)
                seg_name = seg_dir.name
                seg_embeddings[seg_name] = emb
            except Exception as exc:
                logger.warning("Failed to load embedding %s: %s", csv_path, exc)
                continue

        if not seg_embeddings:
            logger.warning("No embeddings loaded for %s", ss)
            continue

        events = annotations_df[annotations_df["code"] == code]
        for _, ev in events.iterrows():
            frame_list = annotation_to_frames(ev, segment_registry, fps,
                                              subject_session_filter=ss)
            if not frame_list:
                continue

            behavior = str(ev.get("behavior", "unknown"))
            if behavior not in behavior_embeddings:
                behavior_embeddings[behavior] = []

            for seg_name, rel_frame in frame_list:
                emb_arr = seg_embeddings.get(seg_name)
                if emb_arr is None or rel_frame >= len(emb_arr):
                    continue
                vec = emb_arr[rel_frame].astype(np.float32)
                behavior_embeddings[behavior].append(vec)

                # Also accumulate per-cluster
                cluster_id = frame_to_cluster.get((seg_name, rel_frame))
                if cluster_id is not None:
                    cid = int(cluster_id)
                    if cid not in cluster_sum:
                        cluster_sum[cid] = np.zeros(128, dtype=np.float64)
                        cluster_count[cid] = 0
                    cluster_sum[cid] += vec
                    cluster_count[cid] += 1

    # Compute annotation centroids
    annotation_centroids: dict[str, np.ndarray] = {}
    for behavior, vecs in behavior_embeddings.items():
        if len(vecs) == 0:
            logger.warning("No embedding vectors for behavior '%s'", behavior)
            continue
        annotation_centroids[behavior] = np.mean(vecs, axis=0).astype(np.float32)
        logger.info(
            "Annotation centroid '%s': computed from %d frames", behavior, len(vecs)
        )

    if not annotation_centroids:
        logger.warning("No annotation centroids computed — check embedding paths and annotations")
        return {}

    # Compute global cluster centroids (from V-record frames only)
    # For a full run, also compute from all records (use cluster_mapping)
    # Here we use what we accumulated from V-records; may be supplemented later.
    cluster_centroids_dict = {
        cid: (cluster_sum[cid] / cluster_count[cid]).astype(np.float32)
        for cid in cluster_sum
    }
    cluster_centroids_df = pd.DataFrame(cluster_centroids_dict).T
    cluster_centroids_df.index.name = "cluster_id"

    # Distance matrix: behavior × cluster
    behaviors = sorted(annotation_centroids.keys())
    cluster_ids = sorted(cluster_centroids_dict.keys())

    dist_data = np.zeros((len(behaviors), len(cluster_ids)), dtype=np.float32)
    for i, behavior in enumerate(behaviors):
        cent = annotation_centroids[behavior]
        for j, cid in enumerate(cluster_ids):
            clust_cent = cluster_centroids_dict[cid]
            dist_data[i, j] = float(np.linalg.norm(cent - clust_cent))

    distance_matrix = pd.DataFrame(dist_data, index=behaviors, columns=cluster_ids)

    # Cluster labeling: nearest annotation centroid per cluster
    label_rows = []
    for cid in cluster_ids:
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
        "Annotation centroids: %d behaviors, %d clusters labeled",
        len(annotation_centroids), len(cluster_ids)
    )

    return {
        "annotation_centroids": annotation_centroids,
        "cluster_centroids": cluster_centroids_df,
        "distance_matrix": distance_matrix,
        "cluster_behavior_labels": cluster_labels_df,
    }


def save_annotation_centroids(
    centroids: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Save annotation centroids dict as a pickle file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "annotation_centroids.pkl"
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
