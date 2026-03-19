"""
Linking utilities: segment registry, subject map, annotation→frame lookup,
prevalence matrix, and data coverage report.

The segment registry maps each segment name to its absolute video frame range,
read from the 'segment_start_frame' and 'segment_end_frame' attributes of the
tracking.nc NetCDF files produced by the pose estimation pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

POSE_RECORD_PREFIX = "results_skeleton_"
TRACKING_FILENAME = "tracking.nc"


# ── Segment registry ─────────────────────────────────────────────────────────

def build_segment_registry(
    pose_records_dir: str | Path,
    subject_session_ids: list[str],
    fps: int = 20,
) -> dict[str, dict[str, int]]:
    """
    Build a mapping from segment_name → absolute video frame range.

    Reads 'segment_start_frame' and 'segment_end_frame' NetCDF attributes from
    each segment's tracking.nc file.

    Parameters
    ----------
    pose_records_dir : Path
        Root directory containing pose record folders.
    subject_session_ids : list[str]
        IDs like "7797_T1a_ADOS1" (without POSE_RECORD_PREFIX).
    fps : int
        Frames per second (used for logging only).

    Returns
    -------
    dict mapping segment_name (e.g. "7797_T1a_ADOS1_seg_001") to
        {"start_abs_frame": int, "end_abs_frame": int}
    """
    pose_records_dir = Path(pose_records_dir)
    registry: dict[str, dict[str, int]] = {}
    skipped_subjects = []

    for ss_id in subject_session_ids:
        record_dir = pose_records_dir / f"{POSE_RECORD_PREFIX}{ss_id}"
        seg_parent = record_dir / "segments"

        if not seg_parent.is_dir():
            logger.warning("Segments directory not found for %s: %s", ss_id, seg_parent)
            skipped_subjects.append(ss_id)
            continue

        seg_dirs = sorted(seg_parent.glob("seg_*"))
        if not seg_dirs:
            logger.warning("No segment directories found under %s", seg_parent)
            skipped_subjects.append(ss_id)
            continue

        for seg_dir in seg_dirs:
            nc_path = seg_dir / TRACKING_FILENAME
            if not nc_path.exists():
                logger.debug("tracking.nc not found in %s, skipping", seg_dir)
                continue

            try:
                with xr.open_dataset(nc_path) as ds:
                    start = int(ds.attrs["segment_start_frame"])
                    end = int(ds.attrs["segment_end_frame"])
                    # B2: validate fps from tracking.nc if the attribute exists
                    for fps_attr in ("fps", "frame_rate", "sample_rate"):
                        if fps_attr in ds.attrs:
                            nc_fps = float(ds.attrs[fps_attr])
                            if abs(nc_fps - fps) > 0.5:
                                logger.warning(
                                    "FPS mismatch in %s: tracking.nc has %s=%.1f "
                                    "but config fps=%d — annotation-to-frame mapping "
                                    "will use config fps",
                                    nc_path, fps_attr, nc_fps, fps
                                )
                            break
            except KeyError as exc:
                logger.warning(
                    "tracking.nc in %s missing attribute %s — skipping segment",
                    seg_dir, exc
                )
                continue
            except Exception as exc:
                logger.warning("Failed to read tracking.nc in %s: %s", seg_dir, exc)
                continue

            # Construct the segment_name as used in the cluster mapping
            seg_label = seg_dir.name  # e.g. "seg_001"
            seg_num = seg_label.replace("seg_", "")
            segment_name = f"{ss_id}_seg_{seg_num}"

            registry[segment_name] = {
                "start_abs_frame": start,
                "end_abs_frame": end,
                "duration_sec": round((end - start) / fps, 2),
            }
            logger.debug(
                "Registry: %s → frames [%d, %d] (%.1fs)",
                segment_name, start, end, (end - start) / fps
            )

    if skipped_subjects:
        logger.warning(
            "Segment registry: skipped %d / %d subjects (missing dirs or tracking.nc)",
            len(skipped_subjects), len(subject_session_ids)
        )

    logger.info(
        "Segment registry built: %d segments from %d subject-sessions",
        len(registry), len(subject_session_ids) - len(skipped_subjects)
    )
    return registry


# ── Subject ↔ segment map ────────────────────────────────────────────────────

def parse_subject_session(segment_name: str) -> tuple[str, str]:
    """
    Parse a segment_name like "7797_T1a_ADOS1_seg_001" into (uuid, session_id).

    The segment suffix is everything after the last "_seg_NNN" pattern.
    The subject session is everything before that.

    Returns
    -------
    (uuid_str, session_str)  e.g.  ("7797", "T1a_ADOS1")
    """
    # Split off "_seg_NNN" suffix
    parts = segment_name.rsplit("_seg_", maxsplit=1)
    subject_session = parts[0]  # e.g. "7797_T1a_ADOS1"
    # uuid = first underscore-separated token
    tokens = subject_session.split("_", maxsplit=1)
    uuid = tokens[0]
    session = tokens[1] if len(tokens) > 1 else ""
    return uuid, session


def build_subject_map(
    cluster_mapping: pd.DataFrame,
    clinical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a per-segment DataFrame that attaches clinical metadata to each segment.

    Parameters
    ----------
    cluster_mapping : pd.DataFrame
        Output of load_cluster_mapping().
    clinical_df : pd.DataFrame
        Output of load_clinical() (indexed by uuid).

    Returns
    -------
    pd.DataFrame with one row per unique segment_name, columns:
        segment_name, subject_session, uuid, session_id,
        n_frames, dominant_cluster,
        + all clinical columns from clinical_df
    """
    # Aggregate cluster_mapping to one row per segment
    seg_agg = (
        cluster_mapping
        .groupby("segment_name")
        .agg(
            n_frames=("cluster_id", "count"),
            dominant_cluster=("cluster_id", lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
    )

    # Parse uuid and session from segment_name
    # Note: for V-records the segment_name starts with the V-code (e.g. "V029_seg_001"),
    # so the parsed uuid will be "V029" — not the numeric uuid in clinical_df.
    # We remap V-codes to numeric uuids via the clinical_df 'code' column.
    parsed = seg_agg["segment_name"].apply(parse_subject_session)
    seg_agg["uuid"] = [p[0] for p in parsed]      # may be "V029" or "7797"
    seg_agg["session_id"] = [p[1] for p in parsed]
    seg_agg["subject_session"] = seg_agg.apply(
        lambda r: r["uuid"] if not r["session_id"] else f"{r['uuid']}_{r['session_id']}",
        axis=1,
    )

    # Build segment_prefix → clinical uuid lookup.
    # segment_prefix is extracted from results_path by load_clinical()
    # and matches the subject_session prefix used in segment_name
    # (e.g. "7797_T1a_ADOS1" or "V012").
    seg_prefix_to_clin_uuid: dict[str, str] = {}
    if "segment_prefix" in clinical_df.columns:
        for clin_uuid, row in clinical_df.iterrows():
            pref = row.get("segment_prefix", "")
            if pref and isinstance(pref, str):
                seg_prefix_to_clin_uuid[pref] = str(clin_uuid)
    else:
        logger.warning(
            "clinical_df has no 'segment_prefix' column — "
            "run load_clinical() with a CSV that has 'results_path'"
        )

    # Map each segment's subject_session → clinical uuid (for the merge key)
    seg_agg["uuid_numeric"] = seg_agg["subject_session"].map(seg_prefix_to_clin_uuid)

    # Join clinical metadata on the clinical uuid
    n_before = len(seg_agg)
    clinical_reset = clinical_df.reset_index().rename(columns={"uuid": "uuid_numeric"})
    clinical_reset["uuid_numeric"] = clinical_reset["uuid_numeric"].astype(str)
    seg_agg["uuid_numeric"] = seg_agg["uuid_numeric"].astype(str)
    seg_agg = seg_agg.merge(clinical_reset, on="uuid_numeric", how="left")

    n_no_clinical = (seg_agg["uuid_numeric"] == "nan").sum()
    if n_no_clinical > 0:
        logger.warning(
            "%d / %d segments have no matching clinical record "
            "(segment_prefix not found in clinical CSV results_path)",
            n_no_clinical, n_before
        )

    logger.info(
        "Subject map: %d segments, %d unique subjects",
        len(seg_agg), seg_agg["uuid"].nunique()
    )
    return seg_agg


# ── Annotation → frame lookup ────────────────────────────────────────────────

def annotation_to_frames(
    annotation_row: pd.Series,
    segment_registry: dict[str, dict[str, int]],
    fps: int,
    subject_session_filter: str | None = None,
) -> list[tuple[str, int]]:
    """
    Convert a single annotation event (row from annotations DataFrame) to a
    list of (segment_name, relative_frame_index) pairs.

    Parameters
    ----------
    annotation_row : pd.Series
        Must have 'start' (seconds). Optionally 'stop' (seconds).
        For POINT status, only 'start' is used (single frame).
    segment_registry : dict
        Output of build_segment_registry().
    fps : int
        Frames per second.
    subject_session_filter : str | None
        If given, only match segments whose name starts with this prefix
        (e.g. "7797_T1a_ADOS1"). Required to avoid cross-subject matches.

    Returns
    -------
    List of (segment_name, relative_frame) tuples covering the annotation.
    Empty list if the annotation falls outside all known segments.
    """
    start_sec = float(annotation_row["start"])
    status = str(annotation_row.get("status", "START")).upper()

    if status == "POINT":
        stop_sec = start_sec
    else:
        stop_sec = float(annotation_row.get("stop", start_sec))
        if stop_sec < start_sec:
            logger.warning(
                "Annotation has stop < start (%.2f < %.2f), treating as POINT",
                stop_sec, start_sec
            )
            stop_sec = start_sec

    abs_start = int(start_sec * fps)
    abs_stop = int(stop_sec * fps)

    results: list[tuple[str, int]] = []

    for seg_name, info in segment_registry.items():
        if subject_session_filter and not seg_name.startswith(subject_session_filter):
            continue

        seg_start = info["start_abs_frame"]
        seg_end = info["end_abs_frame"]

        # Check overlap between annotation [abs_start, abs_stop] and segment [seg_start, seg_end]
        overlap_start = max(abs_start, seg_start)
        overlap_end = min(abs_stop, seg_end)

        if overlap_start > overlap_end:
            continue  # No overlap

        for abs_frame in range(overlap_start, overlap_end + 1):
            rel_frame = abs_frame - seg_start
            results.append((seg_name, rel_frame))

    if not results:
        logger.debug(
            "Annotation [%.2fs, %.2fs] (abs frames %d-%d) not matched to any segment "
            "(filter=%s). Check that tracking.nc attributes are correct.",
            start_sec, stop_sec, abs_start, abs_stop, subject_session_filter
        )

    return results


# ── Prevalence matrix ────────────────────────────────────────────────────────

def compute_prevalence_matrix(
    cluster_mapping: pd.DataFrame,
    subject_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-subject cluster prevalence (fraction of frames per cluster).

    Parameters
    ----------
    cluster_mapping : pd.DataFrame
        Output of load_cluster_mapping().
    subject_map : pd.DataFrame
        Output of build_subject_map() — must have 'segment_name' and 'uuid'.

    Returns
    -------
    pd.DataFrame of shape (N_subjects, N_clusters).
    Index = uuid (string). Columns = cluster IDs (integers).
    Each row sums to 1.0.
    """
    # Add numeric uuid to cluster_mapping via segment_name join
    # Use uuid_numeric so the prevalence matrix index aligns with clinical_df index
    uuid_col = "uuid_numeric" if "uuid_numeric" in subject_map.columns else "uuid"
    seg_to_uuid = subject_map.set_index("segment_name")[uuid_col].to_dict()
    cluster_mapping = cluster_mapping.copy()
    cluster_mapping["uuid"] = cluster_mapping["segment_name"].map(seg_to_uuid)

    n_unmapped = cluster_mapping["uuid"].isna().sum()
    if n_unmapped > 0:
        logger.warning(
            "%d frames (%d%%) have no matching uuid — excluded from prevalence matrix",
            n_unmapped, int(100 * n_unmapped / len(cluster_mapping))
        )
        cluster_mapping = cluster_mapping.dropna(subset=["uuid"])

    # Count frames per (uuid, cluster_id)
    counts = (
        cluster_mapping
        .groupby(["uuid", "cluster_id"])
        .size()
        .unstack(fill_value=0)
    )

    # Normalize to fractions
    prevalence = counts.div(counts.sum(axis=1), axis=0)

    # Sanity check
    row_sums = prevalence.sum(axis=1)
    bad_rows = (row_sums - 1.0).abs() > 1e-6
    if bad_rows.any():
        logger.warning(
            "Prevalence matrix: %d rows do not sum to 1.0 (max error=%.2e)",
            bad_rows.sum(), (row_sums - 1.0).abs().max()
        )

    logger.info(
        "Prevalence matrix: %d subjects × %d clusters",
        *prevalence.shape
    )
    return prevalence


# ── Coverage report ──────────────────────────────────────────────────────────

def build_coverage_report(
    cluster_mapping: pd.DataFrame,
    clinical_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    kinematics_df: pd.DataFrame | None,
    segment_registry: dict[str, dict[str, int]],
    fps: int = 20,
) -> pd.DataFrame:
    """
    Build a per-subject-session coverage report.

    For each subject-session, reports which data sources are available and,
    for annotated (V) records, how many annotation events matched to segments.

    Returns
    -------
    pd.DataFrame with columns:
        subject_session, uuid,
        has_clusters, has_kinematics, has_clinical, has_annotations,
        n_annotation_events, n_events_matched, n_events_unmatched,
        annotation_coverage_frac
    """
    # All subject-sessions from cluster mapping
    all_seg_names = cluster_mapping["segment_name"].unique()
    parsed = [parse_subject_session(s) for s in all_seg_names]
    subject_sessions = sorted(set(f"{u}_{s}" if s else u for u, s in parsed))

    # Build segment_prefix → clinical uuid lookup (same as build_subject_map)
    seg_prefix_to_clin_uuid: dict[str, str] = {}
    if "segment_prefix" in clinical_df.columns:
        for clin_uuid, row in clinical_df.iterrows():
            pref = row.get("segment_prefix", "")
            if pref and isinstance(pref, str):
                seg_prefix_to_clin_uuid[pref] = str(clin_uuid)

    rows = []
    for ss in subject_sessions:
        uuid = ss.split("_", maxsplit=1)[0] if "_" in ss else ss

        # Look up the clinical record for this subject-session via segment_prefix
        clin_uuid = seg_prefix_to_clin_uuid.get(ss, "")
        has_clusters = any(n.startswith(f"{ss}_seg_") for n in all_seg_names)

        if kinematics_df is not None:
            has_kinematics = ss in kinematics_df.index.get_level_values("subject_session")
        else:
            has_kinematics = False

        has_clinical = bool(clin_uuid) and clin_uuid in clinical_df.index

        # Annotation code: look it up from the clinical 'code' column
        # (V-records have code="V012"; non-V records have code=NaN or same as uuid)
        code = None
        if uuid.startswith("V") and not annotations_df.empty and uuid in annotations_df["code"].values:
            code = uuid
        elif has_clinical and "code" in clinical_df.columns:
            code_val = clinical_df.loc[clin_uuid, "code"]
            if isinstance(code_val, str):
                code = code_val

        has_annotations = (
            code is not None
            and not annotations_df.empty
            and code in annotations_df["code"].values
        )

        n_events = 0
        n_matched = 0
        n_unmatched = 0

        if has_annotations and code is not None:
            events = annotations_df[annotations_df["code"] == code]
            n_events = len(events)
            for _, ev in events.iterrows():
                frames = annotation_to_frames(
                    ev, segment_registry, fps, subject_session_filter=ss
                )
                if frames:
                    n_matched += 1
                else:
                    n_unmatched += 1

        rows.append({
            "subject_session": ss,
            "uuid": uuid,           # first token of subject_session (numeric id or V-code)
            "uuid_numeric": clin_uuid,  # full clinical uuid from clinical_df index
            "code": code,
            "has_clusters": has_clusters,
            "has_kinematics": has_kinematics,
            "has_clinical": has_clinical,
            "has_annotations": has_annotations,
            "n_annotation_events": n_events,
            "n_events_matched": n_matched,
            "n_events_unmatched": n_unmatched,
            "annotation_coverage_frac": (n_matched / n_events) if n_events > 0 else float("nan"),
        })

    df = pd.DataFrame(rows)

    # Summary logging
    n_unique_with_clinical = df.loc[df["has_clinical"], "uuid_numeric"].nunique()
    logger.info("=== Data Coverage Report ===")
    logger.info("  Total subject-sessions: %d", len(df))
    logger.info("  Has clusters:    %d (sessions)", df["has_clusters"].sum())
    logger.info("  Has kinematics:  %d (sessions)", df["has_kinematics"].sum())
    logger.info(
        "  Has clinical:    %d (sessions) / %d unique subjects",
        df["has_clinical"].sum(), n_unique_with_clinical
    )
    logger.info("  Has annotations: %d (V-records)", df["has_annotations"].sum())

    annotated = df[df["has_annotations"]]
    if len(annotated) > 0:
        total_events = annotated["n_annotation_events"].sum()
        total_matched = annotated["n_events_matched"].sum()
        frac = total_matched / total_events if total_events > 0 else 0
        logger.info(
            "  Annotation matching: %d / %d events matched (%.1f%%)",
            total_matched, total_events, 100 * frac
        )
        if frac < 0.8:
            logger.warning(
                "Annotation coverage below 80%% — check fps, segment boundaries, or annotation timing"
            )

    return df
