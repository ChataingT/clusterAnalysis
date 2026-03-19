"""
Main orchestration script for the clusterAnalysis pipeline.

Usage
-----
    python -m clusterAnalysis.src.run_analysis \\
        --config clusterAnalysis/configs/default.yaml \\
        --run-name my_run \\
        --log-level INFO

All analyses are gated by the config's `analyses.*` flags.
Each section catches exceptions independently — a failure in one analysis
does not abort the rest.

Outputs per run:
    results/{run_name}/data/         — all .csv and .pkl files
    results/{run_name}/plots/        — all figures
    results/{run_name}/run_summary.json
    results/{run_name}/cluster_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _setup_logging(level: str, log_file: Path | None = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info("Saved %s → %s (%d rows)", label, path, len(df))


def _timed(name: str):
    """Context manager that logs elapsed time for a section."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        t0 = time.perf_counter()
        logger.info("── START: %s ──", name)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            logger.info("── DONE:  %s (%.1fs) ──", name, elapsed)

    return _ctx()


def generate_cluster_report(
    cluster_behavior_labels: pd.DataFrame | None,
    binary_results: dict[str, pd.DataFrame],
    kinematic_profiles: pd.DataFrame | None,
    output_path: Path,
    alpha: float = 0.05,
    top_n_kinematics: int = 3,
) -> None:
    """
    Generate a human-readable Markdown report: one entry per cluster.

    Format:
      ## Cluster {id}
      - Nearest annotation: '{behavior}' (dist={d:.2f})
      - Kinematics: {metric1} (high), {metric2} (low), ...
      - ASD vs TD: p_fdr={p:.3f}, Cohen's d={d:.2f} ({direction})
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine all cluster IDs
    cluster_ids: set = set()
    if cluster_behavior_labels is not None and not cluster_behavior_labels.empty:
        cluster_ids.update(cluster_behavior_labels.index.tolist())
    for df in binary_results.values():
        if not df.empty and "cluster_id" in df.columns:
            cluster_ids.update(df["cluster_id"].tolist())
    if kinematic_profiles is not None and not kinematic_profiles.empty:
        cluster_ids.update(kinematic_profiles.index.tolist())

    cluster_ids = sorted(cluster_ids)

    lines = [
        "# Cluster Report",
        f"\nGenerated: {datetime.now().isoformat(timespec='seconds')}",
        f"\nTotal clusters: {len(cluster_ids)}",
        "\n---\n",
    ]

    for cid in cluster_ids:
        lines.append(f"## Cluster {cid}\n")

        # Annotation label
        if cluster_behavior_labels is not None and cid in cluster_behavior_labels.index:
            row = cluster_behavior_labels.loc[cid]
            lines.append(
                f"- **Nearest annotation behavior**: '{row['nearest_behavior']}' "
                f"(distance = {row['min_distance']:.3f})"
            )
        else:
            lines.append("- **Nearest annotation behavior**: N/A")

        # Kinematics
        if kinematic_profiles is not None and cid in kinematic_profiles.index:
            mean_cols = [c for c in kinematic_profiles.columns if c.endswith("__mean")]
            if mean_cols:
                profile = kinematic_profiles.loc[cid, mean_cols].astype(float)
                # Z-score profile relative to all clusters
                global_mean = kinematic_profiles[mean_cols].astype(float).mean(axis=0)
                global_std = kinematic_profiles[mean_cols].astype(float).std(axis=0) + 1e-8
                z = (profile - global_mean) / global_std
                top_high = z.nlargest(top_n_kinematics).index.str.replace("__mean", "")
                top_low = z.nsmallest(top_n_kinematics).index.str.replace("__mean", "")
                lines.append(
                    f"- **Kinematics (high)**: {', '.join(top_high.tolist())}"
                )
                lines.append(
                    f"- **Kinematics (low)**: {', '.join(top_low.tolist())}"
                )

        # Clinical correlations
        for key, df in binary_results.items():
            if df.empty or "cluster_id" not in df.columns:
                continue
            row_df = df[df["cluster_id"] == cid]
            if len(row_df) == 0:
                continue
            row = row_df.iloc[0]
            sig = "✓" if row.get("significant", False) else ""
            lines.append(
                f"- **{key}**: p_fdr={row['p_fdr']:.4f}, "
                f"Cohen's d={row['cohens_d']:.3f} ({row['direction']}) {sig}"
            )

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Cluster report written to %s", output_path)


def _synthesize_cluster_profiles(
    centroid_results: dict,
    ann_overlap_multilevel: dict[str, dict],
    binary_clinical_results: dict[str, pd.DataFrame],
    continuous_clinical_results: pd.DataFrame,
    frame_kinematic_profiles: pd.DataFrame | None,
    segment_kinematic_profiles: pd.DataFrame | None,
    alpha: float = 0.05,
    top_n_kinematics: int = 3,
    top_n_clinical: int = 3,
) -> pd.DataFrame:
    """
    Synthesize per-cluster profile CSV combining all analyses.

    For each cluster, collects:
    - Nearest annotation label at each of 3 hierarchy levels (from centroids)
    - Annotation overlap enrichment rank at L1 (from overlap analysis)
    - Top kinematic features (high/low z-score) from frame-level profiles
    - Clinical significance: ASD vs TD p_fdr, Cohen's d
    - Top continuous clinical correlations (metric, rho, p_fdr)

    Parameters
    ----------
    centroid_results : dict
        Output of run_annotation_centroids(). Expects 'cluster_behavior_labels'.
    ann_overlap_multilevel : dict
        Output of run_annotation_overlap_multilevel(). Per-level enrichment tables.
    binary_clinical_results : dict[str, pd.DataFrame]
        Keys like "binary_diagnosis".
    continuous_clinical_results : pd.DataFrame
        From clinical_analysis, columns: cluster_id, metric, rho, p_fdr.
    frame_kinematic_profiles : pd.DataFrame | None
        From run_kinematic_frame_analysis(), index=cluster_id, cols=metric__mean/std.
    segment_kinematic_profiles : pd.DataFrame | None
        From run_kinematic_analysis(), same schema. Used as fallback.
    alpha : float
        Significance threshold.

    Returns
    -------
    pd.DataFrame with one row per cluster_id and summary columns.
    """
    # Determine all cluster IDs
    cluster_ids: set[int] = set()
    if frame_kinematic_profiles is not None and not frame_kinematic_profiles.empty:
        cluster_ids.update(frame_kinematic_profiles.index.tolist())
    if segment_kinematic_profiles is not None and not segment_kinematic_profiles.empty:
        cluster_ids.update(segment_kinematic_profiles.index.tolist())
    for df in binary_clinical_results.values():
        if not df.empty and "cluster_id" in df.columns:
            cluster_ids.update(df["cluster_id"].tolist())
    if not continuous_clinical_results.empty and "cluster_id" in continuous_clinical_results.columns:
        cluster_ids.update(continuous_clinical_results["cluster_id"].tolist())
    cluster_behavior_labels: pd.DataFrame = centroid_results.get(
        "cluster_behavior_labels", pd.DataFrame()
    )
    if not cluster_behavior_labels.empty:
        cluster_ids.update(cluster_behavior_labels.index.tolist())

    if not cluster_ids:
        logger.warning("Cluster profiles synthesis: no cluster IDs found from any analysis")
        return pd.DataFrame()

    cluster_ids_sorted = sorted(cluster_ids)

    # Pre-compute z-scored kinematic profiles (prefer frame-level)
    kin_profiles = frame_kinematic_profiles if (
        frame_kinematic_profiles is not None and not frame_kinematic_profiles.empty
    ) else segment_kinematic_profiles

    kin_z: pd.DataFrame | None = None
    if kin_profiles is not None and not kin_profiles.empty:
        mean_cols = [c for c in kin_profiles.columns if c.endswith("__mean")]
        if mean_cols:
            kin_means = kin_profiles[mean_cols].astype(float)
            global_mean = kin_means.mean(axis=0)
            global_std = kin_means.std(axis=0) + 1e-8
            kin_z = (kin_means - global_mean) / global_std

    # Pre-compute annotation enrichment top label per cluster at L1
    enrichment_top_label: dict[int, str] = {}
    l1_enrichment = ann_overlap_multilevel.get("L1", {}).get("enrichment", pd.DataFrame())
    if not l1_enrichment.empty:
        for cid_col in l1_enrichment.columns:
            try:
                cid = int(cid_col)
            except (ValueError, TypeError):
                continue
            col_data = l1_enrichment[cid_col].dropna()
            if col_data.empty:
                continue
            enrichment_top_label[cid] = str(col_data.idxmax())

    rows = []
    for cid in cluster_ids_sorted:
        row: dict = {"cluster_id": cid}

        # Annotation centroid label
        if not cluster_behavior_labels.empty and cid in cluster_behavior_labels.index:
            cb = cluster_behavior_labels.loc[cid]
            row["nearest_behavior_L1"] = str(cb.get("nearest_behavior", ""))
            row["centroid_distance_L1"] = float(cb.get("min_distance", float("nan")))
        else:
            row["nearest_behavior_L1"] = ""
            row["centroid_distance_L1"] = float("nan")

        # Enrichment-based top behavior at L1
        row["enrichment_top_behavior_L1"] = enrichment_top_label.get(cid, "")

        # Kinematic top features
        if kin_z is not None and cid in kin_z.index:
            z_row = kin_z.loc[cid]
            top_high = z_row.nlargest(top_n_kinematics).index.str.replace("__mean", "").tolist()
            top_low = z_row.nsmallest(top_n_kinematics).index.str.replace("__mean", "").tolist()
            row["kinematic_high"] = "; ".join(top_high)
            row["kinematic_low"] = "; ".join(top_low)
        else:
            row["kinematic_high"] = ""
            row["kinematic_low"] = ""

        # Binary clinical results (ASD vs TD)
        for key, df in binary_clinical_results.items():
            if df.empty or "cluster_id" not in df.columns:
                continue
            sub = df[df["cluster_id"] == cid]
            if sub.empty:
                continue
            r = sub.iloc[0]
            group = key.replace("binary_", "")
            row[f"{group}_p_fdr"] = float(r.get("p_fdr", float("nan")))
            row[f"{group}_cohens_d"] = float(r.get("cohens_d", float("nan")))
            row[f"{group}_direction"] = str(r.get("direction", ""))
            row[f"{group}_significant"] = bool(r.get("significant", False))

        # Top continuous clinical correlations
        if not continuous_clinical_results.empty:
            cols_needed = {"cluster_id", "metric", "rho", "p_fdr"}
            if cols_needed.issubset(set(continuous_clinical_results.columns)):
                sub = continuous_clinical_results[
                    (continuous_clinical_results["cluster_id"] == cid) &
                    (continuous_clinical_results["p_fdr"] < alpha)
                ].copy()
                if not sub.empty:
                    sub["abs_rho"] = sub["rho"].abs()
                    sub = sub.sort_values("abs_rho", ascending=False).head(top_n_clinical)
                    row["top_clinical_correlations"] = "; ".join(
                        f"{r['metric']} (ρ={r['rho']:.2f})"
                        for _, r in sub.iterrows()
                    )
                else:
                    row["top_clinical_correlations"] = ""

        rows.append(row)

    profiles_df = pd.DataFrame(rows).set_index("cluster_id")
    logger.info(
        "Cluster profiles synthesis: %d clusters, %d columns",
        len(profiles_df), len(profiles_df.columns)
    )
    return profiles_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the clusterAnalysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", default=None,
                        help="Override run_name from config")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    # ── Config ────────────────────────────────────────────────────────────────
    from .config import load_config
    cfg = load_config(args.config)
    if args.run_name:
        cfg.output.run_name = args.run_name

    # Resolve relative paths relative to the current working directory
    # (the SLURM script cd's to humanLISBET-paper/ before running)
    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else (Path.cwd() / p).resolve()

    cfg.data.cluster_mapping = _resolve(cfg.data.cluster_mapping)
    cfg.data.cross_video_train = _resolve(cfg.data.cross_video_train)
    cfg.data.clinical_csv = _resolve(cfg.data.clinical_csv)
    cfg.data.annotations_csv = _resolve(cfg.data.annotations_csv)
    cfg.data.pose_records_dir = _resolve(cfg.data.pose_records_dir)
    cfg.output.results_dir = _resolve(cfg.output.results_dir)

    output_dir = cfg.output.results_dir / cfg.output.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "run.log"
    _setup_logging(args.log_level, log_file)

    logger.info("=" * 60)
    logger.info("clusterAnalysis pipeline — run: %s", cfg.output.run_name)
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 60)

    run_start = time.perf_counter()
    summary: dict = {
        "run_name": cfg.output.run_name,
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "analyses_enabled": vars(cfg.analyses),
        "errors": [],
    }

    # ── Step 1: Load data ────────────────────────────────────────────────────
    from .data import load_cluster_mapping, load_clinical, load_annotations, load_kinematics_summary
    from .linking import (build_segment_registry, build_subject_map,
                          compute_prevalence_matrix, build_coverage_report,
                          parse_subject_session)

    with _timed("Load cluster mapping"):
        cluster_mapping = load_cluster_mapping(cfg.data.cluster_mapping)

    with _timed("Load clinical data"):
        clinical_df = load_clinical(cfg.data.clinical_csv)

    with _timed("Load annotations"):
        annotations_df = load_annotations(cfg.data.annotations_csv)

    # Derive subject-session IDs from cluster mapping
    all_seg_names = cluster_mapping["segment_name"].unique()
    parsed = [parse_subject_session(s) for s in all_seg_names]
    subject_session_ids = sorted(set(f"{u}_{s}" if s else u for u, s in parsed))
    summary["n_subject_sessions"] = len(subject_session_ids)

    # Load kinematics
    kinematics_df = None
    with _timed("Load kinematics"):
        try:
            kinematics_df = load_kinematics_summary(
                cfg.data.pose_records_dir,
                subject_session_ids,
                use_normalized=cfg.kinematics.use_normalized,
                metrics=cfg.kinematics.metrics,
            )
        except Exception as exc:
            logger.error("Kinematics loading failed: %s", exc, exc_info=True)
            summary["errors"].append(f"kinematics_load: {exc}")

    # ── Step 2: Build segment registry ──────────────────────────────────────
    segment_registry = {}
    with _timed("Build segment registry"):
        try:
            segment_registry = build_segment_registry(
                cfg.data.pose_records_dir,
                subject_session_ids,
                fps=cfg.data.fps,
            )
            summary["n_segments_registered"] = len(segment_registry)
        except Exception as exc:
            logger.error("Segment registry failed: %s", exc, exc_info=True)
            summary["errors"].append(f"segment_registry: {exc}")

    # ── Step 3: Build subject map + prevalence matrix ────────────────────────
    subject_map = pd.DataFrame()
    prevalence_matrix = pd.DataFrame()
    with _timed("Build subject map & prevalence matrix"):
        try:
            subject_map = build_subject_map(cluster_mapping, clinical_df)
            prevalence_matrix = compute_prevalence_matrix(cluster_mapping, subject_map)
            summary["n_subjects"] = len(prevalence_matrix)
            summary["n_clusters"] = len(prevalence_matrix.columns)
            if cfg.output.save_data:
                _save_csv(prevalence_matrix, data_dir / "prevalence_matrix.csv", "prevalence matrix")
        except Exception as exc:
            logger.error("Subject map / prevalence matrix failed: %s", exc, exc_info=True)
            summary["errors"].append(f"prevalence_matrix: {exc}")

    # ── Step 4: Coverage report ──────────────────────────────────────────────
    with _timed("Build coverage report"):
        try:
            coverage_df = build_coverage_report(
                cluster_mapping, clinical_df, annotations_df,
                kinematics_df, segment_registry, fps=cfg.data.fps,
            )
            summary["n_annotated"] = int(coverage_df["has_annotations"].sum())
            summary["annotation_coverage"] = float(
                coverage_df.loc[coverage_df["has_annotations"], "annotation_coverage_frac"].mean()
            ) if coverage_df["has_annotations"].any() else 0.0
            if cfg.output.save_data:
                _save_csv(coverage_df, data_dir / "data_coverage.csv", "coverage report")
            if cfg.output.save_plots:
                from .visualization import plot_coverage_summary
                plot_coverage_summary(coverage_df, output_dir,
                                      formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi)
        except Exception as exc:
            logger.error("Coverage report failed: %s", exc, exc_info=True)
            summary["errors"].append(f"coverage_report: {exc}")

    # Determine V-record UUIDs
    v_uuids: set[str] = set()
    if not coverage_df.empty if "coverage_df" in dir() else False:
        pass  # coverage_df is defined above
    try:
        v_uuids = set(
            coverage_df.loc[coverage_df["has_annotations"], "uuid"].dropna().astype(str)
        )
    except Exception:
        pass

    # ── Analysis 1: Annotation overlap (3-level hierarchy) ──────────────────
    ann_overlap_results: dict = {}      # L1 results (legacy key for cluster report)
    ann_overlap_multilevel: dict = {}   # all 3 levels
    if cfg.analyses.annotation_overlap:
        with _timed("Annotation overlap (3-level)"):
            try:
                from .annotation_analysis import run_annotation_overlap_multilevel
                ann_overlap_multilevel = run_annotation_overlap_multilevel(
                    cluster_mapping, annotations_df,
                    segment_registry, clinical_df,
                    fps=cfg.data.fps,
                    min_frames_level1=cfg.annotation.min_frames_level1,
                    min_frames_level2=cfg.annotation.min_frames_level2,
                    min_frames_level3=cfg.annotation.min_frames_level3,
                )
                # Keep L1 as the canonical result for other analyses
                ann_overlap_results = ann_overlap_multilevel.get("L1", {})

                if cfg.output.save_data:
                    for level, level_res in ann_overlap_multilevel.items():
                        for key in ("contingency", "enrichment", "per_record"):
                            df = level_res.get(key, pd.DataFrame())
                            if not isinstance(df, pd.DataFrame) or df.empty:
                                continue
                            _save_csv(
                                df,
                                data_dir / f"annotation_cluster_{key}_{level}.csv",
                                f"{key}_{level}",
                            )
                if cfg.output.save_plots:
                    from .visualization import (
                        plot_annotation_cluster_heatmap,
                        plot_annotation_cluster_bars,
                    )
                    for level, level_res in ann_overlap_multilevel.items():
                        plot_annotation_cluster_heatmap(
                            level_res.get("enrichment", pd.DataFrame()),
                            output_dir,
                            stem=f"annotation_cluster_heatmap_{level}",
                            formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                        )
                    # Bar charts for L1 only (per-behavior, most interpretable)
                    plot_annotation_cluster_bars(
                        ann_overlap_results.get("contingency", pd.DataFrame()),
                        output_dir, formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                    )
            except Exception as exc:
                logger.error("Annotation overlap failed: %s", exc, exc_info=True)
                summary["errors"].append(f"annotation_overlap: {exc}")

    # ── Analysis 2: Annotation centroids (global cluster centroids) ──────────
    centroid_results: dict = {}
    global_cluster_centroids: pd.DataFrame = pd.DataFrame()
    if cfg.analyses.annotation_centroids:
        with _timed("Annotation centroids"):
            try:
                from .annotation_analysis import (
                    run_annotation_centroids,
                    save_annotation_centroids,
                    compute_global_cluster_centroids,
                )
                # A4: compute global cluster centroids from ALL subjects
                with _timed("Global cluster centroids (all subjects)"):
                    global_cluster_centroids = compute_global_cluster_centroids(
                        cluster_mapping,
                        embeddings_dir=cfg.data.embeddings_dir,
                        segment_registry=segment_registry,
                        subject_session_ids=subject_session_ids,
                    )
                    if cfg.output.save_data and not global_cluster_centroids.empty:
                        _save_csv(
                            global_cluster_centroids,
                            data_dir / "cluster_centroids_global.csv",
                            "global cluster centroids",
                        )

                centroid_results = run_annotation_centroids(
                    cluster_mapping, annotations_df,
                    segment_registry, clinical_df,
                    embeddings_dir=cfg.data.embeddings_dir,
                    fps=cfg.data.fps,
                )
                if centroid_results:
                    if cfg.output.save_data:
                        save_annotation_centroids(
                            centroid_results.get("annotation_centroids", {}),
                            output_dir,
                        )
                        for key in ("distance_matrix", "cluster_behavior_labels"):
                            df = centroid_results.get(key, pd.DataFrame())
                            if not isinstance(df, pd.DataFrame) or df.empty:
                                continue
                            _save_csv(df, data_dir / f"cluster_{key}.csv", key)
                    if cfg.output.save_plots:
                        from .visualization import plot_annotation_centroid_distance
                        plot_annotation_centroid_distance(
                            centroid_results.get("distance_matrix", pd.DataFrame()),
                            output_dir, formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                        )
            except Exception as exc:
                logger.error("Annotation centroids failed: %s", exc, exc_info=True)
                summary["errors"].append(f"annotation_centroids: {exc}")

    # ── Analysis 3: Clinical correlations ───────────────────────────────────
    clinical_results: dict[str, pd.DataFrame] = {}
    if cfg.analyses.clinical_correlations and not prevalence_matrix.empty:
        with _timed("Clinical correlations"):
            try:
                from .clinical_analysis import run_clinical_analysis
                clinical_results = run_clinical_analysis(
                    prevalence_matrix, clinical_df,
                    binary_groups=cfg.clinical.binary_groups,
                    continuous_metrics=cfg.clinical.continuous,
                    alpha=cfg.statistics.alpha,
                    fdr_method=cfg.statistics.fdr_method,
                )
                if cfg.output.save_data:
                    for key, df in clinical_results.items():
                        _save_csv(df, data_dir / f"clinical_{key}_results.csv", key)
                if cfg.output.save_plots:
                    from .visualization import (
                        plot_clinical_volcano,
                        plot_clinical_correlation_heatmap,
                        plot_clinical_violin,
                    )
                    for key, df in clinical_results.items():
                        if key.startswith("binary_"):
                            group_col = key.replace("binary_", "")
                            unique_vals = clinical_df[group_col].dropna().unique() if group_col in clinical_df.columns else []
                            groups = tuple(sorted(unique_vals))[:2] if len(unique_vals) == 2 else ("A", "B")
                            plot_clinical_volcano(
                                df, group_col, groups, output_dir,
                                alpha=cfg.statistics.alpha,
                                formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                            )
                            if not prevalence_matrix.empty:
                                plot_clinical_violin(
                                    prevalence_matrix, clinical_df, df,
                                    group_col, groups, output_dir,
                                    formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                                )
                    if "continuous" in clinical_results:
                        plot_clinical_correlation_heatmap(
                            clinical_results["continuous"], output_dir,
                            alpha=cfg.statistics.alpha,
                            formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                        )
            except Exception as exc:
                logger.error("Clinical correlations failed: %s", exc, exc_info=True)
                summary["errors"].append(f"clinical_correlations: {exc}")

    # ── Analysis 4: Kinematic profiles ──────────────────────────────────────
    kinematic_results: dict[str, pd.DataFrame] = {}
    if cfg.analyses.kinematic_profiles and kinematics_df is not None and not subject_map.empty:
        with _timed("Kinematic profiles"):
            try:
                from .kinematic_analysis import run_kinematic_analysis
                metric_cols = [c for c in kinematics_df.columns if c.endswith("__mean")]
                kinematic_results = run_kinematic_analysis(
                    subject_map, kinematics_df,
                    metric_columns=metric_cols,
                    v_uuids=v_uuids,
                    min_frames_per_cluster=cfg.statistics.min_frames_per_cluster,
                )
                if cfg.output.save_data:
                    for key, df in kinematic_results.items():
                        _save_csv(df, data_dir / f"cluster_kinematic_{key}.csv", f"kinematics_{key}")
                if cfg.output.save_plots:
                    from .visualization import plot_kinematic_heatmap, plot_vsubset_consistency
                    if "global" in kinematic_results:
                        plot_kinematic_heatmap(
                            kinematic_results["global"], output_dir,
                            stem="kinematic_heatmap_global",
                            formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                        )
                    if "vsubset" in kinematic_results and "nonv" in kinematic_results:
                        plot_vsubset_consistency(
                            kinematic_results["vsubset"], kinematic_results["nonv"],
                            output_dir, formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                        )
            except Exception as exc:
                logger.error("Kinematic profiles failed: %s", exc, exc_info=True)
                summary["errors"].append(f"kinematic_profiles: {exc}")

    # ── Analysis 5: Embedding × kinematics ──────────────────────────────────
    if cfg.analyses.embedding_kinematics and kinematics_df is not None and not subject_map.empty:
        with _timed("Embedding × kinematics correlation"):
            try:
                from .embedding_analysis import run_embedding_kinematic_correlation
                metric_cols = [c for c in kinematics_df.columns if c.endswith("__mean")]
                emb_results = run_embedding_kinematic_correlation(
                    cfg.data.embeddings_dir,
                    kinematics_df, subject_map,
                    metric_columns=metric_cols,
                    alpha=cfg.statistics.alpha,
                    fdr_method=cfg.statistics.fdr_method,
                )
                if cfg.output.save_data:
                    for key, df in emb_results.items():
                        _save_csv(df, data_dir / f"embedding_kinematic_{key}.csv", f"emb_{key}")
                if cfg.output.save_plots:
                    from .visualization import plot_embedding_kinematic_heatmap
                    plot_embedding_kinematic_heatmap(
                        emb_results.get("rho", pd.DataFrame()),
                        emb_results.get("significant", pd.DataFrame()),
                        output_dir, formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                    )
            except Exception as exc:
                logger.error("Embedding × kinematics failed: %s", exc, exc_info=True)
                summary["errors"].append(f"embedding_kinematics: {exc}")

    # ── Analysis 6: Frame-level kinematic × cluster profiles (A2) ───────────
    frame_kinematic_results: dict[str, pd.DataFrame] = {}
    if cfg.analyses.kinematic_frame_analysis:
        with _timed("Frame-level kinematic × cluster profiles"):
            try:
                from .kinematic_frame_analysis import run_kinematic_frame_analysis
                frame_kinematic_results = run_kinematic_frame_analysis(
                    cluster_mapping=cluster_mapping,
                    pose_records_dir=cfg.data.pose_records_dir,
                    subject_session_ids=subject_session_ids,
                    use_normalized=cfg.kinematics.use_normalized,
                    metric_cols=cfg.kinematics.metrics,
                    min_frames_per_cluster=cfg.statistics.min_frames_per_cluster,
                    fdr_method=cfg.statistics.fdr_method,
                    alpha=cfg.statistics.alpha,
                )
                if cfg.output.save_data:
                    for key, df in frame_kinematic_results.items():
                        if not isinstance(df, pd.DataFrame) or df.empty:
                            continue
                        _save_csv(
                            df,
                            data_dir / f"cluster_kinematic_frame_{key}.csv",
                            f"frame_kinematic_{key}",
                        )
                if cfg.output.save_plots:
                    from .visualization import (
                        plot_kinematic_frame_heatmap,
                        plot_kruskal_wallis_results,
                    )
                    if "profiles" in frame_kinematic_results:
                        plot_kinematic_frame_heatmap(
                            frame_kinematic_results["profiles"],
                            output_dir,
                            formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                        )
                    if "kruskal" in frame_kinematic_results:
                        plot_kruskal_wallis_results(
                            frame_kinematic_results["kruskal"],
                            output_dir,
                            alpha=cfg.statistics.alpha,
                            formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                        )
            except Exception as exc:
                logger.error("Frame-level kinematic analysis failed: %s", exc, exc_info=True)
                summary["errors"].append(f"kinematic_frame_analysis: {exc}")

    # ── Analysis 7: Annotation kinematics (A3) ───────────────────────────────
    annotation_kinematics_results: dict = {}
    if cfg.analyses.annotation_kinematics and not annotations_df.empty:
        with _timed("Annotation kinematics (frame-level, 3 levels)"):
            try:
                from .annotation_analysis import run_annotation_kinematics
                annotation_kinematics_results = run_annotation_kinematics(
                    annotations_df=annotations_df,
                    segment_registry=segment_registry,
                    clinical_df=clinical_df,
                    pose_records_dir=cfg.data.pose_records_dir,
                    fps=cfg.data.fps,
                    use_normalized=cfg.kinematics.use_normalized,
                    min_frames_level1=cfg.annotation.min_frames_level1,
                    min_frames_level2=cfg.annotation.min_frames_level2,
                    min_frames_level3=cfg.annotation.min_frames_level3,
                )
                if cfg.output.save_data:
                    for level, level_res in annotation_kinematics_results.items():
                        for key, df in level_res.items():
                            if not isinstance(df, pd.DataFrame) or df.empty:
                                continue
                            _save_csv(
                                df,
                                data_dir / f"annotation_kinematics_{level}_{key}.csv",
                                f"ann_kin_{level}_{key}",
                            )
                if cfg.output.save_plots:
                    from .visualization import plot_annotation_kinematics_heatmap
                    for level, level_res in annotation_kinematics_results.items():
                        plot_annotation_kinematics_heatmap(
                            level_res.get("profiles", pd.DataFrame()),
                            level_res.get("background"),
                            output_dir,
                            level=level,
                            formats=cfg.output.plot_formats, dpi=cfg.output.figure_dpi,
                        )
            except Exception as exc:
                logger.error("Annotation kinematics failed: %s", exc, exc_info=True)
                summary["errors"].append(f"annotation_kinematics: {exc}")

    # ── Analysis 8: Cluster profiles synthesis (A5) ──────────────────────────
    cluster_profiles_df: pd.DataFrame = pd.DataFrame()
    if cfg.analyses.cluster_profiles:
        with _timed("Cluster profiles synthesis"):
            try:
                cluster_profiles_df = _synthesize_cluster_profiles(
                    centroid_results=centroid_results,
                    ann_overlap_multilevel=ann_overlap_multilevel,
                    binary_clinical_results={
                        k: v for k, v in clinical_results.items() if k.startswith("binary_")
                    },
                    continuous_clinical_results=clinical_results.get("continuous", pd.DataFrame()),
                    frame_kinematic_profiles=frame_kinematic_results.get("profiles"),
                    segment_kinematic_profiles=kinematic_results.get("global"),
                    alpha=cfg.statistics.alpha,
                )
                if cfg.output.save_data and not cluster_profiles_df.empty:
                    _save_csv(
                        cluster_profiles_df,
                        data_dir / "cluster_profiles.csv",
                        "cluster profiles synthesis",
                    )
            except Exception as exc:
                logger.error("Cluster profiles synthesis failed: %s", exc, exc_info=True)
                summary["errors"].append(f"cluster_profiles: {exc}")

    # ── Cluster report ───────────────────────────────────────────────────────
    with _timed("Generate cluster report"):
        try:
            binary_results_for_report = {
                k: v for k, v in clinical_results.items() if k.startswith("binary_")
            }
            # Prefer frame-level kinematic profiles (more accurate) for report
            kin_for_report = (
                frame_kinematic_results.get("profiles")
                if frame_kinematic_results.get("profiles") is not None
                and not frame_kinematic_results.get("profiles", pd.DataFrame()).empty
                else kinematic_results.get("global")
            )
            generate_cluster_report(
                cluster_behavior_labels=centroid_results.get("cluster_behavior_labels"),
                binary_results=binary_results_for_report,
                kinematic_profiles=kin_for_report,
                output_path=output_dir / "cluster_report.md",
                alpha=cfg.statistics.alpha,
            )
        except Exception as exc:
            logger.error("Cluster report generation failed: %s", exc, exc_info=True)
            summary["errors"].append(f"cluster_report: {exc}")

    # ── Write run summary ────────────────────────────────────────────────────
    summary["elapsed_seconds"] = round(time.perf_counter() - run_start, 1)
    summary["n_errors"] = len(summary["errors"])

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Run summary written to %s", summary_path)

    if summary["errors"]:
        logger.warning(
            "Pipeline completed with %d error(s):\n  %s",
            len(summary["errors"]), "\n  ".join(summary["errors"])
        )
    else:
        logger.info("Pipeline completed successfully in %.1fs", summary["elapsed_seconds"])

    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
