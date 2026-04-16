"""
Visualization module for the clusterAnalysis pipeline.

All plot functions:
- accept an output_dir (Path) and formats list (e.g. ["png", "pdf"])
- save figures to output_dir/plots/
- use a consistent seaborn theme
- include informative titles, axis labels, and colorbars
- log the path of each saved figure

Functions:
    plot_coverage_summary          — data availability per record
    plot_annotation_cluster_heatmap — behaviors × clusters (enrichment)
    plot_annotation_cluster_bars   — per-behavior top-N cluster bar charts
    plot_annotation_centroid_distance — behavior × cluster distance matrix
    plot_annotation_coverage_per_record — matched/unmatched events per V-record
    plot_clinical_volcano          — volcano plot for ASD vs TD
    plot_clinical_correlation_heatmap — Spearman rho heatmap (clusters × metrics)
    plot_clinical_violin           — top discriminative clusters, ASD vs TD violins
    plot_kinematic_heatmap         — clusters × kinematics (z-scored)
    plot_vsubset_consistency       — V vs non-V kinematic scatter
    plot_embedding_kinematic_heatmap — embedding dims × kinematics
    plot_enrichment_significance_overview — volcano + N-events + significance heatmap
    plot_centroid_bootstrap_overview — CI width heatmap + nearest-cluster stability
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for SLURM
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Global theme ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)



PALETTE = {"ASD": "#E84040", "TD": "#4E93C8"}
FIGSIZE_WIDE = (14, 7)
FIGSIZE_SQUARE = (10, 10)
FIGSIZE_TALL = (10, 14)


def _save_fig(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    """Save a matplotlib figure in all requested formats."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = plots_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved plot: %s", path)
    plt.close(fig)


def _clustermap_save(
    g: "sns.matrix.ClusterGrid",
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
) -> None:
    """Save a seaborn ClusterGrid (clustermap) figure."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = plots_dir / f"{stem}.{fmt}"
        g.figure.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved plot: %s", path)
    plt.close(g.figure)


# ── Coverage summary ──────────────────────────────────────────────────────────

def plot_coverage_summary(
    coverage_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str] = ("png", "pdf"),
    dpi: int = 300,
) -> None:
    """
    Horizontal bar chart showing how many records have each data type.
    Annotated records also show annotation matching fraction.
    """
    counts = {
        "Clusters": coverage_df["has_clusters"].sum(),
        "Kinematics": coverage_df["has_kinematics"].sum(),
        "Clinical": coverage_df["has_clinical"].sum(),
        "Annotations (V-records)": coverage_df["has_annotations"].sum(),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: data source availability
    ax = axes[0]
    labels = list(counts.keys())
    values = list(counts.values())
    colors = ["#4E93C8", "#5DB07A", "#F5A623", "#E84040"]
    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    ax.bar_label(bars, padding=3, fmt="%d")
    ax.set_xlabel("Number of subject-sessions")
    ax.set_title("Data availability per source")
    ax.set_xlim(0, max(values) * 1.15)
    ax.invert_yaxis()

    # Right: annotation matching (V-records only)
    ax2 = axes[1]
    ann_df = coverage_df[coverage_df["has_annotations"] & coverage_df["n_annotation_events"] > 0].copy()
    if len(ann_df) > 0:
        ann_df = ann_df.sort_values("annotation_coverage_frac", ascending=True)
        ypos = range(len(ann_df))
        ax2.barh(ypos, ann_df["n_events_matched"], color="#5DB07A", label="Matched")
        ax2.barh(ypos, ann_df["n_events_unmatched"], left=ann_df["n_events_matched"],
                 color="#E84040", label="Unmatched")
        ax2.set_yticks(list(ypos))
        ax2.set_yticklabels(ann_df["code"].fillna(ann_df["subject_session"]).values, fontsize=8)
        ax2.set_xlabel("Number of annotation events")
        ax2.set_title("Annotation event matching per V-record")
        ax2.legend(loc="lower right")
    else:
        ax2.text(0.5, 0.5, "No annotation data available",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Annotation event matching")

    fig.suptitle("Data Coverage Summary", fontsize=13, y=1.01)
    fig.tight_layout()
    _save_fig(fig, output_dir, "coverage_summary", formats, dpi)


# ── Annotation-cluster heatmap ────────────────────────────────────────────────

def plot_annotation_cluster_heatmap(
    enrichment_df: pd.DataFrame,
    output_dir: Path,
    stem: str = "annotation_cluster_heatmap",
    top_n_clusters: int = 30,
    formats: list[str] = ("png", "pdf"),
    dpi: int = 300,
) -> None:
    """Heatmap of annotation enrichment scores (behaviors × top-N clusters) with dendrograms."""
    if enrichment_df.empty:
        logger.warning("Skipping annotation-cluster heatmap: empty data")
        return

    # Select top-N clusters by max enrichment
    max_enrichment = enrichment_df.max(axis=0)
    top_clusters = max_enrichment.nlargest(min(top_n_clusters, len(enrichment_df.columns))).index
    data = enrichment_df[top_clusters].fillna(0)

    n_rows, n_cols = data.shape
    figsize = (max(13, n_cols * 0.4 + 3), max(6, n_rows * 0.5 + 3))

    g = sns.clustermap(
        data,
        method="ward", metric="euclidean",
        row_cluster=(n_rows >= 2), col_cluster=(n_cols >= 2),
        cmap="RdYlBu_r", center=1.0, vmin=0, vmax=max(3, float(data.values.max())),
        linewidths=0.3, linecolor="white",
        xticklabels=True, yticklabels=True,
        figsize=figsize,
        dendrogram_ratio=(0.12, 0.08),
        cbar_kws={"label": "Enrichment (observed / expected)", "shrink": 0.6},
    )
    g.ax_heatmap.set_xlabel("Cluster ID", labelpad=8)
    g.ax_heatmap.set_ylabel("Behavior", labelpad=8)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)
    g.figure.suptitle(
        f"Cluster enrichment during annotations (top {n_cols} clusters)",
        y=1.01, fontsize=11,
    )
    _clustermap_save(g, output_dir, stem, formats, dpi)


# ── Per-behavior bar charts ───────────────────────────────────────────────────

def plot_annotation_cluster_bars(
    contingency_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
    formats: list[str] = ("png", "pdf"),
    dpi: int = 300,
) -> None:
    """One bar chart per behavior: top-N clusters by frame count."""
    if contingency_df.empty:
        return

    bars_dir = output_dir / "plots" / "annotation_cluster_bars"
    bars_dir.mkdir(parents=True, exist_ok=True)

    for behavior in contingency_df.index:
        row = contingency_df.loc[behavior].sort_values(ascending=False)
        top = row.head(top_n)
        if top.sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(top.index.astype(str), top.values, color="#4E93C8", edgecolor="white")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Annotated frames")
        ax.set_title(f"Top {len(top)} clusters for behavior: '{behavior}'")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        stem = f"bar_{behavior.replace(' ', '_')}"
        for fmt in formats:
            path = bars_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    logger.info("Annotation bar charts saved to %s", bars_dir)


# ── Annotation centroid distance matrix ──────────────────────────────────────

def plot_annotation_centroid_distance(
    distance_matrix: pd.DataFrame,
    output_dir: Path,
    top_n_clusters: int = 40,
    title_suffix: str = "",
    filename_suffix: str = "",
    formats: list[str] = ("png", "pdf"),
    dpi: int = 300,
) -> None:
    """Heatmap of 128D centroid distances between annotation centroids and cluster centroids."""
    if distance_matrix.empty:
        return

    # Infer metric from suffix for the colorbar label
    metric_label = "distance (128D)"
    for m in ("euclidean", "cosine", "mahalanobis"):
        if m in title_suffix.lower():
            metric_label = f"{m} distance (128D)"
            break

    # Show only top-N clusters (lowest mean distance across behaviors)
    mean_dist = distance_matrix.mean(axis=0)
    top_clusters = mean_dist.nsmallest(min(top_n_clusters, len(distance_matrix.columns))).index
    data = distance_matrix[top_clusters]

    n_rows, n_cols = data.shape
    figsize = (max(13, n_cols * 0.35 + 3), max(5, n_rows * 0.5 + 3))

    g = sns.clustermap(
        data,
        method="ward", metric="euclidean",
        row_cluster=(n_rows >= 2), col_cluster=(n_cols >= 2),
        cmap="viridis_r",
        xticklabels=True, yticklabels=True,
        figsize=figsize,
        dendrogram_ratio=(0.12, 0.08),
        linewidths=0.2, linecolor="white",
        cbar_kws={"label": metric_label.capitalize(), "shrink": 0.6},
    )
    g.ax_heatmap.set_xlabel("Cluster ID", labelpad=8)
    g.ax_heatmap.set_ylabel("Behavior", labelpad=8)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)
    g.figure.suptitle(
        f"Annotation–cluster centroid distances (top {n_cols} closest clusters){title_suffix}",
        y=1.01, fontsize=11,
    )
    filename = f"annotation_centroid_distance{filename_suffix}"
    _clustermap_save(g, output_dir, filename, formats, dpi)


# ── Clinical volcano plot ─────────────────────────────────────────────────────

def plot_clinical_volcano(
    binary_results: pd.DataFrame,
    group_col: str,
    groups: tuple[str, str],
    output_dir: Path,
    alpha: float = 0.05,
    formats: list[str] = ("png", "pdf"),
    dpi: int = 300,
) -> None:
    """Volcano plot: Cohen's d vs -log10(p_fdr), colored by direction."""
    if binary_results.empty:
        return

    df = binary_results.copy()
    df["-log10_p"] = -np.log10(df["p_fdr"].clip(lower=1e-10))

    higher_a = df["direction"].str.contains(groups[0], na=False)
    higher_b = df["direction"].str.contains(groups[1], na=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df.loc[~higher_a & ~higher_b, "cohens_d"],
               df.loc[~higher_a & ~higher_b, "-log10_p"],
               color="lightgrey", s=20, alpha=0.7, label="Not significant")
    ax.scatter(df.loc[higher_a & (df["p_fdr"] < alpha), "cohens_d"],
               df.loc[higher_a & (df["p_fdr"] < alpha), "-log10_p"],
               color=PALETTE.get(groups[0], "red"), s=30, label=f"Higher in {groups[0]}")
    ax.scatter(df.loc[higher_b & (df["p_fdr"] < alpha), "cohens_d"],
               df.loc[higher_b & (df["p_fdr"] < alpha), "-log10_p"],
               color=PALETTE.get(groups[1], "blue"), s=30, label=f"Higher in {groups[1]}")

    ax.axhline(-np.log10(alpha), color="black", lw=0.8, ls="--", alpha=0.7,
               label=f"FDR={alpha}")
    ax.axvline(0, color="black", lw=0.5, alpha=0.5)

    ax.set_xlabel("Cohen's d (effect size)")
    ax.set_ylabel("-log₁₀(p_fdr)")
    ax.set_title(f"Cluster prevalence: {groups[0]} vs {groups[1]}\n(Mann-Whitney U, BH-FDR)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    _save_fig(fig, output_dir, f"clinical_volcano_{group_col}", formats, dpi)


# ── Clinical correlation heatmap ──────────────────────────────────────────────

def plot_clinical_correlation_heatmap(
    continuous_results: pd.DataFrame,
    output_dir: Path,
    alpha: float = 0.05,
    top_n_clusters: int = 30,
    formats: list[str] = ("png", "pdf"),
    dpi: int = 300,
) -> None:
    """Heatmap of Spearman rho values (clusters × clinical metrics), non-significant masked."""
    if continuous_results.empty:
        return

    pivot_rho = continuous_results.pivot(index="cluster_id", columns="metric", values="rho")
    pivot_sig = continuous_results.pivot(index="cluster_id", columns="metric", values="significant")

    # Select top-N clusters by max |rho| across metrics
    max_rho = pivot_rho.abs().max(axis=1)
    top_clusters = max_rho.nlargest(min(top_n_clusters, len(pivot_rho))).index
    data_rho = pivot_rho.loc[top_clusters]
    data_sig = pivot_sig.loc[top_clusters]

    any_significant = data_sig.fillna(False).any().any()
    n_rows, n_cols = data_rho.shape
    figsize = (max(10, n_cols * 1.2 + 3), max(7, n_rows * 0.35 + 3))

    # Build mask (None when nothing is significant, so raw rho is shown unmasked)
    mask = (~data_sig.fillna(False)) if any_significant else None
    cbar_label = "Spearman ρ" if any_significant else "Spearman ρ (none significant)"

    g = sns.clustermap(
        data_rho.fillna(0),
        method="ward", metric="euclidean",
        row_cluster=(n_rows >= 2), col_cluster=(n_cols >= 2),
        mask=mask,
        cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
        linewidths=0.3, linecolor="white",
        xticklabels=True, yticklabels=True,
        figsize=figsize,
        dendrogram_ratio=(0.10, 0.08),
        cbar_kws={"label": cbar_label, "shrink": 0.5},
    )
    # Grey background highlights non-significant (masked) cells
    if any_significant:
        g.ax_heatmap.set_facecolor("#EEEEEE")

    g.ax_heatmap.set_xlabel("Clinical metric", labelpad=8)
    g.ax_heatmap.set_ylabel("Cluster ID", labelpad=8)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)

    sig_note = f"grey = p_fdr ≥ {alpha}" if any_significant else f"NO SIGNIFICANT CORRELATIONS (p_fdr ≥ {alpha})"
    g.figure.suptitle(
        f"Cluster prevalence ~ clinical metrics\n(Spearman ρ, {sig_note})",
        y=1.01, fontsize=11,
    )
    _clustermap_save(g, output_dir, "clinical_correlation_heatmap", formats, dpi)


# ── Clinical violin plots ─────────────────────────────────────────────────────

def plot_clinical_violin(
    prevalence_matrix: pd.DataFrame,
    clinical_df: pd.DataFrame,
    binary_results: pd.DataFrame,
    group_col: str,
    groups: tuple[str, str],
    output_dir: Path,
    top_n: int = 10,
    formats: list[str] = ("png", "pdf"),
    dpi: int = 300,
) -> None:
    """Violin plots for top-N most discriminative clusters (ASD vs TD)."""
    if binary_results.empty:
        return

    top_clusters = (
        binary_results[binary_results["significant"]]
        .nlargest(top_n, "cohens_d")["cluster_id"]
        .tolist()
    )
    if not top_clusters:
        top_clusters = binary_results.nlargest(min(top_n, len(binary_results)), "cohens_d")["cluster_id"].tolist()

    common_uuids = prevalence_matrix.index.intersection(clinical_df.index)
    prev = prevalence_matrix.loc[common_uuids]
    group_labels = clinical_df.loc[common_uuids, group_col]

    ncols = min(5, len(top_clusters))
    nrows = (len(top_clusters) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, cluster_id in enumerate(top_clusters):
        ax = axes[ax_idx]
        plot_data = pd.DataFrame({
            "prevalence": prev[cluster_id].values,
            "group": group_labels.values,
        })
        sns.violinplot(
            data=plot_data, x="group", y="prevalence",
            order=list(groups),
            palette=PALETTE,
            ax=ax, inner="box", cut=0,
        )
        # Annotate with effect size
        row = binary_results[binary_results["cluster_id"] == cluster_id]
        if len(row) > 0:
            d = row.iloc[0]["cohens_d"]
            p = row.iloc[0]["p_fdr"]
            ax.set_title(f"Cluster {cluster_id}\nd={d:.2f}, p_fdr={p:.3f}", fontsize=9)
        else:
            ax.set_title(f"Cluster {cluster_id}")
        ax.set_xlabel("")
        ax.set_ylabel("Prevalence" if ax_idx % ncols == 0 else "")
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for ax in axes[len(top_clusters):]:
        ax.set_visible(False)

    fig.suptitle(f"Top discriminative clusters: {groups[0]} vs {groups[1]}", y=1.01)
    fig.tight_layout()
    _save_fig(fig, output_dir, f"clinical_top_clusters_violin_{group_col}", formats, dpi)


# ── Kinematic heatmap ─────────────────────────────────────────────────────────

def plot_kinematic_heatmap(
    kinematic_profiles: pd.DataFrame,
    output_dir: Path,
    top_n_metrics: int = 30,
    stem: str = "kinematic_heatmap",
    formats: list[str] = ("png", "pdf"),
    dpi: int = 300,
) -> None:
    """Z-scored kinematic profile heatmap: clusters × metrics."""
    if kinematic_profiles.empty:
        return

    mean_cols = [c for c in kinematic_profiles.columns if c.endswith("__mean")]
    if not mean_cols:
        logger.warning("No __mean columns in kinematic profiles")
        return

    data = kinematic_profiles[mean_cols].copy().astype(float)
    data.columns = [c.replace("__mean", "") for c in data.columns]

    # Select top-N most variable metrics
    std_across_clusters = data.std(axis=0)
    top_metrics = std_across_clusters.nlargest(min(top_n_metrics, len(data.columns))).index
    data = data[top_metrics]

    # Z-score across clusters (per metric)
    data_z = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

    n_rows, n_cols = data_z.shape
    figsize = (max(12, n_cols * 0.4 + 3), max(8, n_rows * 0.25 + 3))

    g = sns.clustermap(
        data_z.fillna(0),
        method="ward", metric="euclidean",
        row_cluster=(n_rows >= 2), col_cluster=(n_cols >= 2),
        cmap="RdBu_r", center=0, vmin=-3, vmax=3,
        xticklabels=True, yticklabels=True,
        figsize=figsize,
        dendrogram_ratio=(0.10, 0.08),
        linewidths=0.1, linecolor="white",
        cbar_kws={"label": "Z-score across clusters", "shrink": 0.5},
    )
    g.ax_heatmap.set_xlabel("Kinematic metric", labelpad=8)
    g.ax_heatmap.set_ylabel("Cluster ID", labelpad=8)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7)
    g.figure.suptitle(
        f"Cluster kinematic profiles (z-scored, top {n_cols} most variable metrics)",
        y=1.01, fontsize=11,
    )
    _clustermap_save(g, output_dir, stem, formats, dpi)


# ── V vs non-V scatter ────────────────────────────────────────────────────────

def plot_vsubset_consistency(
    vsubset_profiles: pd.DataFrame,
    nonv_profiles: pd.DataFrame,
    output_dir: Path,
    top_n_metrics: int = 10,
    formats: list[str] = ("png", "pdf"),
    dpi: int = 300,
) -> None:
    """
    Scatter plot of V-record vs non-V kinematic means per cluster.
    Points on diagonal = consistent profiles.
    """
    if vsubset_profiles.empty or nonv_profiles.empty:
        return

    mean_cols = [c for c in vsubset_profiles.columns if c.endswith("__mean")]
    common_clusters = vsubset_profiles.index.intersection(nonv_profiles.index)
    if len(common_clusters) == 0:
        logger.warning("No common clusters for V vs non-V comparison")
        return

    # Select top-N most variable metrics for readability
    std_v = vsubset_profiles.loc[common_clusters, mean_cols].std(axis=0)
    top_metrics = std_v.nlargest(min(top_n_metrics, len(mean_cols))).index
    short_names = [c.replace("__mean", "") for c in top_metrics]

    ncols = min(5, len(top_metrics))
    nrows = (len(top_metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for i, (metric, name) in enumerate(zip(top_metrics, short_names)):
        ax = axes[i]
        v_vals = vsubset_profiles.loc[common_clusters, metric].values
        nv_vals = nonv_profiles.loc[common_clusters, metric].values
        valid = np.isfinite(v_vals) & np.isfinite(nv_vals)

        ax.scatter(v_vals[valid], nv_vals[valid], s=20, alpha=0.7, color="#4E93C8")
        # Diagonal
        mn = min(v_vals[valid].min(), nv_vals[valid].min())
        mx = max(v_vals[valid].max(), nv_vals[valid].max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, alpha=0.5)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("V-records", fontsize=7)
        ax.set_ylabel("Non-V records", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes[len(top_metrics):]:
        ax.set_visible(False)

    fig.suptitle("V-record vs Non-V kinematic consistency\n(per cluster, each panel = one metric)",
                 y=1.01)
    fig.tight_layout()
    _save_fig(fig, output_dir, "kinematic_vsubset_consistency", formats, dpi)


# ── Embedding × kinematics heatmap ────────────────────────────────────────────

def plot_embedding_kinematic_heatmap(
    rho_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    output_dir: Path,
    top_n_metrics: int = 20,
    formats: list[str] = ("png", "pdf"),
    dpi: int = 300,
) -> None:
    """Heatmap of significant Spearman rho: 128 dims × top-N kinematics."""
    if rho_df.empty:
        return

    # Select top-N metrics by number of significant dimensions
    n_sig_per_metric = sig_df.sum(axis=0)
    top_metrics = n_sig_per_metric.nlargest(min(top_n_metrics, len(rho_df.columns))).index
    data_rho = rho_df[top_metrics]
    data_sig = sig_df[top_metrics]
    mask = ~data_sig.fillna(False)

    n_rows, n_cols = data_rho.shape
    figsize = (max(10, n_cols * 0.5 + 3), max(8, n_rows * 0.12 + 2))

    # Row dendrogram is suppressed: 128 embedding dims produce an unreadable tree.
    # Column dendrogram groups kinematic metrics by their correlation pattern.
    g = sns.clustermap(
        data_rho.fillna(0),
        method="ward", metric="euclidean",
        row_cluster=False, col_cluster=(n_cols >= 2),
        mask=mask,
        cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        xticklabels=True, yticklabels=10,
        figsize=figsize,
        dendrogram_ratio=(0.0, 0.08),
        linewidths=0,
        cbar_kws={"label": "Spearman ρ (significant only)", "shrink": 0.5},
    )
    g.ax_heatmap.set_facecolor("#EEEEEE")
    g.ax_heatmap.set_xlabel("Kinematic metric", labelpad=8)
    g.ax_heatmap.set_ylabel("Embedding dimension", labelpad=8)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    g.figure.suptitle(
        f"Embedding dimensions × kinematic metrics\n"
        f"(top {n_cols} metrics by n_significant_dims, grey = not significant)",
        y=1.01, fontsize=11,
    )
    _clustermap_save(g, output_dir, "embedding_kinematic_heatmap", formats, dpi)


# ── Frame-level kinematic cluster heatmap ─────────────────────────────────────

def plot_kinematic_frame_heatmap(
    profiles_df: pd.DataFrame,
    output_dir: Path,
    stem: str = "kinematic_frame_heatmap",
    top_n_metrics: int = 30,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """
    Heatmap of z-scored frame-level per-cluster kinematic profiles.

    Similar to plot_kinematic_heatmap but uses frame-level statistics
    (from kinematic_frame_analysis.py) rather than segment-level aggregates.
    """
    formats = formats or ["png"]
    if profiles_df is None or profiles_df.empty:
        logger.warning("plot_kinematic_frame_heatmap: empty profiles DataFrame")
        return

    mean_cols = [c for c in profiles_df.columns if c.endswith("__mean")]
    if not mean_cols:
        logger.warning("plot_kinematic_frame_heatmap: no __mean columns found")
        return

    data = profiles_df[mean_cols].copy().astype(float)
    data.columns = [c.replace("__mean", "") for c in data.columns]
    data = data.dropna(axis=1, how="all")

    # Select top N most variable metrics
    col_var = data.var(axis=0)
    top_cols = col_var.nlargest(min(top_n_metrics, len(data.columns))).index
    data = data[top_cols]

    # Z-score across clusters; fillna(0) prevents NaN dissimilarity in clustermap
    z_data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    z_data = z_data.clip(-3, 3).fillna(0)

    n_rows, n_cols = z_data.shape
    figsize = (max(12, n_cols * 0.4 + 4), max(8, n_rows * 0.15 + 3))

    g = sns.clustermap(
        z_data,
        method="ward", metric="euclidean",
        row_cluster=(n_rows >= 2), col_cluster=(n_cols >= 2),
        cmap="RdBu_r", center=0, vmin=-3, vmax=3,
        xticklabels=True, yticklabels=False,
        figsize=figsize,
        dendrogram_ratio=(0.12, 0.08),
        linewidths=0,
        cbar_kws={"label": "Z-score (clipped ±3)", "shrink": 0.5},
    )
    g.ax_heatmap.set_xlabel("Kinematic metric", labelpad=8)
    g.ax_heatmap.set_ylabel("Cluster ID", labelpad=8)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    g.figure.suptitle(
        f"Frame-level kinematic profiles per cluster\n"
        f"(top {n_cols} most variable metrics, z-scored)",
        y=1.01, fontsize=11,
    )
    _clustermap_save(g, output_dir, stem, formats, dpi)


# ── Annotation kinematics heatmap ─────────────────────────────────────────────

def plot_annotation_kinematics_heatmap(
    profiles_df: pd.DataFrame,
    background_df: pd.DataFrame | None,
    output_dir: Path,
    level: str = "L1",
    top_n_metrics: int = 30,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """
    Heatmap of per-annotation-label frame-level kinematic profiles.

    Rows = annotation labels, columns = top N most variable kinematic metrics.
    Z-scored across labels. The background (non-annotated frames) is included
    as a separate row if provided.
    """
    formats = formats or ["png"]
    if profiles_df is None or profiles_df.empty:
        logger.warning("plot_annotation_kinematics_heatmap: empty profiles at level %s", level)
        return

    mean_cols = [c for c in profiles_df.columns if c.endswith("__mean")]
    if not mean_cols:
        return

    data = profiles_df[mean_cols].copy().astype(float)
    data.columns = [c.replace("__mean", "") for c in data.columns]

    # Append background row if available
    if background_df is not None and not background_df.empty:
        bg_cols = [c for c in background_df.columns if c.endswith("__mean")]
        bg_row = background_df[bg_cols].copy().astype(float)
        bg_row.columns = [c.replace("__mean", "") for c in bg_cols]
        bg_row.index = ["_background_"]
        data = pd.concat([data, bg_row], axis=0)

    data = data.dropna(axis=1, how="all")

    # Top N most variable
    col_var = data.var(axis=0)
    top_cols = col_var.nlargest(min(top_n_metrics, len(data.columns))).index
    data = data[top_cols]

    # Z-score across labels; fill NaN with 0 (= cluster mean) so clustermap
    # doesn't receive NaN dissimilarity values (labels may lack data for some metrics)
    z_data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    z_data = z_data.clip(-3, 3).fillna(0)

    n_rows, n_cols = z_data.shape
    if n_rows < 2 or n_cols < 2:
        logger.warning("Too few rows/cols for annotation kinematics heatmap at level %s", level)
        return

    figsize = (max(12, n_cols * 0.4 + 4), max(6, n_rows * 0.4 + 3))

    g = sns.clustermap(
        z_data,
        method="ward", metric="euclidean",
        row_cluster=True, col_cluster=True,
        cmap="RdBu_r", center=0, vmin=-3, vmax=3,
        xticklabels=True, yticklabels=True,
        figsize=figsize,
        dendrogram_ratio=(0.12, 0.08),
        linewidths=0.3,
        cbar_kws={"label": "Z-score (clipped ±3)", "shrink": 0.5},
    )
    g.ax_heatmap.set_xlabel("Kinematic metric", labelpad=8)
    g.ax_heatmap.set_ylabel("Annotation label", labelpad=8)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)
    g.figure.suptitle(
        f"Frame-level kinematic profiles per annotation label ({level})\n"
        f"(top {n_cols} most variable metrics, z-scored, with background row)",
        y=1.01, fontsize=11,
    )
    _clustermap_save(g, output_dir, f"annotation_kinematics_{level}", formats, dpi)


# ── Kruskal-Wallis significance bar chart ────────────────────────────────────

def plot_kruskal_wallis_results(
    kruskal_df: pd.DataFrame,
    output_dir: Path,
    stem: str = "kinematic_frame_kruskal",
    top_n: int = 30,
    alpha: float = 0.05,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """
    Bar chart of Kruskal-Wallis H-statistics for the top N metrics.

    Significant metrics (after FDR correction) are shown in a distinct color.
    """
    formats = formats or ["png"]
    if kruskal_df is None or kruskal_df.empty:
        return

    if "H" not in kruskal_df.columns:
        return

    df = kruskal_df.copy().dropna(subset=["H"])
    df = df.sort_values("H", ascending=False).head(top_n)

    colors = [
        "#E84040" if sig else "#AAAAAA"
        for sig in df.get("significant", pd.Series([False] * len(df)))
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.3 + 2)))
    ax.barh(df.index[::-1], df["H"].values[::-1], color=colors[::-1])
    ax.set_xlabel("Kruskal-Wallis H statistic")
    ax.set_title(
        f"Kinematic metrics differentiating clusters (frame-level)\n"
        f"Red = significant after FDR correction (α={alpha}), top {len(df)} shown"
    )
    ax.axvline(0, color="black", linewidth=0.5)
    sig_patch = mpatches.Patch(color="#E84040", label=f"p_fdr < {alpha}")
    ns_patch = mpatches.Patch(color="#AAAAAA", label="not significant")
    ax.legend(handles=[sig_patch, ns_patch], loc="lower right")
    fig.tight_layout()
    _save_fig(fig, output_dir, stem, formats, dpi)


# ── Significance overview plots ───────────────────────────────────────────────

def plot_enrichment_significance_overview(
    perm_result,
    output_dir: Path,
    level: str = "L1",
    top_n_clusters: int = 30,
    alpha: float = 0.05,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """
    3-panel significance overview for the annotation enrichment permutation test.

    Panels
    ------
    Top-left  : Volcano plot — log2(enrichment) vs -log10(p_fdr).
                Each point = one (behavior, cluster) pair.
                Colored by direction: red = over-represented, blue = under-represented.
                Horizontal dashed line = FDR significance threshold.
                Vertical dashed lines at ±log2(1.5) = 50% fold-change guideline.

    Top-right : N annotation events per behavior (horizontal bar chart).
                Shows the statistical power proxy — behaviors with few events
                will have wide null distributions and low power.

    Bottom    : Significance heatmap — behaviors × top-N clusters (ranked by
                maximum absolute log2-enrichment).
                Cells are colored by log2-enrichment (red=over, blue=under).
                Non-significant cells (p_fdr ≥ α) are shown in light grey.
                FDR-significant cells are annotated with their sig_label (*/***).

    Parameters
    ----------
    perm_result : PermutationEnrichmentResult
        Output of permutation_test_enrichment().
    output_dir : Path
    level : str
        Annotation hierarchy level (used in the figure title and filename).
    top_n_clusters : int
        Number of clusters to show in the heatmap (ranked by |log2_effect|).
    alpha : float
        FDR significance threshold.
    formats, dpi : plot format options.
    """
    from .significance import PermutationEnrichmentResult

    formats = formats or ["png"]
    if perm_result.long_format.empty:
        logger.warning("plot_enrichment_significance_overview: empty result, skipping")
        return

    long_df = perm_result.long_format.copy()
    p_fdr_df = perm_result.p_fdr
    log2_df = perm_result.log2_effect
    sig_df = perm_result.significant
    sig_label_df = perm_result.sig_label
    n_events = perm_result.n_events_per_label

    # ── Select top clusters for heatmap ─────────────────────────────────────
    if log2_df.empty:
        logger.warning("plot_enrichment_significance_overview: no log2_effect data, skipping")
        return

    # Rank clusters by maximum |log2_effect| across all behaviors
    max_abs_effect = log2_df.abs().max(axis=0)
    top_clusters = max_abs_effect.nlargest(top_n_clusters).index.tolist()
    log2_heat = log2_df[top_clusters].copy()
    sig_heat = sig_df[top_clusters].copy() if not sig_df.empty else pd.DataFrame(False, index=log2_heat.index, columns=top_clusters)
    siglbl_heat = sig_label_df[top_clusters].copy() if not sig_label_df.empty else pd.DataFrame("ns", index=log2_heat.index, columns=top_clusters)

    # ── Figure layout ────────────────────────────────────────────────────────
    n_behaviors = len(log2_heat)
    heatmap_h = max(4, n_behaviors * 0.5 + 1)
    fig = plt.figure(figsize=(18, heatmap_h + 6))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[5, max(3, heatmap_h)],
        hspace=0.45, wspace=0.35,
    )
    ax_volcano = fig.add_subplot(gs[0, 0])
    ax_nevents = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[1, :])

    # ── Panel 1: Volcano plot ─────────────────────────────────────────────────
    finite_mask = long_df["p_fdr"].notna() & long_df["log2_enrichment"].notna()
    vdf = long_df[finite_mask].copy()
    vdf["neg_log10_p"] = -np.log10(np.maximum(vdf["p_fdr"].astype(float), 1e-10))
    vdf["color"] = np.where(
        vdf["significant"] & (vdf["log2_enrichment"] > 0), "#D63031",   # over-represented
        np.where(
            vdf["significant"] & (vdf["log2_enrichment"] < 0), "#2980B9",  # under-represented
            "#BBBBBB",  # not significant
        )
    )

    ax_volcano.scatter(
        vdf["log2_enrichment"], vdf["neg_log10_p"],
        c=vdf["color"], s=18, alpha=0.7, linewidths=0,
    )

    threshold_y = -np.log10(alpha)
    ax_volcano.axhline(threshold_y, color="black", linestyle="--", linewidth=0.8,
                       label=f"FDR = {alpha}")
    ax_volcano.axvline(np.log2(1.5), color="#AAAAAA", linestyle=":", linewidth=0.7)
    ax_volcano.axvline(-np.log2(1.5), color="#AAAAAA", linestyle=":", linewidth=0.7)
    ax_volcano.axvline(0, color="black", linewidth=0.5)

    over_patch = mpatches.Patch(color="#D63031", label="Over-represented (sig.)")
    under_patch = mpatches.Patch(color="#2980B9", label="Under-represented (sig.)")
    ns_patch = mpatches.Patch(color="#BBBBBB", label="Not significant")
    ax_volcano.legend(handles=[over_patch, under_patch, ns_patch],
                      fontsize=7, loc="upper left", framealpha=0.8)

    n_sig = int(vdf["significant"].sum())
    n_total = len(vdf)
    ax_volcano.set_xlabel("log₂(enrichment)", fontsize=9)
    ax_volcano.set_ylabel("−log₁₀(p_fdr)", fontsize=9)
    ax_volcano.set_title(
        f"Volcano — level {level}\n"
        f"{n_sig}/{n_total} pairs significant (FDR {alpha}), "
        f"{perm_result.n_permutations} permutations",
        fontsize=9,
    )

    # ── Panel 2: N events per behavior ────────────────────────────────────────
    events_df = n_events[n_events > 0].sort_values(ascending=True)
    # Show only behaviors that appear in the permutation result
    valid_behaviors = set(log2_df.index.tolist())
    events_df = events_df[events_df.index.isin(valid_behaviors)]

    bar_colors = ["#E84040" if n < 5 else "#4E93C8" for n in events_df.values]
    ax_nevents.barh(events_df.index, events_df.values, color=bar_colors, height=0.6)
    ax_nevents.axvline(5, color="#888888", linestyle="--", linewidth=0.8,
                       label="N=5 guideline")
    ax_nevents.set_xlabel("N annotation events", fontsize=9)
    ax_nevents.set_title(f"Sample size (power proxy) — level {level}", fontsize=9)
    ax_nevents.legend(fontsize=7, loc="lower right")
    for i, (lbl, val) in enumerate(events_df.items()):
        ax_nevents.text(val + 0.1, i, str(val), va="center", fontsize=7)

    # ── Panel 3: Significance heatmap ─────────────────────────────────────────
    # Build display matrix: log2_effect where significant, NaN elsewhere → greyed
    display_data = log2_heat.copy().astype(float)
    display_data[~sig_heat] = float("nan")

    # Background for non-significant cells
    bg_data = log2_heat.copy().astype(float)
    bg_data[sig_heat] = float("nan")

    vmax = float(log2_heat.abs().quantile(0.95).max()) or 2.0
    vmax = max(vmax, 0.5)

    ax_heatmap.set_facecolor("#F0F0F0")

    # Non-significant background (light grey)
    im_bg = ax_heatmap.imshow(
        bg_data.values, aspect="auto", cmap="Greys",
        vmin=0, vmax=1, alpha=0.3,
    )

    # Significant cells colored by effect
    im = ax_heatmap.imshow(
        display_data.values, aspect="auto",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )

    ax_heatmap.set_xticks(range(len(top_clusters)))
    ax_heatmap.set_xticklabels(
        [str(c) for c in top_clusters],
        rotation=90, fontsize=6,
    )
    ax_heatmap.set_yticks(range(n_behaviors))
    ax_heatmap.set_yticklabels(log2_heat.index.tolist(), fontsize=8)
    ax_heatmap.set_xlabel(f"Cluster ID (top {top_n_clusters} by |log₂ effect|)", fontsize=9)
    ax_heatmap.set_ylabel("Behavior", fontsize=9)
    ax_heatmap.set_title(
        f"Enrichment significance — level {level} "
        f"(colored = FDR-significant, grey = not significant)",
        fontsize=9,
    )

    # Annotate significant cells with sig_label
    for b_idx, behavior in enumerate(log2_heat.index):
        for c_idx, cid in enumerate(top_clusters):
            if sig_heat.loc[behavior, cid]:
                lbl = siglbl_heat.loc[behavior, cid]
                if lbl != "ns":
                    ax_heatmap.text(
                        c_idx, b_idx, lbl,
                        ha="center", va="center",
                        fontsize=5, color="black", fontweight="bold",
                    )

    plt.colorbar(im, ax=ax_heatmap, label="log₂(enrichment)", shrink=0.6, pad=0.01)

    fig.suptitle(
        f"Annotation-Cluster Enrichment Significance — Level {level}\n"
        f"Event-level permutation test (n={perm_result.n_permutations}), "
        f"FDR-BH correction",
        fontsize=11, fontweight="bold",
    )

    _save_fig(fig, output_dir, f"annotation_enrichment_significance_{level}", formats, dpi)
    logger.info(
        "Enrichment significance overview saved for level %s (%d sig. pairs)",
        level, n_sig
    )


def plot_centroid_bootstrap_overview(
    boot_result,
    output_dir: Path,
    top_n_clusters: int = 40,
    level: str = "L1",
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """
    2-panel bootstrap CI overview for annotation centroid distances.

    Panels
    ------
    Left  : CI width heatmap — behaviors × top-N clusters (by observed distance).
            Narrow CI (blue) = centroid is well-determined by the available events.
            Wide CI (red) = high uncertainty; interpret with caution.

    Right : Nearest-cluster stability bar chart — per behavior, the fraction of
            bootstrap resamples where the same cluster is the nearest neighbor.
            High stability (→1) = reliable cluster assignment.
            Low stability (→0) = assignment is uncertain (consider more events).

    Parameters
    ----------
    boot_result : BootstrapCentroidResult
        Output of bootstrap_centroid_distances().
    output_dir : Path
    top_n_clusters : int
        Number of clusters to show in the CI width heatmap (closest by distance).
    formats, dpi : plot format options.
    """
    formats = formats or ["png"]
    if boot_result.long_format.empty:
        logger.warning("plot_centroid_bootstrap_overview: empty result, skipping")
        return

    obs_dist = boot_result.observed_distance
    ci_width = boot_result.ci_width
    stability_df = boot_result.nearest_cluster_stability
    n_events = boot_result.n_events_per_label

    if obs_dist.empty or ci_width.empty:
        logger.warning("plot_centroid_bootstrap_overview: no distance data, skipping")
        return

    # Select top-N clusters by minimum observed distance across all behaviors
    min_dist_per_cluster = obs_dist.min(axis=0)
    top_clusters = min_dist_per_cluster.nsmallest(top_n_clusters).index.tolist()

    obs_sub = obs_dist[top_clusters]
    ci_sub = ci_width[top_clusters]

    n_behaviors = len(obs_sub)

    # ── Figure layout ────────────────────────────────────────────────────────
    fig_h = max(5, n_behaviors * 0.45 + 2)
    fig, (ax_ci, ax_stab) = plt.subplots(
        1, 2,
        figsize=(18, fig_h),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )

    # ── Panel 1: CI width heatmap ─────────────────────────────────────────────
    ci_arr = ci_sub.values.astype(float)
    vmax_ci = float(np.nanpercentile(ci_arr, 95)) if np.any(np.isfinite(ci_arr)) else 1.0
    vmax_ci = max(vmax_ci, 0.01)

    im = ax_ci.imshow(ci_arr, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=vmax_ci)
    ax_ci.set_xticks(range(len(top_clusters)))
    ax_ci.set_xticklabels([str(c) for c in top_clusters], rotation=90, fontsize=6)
    ax_ci.set_yticks(range(n_behaviors))
    ax_ci.set_yticklabels(obs_sub.index.tolist(), fontsize=8)
    ax_ci.set_xlabel(f"Cluster ID (top {top_n_clusters} closest)", fontsize=9)
    ax_ci.set_ylabel("Behavior", fontsize=9)
    metric = getattr(boot_result, "distance_metric", "euclidean")
    ax_ci.set_title(
        f"Bootstrap CI Width on Centroid Distances ({level}, {metric})\n"
        f"(n={boot_result.n_bootstrap} resamples, 95% CI; "
        f"blue=narrow=precise, red=wide=uncertain)",
        fontsize=9,
    )
    cbar = plt.colorbar(im, ax=ax_ci, label=f"CI width ({metric} distance)", shrink=0.8)
    cbar.ax.tick_params(labelsize=7)

    # ── Panel 2: Nearest-cluster stability ────────────────────────────────────
    if stability_df is not None and not stability_df.empty and \
            "fraction_bootstrap_same_nearest" in stability_df.columns:
        stab = stability_df["fraction_bootstrap_same_nearest"].sort_values(ascending=True)
        stab = stab[stab.index.isin(obs_sub.index)]

        bar_colors = [
            "#2ECC71" if v >= 0.8 else "#F39C12" if v >= 0.5 else "#E74C3C"
            for v in stab.values
        ]
        ax_stab.barh(stab.index.tolist(), stab.values, color=bar_colors, height=0.6)
        ax_stab.axvline(0.8, color="#2ECC71", linestyle="--", linewidth=0.8, label="≥80% stable")
        ax_stab.axvline(0.5, color="#F39C12", linestyle=":", linewidth=0.8, label="≥50% stable")
        ax_stab.set_xlim(0, 1.05)
        ax_stab.set_xlabel("Fraction bootstrap agreeing on nearest cluster", fontsize=9)
        ax_stab.set_title(
            f"Nearest-cluster stability\n(N events per behavior in parentheses)",
            fontsize=9,
        )
        ax_stab.legend(fontsize=7, loc="lower right")

        # Annotate with N events and stability %
        for i, (lbl, val) in enumerate(stab.items()):
            n_ev = int(n_events.get(lbl, 0))
            ax_stab.text(
                val + 0.01, i,
                f"{val:.0%} (N={n_ev})",
                va="center", fontsize=7,
            )
    else:
        ax_stab.text(0.5, 0.5, "No stability data", ha="center", va="center",
                     transform=ax_stab.transAxes, fontsize=10)
        ax_stab.set_title("Nearest-cluster stability", fontsize=9)

    fig.suptitle(
        f"Annotation Centroid Bootstrap — Distance Confidence Intervals ({level}, {metric})\n"
        f"(N={boot_result.n_bootstrap} resamples of annotation events per behavior)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    filename = f"annotation_centroid_bootstrap_overview_{level}"
    _save_fig(fig, output_dir, filename, formats, dpi)
    logger.info("Centroid bootstrap overview saved for level %s (metric=%s)", level, metric)
