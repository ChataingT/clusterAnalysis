"""
Configuration loader for the clusterAnalysis pipeline.

Reads a YAML config file, validates keys, fills defaults, and exposes
a typed ClusterAnalysisConfig dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Allowed top-level sections and their allowed sub-keys ────────────────────
_ALLOWED_KEYS: dict[str, set[str]] = {
    "data": {
        "cluster_mapping",
        "cross_video_train",
        "clinical_csv",
        "annotations_csv",
        "pose_records_dir",
        "embeddings_dir",
        "fps",
    },
    "analyses": {
        "annotation_overlap",
        "annotation_centroids",
        "clinical_correlations",
        "kinematic_profiles",
        "kinematic_frame_analysis",
        "annotation_kinematics",
        "embedding_kinematics",
        "vsubset_consistency",
        "cluster_profiles",
        "annotation_overlap_significance",
        "annotation_centroids_significance",
    },
    "annotation": {"min_frames_level1", "min_frames_level2", "min_frames_level3", "distance_metric"},
    "kinematics": {"use_normalized", "metrics"},
    "clinical": {"binary_groups", "continuous"},
    "statistics": {"fdr_method", "alpha", "min_frames_per_cluster", "significance"},
    "output": {"run_name", "results_dir", "save_plots", "save_data", "plot_formats", "figure_dpi"},
}

_ALLOWED_SIGNIFICANCE_KEYS = {
    "n_permutations",
    "n_bootstrap",
    "seed",
    "min_events_per_label",
    "n_jobs",
}

_REQUIRED_DATA_KEYS = {
    "cluster_mapping",
    "cross_video_train",
    "clinical_csv",
    "annotations_csv",
    "pose_records_dir",
    "embeddings_dir",
}


@dataclass
class DataConfig:
    cluster_mapping: Path
    cross_video_train: Path
    clinical_csv: Path
    annotations_csv: Path
    pose_records_dir: Path
    embeddings_dir: Path
    fps: int = 20


@dataclass
class AnalysesConfig:
    annotation_overlap: bool = True
    annotation_centroids: bool = True
    clinical_correlations: bool = True
    kinematic_profiles: bool = True
    kinematic_frame_analysis: bool = True
    annotation_kinematics: bool = True
    embedding_kinematics: bool = True
    vsubset_consistency: bool = True
    cluster_profiles: bool = True
    annotation_overlap_significance: bool = True
    annotation_centroids_significance: bool = True


_VALID_DISTANCE_METRICS = {"euclidean", "cosine", "mahalanobis"}


@dataclass
class AnnotationConfig:
    min_frames_level1: int = 10  # behavior only
    min_frames_level2: int = 5   # behavior + behavioral_category
    min_frames_level3: int = 3   # behavior + behavioral_category + modifier_1
    distance_metric: str = "euclidean"  # euclidean | cosine | mahalanobis


@dataclass
class KinematicsConfig:
    use_normalized: bool = True
    metrics: list[str] | None = None  # None = use all


@dataclass
class ClinicalConfig:
    binary_groups: list[str] = field(default_factory=lambda: ["diagnosis"])
    continuous: list[str] = field(default_factory=lambda: [
        "ADOS_2_TOTAL",
        "ADOS_G_ADOS_2_TOTAL_score_de_severite",
        "ADOS_2_ADOS_G_REVISED_SA_SEVERITY_SCORE",
        "ADOS_2_ADOS_G_REVISED_RRB_SEVERITY_SCORE_new",
        "ADOS_2_SOCIAL_AFECT_TOTAL",
        "AdSS",
        "TOTAL_DQ",
    ])


@dataclass
class SignificanceConfig:
    n_permutations: int = 1000      # permutations for enrichment test
    n_bootstrap: int = 500          # bootstrap resamples for centroid CI
    seed: int = 42                  # random seed for reproducibility
    min_events_per_label: int = 5   # skip behaviors with fewer events
    n_jobs: int = 1                 # reserved for future parallelization


@dataclass
class StatisticsConfig:
    fdr_method: str = "bh"
    alpha: float = 0.05
    min_frames_per_cluster: int = 100
    significance: SignificanceConfig = field(default_factory=SignificanceConfig)


@dataclass
class OutputConfig:
    run_name: str = "default"
    results_dir: Path = Path("clusterAnalysis/results")
    save_plots: bool = True
    save_data: bool = True
    plot_formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    figure_dpi: int = 300


@dataclass
class ClusterAnalysisConfig:
    data: DataConfig
    analyses: AnalysesConfig
    annotation: AnnotationConfig
    kinematics: KinematicsConfig
    clinical: ClinicalConfig
    statistics: StatisticsConfig
    output: OutputConfig

    @property
    def run_output_dir(self) -> Path:
        return self.output.results_dir / self.output.run_name


def load_config(path: str | Path) -> ClusterAnalysisConfig:
    """
    Load and validate a YAML config file.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    ClusterAnalysisConfig

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the config contains unknown or missing required keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    logger.debug("Raw config loaded from %s: %s", path, raw)

    # ── Validate top-level sections ──────────────────────────────────────────
    unknown_sections = set(raw.keys()) - set(_ALLOWED_KEYS.keys())
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")

    # ── Validate sub-keys per section ────────────────────────────────────────
    for section, allowed in _ALLOWED_KEYS.items():
        section_data = raw.get(section, {})
        if not isinstance(section_data, dict):
            continue
        unknown = set(section_data.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Unknown keys in config section '{section}': {sorted(unknown)}"
            )

    # Validate nested statistics.significance sub-keys
    sig_raw_check = raw.get("statistics", {}).get("significance", {})
    if isinstance(sig_raw_check, dict):
        unknown_sig = set(sig_raw_check.keys()) - _ALLOWED_SIGNIFICANCE_KEYS
        if unknown_sig:
            raise ValueError(
                f"Unknown keys in config section 'statistics.significance': {sorted(unknown_sig)}"
            )

    # ── Parse each section ───────────────────────────────────────────────────
    data_raw = raw.get("data", {})

    # Validate required data keys
    missing = _REQUIRED_DATA_KEYS - set(data_raw.keys())
    if missing:
        raise ValueError(
            f"Missing required keys in config 'data' section: {sorted(missing)}"
        )

    data_cfg = DataConfig(
        cluster_mapping=Path(data_raw["cluster_mapping"]),
        cross_video_train=Path(data_raw["cross_video_train"]),
        clinical_csv=Path(data_raw["clinical_csv"]),
        annotations_csv=Path(data_raw["annotations_csv"]),
        pose_records_dir=Path(data_raw["pose_records_dir"]),
        embeddings_dir=Path(data_raw["embeddings_dir"]),
        fps=int(data_raw.get("fps", 20)),
    )

    analyses_raw = raw.get("analyses", {})
    analyses_cfg = AnalysesConfig(
        annotation_overlap=bool(analyses_raw.get("annotation_overlap", True)),
        annotation_centroids=bool(analyses_raw.get("annotation_centroids", True)),
        clinical_correlations=bool(analyses_raw.get("clinical_correlations", True)),
        kinematic_profiles=bool(analyses_raw.get("kinematic_profiles", True)),
        kinematic_frame_analysis=bool(analyses_raw.get("kinematic_frame_analysis", True)),
        annotation_kinematics=bool(analyses_raw.get("annotation_kinematics", True)),
        embedding_kinematics=bool(analyses_raw.get("embedding_kinematics", True)),
        vsubset_consistency=bool(analyses_raw.get("vsubset_consistency", True)),
        cluster_profiles=bool(analyses_raw.get("cluster_profiles", True)),
        annotation_overlap_significance=bool(
            analyses_raw.get("annotation_overlap_significance", True)
        ),
        annotation_centroids_significance=bool(
            analyses_raw.get("annotation_centroids_significance", True)
        ),
    )

    ann_raw = raw.get("annotation", {})
    distance_metric = str(ann_raw.get("distance_metric", "euclidean"))
    if distance_metric not in _VALID_DISTANCE_METRICS:
        raise ValueError(
            f"Invalid annotation.distance_metric '{distance_metric}'. "
            f"Choose from: {sorted(_VALID_DISTANCE_METRICS)}"
        )
    ann_cfg = AnnotationConfig(
        min_frames_level1=int(ann_raw.get("min_frames_level1", 10)),
        min_frames_level2=int(ann_raw.get("min_frames_level2", 5)),
        min_frames_level3=int(ann_raw.get("min_frames_level3", 3)),
        distance_metric=distance_metric,
    )

    kin_raw = raw.get("kinematics", {})
    kin_cfg = KinematicsConfig(
        use_normalized=bool(kin_raw.get("use_normalized", True)),
        metrics=kin_raw.get("metrics", None),
    )

    clin_raw = raw.get("clinical", {})
    clin_cfg = ClinicalConfig(
        binary_groups=clin_raw.get("binary_groups", ["diagnosis"]),
        continuous=clin_raw.get("continuous", ClinicalConfig.__dataclass_fields__["continuous"].default_factory()),
    )

    stat_raw = raw.get("statistics", {})
    sig_raw = stat_raw.get("significance", {}) if isinstance(stat_raw.get("significance"), dict) else {}
    sig_cfg = SignificanceConfig(
        n_permutations=int(sig_raw.get("n_permutations", 1000)),
        n_bootstrap=int(sig_raw.get("n_bootstrap", 500)),
        seed=int(sig_raw.get("seed", 42)),
        min_events_per_label=int(sig_raw.get("min_events_per_label", 3)),
        n_jobs=int(sig_raw.get("n_jobs", 1)),
    )
    stat_cfg = StatisticsConfig(
        fdr_method=str(stat_raw.get("fdr_method", "bh")),
        alpha=float(stat_raw.get("alpha", 0.05)),
        min_frames_per_cluster=int(stat_raw.get("min_frames_per_cluster", 100)),
        significance=sig_cfg,
    )

    out_raw = raw.get("output", {})
    out_cfg = OutputConfig(
        run_name=str(out_raw.get("run_name", "default")),
        results_dir=Path(out_raw.get("results_dir", "clusterAnalysis/results")),
        save_plots=bool(out_raw.get("save_plots", True)),
        save_data=bool(out_raw.get("save_data", True)),
        plot_formats=list(out_raw.get("plot_formats", ["png", "pdf"])),
        figure_dpi=int(out_raw.get("figure_dpi", 300)),
    )

    cfg = ClusterAnalysisConfig(
        data=data_cfg,
        analyses=analyses_cfg,
        annotation=ann_cfg,
        kinematics=kin_cfg,
        clinical=clin_cfg,
        statistics=stat_cfg,
        output=out_cfg,
    )

    logger.info(
        "Config loaded: run_name=%s, analyses=%s",
        cfg.output.run_name,
        {k: v for k, v in vars(cfg.analyses).items() if v},
    )
    return cfg
