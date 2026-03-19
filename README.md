# clusterAnalysis

Characterizes LISBET behavioral clusters by cross-referencing expert annotations,
clinical metrics, and kinematics from dyadic child–clinician ADOS sessions.

---

## Overview

This module takes as input:

| Source | Content | N |
|---|---|---|
| **Cluster mapping** | Frame-level cluster IDs (127 clusters, ~5.3M frames) | All records |
| **Clinical CSV** | ADOS/MSEL/Vineland scores + ASD/TD diagnosis | ~102 subjects |
| **Annotations CSV** | Expert behavioral labels with timing (seconds) | 28 V-records |
| **Kinematics (segment)** | Segment-level speed, acceleration, social proxies (65 metrics) | ~99 subjects |
| **Kinematics (frame)** | Per-frame kinematic metrics (`metrics_normalised.csv`) | ~99 subjects |
| **Embeddings** | 128-D LISBET embeddings per frame | All records |

And runs nine analyses:

1. **Annotation overlap** — Which clusters co-occur with which annotated behaviors? At three label granularity levels (behavior / behavior+category / behavior+category+modifier).
2. **Annotation centroids** — Label clusters by their nearest behavior in 128-D space, using global cluster centroids computed from all 102 subjects.
3. **Clinical correlations** — Do ASD/TD children differ in cluster usage? Do ADOS/MSEL/Vineland scores correlate with cluster prevalence?
4. **Kinematic profiles (segment)** — Per-cluster kinematic signatures using segment-level dominant cluster assignments.
5. **V vs non-V consistency** — Do cluster kinematic signatures replicate in non-annotated sessions?
6. **Embedding × kinematics** ("Semantic Dictionary") — Which embedding dimensions encode which movements?
7. **Kinematic profiles (frame)** — Per-cluster kinematic signatures at frame resolution, with Kruskal-Wallis significance testing across clusters.
8. **Annotation kinematics** — Frame-level kinematic profiles per annotation label at all three levels, compared to a non-annotated background.
9. **Cluster profiles synthesis** — One-row-per-cluster summary CSV combining annotation labels, clinical significance, and top kinematic features.

---

## Environment

```bash
module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0
source /home/shares/schaerm/schaer2/thibaut/humanlisbet/lisbet_venv/bin/activate
```

Dependencies (beyond standard venv): `xarray`, `netCDF4`, `scipy`, `seaborn`

---

## Running

### Local (interactive)
```bash
cd /srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper

python -m clusterAnalysis.src.run_analysis \
    --config clusterAnalysis/configs/default.yaml \
    --run-name my_run \
    --log-level INFO
```

### SLURM
```bash
cd /srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper

# Default run
sbatch clusterAnalysis/scripts/run_analysis.sh

# Custom config and run name
CONFIG=clusterAnalysis/configs/default.yaml RUN_NAME=exp_v2 \
    sbatch clusterAnalysis/scripts/run_analysis.sh
```

---

## Configuration

Edit `configs/default.yaml` to control:

| Key | Description |
|---|---|
| `data.*` | Paths to all input files |
| `data.fps` | Frame rate for annotation timing (default: 20). Validated against `tracking.nc` attributes; mismatch triggers a WARNING. |
| `analyses.annotation_overlap` | Cluster distribution during annotations (all 3 label levels) |
| `analyses.annotation_centroids` | 128-D annotation centroids + global cluster centroids |
| `analyses.clinical_correlations` | ASD vs TD + continuous clinical score correlations |
| `analyses.kinematic_profiles` | Segment-level per-cluster kinematic profiles |
| `analyses.kinematic_frame_analysis` | Frame-level per-cluster kinematic profiles + Kruskal-Wallis |
| `analyses.annotation_kinematics` | Frame-level kinematic profiles per annotation label (3 levels) |
| `analyses.vsubset_consistency` | V vs non-V kinematic comparison |
| `analyses.embedding_kinematics` | 128-D embedding × kinematic Spearman correlations |
| `analyses.cluster_profiles` | Synthesize all results into `cluster_profiles.csv` |
| `annotation.min_frames_level1` | Minimum annotated frames for inclusion at L1 (behavior only, default 10) |
| `annotation.min_frames_level2` | Minimum annotated frames at L2 (behavior + category, default 5) |
| `annotation.min_frames_level3` | Minimum annotated frames at L3 (+ modifier, default 3) |
| `kinematics.use_normalized` | Use trunk-height-normalized kinematics (recommended) |
| `kinematics.metrics` | Subset of metrics to analyze (`null` = all ~65) |
| `clinical.binary_groups` | Columns for binary group comparisons (e.g. diagnosis) |
| `clinical.continuous` | Clinical score columns for Spearman correlation |
| `statistics.fdr_method` | `bh` (Benjamini-Hochberg) or `bonferroni` |
| `statistics.alpha` | Significance threshold after FDR |
| `statistics.min_frames_per_cluster` | Flag low-frame clusters in reports |
| `output.run_name` | Subdirectory name under `results/` |
| `output.plot_formats` | `[png, pdf]`, `[svg]`, etc. |

---

## Output Structure

```
results/{run_name}/
├── run_summary.json                       # Run metadata + timing + error log
├── cluster_report.md                      # Human-readable per-cluster summary
├── annotation_centroids.pkl               # 128D mean embedding per behavior (for prediction)
│
├── data/
│   ├── data_coverage.csv                  # Per-subject data availability + annotation matching
│   ├── prevalence_matrix.csv              # (N_subjects × N_clusters) fraction of time per cluster
│   │
│   ├── annotation_cluster_contingency_L1.csv  # behavior × cluster frame counts (L1)
│   ├── annotation_cluster_contingency_L2.csv  # composite label × cluster (L2)
│   ├── annotation_cluster_contingency_L3.csv  # fine-grain composite × cluster (L3)
│   ├── annotation_cluster_enrichment_L1.csv   # observed/expected enrichment (L1)
│   ├── annotation_cluster_enrichment_L2.csv
│   ├── annotation_cluster_enrichment_L3.csv
│   ├── annotation_cluster_per_record_L*.csv   # per-V-record annotation matching stats
│   │
│   ├── cluster_centroids_global.csv       # 128D centroids from all subjects (N_clusters × 128)
│   ├── cluster_distance_matrix.csv        # behavior × cluster (128D Euclidean distance)
│   ├── cluster_cluster_behavior_labels.csv # nearest annotation behavior per cluster
│   │
│   ├── clinical_binary_diagnosis_results.csv  # Mann-Whitney: ASD vs TD per cluster
│   ├── clinical_continuous_results.csv        # Spearman: cluster × ADOS/MSEL/Vineland
│   │
│   ├── cluster_kinematic_global.csv       # Segment-level weighted mean kinematics per cluster
│   ├── cluster_kinematic_vsubset.csv      # Same, V-records only
│   ├── cluster_kinematic_nonv.csv         # Same, non-V records only
│   ├── cluster_kinematic_frame_profiles.csv # Frame-level mean/std per cluster
│   ├── cluster_kinematic_frame_kruskal.csv  # Kruskal-Wallis H + p_fdr per metric
│   │
│   ├── annotation_kinematics_L1_profiles.csv  # Frame-level kinematic profile per label (L1)
│   ├── annotation_kinematics_L1_background.csv
│   ├── annotation_kinematics_L2_profiles.csv
│   ├── annotation_kinematics_L2_background.csv
│   ├── annotation_kinematics_L3_profiles.csv
│   ├── annotation_kinematics_L3_background.csv
│   │
│   ├── embedding_kinematic_rho.csv        # 128 dims × 65 metrics Spearman rho
│   ├── embedding_kinematic_significant.csv # Boolean mask of FDR-significant pairs
│   │
│   └── cluster_profiles.csv              # One-row-per-cluster synthesis of all analyses
│
└── plots/
    ├── coverage_summary.png/pdf
    ├── annotation_cluster_heatmap_L1.png/pdf   # enrichment heatmap at behavior level
    ├── annotation_cluster_heatmap_L2.png/pdf   # behavior + category
    ├── annotation_cluster_heatmap_L3.png/pdf   # behavior + category + modifier
    ├── annotation_cluster_bars/                # bar chart per behavior (L1)
    ├── annotation_centroid_distance.png/pdf
    ├── annotation_kinematics_L1.png/pdf        # kinematic profiles per annotation label
    ├── annotation_kinematics_L2.png/pdf
    ├── annotation_kinematics_L3.png/pdf
    ├── clinical_volcano_diagnosis.png/pdf
    ├── clinical_correlation_heatmap.png/pdf
    ├── clinical_top_clusters_violin_diagnosis.png/pdf
    ├── kinematic_heatmap_global.png/pdf
    ├── kinematic_frame_heatmap.png/pdf         # frame-level version (more accurate)
    ├── kinematic_frame_kruskal.png/pdf         # Kruskal-Wallis significance
    ├── kinematic_vsubset_consistency.png/pdf
    └── embedding_kinematic_heatmap.png/pdf
```

---

## Interpreting Results

### `data_coverage.csv`
Check that `has_clusters`, `has_kinematics`, `has_clinical` are `True` for all expected
subjects. The `annotation_coverage_frac` column shows what fraction of annotation events
successfully mapped to embedding frames — values below 0.8 warrant investigation
(check `fps`, segment boundary files, or annotation timing).

### `prevalence_matrix.csv`
Each row (subject) sums to 1.0. A subject spending 20% of their session in cluster 7
has `prevalence[uuid, 7] = 0.20`. Use this to understand how subjects differ in
their behavioral "fingerprint."

### `annotation_cluster_enrichment_L*.csv`
Values > 1 mean a cluster appears more during a behavior than expected by chance.
Values close to 1 = no enrichment. Values >> 2–3 are notable. Compare across L1/L2/L3
to identify sub-types with distinct motor signatures.

### `cluster_kinematic_frame_kruskal.csv`
`H` statistic and `p_fdr` for each kinematic metric. A significant metric means the
frame-level cluster assignments explain substantial variance in that movement measure.
Rank by `H` to identify the most cluster-discriminative kinematics.

### `annotation_kinematics_L1_profiles.csv`
Per-label kinematic profile computed from annotated frames only. The `_background_`
row gives the mean for non-annotated frames. Compare rows to the background to identify
which behaviors have distinctive movement signatures.

### `cluster_profiles.csv`
One row per cluster. Key columns:
- `nearest_behavior_L1` / `centroid_distance_L1` — nearest annotation centroid (embedding space)
- `enrichment_top_behavior_L1` — behavior with highest observed/expected enrichment
- `kinematic_high` / `kinematic_low` — top z-scored kinematic features
- `diagnosis_p_fdr`, `diagnosis_cohens_d`, `diagnosis_direction` — ASD vs TD
- `top_clinical_correlations` — significant Spearman correlations with clinical scores

### `clinical_binary_diagnosis_results.csv`
Key columns:
- `p_fdr` — FDR-corrected p-value
- `cohens_d` — effect size (|d| > 0.5 is medium, > 0.8 is large)
- `direction` — whether cluster is higher in ASD or TD
- `significant` — True if p_fdr < alpha

### `clinical_continuous_results.csv`
Key columns: `rho` (Spearman), `p_fdr`, `significant`.
Positive rho = cluster prevalence increases with the clinical score.

### `cluster_report.md`
A per-cluster summary combining annotation labels, kinematics, and clinical results.
Uses frame-level kinematic profiles when available (falls back to segment-level).

---

## Module Structure

```
clusterAnalysis/
├── src/
│   ├── config.py                — YAML loader + validation
│   ├── data.py                  — load cluster map, clinical, annotations, kinematics
│   ├── linking.py               — segment registry (from tracking.nc, with fps validation),
│   │                              subject map, annotation→frame lookup, prevalence matrix
│   ├── stats.py                 — Mann-Whitney, Spearman, Cohen's d, FDR
│   ├── annotation_analysis.py   — 3-level annotation overlap + centroids
│   │                              + annotation kinematics + global cluster centroids
│   ├── kinematic_frame_analysis.py — frame-level cluster × kinematic profiles
│   │                                  + Kruskal-Wallis significance
│   ├── clinical_analysis.py     — binary + continuous clinical correlations
│   ├── kinematic_analysis.py    — segment-level kinematic profiles + V vs non-V
│   ├── embedding_analysis.py    — 128D × kinematic Spearman ("Semantic Dictionary")
│   ├── visualization.py         — all plots (seaborn/matplotlib)
│   └── run_analysis.py          — CLI orchestrator + cluster profiles synthesis
├── configs/
│   └── default.yaml
├── scripts/
│   └── run_analysis.sh          — SLURM submission script
├── PLOTS.md                     — Plot reference guide (what each figure shows)
└── METHODS.md                   — Statistical methods and data linkage algorithms
```
