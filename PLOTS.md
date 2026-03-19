# Plot Reference Guide

All figures are saved under `results/{run_name}/plots/` in the formats specified by `output.plot_formats` (default: PNG + PDF at 300 dpi).

---

## 1. `coverage_summary.{png,pdf}`

**What it shows:** Two-panel figure summarizing data availability across all subject-sessions.

- **Left panel** — Horizontal bar chart: how many subject-sessions have each data type (clusters, kinematics, clinical metadata, annotations). Ideal outcome: all bars equal (complete overlap).
- **Right panel** — Stacked bar chart per V-record: green = annotation events successfully matched to a segment frame, red = unmatched events. Use this to diagnose annotation timing / segment boundary issues.

**How to interpret:**
- All bars in the left panel should be close to the total number of subjects. A short "Clinical" or "Kinematics" bar indicates a join problem.
- In the right panel, each row is one annotated subject. A large red portion means the annotation timestamps fall outside the known segment boundaries — check FPS settings or tracking.nc attributes.

---

## 2. `annotation_cluster_heatmap.{png,pdf}`

**What it shows:** Heatmap of **enrichment scores** (observed / expected cluster usage) for the top-30 most enriched clusters, one row per annotated behavior.

- Color scale: `1.0` = cluster appears at the expected background rate. `> 1` = over-represented during this behavior. `< 1` = under-represented.
- Only V-records are used (N ≤ 28). Clusters with fewer than `min_frames` annotated frames are excluded.

**How to interpret:**
- Bright (warm) cells = the cluster is strongly associated with that behavior.
- A cluster column that is enriched for a single behavior likely represents a specific motor pattern.
- Multiple behaviors sharing the same cluster suggest overlapping or ambiguous motor signatures.

---

## 3. `annotation_cluster_bars/{behavior}.{png,pdf}`

**What it shows:** One bar chart per annotated behavior, showing the top-10 clusters by raw **frame count** during that behavior.

**How to interpret:**
- Taller bars = more annotated frames fall in that cluster, but this is not corrected for overall cluster prevalence (use the heatmap for enrichment).
- Useful for quickly seeing which cluster IDs are most prominent for a given behavior.

---

## 4. `annotation_centroid_distance.{png,pdf}`

**What it shows:** Heatmap of **128-dimensional Euclidean distances** between annotation centroids (mean embedding per behavior type, V-records only) and global cluster centroids (mean embedding per cluster, V-records only).

- Rows = behavior types. Columns = top-40 clusters closest to any behavior centroid.
- Darker (lower distance) = the cluster's embedding centroid is close to that behavior's mean embedding.

**How to interpret:**
- Each cluster column has one assigned behavior label: the behavior with the minimum distance (see `cluster_behavior_labels.csv`).
- Very small distances indicate the cluster reliably captures a specific behavior.
- Clusters equidistant from multiple behaviors represent transitions or mixed states.
- Note: this analysis uses only V-record frames for cluster centroids, so rare clusters may be noisy.

---

## 5. `clinical_volcano_{group}.{png,pdf}`

**What it shows:** Volcano plot for the binary group comparison (default: ASD vs TD).

- **X-axis** = Cohen's d effect size (positive = higher in ASD, negative = higher in TD, or vice versa depending on group ordering).
- **Y-axis** = −log₁₀(p_fdr) after Benjamini-Hochberg correction. Higher = more significant.
- **Dashed horizontal line** = significance threshold (p_fdr = α = 0.05 by default).
- Each dot is one cluster. Colored dots (red = ASD-higher, blue = TD-higher) pass FDR correction.

**How to interpret:**
- Clusters in the upper-right quadrant: significantly more prevalent in ASD with a large effect.
- Clusters in the upper-left quadrant: significantly more prevalent in TD.
- Points near the x-axis did not survive FDR correction — treat as noise.
- The overall spread of the volcano indicates how strongly diagnosis separates clusters.

---

## 6. `clinical_correlation_heatmap.{png,pdf}`

**What it shows:** Heatmap of **Spearman ρ** correlations between cluster prevalence (fraction of frames per subject in each cluster) and continuous clinical scores (ADOS, MSEL, Vineland).

- Rows = top-30 clusters by maximum |ρ| across all metrics.
- Columns = clinical metrics.
- Color: red = positive correlation (more time in cluster → higher score), blue = negative.
- **Grey cells** = not significant after BH-FDR correction (p_fdr ≥ α).

**If the title says "NO SIGNIFICANT CORRELATIONS":** all correlations failed FDR correction. The raw ρ values are shown unmasked so the plot remains informative. This can happen with small sample sizes (N ≤ 102) and many tests (N_clusters × N_metrics). Interpreting raw ρ values in this case should be done cautiously.

**How to interpret:**
- A positive ρ with an ADOS severity score: spending more time in that cluster correlates with more severe autism symptoms.
- A negative ρ with a Vineland adaptive score: more time in that cluster correlates with lower adaptive functioning.
- Clusters with significant correlations across multiple metrics (multiple colored cells in one row) are clinically informative.

---

## 7. `clinical_top_clusters_violin_{group}.{png,pdf}`

**What it shows:** Violin plots for the top-10 most diagnostically discriminative clusters (highest Cohen's d among FDR-significant clusters, or top-10 by |d| if none are significant).

- Each panel = one cluster. X-axis = group (ASD / TD). Y-axis = cluster prevalence (fraction of frames).
- Annotated with Cohen's d and p_fdr.

**How to interpret:**
- A wide violin at high prevalence for ASD and a narrow violin near zero for TD indicates strong, reliable group separation.
- Effect sizes: d ≈ 0.2 (small), 0.5 (medium), 0.8+ (large).
- Look for clusters where ASD and TD violins do not overlap — these are the most diagnostically informative.

---

## 8. `kinematic_heatmap_global.{png,pdf}`

**What it shows:** Z-scored kinematic profile heatmap: clusters × top-30 most variable kinematic metrics.

- Each cell = how a cluster's mean kinematic value compares to the across-cluster mean (z-score, clipped at ±3).
- Clusters are sorted by hierarchical Ward clustering on kinematic profiles.
- Red = above average, blue = below average for that metric.

**How to interpret:**
- Clusters with many red cells in speed/acceleration columns represent high-movement states.
- Clusters with high `facingness` or `congruent_motion` scores represent periods of strong social coordination.
- Neighboring clusters in the heatmap (after hierarchical sorting) have similar kinematic signatures.
- Use alongside `cluster_report.md` for per-cluster narrative summaries.

---

## 9. `kinematic_vsubset_consistency.{png,pdf}`

**What it shows:** Scatter plots comparing per-cluster kinematic profiles computed on **V-records** (x-axis) vs **non-V records** (y-axis), for the top-10 most variable kinematic metrics.

- Each dot = one cluster. The dashed diagonal = perfect agreement between V and non-V subsets.
- One panel per kinematic metric.

**How to interpret:**
- Points on the diagonal: kinematic signatures are consistent across both subsets — strong evidence the cluster is a real behavioral state rather than V-record-specific noise.
- Points far from the diagonal: the kinematic meaning of the cluster may be different in V-records (annotated ADOS sessions) vs general sessions. This may reflect differences in session structure or annotation-related behaviors.
- Outlier clusters in individual panels are flagged for inspection.

---

## 10. `embedding_kinematic_heatmap.{png,pdf}`

**What it shows:** "Semantic dictionary" — Spearman ρ correlations between each of the **128 LISBET embedding dimensions** (rows) and the top-20 kinematic metrics (columns), computed at the segment level.

- Only significant correlations (p_fdr < α) are colored; non-significant cells are blank.
- Red = the embedding dimension encodes more of that kinematic quantity. Blue = inverse relationship.

**How to interpret:**
- A single embedding dimension correlating with multiple kinematic metrics suggests it captures a high-level behavioral construct (e.g., "activity level").
- Sparse columns (few significant rows) indicate kinematics not well-encoded in the LISBET embedding space.
- This analysis is the most data-intensive (loads all embeddings). Results are at segment level, so correlations reflect medium-timescale structure (10–30 s windows).

---

---

## 11. `annotation_cluster_heatmap_{L1,L2,L3}.{png,pdf}`

**What it shows:** Enrichment heatmap at each of the three annotation hierarchy levels.

- **L1** = behavior only (13 categories)
- **L2** = behavior + behavioral_category composite label (9 × 13 combinations)
- **L3** = behavior + behavioral_category + modifier_1 (finest grain, 65+ composites)

**How to interpret:**
- L1 gives the broadest view; L2 and L3 progressively refine it.
- A behavior that splits into two distinct clusters at L2/L3 but merges at L1 likely contains sub-types with different motor signatures.
- Use L3 to identify highly specific behaviors (e.g., "mannerism | stereotypy | hand flapping") that strongly drive particular clusters.

---

## 12. `kinematic_frame_heatmap.{png,pdf}`

**What it shows:** Z-scored per-cluster kinematic profiles computed at **frame level** — each frame is independently assigned to a cluster and linked to its kinematic values that same frame.

- More accurate than segment-level aggregates (Plot 8) because it avoids averaging over heterogeneous frames within a segment.
- Rows = clusters (hierarchically ordered). Columns = top-30 most variable kinematic metrics.

**How to interpret:**
- Same as Plot 8 but with better temporal precision.
- Discrepancies between this plot and the segment-level heatmap suggest that a cluster's kinematic signature is diluted at the segment level (e.g., short burst behaviors within longer segments).

---

## 13. `kinematic_frame_kruskal.{png,pdf}`

**What it shows:** Bar chart of Kruskal-Wallis H-statistics for the top-30 kinematic metrics that most differentiate clusters at the frame level.

- Red bars = metrics that significantly distinguish clusters after FDR correction.
- Grey bars = not significant.

**How to interpret:**
- High H = the metric takes very different values across clusters (the frame-level cluster assignments explain variance in that metric).
- Useful for prioritizing which kinematic dimensions to report.

---

## 14. `annotation_kinematics_{L1,L2,L3}.{png,pdf}`

**What it shows:** Frame-level kinematic profiles per annotation label (rows), with a `_background_` row showing non-annotated frames.

- One heatmap per hierarchy level.
- Z-scored across labels (including background). Red = above average for that metric. Blue = below.
- Produced only for V-records (annotated sessions).

**How to interpret:**
- Labels far from the background row in z-score space have distinctive kinematics not seen in typical frames.
- Labels that cluster together (dendrogram branches) have similar kinematic signatures — they may represent related motor patterns.
- Comparing L1 and L3 shows whether modifier sub-types have meaningfully different kinematics.

---

## Output files cross-reference

| Plot file | Key data files |
|---|---|
| `coverage_summary` | `data/data_coverage.csv` |
| `annotation_cluster_heatmap_L1/L2/L3` | `data/annotation_cluster_enrichment_L1.csv`, etc. |
| `annotation_cluster_bars/` | `data/annotation_cluster_contingency_L1.csv` |
| `annotation_centroid_distance` | `data/cluster_distance_matrix.csv`, `data/cluster_cluster_behavior_labels.csv` |
| `clinical_volcano_{group}` | `data/clinical_binary_{group}_results.csv` |
| `clinical_correlation_heatmap` | `data/clinical_continuous_results.csv` |
| `clinical_top_clusters_violin_{group}` | `data/clinical_binary_{group}_results.csv` |
| `kinematic_heatmap_global` | `data/cluster_kinematic_global.csv` |
| `kinematic_frame_heatmap` | `data/cluster_kinematic_frame_profiles.csv` |
| `kinematic_frame_kruskal` | `data/cluster_kinematic_frame_kruskal.csv` |
| `kinematic_vsubset_consistency` | `data/cluster_kinematic_vsubset.csv`, `data/cluster_kinematic_nonv.csv` |
| `annotation_kinematics_L1/L2/L3` | `data/annotation_kinematics_L1_profiles.csv`, etc. |
| `embedding_kinematic_heatmap` | `data/embedding_kinematic_rho.csv`, `data/embedding_kinematic_significant.csv` |
| *(no plot)* | `data/cluster_profiles.csv` — per-cluster synthesis |
| *(no plot)* | `data/cluster_centroids_global.csv` — 128D centroids (all subjects) |
