# Cluster × Annotation Analysis

## Overview

Two complementary analyses link LISBET clusters to expert behavioral annotations. Both are restricted to **V-records** (subjects who have annotations). They run as **Analysis Steps 1 and 2** in the main pipeline.

| Analysis | Config flag | Step | Question |
|---|---|---|---|
| Annotation Overlap | `annotation_overlap` | 1 | Which clusters are over-represented *during* annotated behaviors? |
| **Enrichment Significance** | `annotation_overlap_significance` | 1b | Are enrichment scores stronger than expected under a session-wise circular-shift temporal null? |
| Annotation Centroids | `annotation_centroids` | 2 | Which behavior does each cluster *look like* in embedding space? |
| **Centroid Bootstrap CI** | `annotation_centroids_significance` | 2b | How stable and precise are the centroid distance estimates? |

---

## Shared Prerequisites

### Segment Registry

Built by `build_segment_registry()` ([linking.py:27](src/linking.py#L27)). For every segment of every subject-session, reads `tracking.nc` and extracts:
- `segment_start_frame` — absolute video frame where this segment begins
- `segment_end_frame` — absolute video frame where it ends

Result: a dict `segment_name → {start_abs_frame, end_abs_frame}`, e.g.:
```
"7797_T1a_ADOS1_seg_001" → {"start_abs_frame": 0, "end_abs_frame": 2399}
```

### `code → subject_session` Map

Both analyses need to route each annotation (which carries a V-record `code` like `V017`) to the correct segments. The mapping is built by scanning `clinical_df` — first trying the numeric uuid prefix, then the V-code prefix — and resolving to a `subject_session` string like `"7797_T1a_ADOS1"` ([annotation_analysis.py:66–80](src/annotation_analysis.py#L66)).

### Annotation → Frame Resolution

`annotation_to_frames()` ([linking.py:241](src/linking.py#L241)) converts a single annotation event to a list of `(segment_name, relative_frame_index)` pairs:

1. Convert `start` / `stop` timestamps (seconds) to absolute frame indices: `abs_frame = int(time_sec × fps)` (default fps=20).
2. For POINT events: `stop = start` (single frame).
3. For each segment in the registry that belongs to the correct subject-session, check overlap between the annotation interval and the segment's frame range.
4. For overlapping frames: compute `rel_frame = abs_frame − seg_start_abs_frame`.
5. Return all `(segment_name, rel_frame)` pairs.

---

## Analysis 1: Annotation Overlap

**Entry point**: `run_annotation_overlap_multilevel()` → `run_annotation_overlap()` — [annotation_analysis.py:497](src/annotation_analysis.py#L497) and [annotation_analysis.py:33](src/annotation_analysis.py#L33).

**Question**: Among all frames where behavior X is annotated, which clusters appear more than expected by chance?

### Label Hierarchy

Before running the overlap, `build_multilevel_annotations()` ([annotation_analysis.py:408](src/annotation_analysis.py#L408)) creates three composite label sets from the annotations DataFrame:

| Level | Columns combined | Example label |
|---|---|---|
| L1 | `behavior` only | `"vocalization"` |
| L2 | `behavior \| behavioral_category` | `"vocalization \| social"` |
| L3 | `behavior \| behavioral_category \| modifier_1` | `"vocalization \| social \| initiated"` |

Each level gets its own minimum-frames threshold (L1: 10, L2: 5, L3: 3 by default — [config.py:84–86](src/config.py#L84)).

### Frame → Cluster Lookup

A fast dict is built from `cluster_mapping` ([annotation_analysis.py:84–88](src/annotation_analysis.py#L84)):
```python
frame_to_cluster = cluster_mapping
    .set_index(["segment_name", "index"])["cluster_id"]
    .to_dict()
# e.g. ("7797_T1a_ADOS1_seg_001", 47) → cluster_id=12
```

### Counting

For each V-record code, for each annotation event:
1. Resolve the event to `(segment_name, rel_frame)` pairs via `annotation_to_frames()`.
2. Look up each frame's `cluster_id` in `frame_to_cluster`.
3. Accumulate counts per `(behavior_label, cluster_id)` pair ([annotation_analysis.py:125–133](src/annotation_analysis.py#L125)).

Behaviors with fewer than `min_frames` total annotated frames are excluded ([annotation_analysis.py:165–171](src/annotation_analysis.py#L165)).

### Contingency Table

A `(N_behaviors × N_clusters)` integer DataFrame of raw annotated-frame counts ([annotation_analysis.py:154–162](src/annotation_analysis.py#L154)).

### Enrichment Score

The key metric: **observed fraction / expected fraction** ([annotation_analysis.py:173–191](src/annotation_analysis.py#L173)).

- **Observed fraction** for (behavior, cluster): row-normalized contingency → fraction of annotated frames for that behavior that fall in that cluster.
- **Expected fraction** for a cluster: its global prevalence among all V-record frames (not just annotated ones):
  ```python
  global_counts = v_frames["cluster_id"].value_counts()
  global_frac = global_counts / global_counts.sum()
  ```
- **Enrichment** = observed / expected. Value > 1 means the cluster appears more than chance during that behavior; < 1 means under-represented.

### Outputs

Run for L1, L2, L3. Saved as ([run_analysis.py:492–502](src/run_analysis.py#L492)):
```
data/annotation_cluster_contingency_L1.csv   — (behavior × cluster) frame counts
data/annotation_cluster_enrichment_L1.csv    — (behavior × cluster) enrichment scores
data/annotation_cluster_per_record_L1.csv    — per V-record matching stats
data/annotation_cluster_contingency_L2.csv
...
```

### Visualization

- **Heatmap** (`plot_annotation_cluster_heatmap` — [visualization.py:138](src/visualization.py#L138)): enrichment scores for the top-30 clusters (by max enrichment), with Ward hierarchical clustering on both axes. Color scale: `RdYlBu_r`, centered at 1.0 (= expected). One heatmap per level:
  ```
  plots/annotation_cluster_heatmap_L1.{png,pdf}
  plots/annotation_cluster_heatmap_L2.{png,pdf}
  plots/annotation_cluster_heatmap_L3.{png,pdf}
  ```
- **Bar charts** (`plot_annotation_cluster_bars` — [visualization.py:183](src/visualization.py#L183)): one bar chart per L1 behavior showing the top-10 clusters by raw frame count:
  ```
  plots/annotation_cluster_bars/bar_{behavior}.{png,pdf}
  ```

---

## Significance 1: Enrichment Permutation Test

**Entry point**: `build_event_cluster_matrix()` + `permutation_test_enrichment()` — [significance.py](src/significance.py).
**Config flag**: `analyses.annotation_overlap_significance` (default: `true`).
**Config parameters**: `statistics.significance.*`.

### Why a permutation test, and why at the event level?

A naive significance test for enrichment would apply a chi-squared test to the raw contingency table. This fails fundamentally because of **within-event autocorrelation**: a single 5-second behavioral episode recorded at 20 fps contributes 100 frames, all of which belong to the same behavioral act and tend to occupy the same cluster. The effective sample size is therefore the number of *annotation events*, not the number of frames. Using frame-level counts inflates N by a factor of ~10–100, causing nearly all pairs to appear "significant" regardless of whether any true association exists.

The correct unit of observation is the **annotation event** — one indivisible behavioral episode tagged by an expert annotator. The permutation test operationalises this with **circular shifts within each subject-session timeline**, preserving:
- the exact number of frames per event (event duration),
- the exact sequence and transition structure of behaviors within each session,
- the frame-to-cluster assignment within each event.

What is broken under the null is only the temporal alignment between labels and the observed movement trajectory.

### Pre-computation: event-cluster matrix

`build_event_cluster_matrix()` — [significance.py](src/significance.py).

For each annotation event that could be resolved to frames, compute a **cluster count vector** of length `n_clusters`:

```
event_cluster_matrix[i, k] = number of frames in event i that fall in cluster k
```

Shape: `(n_events, n_clusters)` — a dense NumPy float32 array. The matrix is computed once from the existing `frame_to_cluster` dict (already built during the overlap analysis) and reused across all permutations with no further data access. At L1 in our run: **3,732 events × 127 clusters**.

Events that could not be resolved to any matched frame are excluded (1,346 events, i.e. 26.5%, due to the 73.5% annotation coverage). This does not bias the test: unmatched events are simply not part of the analysis.

Labels vector: `labels[i]` = the behavior name (or composite L2/L3 label) for event `i`. This is the vector that is permuted.

Event metadata (`event_meta`) stores `subject_session` and `session_event_order` for each event row, enabling deterministic circular shifts inside each session.

### Observed enrichment

From the event-cluster matrix with the original label vector, compute the enrichment matrix `E` of shape `(n_behaviors, n_clusters)`:

```
E[b, k] = P(cluster k | behavior b) / P(cluster k | V-records)
```

where:
- **Numerator**: `sum(event_cluster_matrix[labels==b, k]) / sum(event_cluster_matrix[labels==b, :])` — fraction of annotated frames for behavior b that fall in cluster k.
- **Denominator**: global cluster prevalence among *all V-record frames* (not only annotated ones), computed from `cluster_mapping` filtered to V-record subject-sessions. This is the same expected distribution used in the original overlap analysis for consistency.

In the current run: 126 clusters covered, 1,142,364 total V-record frames.

Effect size: `log2(E[b, k])`. This is symmetric around 0 (log2 = 0 means no enrichment, log2 = 1 means 2× enrichment, log2 = −1 means 2× depletion).

### Permutation procedure

```python
rng = np.random.default_rng(seed=42)
for i in range(n_permutations):
  perm_labels = circular_shift_within_subject_session(labels, event_meta, rng)
    E_perm = compute_enrichment(event_cluster_matrix, perm_labels, global_frac)
    null_deviation[i, b, k] = |E_perm[b,k] - 1|
```

At each permutation, labels are circularly shifted **within each subject-session** (with independent random offsets per session). This preserves realistic behavior duration and transition dynamics while removing true temporal alignment to movement clusters.

The test statistic per cell is the absolute deviation from 1.0 (the null value of enrichment): `|E - 1|`. This makes the test two-sided — it detects both over-representation (E >> 1) and under-representation (E << 1) with equal power.

**p-value per cell**:
```
p_raw[b, k] = #{permutations where |E_perm[b,k] - 1| >= |E_obs[b,k] - 1|} / n_permutations
```

Minimum achievable p-value: `1/n_permutations` = 0.001 at the default of 1000 permutations. Increase `n_permutations` to 5000 for precision at p < 0.001.

Null distribution statistics (mean and std over permutations per cell) are saved alongside p-values as diagnostic output — they allow visual inspection of the null distribution for any cell of interest.

### Multiple testing correction

FDR correction is applied **per annotation level** (L1, L2, L3 separately) using the Benjamini-Hochberg procedure ([stats.py:fdr_correct](src/stats.py)). Correcting per level is appropriate here because L1/L2/L3 are nested (L2 is derived from L1, so they share annotation events) — correcting jointly would be over-conservative. Within each level, all `n_behaviors × n_clusters` pairs are corrected simultaneously.

At L1: **1,524 pairs** (12 behaviors × 127 clusters). All p_raw values are corrected jointly, and each p_fdr ≥ p_raw by construction.

### Run results (run_20260330_162854)

| Level | Behaviors | Clusters | Pairs tested | Significant (FDR-BH, α=0.05) |
|-------|-----------|----------|--------------|-------------------------------|
| L1 | 12 | 127 | 1,524 | **0** |
| L2 | 12 | 127 | 1,524 | **0** |
| L3 | 65 | 127 | 8,255 | **0** |

All 12 behaviors at L1 had zero significant clusters. This result is informative — see the section **Interpreting zero significance** below.

Event counts per behavior at L1 (from the run):

| Behavior | N events |
|---|---|
| Mannerisms | 788 |
| Sensory stimulation | 637 |
| Shared enjoyment | 559 |
| IC gestures | 403 |
| Show | 378 |
| Request | 342 |
| Point | 222 |
| Give | 211 |
| IJA | 108 |
| Use of another's body | 37 |
| Name calling | 23 |
| RJA | 22 |
| Sensory aversion | 2 (skipped, < 5 events) |

### Interpreting zero significance

Zero significant pairs across all levels after FDR correction does not mean "the clusters have nothing to do with behavior." It has a precise statistical interpretation:

**The null hypothesis cannot be rejected at α=0.05 after correcting for 1,524 simultaneous tests.**

Several factors can explain this result:

1. **Low power from 1,000 permutations**: at 1,000 permutations, the minimum achievable p-value per cell is 0.001. After BH correction over 1,524 tests, a cell needs a raw p-value of roughly ≤ 0.001 × 1524/1524 × ... which approaches 0.001 at rank 1 — essentially, at least one permutation would have to exceed the observed value in *zero* runs. This is a hard floor. **Increase `n_permutations` to 5,000–10,000 to detect moderate effects**. At 5,000 permutations, cells with raw p = 0.0002 become detectable.

2. **High multiple testing burden**: 1,524 pairs at L1 is a large family. BH correction at this scale requires strong signals. Behaviors with fewer events (RJA: 22, Name calling: 23) have inherently noisy enrichment estimates and cannot drive significance.

3. **Event-level N is small for rare behaviors**: RJA with 22 events × ~10 frames/event ≈ 220 frames total. The permutation null distribution for such a behavior has high variance, making small-to-moderate enrichment ratios indistinguishable from noise.

4. **Cluster granularity vs behavior breadth**: with 127 clusters and 12 broad L1 behaviors, each behavior spans many clusters. The enrichment signal may be distributed across several clusters at sub-threshold levels rather than concentrated in one.

**Recommended next steps**:
- Increase `n_permutations` to 5000 in the config and re-run.
- Inspect the `p_raw` values in `annotation_cluster_significance_L1.csv` — even without FDR significance, cells with p_raw ≤ 0.01 (approaching significance) indicate candidate pairs worth investigating.
- Examine the `log2_enrichment` column: large effect sizes with marginally non-significant p-values are scientifically interesting even if not statistically significant after FDR correction.
- For rare behaviors (N < 30 events), consider pooling across similar subcategories or reporting results without FDR correction for exploratory purposes.

### Outputs

Per level (L1, L2, L3):

| File | Format | Description |
|---|---|---|
| `annotation_cluster_significance_L{1,2,3}.csv` | Long (behavior × cluster pairs) | All statistics per pair: enrichment, log2_enrichment, p_raw, p_fdr, significant, sig_label, n_events_behavior, null_mean, null_std |
| `annotation_cluster_pfdr_L{1,2,3}.csv` | Wide (behaviors × clusters) | p_fdr matrix for plotting |
| `annotation_cluster_log2effect_L{1,2,3}.csv` | Wide (behaviors × clusters) | log2(enrichment) matrix for plotting |

Key columns in the long-format CSV:

| Column | Description |
|---|---|
| `behavior` | Behavior label (L1: behavior name; L2/L3: composite) |
| `cluster_id` | LISBET cluster ID (0–126) |
| `enrichment` | Observed / expected = P(cluster\|behavior) / P(cluster\|V-records) |
| `log2_enrichment` | log2(enrichment). 0 = no enrichment, 1 = 2×, −1 = 0.5× |
| `p_raw` | Raw permutation p-value (minimum = 1/n_permutations) |
| `p_fdr` | BH-corrected p-value (per level) |
| `significant` | True if p_fdr < α (default 0.05) |
| `sig_label` | "***" (p<0.001), "**" (p<0.01), "*" (p<0.05), "ns" |
| `n_events_behavior` | Number of annotation events for this behavior |
| `null_mean` | Mean enrichment across permutations (diagnostic) |
| `null_std` | Std of enrichment across permutations (diagnostic) |

### Visualization

`plot_enrichment_significance_overview()` — [visualization.py](src/visualization.py).
Output: `plots/annotation_enrichment_significance_L{1,2,3}.{png,pdf}` — one figure per level.

**Three panels**:

1. **Volcano plot** (top-left): Each point = one (behavior, cluster) pair. X-axis = log2(enrichment), Y-axis = −log10(p_fdr). Horizontal dashed line = FDR threshold (−log10(0.05) ≈ 1.3). Vertical dotted lines at ±log2(1.5) = ±0.585 as a 50%-fold-change guideline. Color: red = significantly over-represented, blue = significantly under-represented, grey = not significant. Allows simultaneous assessment of effect size and significance for all pairs.

2. **N events per behavior** (top-right): Horizontal bar chart, one bar per behavior, colored red if N < 5 (low power). Shows the primary determinant of statistical power — small-N behaviors cannot achieve significance in this test.

3. **Significance heatmap** (bottom): Behaviors × top-30 clusters (ranked by maximum |log2 enrichment|). Cells colored by log2(enrichment) only where p_fdr < α; non-significant cells shown in light grey. FDR-significant cells annotated with sig_label (*/***). Provides a compact overview of which behaviors have cluster-specific associations.

---

## Significance 2: Bootstrap CI on Centroid Distances

> **Note**: This section documents the significance extension of [Analysis 2: Annotation Centroids](#analysis-2-annotation-centroids), which is described further below. It relies on the per-event mean embeddings computed during that analysis.

**Entry point**: `bootstrap_centroid_distances()` — [significance.py](src/significance.py).
**Config flag**: `analyses.annotation_centroids_significance` (default: `true`).

### What question does this test answer?

The distance matrix from Analysis 2 reports 128D Euclidean distances between each annotation centroid and each cluster centroid. These distances are point estimates — they depend on the particular annotation events available. The bootstrap CI addresses: **how stable is this distance estimate?**

- **Narrow CI**: the centroid is robustly determined from the available events; the distance can be trusted.
- **Wide CI**: the centroid is unstable (few events, or high within-behavior embedding variance); the distance is imprecise and should be interpreted cautiously.

This is a precision / confidence interval test, not an inferential test against a null hypothesis. It does not compute whether the distance is "significantly smaller than chance" — it quantifies how much the observed distance would vary if you had observed a different set of annotation events for the same behavior.

### Unit of resampling: annotation events (not frames)

The bootstrap resamples at the **event level**: for each behavior, the list of per-event mean embeddings is resampled with replacement. Each event mean is the arithmetic mean of all 128D frame embeddings within that event — i.e., it summarizes the motion state during one behavioral episode.

Why event-level rather than frame-level? For the same reason as the permutation test: frames within an event are autocorrelated. Resampling frames would underestimate the true variability in the centroid estimate. Resampling events gives a proper bootstrap distribution that reflects how much the centroid would change if you had observed a different set of behavioral episodes.

Per-event mean embeddings are computed as a by-product of `run_annotation_centroids()` ([annotation_analysis.py:run_annotation_centroids](src/annotation_analysis.py)) and stored in the `event_embedding_means` key of the results dict. No re-loading of embedding files is needed.

### Algorithm

```python
for each behavior b:
    means_mat = np.array(event_embedding_means[b])  # shape (n_events, 128)
    centroid_obs = means_mat.mean(axis=0)             # observed centroid, shape (128,)

    for each bootstrap resample j in 1..n_bootstrap:
        idx = rng.integers(0, n_events, size=n_events)  # sample with replacement
        boot_centroid = means_mat[idx].mean(axis=0)
        boot_dist[j, k] = ||boot_centroid - cluster_centroid[k]||_2  for each cluster k

    ci_low[b, k]  = percentile(boot_dist[:, k], 2.5)
    ci_high[b, k] = percentile(boot_dist[:, k], 97.5)
    ci_width[b, k] = ci_high[b, k] - ci_low[b, k]
```

Default: 500 bootstrap resamples, 95% CI, seed=42.

### Nearest-cluster stability

As a by-product of each bootstrap resample, the function also records whether the same cluster is the nearest neighbor as in the observed data:

```
stability[b] = #{bootstrap resamples where argmin_k boot_dist[j, k] == argmin_k dist_obs[b, k]}
               / n_bootstrap
```

A stability of 1.0 means the nearest cluster is the same in every bootstrap resample — the cluster labeling is highly reliable. A stability of 0.5 means the nearest cluster changes in half of the resamples — the assignment is uncertain and should not be over-interpreted.

Stability is primarily driven by N events: behaviors with many events have stable centroids; rare behaviors with few events may have unstable assignments.

### Outputs

| File | Format | Description |
|---|---|---|
| `annotation_centroid_bootstrap_summary_L{1,2,3}.csv` | Long (behavior × cluster pairs) | observed_distance, ci_low, ci_high, ci_width, n_events |
| `annotation_centroid_ci_low_L{1,2,3}.csv` | Wide (behaviors × clusters) | CI lower bound per pair |
| `annotation_centroid_ci_high_L{1,2,3}.csv` | Wide (behaviors × clusters) | CI upper bound per pair |
| `annotation_centroid_nearest_stability_L{1,2,3}.csv` | Per behavior | nearest_cluster_id, observed_distance, fraction_bootstrap_same_nearest, n_events, n_bootstrap |

Key columns in `annotation_centroid_nearest_stability_L{1,2,3}.csv`:

| Column | Description |
|---|---|
| `nearest_cluster_id` | The cluster closest to this behavior's centroid |
| `observed_distance` | 128D Euclidean distance from the behavior centroid to nearest cluster |
| `fraction_bootstrap_same_nearest` | Stability: fraction of 500 resamples agreeing on nearest cluster |
| `n_events` | Number of annotation events used to compute the centroid |
| `n_bootstrap` | Number of bootstrap resamples |

### Visualization

`plot_centroid_bootstrap_overview()` — [visualization.py](src/visualization.py).
Output: `plots/annotation_centroid_bootstrap_overview_L{1,2,3}.{png,pdf}`.

**Two panels**:

1. **CI width heatmap** (left): Behaviors × top-40 clusters (by minimum observed distance, i.e., the closest clusters). Color = CI width (95% interval). Blue = narrow = precise centroid estimate; red = wide = uncertain. This panel allows at a glance to see which (behavior, cluster) distances can be trusted and which cannot. Behaviors with few events will show red across all clusters.

2. **Nearest-cluster stability** (right): Horizontal bar chart, one bar per behavior. Color: green (≥80%), orange (50–80%), red (<50%). Each bar is annotated with the stability fraction and N events in parentheses. Behaviors with stability > 80% have reliable cluster assignments; those below 50% should not be used to label clusters.

---

## Significance Configuration

All significance parameters are in the `statistics.significance` block of the config file:

```yaml
statistics:
  fdr_method: bh          # applied to permutation test p-values
  alpha: 0.05             # FDR significance threshold
  significance:
    n_permutations: 1000  # → increase to 5000 for precision at p < 0.001
    n_bootstrap: 500      # bootstrap resamples for centroid CI
    seed: 42              # random seed (full reproducibility)
    min_events_per_label: 5  # behaviors with fewer events are skipped
    n_jobs: 1             # parallelization (future use)
```

The two significance analyses are independently gated:
```yaml
analyses:
  annotation_overlap_significance: true
  annotation_centroids_significance: true
```

Both require their parent analyses to have run (`annotation_overlap` and `annotation_centroids` respectively).

---

## Analysis 2: Annotation Centroids

**Entry point**: `run_annotation_centroids()` — [annotation_analysis.py:209](src/annotation_analysis.py#L209).

**Question**: In 128D LISBET embedding space, how close is each cluster's centroid to each behavior's centroid?

This analysis operates frame-by-frame on raw embeddings — unlike the embedding × kinematics analysis which averaged embeddings per segment.

### Embedding Loading

For each V-record subject-session, all segment embedding CSVs are loaded ([annotation_analysis.py:275–296](src/annotation_analysis.py#L275)):
```
embeddings_dir/<segment_name>/features_lisbet_embedding.csv
```
Each CSV is a `(N_frames × 128)` matrix (float16). It is stored in memory as a dict `segment_name → np.ndarray`.

### Accumulation (Frame Level)

For each annotation event, `annotation_to_frames()` yields `(segment_name, rel_frame)` pairs. For each such frame:

1. Look up the embedding vector: `vec = seg_embeddings[seg_name][rel_frame]` — a single 128D vector ([annotation_analysis.py:317](src/annotation_analysis.py#L317)).
2. Append to `behavior_embeddings[behavior]` — accumulating all 128D frame-level vectors for that behavior across all V-records.
3. Also accumulate into `cluster_sum[cluster_id]` and `cluster_count[cluster_id]` for computing cluster centroids from V-record frames ([annotation_analysis.py:321–328](src/annotation_analysis.py#L321)).

### Annotation Centroids

For each behavior, centroids are **event-weighted**: first compute one mean embedding per event, then average event means ([annotation_analysis.py](src/annotation_analysis.py)):
```python
means_mat = np.array(event_embedding_means[behavior])   # (n_events, 128)
annotation_centroids[behavior] = means_mat.mean(axis=0) # shape: (128,)
```

This matches the bootstrap unit of resampling (events), so the bootstrap distribution is centered on the point estimate.

### Global Cluster Centroids (All Subjects)

A separate, more accurate computation is run first via `compute_global_cluster_centroids()` ([annotation_analysis.py:775](src/annotation_analysis.py#L775)), using **all** subjects (not just V-records). It iterates over every segment in the registry, loads its embedding CSV, and accumulates frame vectors per cluster. The global centroid per cluster is the mean over all frames assigned to that cluster across the full dataset. Saved to:
```
data/cluster_centroids_global.csv   — (N_clusters × 128)
```

The centroid results from `run_annotation_centroids()` (V-records only) also produce cluster centroids as a by-product, used for the distance matrix.

### Distance Matrix

Cosine distance in 128D between each annotation centroid and each cluster centroid ([annotation_analysis.py:359–365](src/annotation_analysis.py#L359)):
```python
dist = np.linalg.norm(annotation_centroid[behavior] - cluster_centroid[cluster_id])
```

Result: a `(N_behaviors × N_clusters)` float32 DataFrame.

### Cluster Labeling

Each cluster is assigned its **nearest annotation behavior** = the behavior whose centroid is closest in embedding space ([annotation_analysis.py:369–379](src/annotation_analysis.py#L369)):
```python
nearest_behavior = distance_matrix[cluster_id].idxmin()
min_distance = distance_matrix[cluster_id].min()
```

### Outputs

```
annotation_centroids_L{1,2,3}.pkl          — dict: behavior → np.ndarray (128,)
data/cluster_centroids_global.csv           — (N_clusters × 128) all-subject centroids
data/cluster_distance_matrix_L{1,2,3}.csv   — (N_behaviors × N_clusters) distances
data/cluster_cluster_behavior_labels_L{1,2,3}.csv
                                            — (N_clusters) nearest_behavior + min_distance
```

### Visualization

`plot_annotation_centroid_distance()` ([visualization.py:221](src/visualization.py#L221)): hierarchically clustered heatmap (Ward, Euclidean) of the distance matrix, restricted to the top-40 clusters by lowest mean distance. Color scale: `viridis_r` (dark = close = similar in embedding space).
```
plots/annotation_centroid_distance_L{1,2,3}.{png,pdf}
```

---

## Summary Flow

```
annotations CSV (code, behavior, start, stop)
    │
    ├─ build_multilevel_annotations() → L1/L2/L3 label sets
    │
    └─ for each V-record code → resolve to subject_session
            │
            ├─ annotation_to_frames()
            │     annotation time (sec) × fps → abs frame → rel_frame in segment
            │     segment_registry: tracking.nc → abs frame ranges per segment
            │
            │   ANALYSIS 1: OVERLAP
            │   ├─ frame_to_cluster lookup: (segment_name, rel_frame) → cluster_id
            │   ├─ count (behavior, cluster_id) co-occurrences
            │   ├─ contingency table (N_behaviors × N_clusters)
            │   └─ enrichment = observed_frac / global_cluster_frac
            │       → annotation_cluster_{contingency,enrichment,per_record}_{L1,L2,L3}.csv
            │       → annotation_cluster_heatmap_{L1,L2,L3}.{png,pdf}
            │
            │   SIGNIFICANCE 1: ENRICHMENT PERMUTATION TEST (per level)
            │   ├─ build_event_cluster_matrix()
            │   │     for each event: count frames per cluster → (n_events × n_clusters) matrix
            │   │     (reuses frame_to_cluster dict, no re-loading)
            │   ├─ permutation_test_enrichment()
            │   │     for n_permutations: circular shift labels within each session timeline
            │   │     → recompute enrichment matrix
            │   │     p_raw[b,k] = #{perm: |E_perm - 1| >= |E_obs - 1|} / n_permutations
            │   │     effect size: log2(enrichment)
            │   │     FDR-BH correction across all (behavior × cluster) pairs per level
            │   └─ → annotation_cluster_significance_{L1,L2,L3}.csv  (long format)
            │       → annotation_cluster_pfdr_{L1,L2,L3}.csv          (wide matrix)
            │       → annotation_cluster_log2effect_{L1,L2,L3}.csv    (wide matrix)
            │       → annotation_enrichment_significance_{L1,L2,L3}.{png,pdf}
            │
            └─ ANALYSIS 2: CENTROIDS (embedding space)
               ├─ load features_lisbet_embedding.csv per segment (N_frames × 128)
               ├─ for each annotated frame: collect 128D vec → behavior_embeddings[behavior]
               │   + per-event mean embedding → event_embedding_means[behavior]  (NEW)
               ├─ annotation centroid = mean of per-event means (event-weighted)
               ├─ cluster centroid = mean over all frames per cluster (V-records only)
               │   + global centroids from all subjects → cluster_centroids_global.csv
               ├─ distance matrix = Euclidean dist (128D) between each (behavior, cluster) pair
               └─ cluster label = nearest behavior centroid (argmin distance)
                   → annotation_centroids.pkl
                   → cluster_distance_matrix.csv
                   → cluster_cluster_behavior_labels.csv
                   → annotation_centroid_distance.{png,pdf}

               SIGNIFICANCE 2: BOOTSTRAP CI ON CENTROID DISTANCES
               ├─ bootstrap_centroid_distances()
               │     for n_bootstrap: resample events with replacement per behavior
               │     → boot_centroid = mean of resampled event means
               │     → boot_dist[j, k] = ||boot_centroid - cluster_centroid[k]||_2
               │     CI: percentiles of boot_dist distribution (default 95%)
               │     stability: fraction of resamples where same cluster is nearest
                 └─ → annotation_centroid_bootstrap_summary_L{1,2,3}.csv  (long format)
                   → annotation_centroid_ci_{low,high}_L{1,2,3}.csv     (wide matrices)
                   → annotation_centroid_nearest_stability_L{1,2,3}.csv (per behavior)
                   → annotation_centroid_bootstrap_overview_L{1,2,3}.{png,pdf}
```
