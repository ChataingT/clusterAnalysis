# Methods

Statistical methods, data linkage algorithms, and assumptions used by clusterAnalysis.

---

## Cluster Prevalence

For each subject, the **cluster prevalence vector** is computed as:

```
prevalence[subject, k] = frames_in_cluster_k / total_frames_subject
```

Each row of the prevalence matrix sums to 1.0. This representation captures both
*presence* (is this cluster used at all?) and *intensity* (what fraction of time?).

Clusters are defined at the video level: all frames from all segments of one subject
are pooled before computing proportions.

---

## Annotation → Frame Mapping

Expert annotations are stored in seconds (absolute video time). LISBET embeddings
and cluster assignments are indexed per-frame within each video segment (each segment
starts at frame 0).

**Mapping procedure:**
1. Convert annotation `start_sec` and `stop_sec` to absolute frame indices:
   `abs_frame = int(time_sec × fps)` (fps = 20 by default)
2. For each segment of the annotated subject: read `segment_start_frame` and
   `segment_end_frame` from `tracking.nc` (NetCDF attributes).
3. Compute segment-relative frame: `rel_frame = abs_frame - segment_start_frame`
4. Look up `(segment_name, rel_frame)` in the cluster mapping.

**FPS validation:** When building the segment registry, if `tracking.nc` contains
an `fps`, `frame_rate`, or `sample_rate` attribute, it is compared to the configured
fps. A mismatch > 0.5 fps triggers a WARNING so annotation timing errors can be caught
early.

Segments overlap (window=1200 frames, overlap=600 frames in the BIRCH pipeline).
A given video frame may therefore appear in multiple segments with potentially different
cluster assignments. All matching (segment, frame) pairs are included, which increases
annotation coverage and does not bias enrichment scores because both numerator and
denominator are computed consistently.

**Coverage check:** The pipeline logs the fraction of annotation events that
successfully map to at least one segment. Values below 80% trigger a WARNING.
Common causes: annotation timestamps outside the embedded portion of the video,
incorrect fps setting.

---

## Annotation Hierarchy (Three Levels)

Expert annotations carry three levels of label granularity:

| Level | Columns used | Description |
|---|---|---|
| **L1** | `behavior` | Broad behavioral category (13 unique values) |
| **L2** | `behavior` + `behavioral_category` | Concatenated composite label |
| **L3** | `behavior` + `behavioral_category` + `modifier_1` | Finest-grain label (65+ composites) |

Composite labels at L2 and L3 are formed by joining non-empty fields with ` | ` separator.
All annotation analyses (overlap, kinematics) are run at all three levels independently,
with level-specific minimum-frame thresholds for inclusion (default: 10 / 5 / 3 frames).

---

## Annotation Enrichment

For a given label L and cluster K at any hierarchy level:

```
enrichment(L, K) = P(K | L) / P(K | V-records)
```

where:
- `P(K | L)` = fraction of annotated frames (for label L) assigned to cluster K
- `P(K | V-records)` = global fraction of frames in cluster K, computed **over V-records only**
  (not the full dataset), to avoid diluting the denominator.

`enrichment > 1` means cluster K appears more during label L than expected by chance.
`enrichment = 1` means no association.

---

## Annotation Centroids and Global Cluster Centroids

For each behavior type, the **annotation centroid** is event-weighted: first compute
one mean 128-D LISBET embedding per annotated event, then average those event means
(V-records only):

```
event_mean(e) = mean( embedding[frame] for frames in event e )
centroid(B) = mean( event_mean(e) for events e with behavior B )
```

This matches the bootstrap resampling unit (events), preventing weighting mismatch
between point estimates and confidence intervals.

**Global cluster centroids** are computed from **all subjects** (not just V-records),
loading all available segment embeddings from `embeddings_dir`:

```
cluster_centroid(K) = mean( embedding[frame] for all frames assigned to cluster K,
                             across all records )
```

Using all subjects rather than V-records only produces more stable centroids for rare
clusters (which may be poorly sampled in the 28 V-records).

**Cluster labeling:** Each cluster is assigned the behavior whose centroid is
nearest in Euclidean distance (128-D space):

```
label(K) = argmin_B  ||centroid(B) − cluster_centroid(K)||₂
```

Annotation centroids are saved as a `.pkl` file for reuse: e.g., predicting likely
behaviors in unannotated records.

---

## Clinical Correlations

### Binary (ASD vs TD)
- **Test:** Mann-Whitney U (non-parametric, no normality assumption)
- **Effect size:** Cohen's d (pooled standard deviation)
- **Multiple testing:** Benjamini-Hochberg FDR correction over all cluster tests
- **Input:** Per-subject prevalence vector; groups defined by `diagnosis` column

### Continuous metrics (ADOS, MSEL, Vineland)
- **Test:** Spearman rank correlation (non-parametric)
- **Confidence interval:** Bootstrap percentile CI (500 resamples, 95%)
- **Multiple testing:** BH-FDR over all (cluster × metric) pairs simultaneously
- **Input:** Per-subject prevalence per cluster vs. clinical score

**Statistical power note:** Using all N≈102 subjects maximizes power for detecting
clinical associations. The V-subset (N≈28) is used only for annotation analyses
where ground-truth behavioral labels are required.

---

## Kinematic Profiles (Segment Level)

Kinematics are measured at segment level (`metrics_summary.csv`): for each
segment, the `norm_mean` of each kinematic metric is used (normalized by trunk
height for within-subject comparability).

The **dominant cluster** for each segment is taken from `cross_video_train.csv`
(the cluster ID that covers the most frames in that segment).

Per-cluster profiles are computed as **frame-count-weighted means**:

```
profile_mean(K, metric) = Σ_seg [ n_frames(seg) × metric_mean(seg) ] / Σ_seg n_frames(seg)
      (sum over segments whose dominant cluster is K)
```

This weighting ensures longer segments contribute proportionally more.

**V vs non-V consistency:** The same profiles are computed separately for V-records
and non-V records, then compared. High correlation (ρ > 0.7) confirms that clusters
have consistent kinematic signatures regardless of annotation availability, supporting
the validity of using V-record annotations to interpret globally-derived clusters.

---

## Kinematic Profiles (Frame Level)

Unlike the segment-level approach, frame-level kinematic profiling directly aligns
individual frames with their cluster assignment by loading `metrics_normalised.csv`
per segment (65 kinematic columns × N frames per segment).

**Procedure:**
1. For each segment, load `metrics_normalised.csv` (65 columns × N frames).
2. Join each frame with its cluster ID from the cluster mapping via `(segment_name, frame_index)`.
3. Accumulate per-cluster statistics using running sum and sum-of-squares (memory-efficient;
   processes one segment at a time without loading all frames simultaneously):
   ```
   mean(K, metric) = Σ frames_in_K  metric_value  /  n_frames(K)
   std(K, metric)  = sqrt( E[X²] - E[X]² )
   ```
4. For significance testing, up to 5,000 frame samples per cluster are retained in
   memory (subsampled uniformly as frames are processed).

**Kruskal-Wallis test:** For each kinematic metric, a Kruskal-Wallis H-test is run
with clusters as groups (using the retained samples). BH-FDR correction is applied
over all metrics simultaneously. A significant result means the metric takes
significantly different values across clusters at the frame level.

Frame-level profiles are more accurate than segment-level profiles because they avoid
the "dominant cluster" approximation — every frame is matched to its own cluster
rather than the segment's most common one.

---

## Annotation Kinematics

For each annotated V-record frame, the corresponding kinematic values from
`metrics_normalised.csv` are collected and grouped by annotation label.

**Per-label profile:**
```
profile(L, metric) = mean( metric[frame] for all frames annotated with label L )
```

A **background profile** is computed from non-annotated frames in the same V-record
sessions, providing a reference for what "typical" movement looks like outside of
annotated behavioral episodes.

This analysis is run at all three annotation hierarchy levels (L1, L2, L3) with
level-specific minimum-frame thresholds. Results enable direct comparison of
kinematic signatures across behaviors at increasing granularity, and between
behaviors and the non-annotated background.

---

## Cluster Profiles Synthesis

After all individual analyses complete, a single summary CSV (`cluster_profiles.csv`)
is generated with one row per cluster, combining:

- **Annotation label**: nearest annotation centroid at L1 (embedding distance)
- **Enrichment top behavior**: L1 behavior with highest observed/expected enrichment
- **Kinematic signature**: top-3 high and top-3 low z-scored metrics (frame-level
  profiles preferred; falls back to segment-level if frame-level unavailable)
- **Clinical significance**: p_fdr and Cohen's d for each binary group comparison
- **Top clinical correlations**: up to 3 significant Spearman correlations with
  continuous clinical scores (sorted by |ρ|)

This CSV is the primary entry point for interpreting clusters without opening
individual analysis files.

---

## Semantic Dictionary (Embedding × Kinematics)

For each segment, the mean 128-D embedding vector is computed (mean over all frames
in the segment). This gives a (N_segments × 128) matrix.

Spearman correlation is computed between each of the 128 dimensions and each
kinematic metric across segments. This produces a (128 × N_metrics) correlation
matrix, corrected for multiple testing (BH-FDR over all 128 × 65 = 8,320 pairs).

Significant correlations reveal which embedding dimensions "encode" which physical
movements — forming a human-interpretable "dictionary" of what the transformer has
learned to represent.

---

## FDR Correction

Benjamini-Hochberg (BH) procedure is used throughout:
1. Sort p-values in ascending order: p(1) ≤ p(2) ≤ ... ≤ p(m)
2. BH-corrected p-value for rank i: `p_fdr[i] = min(p[i] × m / rank[i], 1.0)`
3. Enforce monotonicity: `p_fdr[i] = min(p_fdr[i], p_fdr[i+1])`

The FDR threshold controls the expected proportion of false positives among all
rejected null hypotheses (e.g., alpha=0.05 → at most 5% of significant findings
are expected to be false).

---

## Known Limitations

1. **Annotated subset (N=28):** All annotation-based analyses have low statistical
   power. Enrichment scores and annotation-centroid labels are descriptive, not
   inferential. Results should be interpreted as hypotheses for future validation.

2. **Segment overlap:** Overlapping segments mean the same video frame may appear
   in multiple segments with potentially different cluster assignments (due to
   smoothing boundary effects). This is accounted for in the frame-level kinematic
   analysis by processing each (segment, frame) independently; aggregate cluster
   statistics are robust to this.

3. **Dominant cluster approximation (segment-level only):** The segment-level
   kinematic analysis uses the per-segment dominant cluster, ignoring within-segment
   cluster variation. The frame-level kinematic analysis (Analysis 7) does not have
   this limitation and should be preferred for interpretation.

4. **Annotation centroid noise:** Annotation centroids for rare behaviors are computed
   from few frames and may not be representative of the full embedding distribution.
   The `min_distance` column in `cluster_behavior_labels.csv` indicates reliability:
   large distances suggest uncertain labeling.

5. **Annotation kinematics confound:** V-records are ADOS sessions; the clinician
   deliberately elicits behaviors. Kinematics during annotated behaviors may reflect
   session structure (e.g., clinician positioning) as much as the behavior itself.
   The background comparison (non-annotated V-record frames) partially controls for this.

6. **Frame-level kinematic memory:** The Kruskal-Wallis test uses up to 5,000
   subsampled frames per cluster (not the full frame set) to stay within memory limits.
   For very large clusters this is conservative; the effect estimate (mean/std) uses
   all frames via running statistics.
