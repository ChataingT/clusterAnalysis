# Embedding × Kinematics Analysis

## Overview

This analysis correlates the 128-dimensional LISBET behavioral embedding with kinematic metrics computed from skeleton pose data, **at the segment level**. It is gated behind the `analyses.embedding_kinematics` config flag ([config.py:77](src/config.py#L77)) and runs as **Analysis Step 5** in the main pipeline ([run_analysis.py:658](src/run_analysis.py#L658)).

The outputs are four CSV files:
- `data/embedding_kinematic_rho.csv` — Spearman ρ matrix
- `data/embedding_kinematic_p_raw.csv` — raw p-value matrix
- `data/embedding_kinematic_p_fdr.csv` — FDR-corrected p-value matrix
- `data/embedding_kinematic_significant.csv` — boolean significance mask

And one plot:
- `plots/embedding_kinematic_heatmap.{png,pdf}` — clustered heatmap of significant ρ values

---

## Data Sources

### 1. Embedding Data (`embeddings_dir`)

- **Origin**: `cfg.data.embeddings_dir` — a directory configured in the YAML config ([config.py:65](src/config.py#L65)).
- **Structure**: One subdirectory per segment, named by `segment_name` (e.g. `7797_T1a_ADOS1_seg_001/`). Each subdirectory contains:
  ```
  features_lisbet_embedding.csv
  ```
  This CSV has frame-level LISBET embeddings: each row is a video frame, each column is one of the 128 embedding dimensions.
- **State when loaded**: Raw float32 matrix of shape `(N_frames, 128)`.
- **Transformation**: The per-frame rows are averaged to produce **one 128-dimensional mean vector per segment** ([embedding_analysis.py:66](src/embedding_analysis.py#L66)):
  ```python
  mean_vec = emb.mean(axis=0).astype(dtype)  # float16 for storage
  ```
- **Result**: A DataFrame of shape `(N_segments, 128)`, indexed by `segment_name` ([embedding_analysis.py:76–85](src/embedding_analysis.py#L76)).

### 2. Kinematic Data (`pose_records_dir`)

- **Origin**: `cfg.data.pose_records_dir` — the pose record directories ([config.py:64](src/config.py#L64)).
- **Structure**: One directory per subject-session, named `results_skeleton_<subject_session_id>/`. Inside:
  ```
  segments/seg_XXX/metrics_summary.csv
  ```
  `metrics_summary.csv` has one row per kinematic metric (e.g. `child_speed_centroid`, `clinician_speed_trunk`, `facingness`, etc.) with columns `norm_mean`, `norm_std`, `raw_mean`, `raw_std`, `raw_count`, `pct_valid_raw`.
- **State when loaded**: Pre-computed per-segment summary statistics (means and stds over all valid frames in that segment).
- **Transformation**: The loader ([data.py:280–283](src/data.py#L280)) renames columns to `{metric}__mean` and `{metric}__std` and assembles a DataFrame with MultiIndex `(subject_session, seg_label)`:
  ```python
  row = summary[mean_col].rename(lambda m: f"{m}__mean")
  row_std = summary[std_col].rename(lambda m: f"{m}__std")
  ```
  Whether `norm_mean` or `raw_mean` is used is controlled by `cfg.kinematics.use_normalized` (default: `True`) ([config.py:91](src/config.py#L91)).
- **Result**: A DataFrame of shape `(N_total_segments, N_metrics*2)` with MultiIndex `(subject_session, seg_label)`.

Only `__mean` columns are passed to the embedding analysis ([run_analysis.py:663](src/run_analysis.py#L663)):
```python
metric_cols = [c for c in kinematics_df.columns if c.endswith("__mean")]
```

---

## Joining Embeddings and Kinematics

The join is done inside `run_embedding_kinematic_correlation()` ([embedding_analysis.py:89](src/embedding_analysis.py#L89)).

The challenge is that embeddings are indexed by `segment_name` (e.g. `7797_T1a_ADOS1_seg_001`) while kinematics use a MultiIndex `(subject_session, seg_label)` (e.g. `("7797_T1a_ADOS1", "seg_001")`).

The **subject_map** (built earlier by `build_subject_map()` in `linking.py`) provides the bridge. It has columns `segment_name`, `subject_session`, and `seg_label`. A `seg_label` column is extracted from `segment_name` via regex ([embedding_analysis.py:137](src/embedding_analysis.py#L137)):
```python
sm["seg_label"] = subject_map["segment_name"].str.extract(r"_(seg_\d+)$")[0]
```

Kinematics are merged onto this map using `(subject_session, seg_label)` as keys ([embedding_analysis.py:140](src/embedding_analysis.py#L140)), producing a DataFrame indexed by `segment_name`.

Finally, only segments present in **both** the embedding DataFrame and the kinematic DataFrame are kept ([embedding_analysis.py:148](src/embedding_analysis.py#L148)):
```python
common_segs = emb_df.index.intersection(kin_by_segment.index)
```

---

## What Is Computed

### Spearman Correlation Matrix

For each of the ~65 kinematic metrics (the `__mean` columns) × 128 embedding dimensions, a **Spearman rank correlation** is computed ([embedding_analysis.py:167–184](src/embedding_analysis.py#L167)):

```python
for j, metric in enumerate(valid_metrics):
    y = kin[:, j]
    valid_mask = np.isfinite(y)          # skip NaN frames
    if valid_mask.sum() < 10: continue   # skip if too few valid segments

    for i in range(N_EMBEDDING_DIMS):    # 128 dims
        x = emb_valid[:, i]
        if np.std(x) < 1e-10: continue  # skip constant embedding dim
        rho, p = spearmanr(x, y_valid)
```

This yields a `(128, N_metrics)` matrix of ρ values and a `(128, N_metrics)` matrix of raw p-values. The total number of statistical tests is **128 × N_metrics** (reported in the log: e.g. 128 × 65 = 8,320 tests).

### FDR Correction

All p-values are corrected together (over the full flattened 128 × N_metrics array) using the **Benjamini-Hochberg** procedure by default ([embedding_analysis.py:190–192](src/embedding_analysis.py#L190)):

```python
p_flat = p_matrix.flatten()
p_fdr_flat = fdr_correct(p_flat, method=fdr_method)   # stats.py:174
p_fdr_matrix = p_fdr_flat.reshape(p_matrix.shape)
```

The BH procedure is implemented manually in [stats.py:206–212](src/stats.py#L206) to enforce monotonicity.

A significance mask is then derived: `p_fdr < alpha` (default `alpha = 0.05`, [config.py:112](src/config.py#L112)).

### Output DataFrames

All four result DataFrames share the same structure ([embedding_analysis.py:196–200](src/embedding_analysis.py#L196)):
- **Index**: embedding dimension label (strings `"0"` to `"127"`)
- **Columns**: kinematic metric names (e.g. `child_speed_centroid__mean`)

| File | Content |
|---|---|
| `embedding_kinematic_rho.csv` | Spearman ρ ∈ [−1, 1] |
| `embedding_kinematic_p_raw.csv` | Raw p-values ∈ [0, 1] |
| `embedding_kinematic_p_fdr.csv` | FDR-corrected p-values (BH) |
| `embedding_kinematic_significant.csv` | Boolean: `p_fdr < 0.05` |

---

## Visualization

The heatmap ([visualization.py:548](src/visualization.py#L548)) shows only the **top-N kinematic metrics** ranked by the number of embedding dimensions that are significantly correlated with them ([visualization.py:561–562](src/visualization.py#L561)):

```python
n_sig_per_metric = sig_df.sum(axis=0)
top_metrics = n_sig_per_metric.nlargest(min(top_n_metrics, ...)).index
```

Non-significant cells are masked (shown in grey). Kinematic columns are hierarchically clustered (Ward linkage, Euclidean distance). Embedding dimension rows are **not** clustered (128 rows would produce an unreadable dendrogram — [visualization.py:570](src/visualization.py#L570)).

Color scale: `RdBu_r`, centered at 0, clipped at ±0.5.

---

## Summary Flow

```
embeddings_dir/
  <segment_name>/features_lisbet_embedding.csv   ← (N_frames × 128) per-frame embeddings
    └─ mean over frames ──────────────────────→  mean embedding vector (128D) per segment

pose_records_dir/
  results_skeleton_<ss_id>/segments/seg_XXX/metrics_summary.csv  ← pre-computed kinematic stats
    └─ norm_mean column, renamed to {metric}__mean ──────────→  one value per segment per metric

subject_map (from linking.py)
    └─ bridges segment_name ↔ (subject_session, seg_label)

                  ↓ join on common segments ↓

  emb: (N_common × 128)     kin: (N_common × N_metrics)
          └──────── Spearman ρ for each (dim, metric) pair ────────┘
                     128 × N_metrics tests
                     FDR correction (BH) over all pairs
                     significance at p_fdr < 0.05

Output: embedding_kinematic_{rho, p_raw, p_fdr, significant}.csv
        embedding_kinematic_heatmap.{png, pdf}
```
