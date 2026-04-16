# Cluster × Clinical Analysis

## Overview

This analysis tests whether LISBET cluster usage (how much time a subject spends in each cluster) relates to clinical diagnosis and scores. It runs as **Analysis Step 3** ([run_analysis.py:577](src/run_analysis.py#L577)), gated by `analyses.clinical_correlations`. It uses **all subjects** with both cluster data and clinical data (not restricted to V-records).

Two sub-analyses are run:

| Sub-analysis | Config key | Test | Question |
|---|---|---|---|
| Binary | `clinical.binary_groups` (default: `["diagnosis"]`) | Mann-Whitney U + Cohen's d | Is cluster prevalence different between ASD and TD? |
| Continuous | `clinical.continuous` (7 scores) | Spearman ρ + bootstrap CI | Does cluster prevalence correlate with severity scores? |

---

## Input: Prevalence Matrix

The central input to both analyses is the **prevalence matrix**, computed by `compute_prevalence_matrix()` ([linking.py:318](src/linking.py#L318)) before the clinical step.

### What it is

A `(N_subjects × N_clusters)` DataFrame where:
- **Index**: clinical uuid (numeric, e.g. `"7797"`)
- **Columns**: cluster IDs (integers)
- **Values**: fraction of frames spent in each cluster per subject — each row sums to 1.0

### How it is built

1. The `subject_map` (from `build_subject_map()`) provides the bridge from `segment_name` → `uuid_numeric`.
2. A dict `segment_name → uuid_numeric` is built and joined onto `cluster_mapping` ([linking.py:341–343](src/linking.py#L341)).
3. Frames are grouped by `(uuid, cluster_id)` and counted.
4. Each row is normalized by the subject's total frame count ([linking.py:362](src/linking.py#L362)):
   ```python
   prevalence = counts.div(counts.sum(axis=1), axis=0)
   ```

### State at entry to clinical analysis

- Shape: `(N_subjects, N_clusters)`, e.g. 120 × K clusters
- All values ∈ [0, 1], each row sums to 1.0
- Indexed by the same uuid as `clinical_df`
- Saved as: `data/prevalence_matrix.csv`

---

## Input: Clinical Data

Loaded from the CSV at `cfg.data.clinical_csv` by `load_clinical()` ([data.py:92](src/data.py#L92)). Indexed by `uuid`. Relevant columns:

**Binary grouping** (default column: `diagnosis`):
- Values: `"ASD"` / `"TD"`

**Continuous scores** (default list from [config.py:98–106](src/config.py#L98)):
```
ADOS_2_TOTAL
ADOS_G_ADOS_2_TOTAL_score_de_severite
ADOS_2_ADOS_G_REVISED_SA_SEVERITY_SCORE
ADOS_2_ADOS_G_REVISED_RRB_SEVERITY_SCORE_new
ADOS_2_SOCIAL_AFECT_TOTAL
AdSS
TOTAL_DQ
```

---

## Sub-analysis 1: Binary (ASD vs TD)

**Function**: `run_binary_analysis()` — [clinical_analysis.py:33](src/clinical_analysis.py#L33)

### Alignment

Only subjects present in **both** the prevalence matrix and `clinical_df` are kept ([clinical_analysis.py:69](src/clinical_analysis.py#L69)):
```python
common_uuids = prevalence_matrix.index.intersection(clinical_df.index)
```

### Test: Mann-Whitney U per cluster

For each cluster column in the prevalence matrix, the subjects are split into two groups by their `diagnosis` value (e.g. `"ASD"` vs `"TD"`). `mann_whitney_with_effect()` ([stats.py:46](src/stats.py#L46)) is called:

```python
vals_a = prev.loc[group_a_mask, cluster_id]   # ASD subjects' prevalence for this cluster
vals_b = prev.loc[group_b_mask, cluster_id]   # TD subjects' prevalence
U, p = mannwhitneyu(a, b, alternative="two-sided")
```

NaN values are dropped. Groups with fewer than 3 samples trigger a warning.

### Effect Size: Cohen's d

Computed as the standardized mean difference using pooled standard deviation ([stats.py:92–101](src/stats.py#L92)):
```python
pooled_std = sqrt(((n_a-1)*std_a² + (n_b-1)*std_b²) / (n_a + n_b - 2))
d = (mean_a - mean_b) / pooled_std
```
Direction is recorded as `"higher_in_ASD"` or `"higher_in_TD"`.

### FDR Correction

After all clusters are tested, Benjamini-Hochberg FDR is applied over all cluster p-values simultaneously ([clinical_analysis.py:113](src/clinical_analysis.py#L113)):
```python
results["p_fdr"] = fdr_correct(results["p_raw"].values, method=fdr_method)
```

Significance flags: `p_fdr < alpha` (default 0.05), with `sig_label` (`*`, `**`, `***`, `ns`).

### Output

Sorted by `p_fdr`. One row per cluster:
```
data/clinical_binary_diagnosis_results.csv
```
Columns: `cluster_id, U, p_raw, p_fdr, cohens_d, direction, n_a, n_b, significant, sig_label`

---

## Sub-analysis 2: Continuous Correlations

**Function**: `run_continuous_correlations()` — [clinical_analysis.py:122](src/clinical_analysis.py#L122)

### Test: Spearman ρ per (cluster, metric) pair

For every combination of cluster × clinical metric, `spearman_with_ci()` ([stats.py:116](src/stats.py#L116)) is called:

```python
x = prev[cluster_id].values     # prevalence across subjects (N,)
y = clin[metric].values         # clinical score across subjects (N,)
rho, p = spearmanr(x, y)        # NaN pairs are dropped before call
```

Minimum 3 valid pairs required; otherwise returns NaN.

### Bootstrap 95% CI

500 bootstrap resamples (with replacement) of the paired `(x, y)` observations are drawn, Spearman ρ is computed on each, and the 2.5th / 97.5th percentiles give the confidence interval ([stats.py:153–166](src/stats.py#L153)):
```python
idx = rng.integers(0, n, size=n)
r, _ = spearmanr(x[idx], y[idx])
ci_low, ci_high = percentile(boot_rhos, [2.5, 97.5])
```

### FDR Correction

Applied over **all** (cluster, metric) pairs simultaneously — e.g. K clusters × 7 metrics tests ([clinical_analysis.py:193](src/clinical_analysis.py#L193)):
```python
results["p_fdr"] = fdr_correct(results["p_raw"].values, method=fdr_method)
```

### Output

Sorted by `p_fdr`. One row per (cluster, metric) pair:
```
data/clinical_continuous_results.csv
```
Columns: `cluster_id, metric, rho, p_raw, p_fdr, ci_low, ci_high, n, significant, sig_label`

---

## Visualizations

### Volcano Plot — `plot_clinical_volcano()` ([visualization.py:264](src/visualization.py#L264))

One plot per binary group column (default: `diagnosis`). X-axis = Cohen's d, Y-axis = −log₁₀(p_fdr). Points colored by direction (ASD vs TD). Horizontal dashed line at −log₁₀(alpha).
```
plots/clinical_volcano_diagnosis.{png,pdf}
```

### Violin Plots — `plot_clinical_violin()` ([visualization.py:368](src/visualization.py#L368))

Top-10 most discriminative clusters (significant + largest |Cohen's d|, falling back to largest d if none significant). One violin panel per cluster showing the distribution of prevalence values for ASD vs TD subjects. Annotated with d and p_fdr.
```
plots/clinical_violin_diagnosis.{png,pdf}
```

### Correlation Heatmap — `plot_clinical_correlation_heatmap()` ([visualization.py:308](src/visualization.py#L308))

Spearman ρ values for the top-30 clusters (by max |ρ| across metrics). Non-significant cells masked in grey. Ward hierarchical clustering on both axes. Color scale: `RdBu_r`, centered at 0, clipped at ±0.6.
```
plots/clinical_correlation_heatmap.{png,pdf}
```

---

## Summary Flow

```
cluster_mapping (frame → cluster_id)
    + subject_map (segment_name → uuid_numeric)
        └─ compute_prevalence_matrix()
               group by (uuid, cluster_id), count frames
               normalize per subject → fraction ∈ [0,1]
               shape: (N_subjects × N_clusters), each row sums to 1.0
               → data/prevalence_matrix.csv

clinical_df (indexed by uuid)
    ├─ diagnosis column (ASD / TD)
    └─ continuous scores (ADOS_2_TOTAL, ADOS severity, AdSS, TOTAL_DQ, ...)

                  ↓ align on common uuids ↓

  SUB-ANALYSIS 1: BINARY (ASD vs TD)
  For each cluster:
      Mann-Whitney U (two-sided) on prevalence values
      Cohen's d (pooled std)
  FDR correction (BH) over all clusters
  → clinical_binary_diagnosis_results.csv
  → clinical_volcano_diagnosis.{png,pdf}
  → clinical_violin_diagnosis.{png,pdf}   (top-10 discriminative clusters)

  SUB-ANALYSIS 2: CONTINUOUS
  For each (cluster × clinical metric) pair:
      Spearman ρ + 500-sample bootstrap CI
  FDR correction (BH) over all (cluster, metric) pairs
  → clinical_continuous_results.csv
  → clinical_correlation_heatmap.{png,pdf}
```
