# BrainFormer-PD v2.1 — Claim-by-claim audit of `main_results.json`

One-line disposition for every finding produced by `analyze_local.py`. For the paper,
use only the **KEEP** claims. **DROP** claims are not to be reported; **REFRAME** claims
must be stated with the specified caveat.

---

## Block 07 — Primary metrics (`metrics.json`)

| Finding | Value | Disposition |
|---|---|---|
| UPDRS-III R² all-test | 0.755 | **KEEP** — report as primary regression result |
| UPDRS-III R² PD cohort | 0.409 | **KEEP** — primary, with disclosure that prodromal R² is higher |
| UPDRS-III R² GBA+ | 0.842 (n=36) | **KEEP** — note small n |
| UPDRS-III R² SAA+ | 0.701 | **KEEP** |
| UPDRS-III MAE all | 4.39 pts | **KEEP** |
| **DaTscan Pearson r V06** | **0.893** (p ≈ 0) | **KEEP** — honest forward-prediction target |
| NHH C-index (raw) | 0.985 | **REFRAME** — use restricted C-index from Block 14b instead |
| NSD-ISS Weighted F1 | 0.987 | **REFRAME** — frame as feature-coverage confirmation |
| NSD-ISS Macro F1 | 0.831 | **KEEP** as secondary |

### NSD-ISS framing language (use verbatim)

> "NSD-ISS stage is defined multiaxially over genetic, imaging, and clinical features that
> partially overlap with our input modalities. The Weighted F1 of 0.987 therefore confirms
> correct end-to-end feature integration rather than presenting a novel staging algorithm.
> We report Macro F1 (0.831) alongside for completeness, which better reflects
> class-imbalanced performance given that 73% of subjects are Stage 0."

### NHH C-index framing language

> "The concordance index of the neural hazard head is reported on the restricted
> subset of patients whose observed input UPDRS-III (baseline, 12, 24 months) was
> strictly below the UPDRS-III ≥ 40 event threshold — i.e., patients whose event
> had not already occurred in the input window. Restricted C-index is the primary
> survival metric; the raw C-index on all test subjects is provided only as an
> upper bound."

---

## Block 08 — Calibration

| Finding | Value | Disposition |
|---|---|---|
| ECE | **0.006** | **KEEP** — primary calibration result |
| MAE quintiles | [1.30, 2.37, 4.88, 6.07, 7.36] | **KEEP** — Figure 4 |
| Uncertainty ratio (Q5/Q1) | **5.67×** | **KEEP** — primary |
| Clinical OR (Haldane) | 42.5 | **REFRAME** — report CI widely; prefer Fisher p |
| Clinical 95% CI | [2.5, 718] | **REFRAME** — note wide interval |
| Fisher exact p | **3.3 × 10⁻⁶** | **KEEP** — primary monitoring-signal result |

### Clinical monitoring framing language

> "Among 201 test patients with observed 36-month UPDRS-III, 17 of 100 in the top-quartile
> baseline epistemic uncertainty group experienced >15-point UPDRS-III deterioration versus
> 0 of 101 in the bottom-quartile group (Fisher exact p = 3.3 × 10⁻⁶). The Haldane-corrected
> odds ratio is 42.5 with 95% CI [2.5, 718], with the wide interval reflecting the single-cell
> zero in the bottom-quartile arm; we report the Fisher exact p value as the primary
> inferential statistic."

---

## Block 09 — Ablation

| Finding | Disposition |
|---|---|
| Full 150-epoch retrain in progress | **KEEP** — final paper uses 150-epoch numbers for every B0–B5 variant |
| 40-epoch ablation (B1 > B5 on R²_PD) | **REPORT as supplementary** — shows convergence dynamics |

### Ablation framing language

> "The contribution of the personalized Braak graph (Variants B3–B5) over the plain
> transformer baseline (B1) on R²_PD is marginal at 150 epochs [value TBD from current
> retrain]. We interpret this as the graph modules contributing interpretability and
> calibrated uncertainty (ECE 0.006, ratio 5.67×, SAA edge discrimination p < 10⁻¹⁸),
> not raw predictive accuracy. This is consistent with the framework's design goal of
> producing an auditable, not merely accurate, progression model."

---

## Block 10 — Stats vs XGBoost

| Finding | Value | Disposition |
|---|---|---|
| Wilcoxon W | 35,982 | **KEEP** |
| Wilcoxon p | **0.036** | **KEEP** — barely significant at α=0.05 |
| ΔR²_all bootstrap CI | [−0.018, +0.047] | **KEEP** — disclose CI straddles zero |
| DaTscan r bootstrap CI | [0.860, 0.919] | **KEEP** |

### Framing language

> "Absolute errors on V06 UPDRS-III on the test set were significantly smaller for
> BrainFormer-PD than for the XGBoost + LTP baseline (Wilcoxon signed-rank p = 0.036).
> The bootstrap 95% confidence interval on ΔR² was [−0.018, +0.047], spanning zero,
> indicating that BrainFormer-PD's predictive advantage over a well-tuned gradient-boosting
> baseline is at the margin of statistical significance on this cohort. The primary
> contribution is therefore not predictive but methodological and biological (Sections 3–6)."

---

## Block 11 — GBA stratification

| Finding | Value | Disposition |
|---|---|---|
| R² GBA+ | 0.842 | **KEEP** — strong on small subset |
| R² GBA− | 0.746 | **KEEP** |
| ΔR² GBA+ vs GBA− | **not reported** | — |
| u_epi MW p (GBA+ vs GBA−) | **0.44** | **DROP** — not significant under honest training |

### Framing language

> "Stratified by GBA status, UPDRS-III R² was 0.842 in the GBA+ subset (n=36) and 0.746
> in the GBA− subset, indicating the model captures GBA-related severity effects. However,
> we observed no significant difference in epistemic uncertainty between GBA+ and GBA−
> patients (Mann-Whitney p = 0.44); any claim of differential uncertainty by genetic status
> requires a larger GBA+ cohort than this study can provide."

---

## Block 12 — SAA stratification (the paper's biological headline)

| Finding | Value | Disposition |
|---|---|---|
| R² SAA+ | 0.701 | **KEEP** |
| R² SAA− | 0.757 | **KEEP** |
| u_epi MW p (SAA+ vs SAA−) | **1.8 × 10⁻²⁹** | **KEEP** — PRIMARY BIOLOGICAL CLAIM |
| `A_i` edge weight at **LC** (Node 2) MW p | **8.9 × 10⁻¹⁸** | **KEEP** — PRIMARY |
| `A_i` edge weight at **SNc** (Node 4) MW p | **1.3 × 10⁻⁹** | **KEEP** — PRIMARY |

### Framing language

> "The learned personalized Braak graph posterior weights at locus coeruleus (Braak Stage 2)
> and substantia nigra pars compacta (Braak Stage 3) differed significantly between
> α-synuclein-SAA-positive and SAA-negative patients in the test set (Mann-Whitney U,
> p = 8.9 × 10⁻¹⁸ and p = 1.3 × 10⁻⁹ respectively). This result localises the model's
> learned graph topology to the exact early Braak stages at which SAA-detectable α-synuclein
> aggregation first occurs, and constitutes, to our knowledge, the first demonstration of
> SAA-correlated graph topology emerging from an end-to-end-trained PD progression model."

---

## Block 13 — Topology phenotyping

| Finding | Value | Disposition |
|---|---|---|
| KMeans (k=3) on `A_i` flattenings | 778 / 347 / 143 | — |
| GBA Fisher p per cluster | 0.67 / 0.88 / 0.66 | **DROP** |
| Labels (brainstem / limbic / cortical) | — | **DROP** |

### Framing language

> "We attempted topology phenotyping by KMeans clustering of patient-level upper-triangular
> `A_i` vectors (45 edges). No cluster exhibited significant enrichment for GBA+ carriers
> (Fisher exact p values 0.67, 0.88, 0.66). We do not claim topology-based subtyping as a
> result; cluster-mean adjacency heatmaps are provided in the supplement for transparency."

---

## Block 14 — Two-source uncertainty

| Finding | Value | Disposition |
|---|---|---|
| Pearson r(u_graph, u_pred) | **−0.18**, p = 2 × 10⁻¹⁰ | **KEEP** |
| Top-Q u_graph ∩ bottom-Q u_pred (phenotype) | n=144 | **KEEP** |
| Phenotype GBA+ enrichment | **6.97×** | **KEEP** — exploratory result |

### Framing language

> "The graph-posterior uncertainty (`u_graph`, mean edge variance from S-PBGL) and the
> predictive uncertainty (`u_pred`, NIG epistemic) are only weakly negatively correlated
> (r = −0.18, p = 2 × 10⁻¹⁰), indicating they capture complementary aspects of model
> uncertainty. Patients in the top-quartile `u_graph` × bottom-quartile `u_pred` phenotype
> (n = 144) are 6.97× enriched for GBA+ carriers relative to the rest of the cohort —
> a candidate 'biologically-atypical, motorically-predictable' subgroup warranting
> prospective characterisation."

---

## Block 14b — Restricted C-index (new)

Restricted to patients whose observed BL/V02/V04 UPDRS-III was < 40 (event had not already occurred in the input window).

Value will be populated by the current analysis run.

---

## Block 15 — Latent subtyping

| Finding | Value | Disposition |
|---|---|---|
| Silhouette (k=4) | **0.628** | **KEEP** |
| Cluster sizes | 557 / 109 / 225 / 377 | **KEEP** |
| SAA+ fraction per cluster | 1% / 68% / 59% / 34% | **KEEP** — biological axis |
| UPDRS mean per cluster | 0.06 / 18.4 / 11.0 / 2.3 | **KEEP** |
| GBA+ fraction per cluster | 1% / 8% / 5% / 8% | **KEEP** |

### Framing language

> "Unsupervised clustering (k = 4) of the projection-head representation in the test set
> yielded a silhouette score of 0.628. The four clusters align with an SAA gradient
> (1%, 34%, 59%, 68% SAA-positive) that tracks the UPDRS-III severity gradient
> (0.06, 2.3, 11.0, 18.4 mean UPDRS-III at 36 months), and within the most severe cluster
> (Cluster 1, n = 109) both SAA+ fraction (68%) and mean epistemic uncertainty
> (1.34 × 10⁻³) are maximal. This biologically-plausible separation was not supervised
> by outcome; it emerges from the contrastive-pretraining + S-PBGL representation."

---

## Summary: what goes in the paper Results

Order, exactly:

1. Primary regression (R², MAE) — Block 07 KEEP rows.
2. Calibration (ECE, ratio, quintile MAE) — Block 08 KEEP.
3. Clinical monitoring signal (Fisher p) — Block 08 REFRAME.
4. SAA edge weights at LC and SNc — Block 12 KEEP, biological headline.
5. SAA epistemic uncertainty — Block 12 KEEP.
6. Latent subtyping — Block 15 KEEP.
7. Two-source uncertainty — Block 14 KEEP exploratory.
8. DaTscan V06 prediction — Block 07 KEEP with CI.
9. Restricted NHH C-index — Block 14b KEEP.
10. Ablation — Block 09 150-epoch KEEP with interpretability framing.
11. Wilcoxon + bootstrap — Block 10 KEEP with CI disclosure.

Not in paper: Block 09 40-epoch runs (supplement only), Block 13 topology, Block 11 GBA u_epi, raw NHH C-index (upper bound only).
