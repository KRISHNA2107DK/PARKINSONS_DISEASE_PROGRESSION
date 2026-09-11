# Limitations

Honest list of what this study does NOT do, and why.

## 1. No external validation

- All results are on the canonical PPMI splits. Generalisation to other PD cohorts (PDBP, UK Biobank, LuxPARK, BioFIND, etc.) is not tested.
- This is the single biggest limitation for Q1 venues. Plan this as a follow-up study or an explicit "future work" paragraph.

## 2. Marginal predictive advantage over tabular baselines

- Wilcoxon signed-rank on absolute UPDRS-III errors, BrainFormer-PD vs XGBoost+LTP: **p = 0.036** on the full test set.
- Bootstrap 95% CI on ΔR²_all: **[−0.018, +0.047]** — straddles zero.
- On the PD cohort, full model R² = 0.409 vs plain-transformer R² = 0.443 at 40 epochs; 150-epoch re-run of ablation is in progress.
- **Do not frame the paper as a predictive-SOTA paper.** The contribution is calibrated uncertainty and interpretability.

## 3. Failed claims that should NOT be reported

- **Topology phenotyping**: KMeans on learned `A_i` produced Fisher exact p values of 0.67 / 0.88 / 0.66 for GBA enrichment — no significant clusters. Dropped.
- **GBA-stratified epistemic uncertainty**: Mann-Whitney p = 0.44 under honest training. Dropped.

## 4. Framing-required findings

- **NSD-ISS Weighted F1 = 0.987** is inflated by feature-coverage: NSD-ISS stage is defined over features overlapping the model's inputs. Report with the feature-coverage framing (see CAVEATS_AND_CLAIMS.md).
- **NHH C-index = 0.985 (raw)** is inflated by events observable in the input window. Use restricted C-index from Block 14b as the primary survival metric.

## 5. Cohort bias

- PPMI is predominantly white, enrolled in developed countries (US, EU). Demographic skew limits generalisation to underrepresented populations.
- 73% Stage-0 NSD-ISS dominates macroscale metrics; weighted F1 masks per-stage performance. Always report Macro F1 alongside Weighted F1.

## 6. Small genetic subset for GBA claims

- Only 357 GBA+ subjects total; 36 in test. R² on that subset (0.842) is strong but confidence intervals are wide.
- Any claim about GBA-specific model behaviour should be treated as exploratory, not confirmatory.

## 7. Event rate limits survival inference

- UPDRS-III ≥ 40 milestone event rate: 3.3% over 36 months.
- Low event count widens all survival-related confidence intervals (C-index, clinical monitoring OR).
- Report Fisher exact p values rather than OR magnitudes for inference.

## 8. Biological prior is canonical Braak, not patient-specific prior

- `A_braak` from `data/braak_graph/A_symmetric.npy` is the consensus 10-region graph.
- Braak staging is a population average; individual patients may follow caudo-rostral or limbic-first or cortical-first progression patterns. S-PBGL personalises away from this prior but the prior shape constrains the posterior.

## 9. Uncertainty calibration is on test-set, not prospectively validated

- ECE = 0.006 and uncertainty ratio 5.67× are measured on the PPMI test set.
- Prospective monitoring-interval studies would be needed to validate the clinical monitoring OR claim. This study is observational.

## 10. No causal claims

- Edge weights at LC and SNc differ between SAA+ and SAA− at p < 10⁻¹⁸.
- This is a **representation correlation**, not a causal statement about synuclein propagation through the brain. The model learns what correlates with the label; causal direction remains to be established.

## 11. Reproducibility caveats

- cuDNN kernel selection introduces small (<0.01 R²) non-determinism even with fixed seed. To rule this out, set `torch.use_deterministic_algorithms(True)` at the cost of ~2× slowdown.
- Blackwell (sm_120) requires PyTorch nightly cu128. Older GPUs may behave slightly differently in floating-point precision.
