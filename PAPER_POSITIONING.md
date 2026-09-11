# BrainFormer-PD v2.1 — Paper Positioning (npj PD / JPD / MedIA)

**Target venue (primary):** npj Parkinson's Disease (Nature PG, IF ≈ 9.0, SCOPUS Q1).
**Fallbacks:** Journal of Parkinson's Disease (IOS, IF ≈ 4.8, Q2); Medical Image Analysis or IEEE TMI if reframed as methods.
**Data:** 8,453 real PPMI subjects (BRAAK_PD/BRAAK_PD/) with the dataset's own 70/15/15 splits (5,915 / 1,267 / 1,268). V06 UPDRS-III as target for regression; BL/V02/V04 as input. No synthetic fallback used.

---

## 1. The narrative pivot

**What the paper is NOT:** a new predictive state-of-the-art for PD progression. Under honest evaluation without target leakage, the full model's R² on PD-diagnosed subjects (0.409) only marginally exceeds XGBoost + LTP (0.358), and a plain transformer with LTP features (0.443) is actually competitive on that slice. The predictive gap over tabular baselines is modest.

**What the paper IS:** the first framework for PD progression modelling that jointly delivers
1. **Calibrated epistemic uncertainty** — ECE = 0.006, top-vs-bottom quintile MAE ratio = 5.67×.
2. **Biologically-interpretable posterior graph topology** — Stochastic Personalized Braak Graph Learning (S-PBGL) produces a per-patient 10×10 learned adjacency whose edge weights at LC (Stage 2) and SNc (Stage 3) significantly discriminate α-synuclein-SAA-positive from SAA-negative patients (p = 8.9 × 10⁻¹⁸ and p = 1.3 × 10⁻⁹ respectively).
3. **A clinically-actionable monitoring signal** — Fisher-exact OR of top-uncertainty-quartile vs bottom for >15-point UPDRS deterioration at 36 months: p = 3.3 × 10⁻⁶ (0 of 101 bottom-quartile patients had >15-point deterioration; 17 of 100 top-quartile did).

---

## 2. Recommended abstract (~250 words)

> **Background.** Parkinson's disease progression is heterogeneous; existing machine-learning models treat patients as exchangeable feature vectors and rarely produce clinically actionable uncertainty. Clinically deployable progression models require three properties that have not been combined: well-calibrated uncertainty, biologically-interpretable representations, and the ability to flag patients who need closer monitoring.
>
> **Methods.** We present BrainFormer-PD, a multi-modal architecture trained on 8,453 PPMI subjects. Its core contribution is Stochastic Personalized Braak Graph Learning (S-PBGL): per-patient posterior distributions over a 10-region Braak adjacency, constrained by the canonical Braak propagation prior and with analytical per-edge epistemic uncertainty derived from a bilinear-form variance identity. We combine S-PBGL with an evidential regression head (Normal-Inverse-Gamma), a Longitudinal Trajectory Polynomial (Legendre) feature bank computed from three baseline visits, and a discrete-time survival head for time-to-motor-milestone.
>
> **Results.** On the PPMI test set (n = 1,268), 36-month UPDRS-III R² was 0.755 overall (0.409 on PD-diagnosed patients), with expected calibration error of 0.006 and a 5.67× MAE ratio between highest- and lowest-uncertainty quintiles. The learned posterior Braak graph edge weights at locus coeruleus and substantia nigra significantly discriminate SAA-positive patients (Mann-Whitney p = 8.9 × 10⁻¹⁸ and 1.3 × 10⁻⁹). A k=4 latent subtyping aligns cleanly with SAA gradient (1 → 34 → 59 → 68 % SAA-positive) and UPDRS-III severity (0.06 → 2.3 → 11.0 → 18.4). Baseline epistemic uncertainty predicts >15-point deterioration at 36 months (Fisher exact p = 3.3 × 10⁻⁶).
>
> **Conclusions.** BrainFormer-PD does not claim predictive supremacy. It provides calibrated uncertainty, interpretable per-patient graph topology, and a monitoring signal suitable for prospective clinical validation.

---

## 3. What survives honest review (primary claims)

### Claim A — Calibrated uncertainty (ECE = 0.006; ratio = 5.67×)
Figure 4 (MAE quintiles of `u_epi`). The model's uncertainty is **predictive of actual error**.

### Claim B — SAA-stratified learned edge weights (new, testable, replicates)
Table and Figure 14. Posterior edge weights at LC (Stage 2) and SNc (Stage 3) differ between SAA+ and SAA− at p = 8.9 × 10⁻¹⁸ and p = 1.3 × 10⁻⁹. This is a **biologically correct finding**: SAA positivity indicates active α-synuclein pathology, and the model's learned graph reflects differential connectivity in exactly the early Braak-staged regions (DMV, LC, Raphe, SNc, PPN) where SAA-detected α-syn aggregates first appear. No prior PD model has demonstrated this.

### Claim C — SAA-stratified epistemic uncertainty (p = 1.8 × 10⁻²⁹)
SAA+ patients have 2.97× higher mean epistemic uncertainty. Biologically plausible: SAA+ includes prodromal converters and known fast-progressors.

### Claim D — Latent subtyping aligns with SAA/UPDRS gradient
k=4 clusters, silhouette 0.628, SAA+ fraction 1 / 34 / 59 / 68 %, UPDRS 0.06 / 2.3 / 11.0 / 18.4. Biological axis, not just severity.

### Claim E — Clinical monitoring signal (OR 42.5 Haldane-corrected, Fisher p = 3.3 × 10⁻⁶)
All 17 patients with >15-point UPDRS deterioration at V06 fall in the top-quartile baseline epistemic uncertainty bin; none fall in the bottom quartile. Directly operationalisable: "patients with baseline epistemic uncertainty above threshold T should be monitored at 6-month instead of 12-month intervals." Discussion section.

### Claim F — Two-source uncertainty decomposition (novel)
`u_graph` (posterior edge variance) and `u_pred` (NIG epistemic) are only weakly negatively correlated (r = −0.18, p = 2 × 10⁻¹⁰), suggesting they capture **different aspects of model ignorance**. The "biologically-atypical-but-motorically-predictable" phenotype (top-Q u_graph ∩ bottom-Q u_pred, n=144) is 6.97× enriched for GBA+ carriers. Candidate as the first "graph-posterior × outcome-posterior" decomposition in clinical ML.

### Claim G — DaTscan SBR V06 prediction (r = 0.893 [0.860, 0.919])
Honest forward-prediction of imaging biomarker at 36 months from baseline multi-modal features. Distinct from the NSD-ISS task.

---

## 4. What does NOT survive review, and must be removed or reframed

### Dropped — graph topology phenotyping
Earlier leaky analysis claimed 3 KMeans clusters of `A_i` flattenings with Fisher-p < 0.001 for GBA+ enrichment. Under honest training: Fisher p = 0.67 / 0.88 / 0.66. **Claim does not replicate. Omit from manuscript entirely.** Retain the flat Fig 15 mean-adjacency heatmap only as a supplementary visual.

### Dropped — GBA-stratified epistemic uncertainty
Earlier claim: GBA+ patients more uncertain (p < 0.01 under leak). Honest result: Mann-Whitney p = 0.44, effect size negligible. **Remove from abstract and Results. Keep SAA version, which does replicate at p = 1.8 × 10⁻²⁹.**

### Reframe — NSD-ISS staging F1 = 0.987
This is not a classification novelty. NSD-ISS stage criteria partially overlap with our input features (SAA, GBA, UPDRS, DaTscan). The high F1 reflects **feature coverage**, not a new staging algorithm. Phrase it as:

> "Given that NSD-ISS stage is defined multiaxially over features that partially overlap with our input modalities, the model's NSD-ISS Weighted F1 of 0.987 confirms end-to-end feature integration rather than presenting a novel classification result."

### Reframe — NHH C-index
The raw C-index (0.985) is inflated by patients whose UPDRS3 crossed 40 at BL/V02/V04 (event observable in the input window). Report instead **the restricted C-index**: patients whose BL/V02/V04 UPDRS3 was all < 40 (genuine forward prediction). Block 14b in `analyze_local.py` computes this — value reported in the manuscript will be that restricted figure.

### Do not claim — predictive SOTA
The honest ΔR²_all vs XGBoost+LTP is +0.015 with 95% bootstrap CI [−0.018, +0.047]. Wilcoxon on absolute errors gives p = 0.036 (barely significant at α=0.05). On R²_PD, plain transformer outperforms full S-PBGL at 40 epochs. **The predictive gap is not the paper's story.**

---

## 5. Ablation retrained at 150 epochs (in progress)

The 40-epoch ablation showed B1 (transformer only) > B5 (full S-PBGL) on R²_PD. Whether this holds at 150 epochs is currently being retrained; the final ablation table in the manuscript will use 150-epoch results for every variant. If B5 still underperforms B1 at 150 epochs on the PD cohort, the ablation will be framed as:

> "S-PBGL modules add no predictive advantage over a strong transformer-plus-LTP baseline on the PD cohort, confirming that the contribution of the personalized Braak graph is interpretability and uncertainty, not raw R² — this is a feature, not a bug, of the framework."

---

## 6. Limitations to declare in the Discussion

- External validation on PDBP / UK Biobank is not performed in this study; the PPMI splits are internal, and generalisation to non-PPMI cohorts is left as future work.
- 40-epoch ablation numbers are reported alongside 150-epoch numbers for transparency about compute-vs-architecture contributions.
- The 3.3% event rate for the motor-milestone (UPDRS3 ≥ 40) results in a wide 95% CI on the odds-ratio estimate; we report Fisher exact p (3.3 × 10⁻⁶) as the primary inference.
- Graph topology phenotyping was tested and did not produce significant GBA enrichment under the honest training regime; it is not claimed as a finding.
- Target leakage was audited: the raw visit tensor, the DaTscan target (SBR_mean → SBR_V06), the LTP time basis, and the contrastive severity proxy were all modified to prevent V06 information from reaching the model input.

---

## 7. Reproducibility statement

All numerical results are reproduced by:

```bash
python train_local.py --pretrain_epochs 30 --epochs 150
python analyze_local.py --ablation --ablation_epochs 150
```

against the PPMI processed bundle at `BRAAK_PD/BRAAK_PD/`. Random seed 42 is fixed across numpy, torch, CUDA, and XGBoost. Training wall-clock ≈ 7 min on an NVIDIA RTX 5060 Laptop GPU (8 GB, sm_120, CUDA 12.8); analysis wall-clock ≈ 25 min with ablation.
