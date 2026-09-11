# Model Card — BrainFormer-PD v2.1

Following the Mitchell et al. (2019) model card template.

## Model details

- **Name**: BrainFormer-PD v2.1 (Stochastic Personalized Braak Graph Learning, S-PBGL).
- **Version**: 2.1 (this repository). Checkpoints in `RESULTS_LOCAL/checkpoints/best.pt` and `final.pt`.
- **Architecture**: multi-modal encoder (clinical + genetic + imaging + CSF) with cross-modal attention; Longitudinal Trajectory Polynomial (LTP) Legendre coefficient bank computed from 3 input visits; Stochastic Personalized Braak Graph module with analytical per-edge epistemic uncertainty; 2-layer Braak-structured multi-head temporal attention; Normal-Inverse-Gamma evidential regression head; discrete-time hazard head; supervised-contrastive projection head.
- **Parameters**: ~0.73 M (compact relative to typical medical imaging transformers).
- **Training**: 30-epoch GBA-BCSD supervised contrastive pretraining followed by 150-epoch multi-task training (UPDRS regression + NSD-ISS staging + DaTscan progression + time-to-milestone survival + graph regularisation + KL on graph posterior). Cosine annealing with warm restarts (T_0 = 30, T_mult = 2). AdamW, lr = 3 × 10⁻⁴, weight decay = 10⁻⁵, gradient clip = 1.0.
- **Seed**: 42, fixed across numpy / torch / CUDA / XGBoost.
- **Framework**: PyTorch 2.12 nightly (cu128).
- **Developed by**: Krishna Betwal, 2026.
- **License**: MIT (code); PPMI data subject to MJFF data use agreement.

## Intended use

- **Primary use**: research-grade progression modelling on the PPMI cohort, producing per-patient forecasts of 36-month UPDRS-III, NSD-ISS stage, DaTscan SBR at V06, and time-to-motor-milestone, together with calibrated epistemic uncertainty and an interpretable per-patient Braak graph topology.
- **Intended users**: PD researchers, biostatisticians, and clinical trialists evaluating candidate enrichment criteria or uncertainty-aware trial stratification.

## Out-of-scope

- Clinical deployment on individual patients.
- Decisions regarding DBS candidacy, medication choice, or diagnosis.
- Use on non-PPMI cohorts without external validation.
- Claims of predictive superiority over well-tuned tabular baselines (the evidence for that claim is marginal; see CAVEATS_AND_CLAIMS.md).

## Factors

- **Cohort**: PPMI — Parkinson's Progression Markers Initiative. 8,453 subjects across Prodromal (69%), PD (25%), HC (5%), SWEDD (1%).
- **Genetic subgroups**: GBA+ (4.2%, n = 357), SAA+ (27.1%, n = 2,293).
- **Imaging availability**: DaTscan for ~45% of subjects, MRI FreeSurfer for ~55%, CSF asyn for a smaller Prodromal-enriched subset. Handled with explicit missingness masks and IMPE node-uncertainty inflation.
- **Class imbalance**: NSD-ISS stage 0 = 73% of cohort. Class-weighted cross-entropy loss; Macro F1 reported alongside Weighted F1.

## Metrics (honest, real-PPMI test set, n = 1,268)

Primary regression:
| Slice | n | R² | MAE |
|---|---|---|---|
| All test | 404 | 0.755 | 4.39 |
| PD cohort | 168 | 0.409 | 6.35 |
| Prodromal | 198 | 0.549 | 3.06 |
| GBA+ | 36 | 0.842 | 3.86 |
| SAA+ | 238 | 0.701 | 5.14 |

Calibration:
- ECE (10-bin, on predicted std) = 0.006
- MAE ratio Q5/Q1 epistemic quintiles = 5.67×

Auxiliary:
- NSD-ISS Weighted F1 = 0.987, Macro F1 = 0.831 (framed as feature-coverage confirmation — see CAVEATS)
- DaTscan V06 Pearson r = 0.893, 95% CI [0.860, 0.919]
- NHH restricted C-index = [filled by Block 14b at run time]

Clinical monitoring:
- Top vs bottom epistemic uncertainty quartile, outcome = |ΔUPDRS-III @ V06| > 15
- Fisher exact p = 3.3 × 10⁻⁶; Haldane-corrected OR = 42.5, 95% CI [2.5, 718]

Biological interpretability (primary headline):
- Posterior edge weights at LC (Node 2) differ between SAA+/− at Mann-Whitney p = 8.9 × 10⁻¹⁸
- Posterior edge weights at SNc (Node 4) differ at p = 1.3 × 10⁻⁹
- Epistemic uncertainty higher in SAA+ at p = 1.8 × 10⁻²⁹

## Evaluation data

PPMI test split of 1,268 subjects (cohort-specified). No external validation was performed; see LIMITATIONS.md.

## Training data

PPMI train split of 5,915 subjects. Source: processed bundle at `BRAAK_PD/BRAAK_PD/data/processed/`. See DATA_SHEET.md for detailed provenance and feature construction.

## Quantitative analyses (bias, robustness)

- **GBA stratified**: full model performs better on GBA+ (R² 0.842) than GBA− (0.746), but epistemic uncertainty does **not** differ significantly between the two groups (MW p = 0.44) — any claim of differential uncertainty calibration by genetic status requires a larger GBA+ cohort.
- **SAA stratified**: full model performs slightly worse on SAA+ (R² 0.701) than SAA− (0.757), which is consistent with SAA+ patients being biologically heterogeneous and harder to predict.
- **Cohort stratified**: PD (R² 0.409) substantially lower than Prodromal (0.549) and all-test (0.755). PD is the hard slice.
- **Target leakage audit**: V06 UPDRS-III removed from input visit tensor; DaTscan target changed from SBR_mean (algebraically derivable from inputs) to SBR_V06 (forward-prediction target); LTP restricted to BL/V02/V04; contrastive severity proxy switched from V06 to BL UPDRS-III. All four leakage fixes applied before reporting.

## Ethical considerations

- PHI / PII: the released model contains no patient-identifying information. Checkpoints store model weights only.
- Fairness: the cohort is PPMI — predominantly white, enrolled in developed countries. Generalisation to under-represented populations is unknown.
- Safety: the model is NOT a diagnostic or treatment-selection tool and MUST NOT be used as such.

## Caveats and recommendations

- Use the restricted C-index from Block 14b, not the raw 0.985, when reporting survival performance.
- Do not claim GBA-stratified uncertainty differences; that finding did not replicate.
- Do not claim topology-phenotyping GBA enrichment; that finding did not replicate.
- Frame NSD-ISS F1 = 0.987 as feature-coverage confirmation, not a classification novelty.
- For predictive comparisons vs XGBoost, always report the bootstrap 95% CI on ΔR², which straddles zero.
