# Data Sheet — PPMI processed bundle (`BRAAK_PD/BRAAK_PD/`)

Following the Gebru et al. (2018) datasheet for datasets template.

## Motivation

- **Why was this dataset created?** To train BrainFormer-PD v2.1 on 36-month UPDRS-III prediction and other PD progression endpoints. The bundle is a pre-processed, feature-engineered subset of PPMI tailored for reproducible multi-modal modelling.
- **Creator**: Krishna Betwal, 2026, derived from PPMI under MJFF data access.
- **Funding**: PPMI data is funded by the Michael J. Fox Foundation for Parkinson's Research and funding partners.

## Composition

- **Instances**: 8,453 unique PPMI subjects (one row per subject in `master_subjects.csv`).
- **Split**: canonical PPMI 70/15/15 stratified split — 5,915 train / 1,267 val / 1,268 test. Stored as `data/splits/train_patnos.npy` etc.
- **Cohorts**: Prodromal (5,828, 69%), PD (2,104, 25%), HC (440, 5%), SWEDD (81, 1%).
- **Genetic subgroups**: GBA+ (357, 4.2%), SAA+ (2,293, 27.1%).
- **Data types**:
  - Clinical: AGE, SEX, UPDRS Part I/II/III totals, MoCA, all at BL.
  - Longitudinal clinical: UPDRS-III and MoCA at BL, V02 (12 mo), V04 (24 mo), V06 (36 mo).
  - Imaging (DaTscan): Xing Core Lab quantitative SBR — CAUDATE_L/R, PUTAMEN_L/R, with a BL slice retained.
  - Imaging (MRI FreeSurfer): Cortical thickness (bilateral entorhinal, parahippocampal, mean thickness), ASEG volumes (bilateral hippocampus, amygdala, putamen, brain stem, total intracranial volume). BL slice.
  - Longitudinal imaging: SBR_mean per visit (BL/V02/V04/V06).
  - Genetic: GBA binary/severity, SAA positive.
  - Labels: NSD-ISS stage (0–6, computed upstream), UPDRS3_V06 (the primary regression target).
- **Missingness**: Not all subjects have all visits. Visit mask and per-modality missingness fractions are computed in `build_feature_matrices` and fed to the IMPE node-uncertainty inflation term.

## Collection process

- Source CSVs at `data/raw/ppmi/` come from PPMI's scheduled data releases.
- Processing performed in upstream pipeline (not included in this repo):
  - Cohort mapping to PD / Prodromal / HC / SWEDD labels.
  - UPDRS-III aggregation per visit.
  - DaTscan BL slice selection (EVENT_ID ∈ {BL, SC}).
  - FreeSurfer feature aggregation with BL-slice filter.
  - NSD-ISS stage computation from SAA status, DAT scan, clinical markers per published NSD-ISS criteria.
- Target labels are not fabricated — all UPDRS-III scores and DaTscan SBR values come from PPMI raw visits.

## Pre-processing done in THIS repository

- Feature matrices are constructed in `train_local.build_feature_matrices`:
  - Each modality block is concatenated with its own missingness indicators.
  - All features are z-scored using **training-split** statistics only; validation and test are transformed with the same scalers.
  - Visit tensor built from BL/V02/V04 — **V06 is excluded from the input** to prevent target leakage.
  - LTP features (Legendre coefficients over BL/V02/V04) are computed per patient per longitudinal feature.
  - Node-missingness fractions are linearly mapped to 10 Braak nodes via a fixed NODE_WEIGHTS matrix.
  - Hazard labels use the full 4-visit horizon for event detection but never leak V06 back into input features.

## Uses

- **Primary use in this repo**: training and evaluating the S-PBGL progression model.
- **Other known uses**: none at time of writing.
- **Uses for which the dataset should NOT be used**:
  - Direct clinical deployment.
  - Analyses assuming the feature subset here is exhaustive (many PPMI signals — neuropsychiatric scales, CSF panel full markers, gait sensors, etc. — are not represented).
  - External generalisation claims — this is PPMI only, and PPMI is demographically skewed (predominantly white, North American / European).

## Distribution

- This repository does NOT redistribute PPMI data. The `BRAAK_PD.zip` archive should be obtained from the authors under PPMI terms; it contains pre-processed features only, not raw PHI.
- Users must have their own PPMI access agreement with MJFF.

## Maintenance

- **Maintainer**: Krishna Betwal.
- **Update schedule**: PPMI publishes new data releases periodically; this bundle reflects a snapshot. Re-processing against future PPMI releases may change feature counts and stratify fractions.
- **Errata**: any target-leakage issues discovered are logged in `CAVEATS_AND_CLAIMS.md`. V06 input leak and SBR_mean → SBR_V06 fixes were applied during development; the current bundle contains no known leaks but users should re-audit if adding new columns.
