# BrainFormer-PD v2.1

Stochastic Personalized Braak Graph Learning for uncertainty-quantified
Parkinson's disease progression modelling on PPMI.

## Highlights

- **Calibrated uncertainty** — ECE 0.006, 5.67× MAE ratio between highest and lowest epistemic uncertainty quintile.
- **Biologically-interpretable posterior Braak graph** — per-patient edge weights at locus coeruleus and substantia nigra significantly discriminate α-synuclein-SAA-positive patients (Mann-Whitney p = 8.9 × 10⁻¹⁸ at LC, 1.3 × 10⁻⁹ at SNc).
- **Clinical monitoring signal** — baseline epistemic uncertainty predicts >15-point UPDRS-III deterioration at 36 months with Fisher exact p = 3.3 × 10⁻⁶.
- **End-to-end reproducible on RTX 5060** — 7 minutes to train, 25 minutes to run the full ablation + analysis on real PPMI data (n = 8,453).

## Model card and data sheet

- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- [`docs/DATA_SHEET.md`](docs/DATA_SHEET.md)

## Paper positioning and caveats

- [`PAPER_POSITIONING.md`](PAPER_POSITIONING.md) — abstract, narrative, and claim disposition for manuscript preparation.
- [`CAVEATS_AND_CLAIMS.md`](CAVEATS_AND_CLAIMS.md) — claim-by-claim KEEP / DROP / REFRAME audit.

## Quick start

### 1. Environment

Windows / Linux with Python 3.11 and CUDA 12.x. On an RTX 40/50-series (Blackwell, sm_120)
you need PyTorch nightly with CUDA 12.8.

```bash
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 2. Data

Unzip `BRAAK_PD.zip` into the project root so that
`BRAAK_PD/BRAAK_PD/data/processed/master_subjects.csv` exists. The bundle is the
PPMI-derived processed dataset with 8,453 subjects, the canonical
train/val/test splits (5,915 / 1,267 / 1,268), and the 10-region Braak graph prior.

### 3. Train and evaluate

```bash
python train_local.py --pretrain_epochs 30 --epochs 150 --batch 64
python analyze_local.py --ablation --ablation_epochs 150
```

Outputs land in [`RESULTS_LOCAL/`](RESULTS_LOCAL/):

- `metrics.json` — primary regression metrics
- `analysis/block_*.json` — calibration, stratified analyses, topology, two-source uncertainty, subtypes, ablation, stats
- `figures/*.pdf` and `figures/*.png` — all 16 paper figures
- `checkpoints/{pretrain,best,final}.pt` — model weights

See [`REPRODUCE.md`](REPRODUCE.md) for seed-controlled replication instructions.

## Method — what S-PBGL actually does

Each patient is assigned a **posterior distribution** over the 10×10 Braak adjacency rather
than a point estimate. Per-edge variance is derived **analytically** (not Monte-Carlo):

```
F_i[j] ~ N(mu_F[j], diag(sigma_F[j]^2))      # learned, per-patient, per-node
S_i[j,k] = F_i[j] · F_i[k]^T / sqrt(d_node)  # bilinear node-pair affinity
Var[S_i[j,k]] = (||sigma_F[j]||^2 * ||mu_F[k]||^2
              +  ||mu_F[j]||^2    * ||sigma_F[k]||^2
              +  ||sigma_F[j]||^2 * ||sigma_F[k]||^2) / d_node
u_edge[j,k]   = scale^2 * p_bar^2 * (1-p_bar)^2 * Var[S] * (A_braak[j,k]+eps)^2
```

Missing modalities inflate node-level sigma through a learned `kappa`:
`sigma_F[j] = sigma_net[j] * exp(kappa * node_miss[j])`.
A KL divergence to a standard normal prior with cosine annealing keeps the posterior stable.

See [`docs/METHOD.md`](docs/METHOD.md) for the full derivation and symbol table.

## Repository layout

```
.
├── train_local.py               # End-to-end training (data + model + eval)
├── analyze_local.py             # Post-hoc analyses + ablation + figures
├── BrainFormer_PD_v21.ipynb     # Colab-targeted notebook (legacy; use .py for real data)
├── BRAAK_PD/                    # PPMI-derived processed data (not in git; unzip locally)
├── RESULTS_LOCAL/               # Training outputs (reproducible)
│   ├── metrics.json
│   ├── analysis/
│   ├── figures/
│   └── checkpoints/
├── docs/
│   ├── MODEL_CARD.md
│   ├── DATA_SHEET.md
│   ├── METHOD.md
│   └── LIMITATIONS.md
├── PAPER_POSITIONING.md
├── CAVEATS_AND_CLAIMS.md
├── REPRODUCE.md
├── LICENSE
└── requirements.txt
```

## Limitations

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for the full list. Headline:

- **External validation** on PDBP, UK Biobank, or other cohorts is left as future work.
- Predictive gap over XGBoost + LTP is at the margin of statistical significance; the paper's contribution is **calibrated uncertainty + biological interpretability**, not predictive SOTA.
- Topology phenotyping (KMeans on learned `A_i`) did not produce significant GBA enrichment in the honest evaluation and is not claimed.
- 3.3% event rate for UPDRS-III ≥ 40 milestone produces wide odds-ratio CIs; Fisher exact p is the reported inferential statistic.

## Citation

Manuscript in preparation for *npj Parkinson's Disease*. Please cite the preprint (to be posted)
and acknowledge PPMI (Michael J. Fox Foundation).

## License

MIT — see [`LICENSE`](LICENSE).
