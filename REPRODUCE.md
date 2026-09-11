# Reproducibility instructions

Every number in the paper is reproduced by the two commands below, running on
fixed random seeds against the PPMI-derived processed bundle.

## Hardware tested

- **GPU**: NVIDIA GeForce RTX 5060 Laptop, 8 GB, Blackwell (sm_120), driver 592.01, CUDA 13.1
  (the GPU supports up to CUDA 13; PyTorch nightly ships cu128 binaries).
- **CPU**: x86_64, 12 cores — CPU fallback works but is ~10× slower.
- **OS**: Windows 11 Home Single Language 10.0.26200.

## Environment

```bash
# 1. Python 3.11
python --version   # 3.11.x

# 2. Install torch nightly for Blackwell sm_120
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt

# Sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected output on the test system:
```
2.12.0.dev20260408+cu128 True NVIDIA GeForce RTX 5060 Laptop GPU
```

## Data

Unzip `BRAAK_PD.zip` to the repository root so that:

```
./BRAAK_PD/BRAAK_PD/data/processed/master_subjects.csv
./BRAAK_PD/BRAAK_PD/data/splits/{train,val,test}_patnos.npy
./BRAAK_PD/BRAAK_PD/data/braak_graph/A_symmetric.npy
./BRAAK_PD/BRAAK_PD/data/raw/ppmi/...
```

exists. Splits are the canonical 70/15/15 PPMI split (5,915 / 1,267 / 1,268).

## Determinism

Random seed is fixed in `train_local.py` at `SEED = 42` across:
- Python `random`
- `numpy.random`
- `torch.manual_seed`
- `torch.cuda.manual_seed_all`
- XGBoost `random_state`

`torch.backends.cudnn.benchmark = True` is set for speed (marginal non-determinism
at the kernel-selection level). For strictly deterministic runs, uncomment
`torch.use_deterministic_algorithms(True)` at the top of `train_local.py`.

## Primary training + evaluation

```bash
python train_local.py --pretrain_epochs 30 --epochs 150 --batch 64
```

Wall-clock on the reference hardware: ~7 minutes.

Outputs: `RESULTS_LOCAL/metrics.json`, `RESULTS_LOCAL/checkpoints/{pretrain,best,final}.pt`,
`RESULTS_LOCAL/test_outputs.npz`.

## Full analysis including B0–B5 ablation

```bash
python analyze_local.py --ablation --ablation_epochs 150
```

Wall-clock on the reference hardware: ~25 minutes.

Outputs: `RESULTS_LOCAL/analysis/block_*.json`, all 16 figures as `.pdf` + `.png`,
consolidated `RESULTS_LOCAL/main_results.json`.

## Expected numbers

After both commands, the key numbers in `RESULTS_LOCAL/metrics.json` should match
(within 0.01 R² from stochastic cuDNN kernel selection):

| Metric | Expected |
|---|---|
| UPDRS_all R² | 0.745–0.770 |
| UPDRS_PD R² | 0.38–0.45 |
| UPDRS_GBA+ R² | 0.82–0.87 |
| DaTscan Pearson r (V06) | 0.88–0.91 |
| NSD-ISS Weighted F1 | 0.98–0.99 |
| ECE | 0.003–0.010 |
| Uncertainty ratio | 4.5–6.5× |

And in `RESULTS_LOCAL/analysis/block_12_saa.json`:

| Metric | Expected |
|---|---|
| SAA edge MW p at LC | < 10⁻¹⁵ |
| SAA edge MW p at SNc | < 10⁻⁸ |
| SAA u_epi MW p | < 10⁻²⁵ |

## Resuming from checkpoints

All `analyze_local.py` blocks regenerate their outputs on every run. If you want
to skip retraining and just re-run the analysis on an existing checkpoint, run
`analyze_local.py` **without** `--ablation` (~3 min, uses the existing
`RESULTS_LOCAL/test_outputs.npz`).

## Known issues

- **tqdm spam on Windows**: `\r`-based progress bars render as separate lines in the
  Windows console. Use `python -u` and redirect to a log file for clean output.
- **Unicode logging on cp1252 console**: if the console codepage is legacy, subscript
  characters in log output may appear as `?`. The underlying numbers and JSON are
  unaffected.
