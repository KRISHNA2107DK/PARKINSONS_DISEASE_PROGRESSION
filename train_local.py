"""BrainFormer-PD v2.1 local training on real PPMI data (BRAAK_PD folder).

Architecture: S-PBGL (stochastic personalised Braak graph) + LTP + IMPE +
BSMTA + EDP/UG-HEM + NHH + supervised-contrastive GBA pre-training.

Usage:
    python train_local.py
    python train_local.py --phase full      # skip pretraining
    python train_local.py --epochs 150      # longer run

Outputs land in ./RESULTS_LOCAL/.
"""
from __future__ import annotations
import argparse, json, math, os, pickle, random, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, f1_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import pearsonr, mannwhitneyu

warnings.filterwarnings("ignore")

# ============================ Paths & config ============================
HERE  = Path(__file__).resolve().parent
ROOT  = HERE / "BRAAK_PD" / "BRAAK_PD"
RESULTS     = HERE / "RESULTS_LOCAL"
CHECKPOINTS = RESULTS / "checkpoints"
FIGURES     = RESULTS / "figures"
DATA_PROC   = RESULTS / "data_proc"
for p in (RESULTS, CHECKPOINTS, FIGURES, DATA_PROC):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
def set_seed(s: int = SEED) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = dict(
    d_model=128, d_node=8, n_nodes=10, n_heads=8, n_tf_layers=2,
    VISIT_T=3,        # Input visits: BL, V02, V04 (V06 is the target — MUST be excluded)
    HAZARD_BINS=4,    # Hazard bins over full 4-visit horizon: BL, V02, V04, V06
    UPDRS_MAX=132.0,
    batch_size=64, lr=3e-4, weight_decay=1e-5, grad_clip=1.0,
    pretrain_epochs=20, train_epochs=80,
    w_updrs=1.0, w_stage=0.5, w_datscan=0.3, w_graph=0.01, w_hazard=0.1,
    kl_anneal_epochs=30, kl_max=1e-4,
    ughem_warmup=15, ughem_gamma=0.5,
    graph_scale_init=0.1, kappa_init=0.5, logvar_clip=(-4.0, 2.0), braak_eps=0.05,
)

VISIT_PREFIX_ALL   = ["BL", "V02", "V04", "V06"]   # full horizon (hazard labels)
VISIT_PREFIX_INPUT = ["BL", "V02", "V04"]           # model input only — NO V06
NODE_NAMES = ["DMV","OlfBulb","LC","Raphe","SNc","PPN","Amygdala","Hippocampus","TempMeso","PFC"]

# =============================== Data =================================
def find_col(df, *candidates, default=None):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c is None: continue
        if c in df.columns: return c
        lc = c.lower()
        if lc in lower: return lower[lc]
        for k, orig in lower.items():
            if lc in k: return orig
    return default

def safe_numeric(df, col):
    if col is None or col not in df.columns:
        return np.zeros(len(df), dtype=np.float32), np.ones(len(df), dtype=np.float32)
    v = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
    miss = np.isnan(v).astype(np.float32)
    v = np.nan_to_num(v, nan=0.0)
    return v, miss

def build_block(df, cols):
    xs, ms = [], []
    for c in cols:
        v, m = safe_numeric(df, c); xs.append(v); ms.append(m)
    X = np.stack(xs, axis=1)
    M = np.stack(ms, axis=1)
    return np.concatenate([X, M], axis=1).astype(np.float32)

def fit_transform(name, X, tr_mask):
    scaler = StandardScaler().fit(X[tr_mask])
    Xs = np.nan_to_num(scaler.transform(X).astype(np.float32), nan=0.0)
    with open(DATA_PROC / f"scaler_{name}.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return Xs

def legendre_fit(y, t):
    obs = ~np.isnan(y)
    if obs.sum() < 2:
        return np.zeros(3, dtype=np.float32)
    P0 = np.ones_like(t[obs]); P1 = t[obs]; P2 = 0.5 * (3*t[obs]**2 - 1)
    Phi = np.stack([P0, P1, P2], axis=1)
    if obs.sum() == 2:
        c, *_ = np.linalg.lstsq(Phi[:, :2], y[obs], rcond=None)
        return np.concatenate([c, [0.0]]).astype(np.float32)
    c, *_ = np.linalg.lstsq(Phi, y[obs], rcond=None)
    return c.astype(np.float32)

def long_to_wide(df, value_col, visits=("BL","V02","V04","V06")):
    df = df[df["EVENT_ID"].isin(visits)][["PATNO", "EVENT_ID", value_col]]
    w = df.pivot_table(index="PATNO", columns="EVENT_ID", values=value_col, aggfunc="mean")
    for v in visits:
        if v not in w.columns: w[v] = np.nan
    return w[list(visits)]

def load_and_build():
    print("[data] reading CSVs ...")
    master = pd.read_csv(ROOT / "data/processed/master_subjects.csv")
    updrs  = pd.read_csv(ROOT / "data/processed/updrs_longitudinal.csv")
    moca   = pd.read_csv(ROOT / "data/processed/moca_longitudinal.csv")
    sbr_l  = pd.read_csv(ROOT / "data/processed/sbr_longitudinal.csv")
    # UPDRS Part I and II (scalar baseline)
    updrs1 = pd.read_csv(ROOT / "data/raw/ppmi/clinical/MDS_UPDRS_Part_I.csv")
    updrs2 = pd.read_csv(ROOT / "data/raw/ppmi/clinical/MDS_UPDRS_Part_II.csv")
    cth = pd.read_csv(ROOT / "data/raw/ppmi/imaging/mri_freesurfer/Cortical_Thickness_CTh.csv")
    aseg= pd.read_csv(ROOT / "data/raw/ppmi/imaging/mri_freesurfer/Volume_ASEG.csv")
    sbr_raw = pd.read_csv(ROOT / "data/raw/ppmi/imaging/datscan/Xing_Core_Lab_Quant_SBR.csv")

    # -------- longitudinal wide tables keyed by PATNO --------
    up3_w = long_to_wide(updrs, "UPDRS3")
    moca_w= long_to_wide(moca,  "MOCA")
    sbr_w = long_to_wide(sbr_l, "SBR_mean")

    # Baseline UPDRS1/2 — use NP1 / NP2 total columns if present, else zeros
    def _baseline_total(df, *totcol_candidates):
        c = find_col(df, *totcol_candidates)
        if c is None: return {}
        tmp = df[df["EVENT_ID"] == "BL"][["PATNO", c]] if "EVENT_ID" in df.columns else df[["PATNO", c]]
        tmp = tmp.groupby("PATNO")[c].mean()
        return tmp.to_dict()
    u1_bl_map = _baseline_total(updrs1, "NP1RTOT", "NP1TOT", "NP1PTOT")
    u2_bl_map = _baseline_total(updrs2, "NP2PTOT", "NP2TOT")

    # -------- FreeSurfer BL slices for imaging features --------
    def _bl_slice(df, cols):
        if "EVENT_ID" in df.columns:
            df = df[df["EVENT_ID"].isin(["BL","SC"])]
        df = df.groupby("PATNO")[cols].mean().reset_index()
        return df
    cth_cols = [c for c in ["lh_entorhinal","rh_entorhinal",
                            "lh_parahippocampal","rh_parahippocampal",
                            "lh_MeanThickness","rh_MeanThickness"] if c in cth.columns]
    cth_bl = _bl_slice(cth, cth_cols)

    aseg_cols = [c for c in ["Left_Hippocampus","Right_Hippocampus",
                             "Left_Amygdala","Right_Amygdala",
                             "Left_Putamen","Right_Putamen",
                             "Brain_Stem","EstimatedTotalIntraCranialVol"] if c in aseg.columns]
    aseg_bl = _bl_slice(aseg, aseg_cols)

    sbr_sub_cols = [c for c in ["CAUDATE_L_REF_CWM","CAUDATE_R_REF_CWM",
                                "PUTAMEN_L_REF_CWM","PUTAMEN_R_REF_CWM"] if c in sbr_raw.columns]
    sbr_sub_bl = _bl_slice(sbr_raw, sbr_sub_cols)

    # -------- merge everything onto master --------
    m = master.copy()
    m["UPDRS1_BL"] = m["PATNO"].map(u1_bl_map).astype(float)
    m["UPDRS2_BL"] = m["PATNO"].map(u2_bl_map).astype(float)
    m["MOCA_BL"]   = m["PATNO"].map(moca_w["BL"].to_dict()).astype(float)
    for v in VISIT_PREFIX_ALL:
        m[f"UPDRS3_{v}"] = m["PATNO"].map(up3_w[v].to_dict())
        m[f"MOCA_{v}"]   = m["PATNO"].map(moca_w[v].to_dict())
        m[f"SBR_mean_{v}"] = m["PATNO"].map(sbr_w[v].to_dict()) if v in sbr_w.columns else np.nan
    # Normalise master UPDRS3_VNN names to our convention if missing
    for v in VISIT_PREFIX_ALL:
        if f"UPDRS3_{v}" not in m.columns or m[f"UPDRS3_{v}"].isna().all():
            alt = f"UPDRS3_{v}"
            if alt in master.columns:
                m[f"UPDRS3_{v}"] = master[alt]

    m = m.merge(cth_bl,     on="PATNO", how="left", suffixes=("","_CTh"))
    m = m.merge(aseg_bl,    on="PATNO", how="left", suffixes=("","_ASEG"))
    m = m.merge(sbr_sub_bl, on="PATNO", how="left", suffixes=("","_SBRsub"))

    # -------- splits --------
    tr = np.load(ROOT / "data/splits/train_patnos.npy")
    va = np.load(ROOT / "data/splits/val_patnos.npy")
    te = np.load(ROOT / "data/splits/test_patnos.npy")

    # -------- Braak graph --------
    A_sym = np.load(ROOT / "data/braak_graph/A_symmetric.npy").astype(np.float32)
    np.fill_diagonal(A_sym, 1.0)

    return m, (tr, va, te), A_sym, (cth_cols, aseg_cols, sbr_sub_cols)

def build_feature_matrices(m, imaging_extra_cols):
    cth_cols, aseg_cols, sbr_sub_cols = imaging_extra_cols

    CLINICAL = ["AGE","SEX_binary","UPDRS1_BL","UPDRS2_BL","UPDRS3_BL","MOCA_BL"]
    GENETIC  = ["GBA_binary","SAA_positive"]
    IMAGING  = sbr_sub_cols + [
        "SBR_BL","CAUDATE_mean","PUTAMEN_mean",
    ] + cth_cols + aseg_cols
    BIO      = []   # CSF asyn not in this dump; rely on SAA + imaging

    N = len(m)
    X_clin = build_block(m, CLINICAL)
    X_gen  = build_block(m, GENETIC)
    X_img  = build_block(m, IMAGING)
    X_bio  = np.zeros((N, 2), dtype=np.float32) if not BIO else build_block(m, BIO)

    # Visit tensor [N, VISIT_T=3, 5] — (UPDRS3, UPDRS1, UPDRS2, MoCA, SBR)
    # Only BL/V02/V04 enter the model input. V06 is the target and MUST NOT leak in.
    # UPDRS1/UPDRS2 are only at baseline; fill forward as constant (model has missing mask anyway).
    T_in = CFG["VISIT_T"]
    X_visit = np.zeros((N, T_in, 5), dtype=np.float32)
    V_mask  = np.zeros((N, T_in), dtype=np.float32)
    U1 = m["UPDRS1_BL"].astype(float).values
    U2 = m["UPDRS2_BL"].astype(float).values
    for ti, v in enumerate(VISIT_PREFIX_INPUT):
        up3  = pd.to_numeric(m.get(f"UPDRS3_{v}", np.nan), errors="coerce").values
        mocv = pd.to_numeric(m.get(f"MOCA_{v}",   np.nan), errors="coerce").values
        sbrv = pd.to_numeric(m.get(f"SBR_mean_{v}", np.nan), errors="coerce").values
        obs = ~np.isnan(up3) | ~np.isnan(mocv) | ~np.isnan(sbrv)
        X_visit[:, ti, 0] = np.nan_to_num(up3,  nan=0.0)
        X_visit[:, ti, 1] = np.nan_to_num(U1,   nan=0.0)
        X_visit[:, ti, 2] = np.nan_to_num(U2,   nan=0.0)
        X_visit[:, ti, 3] = np.nan_to_num(mocv, nan=0.0)
        X_visit[:, ti, 4] = np.nan_to_num(sbrv, nan=0.0)
        V_mask[obs, ti] = 1.0

    # LTP (Fix 1: use BL/V02/V04 only — never V06)
    X_ltp = np.zeros((N, 15), dtype=np.float32)
    t_grid = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    long_feats = [
        [f"UPDRS3_{v}" for v in ["BL","V02","V04"]],
        None,                                             # UPDRS1 — only baseline
        None,                                             # UPDRS2 — only baseline
        [f"MOCA_{v}"   for v in ["BL","V02","V04"]],
        [f"SBR_mean_{v}" for v in ["BL","V02","V04"]],
    ]
    for fi, cols in enumerate(long_feats):
        if cols is None:
            continue
        if not all(c in m.columns for c in cols):
            continue
        Y = m[cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        for i in range(N):
            X_ltp[i, 3*fi:3*fi+3] = legendre_fit(Y[i], t_grid)

    # IMPE node missingness
    def modality_miss_frac(feat_list):
        if not feat_list: return np.zeros(N, dtype=np.float32)
        miss = np.zeros(N, dtype=np.float32); cnt = 0
        for f in feat_list:
            if f not in m.columns: continue
            miss = miss + pd.to_numeric(m[f], errors="coerce").isna().astype(float).values
            cnt += 1
        return (miss / max(cnt, 1)).astype(np.float32)

    miss_clin = modality_miss_frac(CLINICAL)
    miss_gen  = modality_miss_frac(GENETIC)
    miss_img  = modality_miss_frac(IMAGING)
    miss_bio  = np.zeros(N, dtype=np.float32)
    miss_mod  = np.stack([miss_clin, miss_gen, miss_img, miss_bio], axis=1)

    NODE_WEIGHTS = np.array([
        [0.1,0.0,0.3,0.6],  # DMV
        [0.1,0.0,0.3,0.6],  # OlfBulb
        [0.1,0.0,0.4,0.5],  # LC
        [0.1,0.0,0.4,0.5],  # Raphe
        [0.0,0.0,1.0,0.0],  # SNc
        [0.2,0.0,0.5,0.3],  # PPN
        [0.2,0.0,0.6,0.2],  # Amygdala
        [0.2,0.0,0.6,0.2],  # Hippocampus
        [0.3,0.0,0.7,0.0],  # TempMeso
        [0.4,0.0,0.6,0.0],  # PFC
    ], dtype=np.float32)
    node_miss = (miss_mod @ NODE_WEIGHTS.T).clip(0, 1).astype(np.float32)

    # Hazard labels — last observed UPDRS3 visit, event = first visit with UPDRS3 >= 40
    # This uses the full 4-visit horizon (BL, V02, V04, V06) because the event can occur at V06.
    # Using V06 for *label* is fine; what must not leak is V06 *inputs*, handled above.
    THRESH = 40.0
    ev_t = np.zeros(N, dtype=np.int64); ev_d = np.zeros(N, dtype=np.float32)
    u3_cols = [f"UPDRS3_{v}" for v in VISIT_PREFIX_ALL]
    U = m[u3_cols].apply(pd.to_numeric, errors="coerce").values
    for i in range(N):
        obs_t = np.where(~np.isnan(U[i]))[0]
        if len(obs_t) == 0:
            ev_t[i] = 0; ev_d[i] = 0; continue
        ev = np.where((~np.isnan(U[i])) & (U[i] >= THRESH))[0]
        if len(ev) > 0:
            ev_t[i] = int(ev[0]); ev_d[i] = 1.0
        else:
            ev_t[i] = int(obs_t.max()); ev_d[i] = 0.0

    # Target UPDRS3 @ V06 (normalised to [0,1])
    y_u3 = pd.to_numeric(m["UPDRS3_V06"], errors="coerce").values.astype(np.float32)
    y_obs = (~np.isnan(y_u3)).astype(np.float32)
    y_norm = np.nan_to_num(y_u3, nan=0.0) / CFG["UPDRS_MAX"]

    y_stage = pd.to_numeric(m["NSD_ISS_stage"], errors="coerce").values
    y_stage = np.nan_to_num(y_stage, nan=0).astype(np.int64)

    # DaTscan target: SBR at V06 (future imaging) — a true progression target.
    # The prior SBR_mean target was algebraically leaked: SBR_mean ≡ (CAUDATE_mean + PUTAMEN_mean)/2,
    # both of which are in the imaging input block → trivial r ≈ 1.0.
    y_sbr = pd.to_numeric(m.get("SBR_V06", np.nan), errors="coerce").values.astype(np.float32)
    y_sbr_obs = (~np.isnan(y_sbr)).astype(np.float32)
    y_sbr = np.nan_to_num(y_sbr, nan=0.0)

    GBA = pd.to_numeric(m["GBA_binary"], errors="coerce").fillna(0).astype(int).values
    SAA = pd.to_numeric(m["SAA_positive"], errors="coerce").fillna(0).astype(int).values
    cohort = m["COHORT_LABEL"].values

    # Severity proxy for contrastive pretraining — MUST be BL, never V06 (target).
    updrs3_bl = pd.to_numeric(m["UPDRS3_BL"], errors="coerce").fillna(0).values.astype(np.float32)

    return dict(
        X_clin=X_clin, X_gen=X_gen, X_img=X_img, X_bio=X_bio,
        X_ltp=X_ltp, X_visit=X_visit, V_mask=V_mask,
        node_miss=node_miss,
        event_time=ev_t, event_ind=ev_d,
        y_u3_norm=y_norm, y_obs=y_obs,
        y_stage=y_stage,
        y_sbr=y_sbr, y_sbr_obs=y_sbr_obs,
        GBA=GBA, SAA=SAA, cohort=cohort,
        updrs3_bl=updrs3_bl,
        patno=m["PATNO"].values,
    )

# =============================== Model =================================
class MultiModalEncoder(nn.Module):
    def __init__(self, d_clin, d_gen, d_img, d_bio, d_model=128):
        super().__init__()
        self.e_c = nn.Sequential(nn.Linear(d_clin, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d_model))
        self.e_g = nn.Sequential(nn.Linear(d_gen,  128), nn.GELU(), nn.Linear(128, d_model))
        self.e_i = nn.Sequential(nn.Linear(d_img,  256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d_model))
        self.e_b = nn.Sequential(nn.Linear(d_bio,  128), nn.GELU(), nn.Linear(128, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ln   = nn.LayerNorm(d_model)
    def forward(self, c, g, i, b):
        seq = torch.stack([self.e_c(c), self.e_g(g), self.e_i(i), self.e_b(b)], dim=1)
        out, _ = self.attn(seq, seq, seq)
        return self.ln(out + seq).mean(dim=1)

class LTPEncoder(nn.Module):
    def __init__(self, d_in=15, d_model=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d_model))
    def forward(self, x): return self.net(x)

class StochasticBraakGraph(nn.Module):
    """S-PBGL — Gaussian posterior over node features, analytical edge variance."""
    def __init__(self, A_braak, d_model=128, n_nodes=10, d_node=8,
                 braak_eps=0.05, scale_init=0.1, kappa_init=0.5, logvar_clip=(-4.0, 2.0)):
        super().__init__()
        self.register_buffer("A_braak", A_braak.float())
        self.n_nodes, self.d_node, self.eps = n_nodes, d_node, braak_eps
        self.logvar_clip = logvar_clip
        self.mu_proj     = nn.Sequential(nn.Linear(d_model, n_nodes * d_node), nn.Tanh())
        self.logvar_proj = nn.Linear(d_model, n_nodes * d_node)
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, z, node_miss=None):
        B = z.shape[0]
        mu     = self.mu_proj(z).reshape(B, self.n_nodes, self.d_node)
        logvar = self.logvar_proj(z).reshape(B, self.n_nodes, self.d_node)
        logvar = logvar.clamp(*self.logvar_clip)
        sigma  = torch.exp(0.5 * logvar)
        if node_miss is not None:
            sigma = sigma * torch.exp(self.kappa * node_miss).unsqueeze(-1)
        F_i = mu + sigma * torch.randn_like(sigma) if self.training else mu

        A_prior = (self.A_braak + self.eps).unsqueeze(0)
        S = torch.bmm(F_i, F_i.transpose(1, 2)) / (self.d_node ** 0.5)
        A_raw = torch.sigmoid(self.scale * S) * A_prior
        d = A_raw.sum(-1, keepdim=True).clamp(min=1e-6)
        d_is = d.pow(-0.5)
        A_i = d_is * A_raw * d_is.transpose(1, 2)

        mu_n  = (mu ** 2).sum(-1)
        sig_n = (sigma ** 2).sum(-1)
        var_S = (sig_n.unsqueeze(2) * mu_n.unsqueeze(1)
               + mu_n.unsqueeze(2)  * sig_n.unsqueeze(1)
               + sig_n.unsqueeze(2) * sig_n.unsqueeze(1)) / self.d_node
        S_mean = torch.bmm(mu, mu.transpose(1, 2)) / (self.d_node ** 0.5)
        p_bar  = torch.sigmoid(self.scale * S_mean) * A_prior
        dp     = (self.scale ** 2) * (p_bar ** 2) * ((1 - p_bar) ** 2)
        u_edge = dp * var_S * (A_prior ** 2)

        kl = 0.5 * (mu**2 + sigma**2 - 1 - logvar).sum(dim=[1, 2]).mean()
        return A_i, u_edge, kl

    def reg_loss(self, A_i):
        tgt = self.A_braak.unsqueeze(0).expand_as(A_i)
        return F.mse_loss(A_i, tgt)

class BraakGraphConv(nn.Module):
    def __init__(self, d_img, d_model=128, n_nodes=10, d_h=16):
        super().__init__()
        self.n_nodes, self.d_h = n_nodes, d_h
        self.init_proj = nn.Linear(d_img, n_nodes * d_h)
        self.W1 = nn.Linear(d_h, d_h, bias=False); self.W1s = nn.Linear(d_h, d_h)
        self.W2 = nn.Linear(d_h, d_h, bias=False); self.W2s = nn.Linear(d_h, d_h)
        self.ln1 = nn.LayerNorm(d_h); self.ln2 = nn.LayerNorm(d_h)
        self.out = nn.Linear(d_h, d_model)
    def forward(self, x_img, A_i):
        B = x_img.shape[0]
        H0 = self.init_proj(x_img).reshape(B, self.n_nodes, self.d_h)
        H1 = F.gelu(self.ln1(torch.bmm(A_i, self.W1(H0)) + self.W1s(H0)))
        H2 = F.gelu(self.ln2(torch.bmm(A_i, self.W2(H1)) + self.W2s(H1)))
        return self.out(H2.mean(dim=1))

class BSMTALayer(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_nodes=10, T_max=4, dropout=0.1):
        super().__init__()
        self.h = n_heads; self.dh = d_model // n_heads
        self.n_nodes = n_nodes; self.T_max = T_max
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.tnq = nn.Parameter(torch.randn(n_heads, T_max, n_nodes) * 0.02)
        self.head_scale = nn.Parameter(torch.ones(n_heads) * 0.1)
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(),
                                 nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)
    def forward(self, x, A_i, visit_mask=None):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.reshape(B, T, self.h, self.dh).transpose(1, 2)
        k = k.reshape(B, T, self.h, self.dh).transpose(1, 2)
        v = v.reshape(B, T, self.h, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.dh ** 0.5)
        Q_h = self.tnq[:, :T, :]
        C = torch.einsum("htn,bnm->bhtm", Q_h, A_i)
        B_bias = torch.einsum("bhtn,bhsn->bhts", C, C) / (self.n_nodes ** 0.5)
        scores = scores + self.head_scale.view(1, -1, 1, 1) * B_bias
        if visit_mask is not None:
            m = visit_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(m == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)
        out = x + self.proj(out)
        out = out + self.mlp(self.ln2(out))
        return out, attn

class BSMTA(nn.Module):
    def __init__(self, d_model=128, d_visit=5, n_heads=8, n_nodes=10, T_max=4, n_layers=2):
        super().__init__()
        self.visit_embed = nn.Linear(d_visit, d_model)
        self.pos = nn.Parameter(torch.randn(T_max, d_model) * 0.02)
        self.layers = nn.ModuleList([BSMTALayer(d_model, n_heads, n_nodes, T_max) for _ in range(n_layers)])
    def forward(self, x_visit, A_i, z_ctx=None, visit_mask=None):
        B, T, _ = x_visit.shape
        h = self.visit_embed(x_visit) + self.pos[:T].unsqueeze(0)
        if z_ctx is not None: h = h + z_ctx.unsqueeze(1)
        for layer in self.layers:
            h, _ = layer(h, A_i, visit_mask)
        if visit_mask is not None:
            last_idx = (visit_mask.sum(dim=1).long() - 1).clamp(min=0)
            idx = last_idx.view(-1, 1, 1).expand(-1, 1, h.shape[-1])
            h_last = torch.gather(h, 1, idx).squeeze(1)
        else:
            h_last = h[:, -1, :]
        if z_ctx is not None: h_last = h_last + z_ctx
        return h_last

class EDPHead(nn.Module):
    def __init__(self, d_in=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.GELU(), nn.Linear(128, 4))
    def forward(self, h):
        o = self.net(h)
        g = o[:, 0]
        nu = F.softplus(o[:, 1]) + 1e-6
        al = F.softplus(o[:, 2]) + 1.0 + 1e-6
        be = F.softplus(o[:, 3]) + 1e-6
        return g, nu, al, be

class StageHead(nn.Module):
    def __init__(self, d_in=128, n_stages=7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.GELU(), nn.Linear(128, n_stages))
    def forward(self, h): return self.net(h)

class DaTscanHead(nn.Module):
    def __init__(self, d_in=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, h): return self.net(h).squeeze(-1)

class NHH(nn.Module):
    def __init__(self, d_in=128, n_bins=4, d_emb=8):
        super().__init__()
        self.n_bins = n_bins
        self.t_emb = nn.Embedding(n_bins, d_emb)
        self.mlp  = nn.Sequential(nn.Linear(d_in + d_emb, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, h):
        B = h.shape[0]
        t = torch.arange(self.n_bins, device=h.device)
        t_e = self.t_emb(t).unsqueeze(0).expand(B, -1, -1)
        h_e = h.unsqueeze(1).expand(-1, self.n_bins, -1)
        return self.mlp(torch.cat([h_e, t_e], dim=-1)).squeeze(-1)
    def loss(self, logits, event_time, event_ind):
        B, T = logits.shape
        tr = torch.arange(T, device=logits.device).unsqueeze(0)
        before = (tr < event_time.unsqueeze(1)).float()
        at_T   = (tr == event_time.unsqueeze(1)).float()
        t1 = (F.logsigmoid(-logits) * before).sum(1)
        t2 = (F.logsigmoid( logits) * at_T * event_ind.unsqueeze(1)).sum(1)
        return -(t1 + t2).mean()

class ProjectionHead(nn.Module):
    def __init__(self, d_in=128, d_out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, d_out))
    def forward(self, z): return F.normalize(self.net(z), dim=-1)

class BrainFormerPD(nn.Module):
    def __init__(self, d_clin, d_gen, d_img, d_bio, d_ltp, A_braak, n_stages=7, cfg=CFG):
        super().__init__()
        self.d_img = d_img
        self.mod  = MultiModalEncoder(d_clin, d_gen, d_img, d_bio, cfg["d_model"])
        self.ltp  = LTPEncoder(d_ltp, cfg["d_model"])
        self.graph= StochasticBraakGraph(A_braak, cfg["d_model"], cfg["n_nodes"], cfg["d_node"],
                                         cfg["braak_eps"], cfg["graph_scale_init"],
                                         cfg["kappa_init"], cfg["logvar_clip"])
        self.gcn  = BraakGraphConv(d_img, cfg["d_model"], cfg["n_nodes"])
        self.tmp  = BSMTA(cfg["d_model"], 5, cfg["n_heads"], cfg["n_nodes"], cfg["VISIT_T"], cfg["n_tf_layers"])
        self.h_edp= EDPHead(cfg["d_model"])
        self.h_stg= StageHead(cfg["d_model"], n_stages)
        self.h_sbr= DaTscanHead(cfg["d_model"])
        self.h_nhh= NHH(cfg["d_model"], cfg["HAZARD_BINS"])
        self.proj = ProjectionHead(cfg["d_model"], 64)

    def encode(self, b):
        z = self.mod(b["clin"], b["gen"], b["img"], b["bio"])
        z = z + self.ltp(b["ltp"])
        A_i, u_edge, kl = self.graph(z, b.get("node_miss"))
        z = z + self.gcn(b["img_raw"], A_i)
        return z, A_i, u_edge, kl

    def forward(self, b):
        z, A_i, u_edge, kl = self.encode(b)
        h = self.tmp(b["visit"], A_i, z_ctx=z, visit_mask=b.get("visit_mask"))
        g, nu, al, be = self.h_edp(h)
        return dict(gamma=g, nu=nu, alpha=al, beta=be,
                    stage_logits=self.h_stg(h), sbr_pred=self.h_sbr(h),
                    hazard_logits=self.h_nhh(h),
                    z=z, proj=self.proj(z),
                    A_i=A_i, u_edge=u_edge, kl_graph=kl)

# ============================ Dataset / Loss ===========================
class PPMIDataset(Dataset):
    def __init__(self, indices, A):
        self.idx = np.asarray(indices); self.A = A
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        k = int(self.idx[i]); A = self.A
        return dict(
            clin=torch.from_numpy(A["X_clin"][k]),
            gen =torch.from_numpy(A["X_gen"][k]),
            img =torch.from_numpy(A["X_img"][k]),
            bio =torch.from_numpy(A["X_bio"][k]),
            ltp =torch.from_numpy(A["X_ltp"][k]),
            visit=torch.from_numpy(A["X_visit"][k]),
            visit_mask=torch.from_numpy(A["V_mask"][k]),
            node_miss=torch.from_numpy(A["node_miss"][k]),
            img_raw=torch.from_numpy(A["X_img"][k]),
            y_updrs=torch.tensor(A["y_u3_norm"][k], dtype=torch.float32),
            y_obs  =torch.tensor(A["y_obs"][k], dtype=torch.float32),
            y_stage=torch.tensor(int(A["y_stage"][k]), dtype=torch.long),
            y_sbr  =torch.tensor(A["y_sbr"][k], dtype=torch.float32),
            y_sbr_obs=torch.tensor(A["y_sbr_obs"][k], dtype=torch.float32),
            event_time=torch.tensor(int(A["event_time"][k]), dtype=torch.long),
            event_ind =torch.tensor(A["event_ind"][k], dtype=torch.float32),
            GBA=torch.tensor(int(A["GBA"][k]), dtype=torch.long),
            SAA=torch.tensor(int(A["SAA"][k]), dtype=torch.long),
            updrs3_bl=torch.tensor(A["updrs3_bl"][k], dtype=torch.float32),
            patno=torch.tensor(int(A["patno"][k]), dtype=torch.long),
        )

def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

def supervised_contrastive(proj, labels, severity, tau=0.1, eta=1.5, sigma=5.0):
    """Supervised InfoNCE (Khosla et al. 2020) with UPDRS-matched hard negatives.
    Positives: same GBA label AND |Δseverity| < sigma.
    Hard negatives: different GBA label, weighted by severity similarity.
    Uses -1e9 instead of -inf on diagonal to avoid (-inf * 0 = NaN) when summing.
    """
    B = proj.shape[0]
    sim = proj @ proj.t() / tau                           # [B, B]
    eye = torch.eye(B, device=proj.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)                      # FIX: finite sentinel, not -inf
    dU  = (severity.unsqueeze(0) - severity.unsqueeze(1)).abs()
    sg  = (labels.unsqueeze(0) == labels.unsqueeze(1))
    pos = sg & (dU < sigma) & (~eye)
    valid = pos.any(dim=1)
    if valid.sum() < 2:
        return proj.sum() * 0.0
    diff = ~sg
    w_hard = 1.0 + eta * torch.exp(-dU / (sigma * 2.0)) * diff.float()
    logits = sim + torch.log(w_hard.clamp(min=1e-6))
    logits = logits[valid]; pv = pos[valid]
    lp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # Explicit zero-masking at non-positive entries prevents NaN propagation.
    contrib = torch.where(pv, lp, torch.zeros_like(lp))
    mp = contrib.sum(1) / pv.float().sum(1).clamp(min=1)
    return -mp.mean()

def nig_loss_each(y, g, nu, al, be, lam=0.01):
    two_b = 2.0 * be * (1 + nu)
    nll = 0.5 * torch.log(math.pi / nu) - al * torch.log(two_b) \
        + (al + 0.5) * torch.log(nu * (y - g) ** 2 + two_b) \
        + torch.lgamma(al) - torch.lgamma(al + 0.5)
    reg = torch.abs(y - g) * (2 * nu + al)
    return nll + lam * reg

# ============================ Training =================================
def eval_r2(model, loader, dev):
    model.eval()
    ys, ps, obs = [], [], []
    with torch.no_grad():
        for b in loader:
            b = move(b, dev)
            out = model(b)
            ys.append(b["y_updrs"].cpu().numpy())
            ps.append(out["gamma"].cpu().numpy())
            obs.append(b["y_obs"].cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps); m = np.concatenate(obs).astype(bool)
    if m.sum() < 10: return float("nan")
    y = y[m] * CFG["UPDRS_MAX"]; p = p[m] * CFG["UPDRS_MAX"]
    return float(r2_score(y, p))

def pretrain(model, loader_tr, dev, epochs):
    params = list(model.mod.parameters()) + list(model.ltp.parameters()) \
           + list(model.graph.parameters()) + list(model.gcn.parameters()) \
           + list(model.proj.parameters())
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    for epoch in range(epochs):
        model.train(); losses = []
        pbar = tqdm(loader_tr, desc=f"[pretrain {epoch+1}/{epochs}]", leave=False)
        for b in pbar:
            b = move(b, dev)
            z, A_i, u_edge, kl = model.encode(b)
            proj = model.proj(z)
            # Severity proxy MUST be baseline UPDRS, never V06 (the target).
            sev = b["updrs3_bl"]
            loss_c = supervised_contrastive(proj, b["GBA"], sev, tau=0.1, eta=1.5, sigma=5.0)
            anneal = min((epoch + 1) / CFG["kl_anneal_epochs"], 1.0) * CFG["kl_max"]
            loss = loss_c + anneal * kl + CFG["w_graph"] * model.graph.reg_loss(A_i)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CFG["grad_clip"]); opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=np.mean(losses))
        print(f"[pretrain] epoch {epoch+1}  loss={np.mean(losses):.4f}")
    torch.save(dict(model_state=model.state_dict()), CHECKPOINTS / "pretrain.pt")

def ughem_w(u_epi, gamma=0.5):
    with torch.no_grad():
        un = u_epi / (u_epi.mean() + 1e-6)
        return 1.0 + gamma * torch.tanh(un - 1.0)

def train_full(model, loader_tr, loader_va, dev, class_weights, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)
    best = -1e9; history = []
    for epoch in range(epochs):
        model.train(); losses = []; diags = dict(updrs=[], stage=[], sbr=[], haz=[], kl=[])
        pbar = tqdm(loader_tr, desc=f"[train {epoch+1}/{epochs}]", leave=False)
        for b in pbar:
            b = move(b, dev)
            out = model(b)
            y, y_obs = b["y_updrs"], b["y_obs"]
            l_each = nig_loss_each(y, out["gamma"], out["nu"], out["alpha"], out["beta"])
            u_epi = out["beta"] / (out["nu"] * (out["alpha"] - 1).clamp(min=1e-6))
            w = ughem_w(u_epi, CFG["ughem_gamma"]) if epoch >= CFG["ughem_warmup"] else torch.ones_like(u_epi)
            L_u = (l_each * w * y_obs).sum() / y_obs.sum().clamp(min=1)
            L_s = F.cross_entropy(out["stage_logits"], b["y_stage"], weight=class_weights)
            sbr_err = (out["sbr_pred"] - b["y_sbr"]) ** 2 * b["y_sbr_obs"]
            L_sbr = sbr_err.sum() / b["y_sbr_obs"].sum().clamp(min=1)
            L_h = model.h_nhh.loss(out["hazard_logits"], b["event_time"], b["event_ind"])
            L_g = model.graph.reg_loss(out["A_i"])
            kl_w = min((epoch + 1) / CFG["kl_anneal_epochs"], 1.0) * CFG["kl_max"]
            loss = (CFG["w_updrs"] * L_u + CFG["w_stage"] * L_s + CFG["w_datscan"] * L_sbr
                    + CFG["w_hazard"] * L_h + CFG["w_graph"] * L_g + kl_w * out["kl_graph"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"]); opt.step()
            losses.append(float(loss.item()))
            diags["updrs"].append(float(L_u.item())); diags["stage"].append(float(L_s.item()))
            diags["sbr"].append(float(L_sbr.item())); diags["haz"].append(float(L_h.item()))
            diags["kl"].append(float(out["kl_graph"].item()))
            pbar.set_postfix(loss=np.mean(losses))
        sched.step()
        r2v = eval_r2(model, loader_va, dev)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), r2_val=r2v,
                            **{k: float(np.mean(v)) for k, v in diags.items()}))
        print(f"[train] epoch {epoch+1}  loss={np.mean(losses):.4f}  r2_val={r2v:.4f}")
        if r2v > best and not np.isnan(r2v):
            best = r2v
            torch.save(dict(model_state=model.state_dict(),
                            epoch=epoch, r2_val=r2v),
                       CHECKPOINTS / "best.pt")
    torch.save(dict(model_state=model.state_dict(), history=history),
               CHECKPOINTS / "final.pt")
    with open(RESULTS / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return best

# ============================= Evaluation ==============================
def run_eval(model, loader, dev):
    model.eval()
    acc = {k: [] for k in ["gamma","nu","alpha","beta","stage","sbr","hazard",
                           "A","u_edge","z","proj","y_u3","y_obs","y_stage","y_sbr","sbr_obs",
                           "GBA","SAA","patno","event_time","event_ind"]}
    with torch.no_grad():
        for b in loader:
            b = move(b, dev); r = model(b)
            acc["gamma"].append(r["gamma"].cpu().numpy())
            acc["nu"].append(r["nu"].cpu().numpy())
            acc["alpha"].append(r["alpha"].cpu().numpy())
            acc["beta"].append(r["beta"].cpu().numpy())
            acc["stage"].append(r["stage_logits"].cpu().numpy())
            acc["sbr"].append(r["sbr_pred"].cpu().numpy())
            acc["hazard"].append(r["hazard_logits"].cpu().numpy())
            acc["A"].append(r["A_i"].cpu().numpy())
            acc["u_edge"].append(r["u_edge"].cpu().numpy())
            acc["z"].append(r["z"].cpu().numpy())
            acc["proj"].append(r["proj"].cpu().numpy())
            acc["y_u3"].append(b["y_updrs"].cpu().numpy())
            acc["y_obs"].append(b["y_obs"].cpu().numpy())
            acc["y_stage"].append(b["y_stage"].cpu().numpy())
            acc["y_sbr"].append(b["y_sbr"].cpu().numpy())
            acc["sbr_obs"].append(b["y_sbr_obs"].cpu().numpy())
            acc["GBA"].append(b["GBA"].cpu().numpy())
            acc["SAA"].append(b["SAA"].cpu().numpy())
            acc["patno"].append(b["patno"].cpu().numpy())
            acc["event_time"].append(b["event_time"].cpu().numpy())
            acc["event_ind"].append(b["event_ind"].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}

def compute_metrics(out, arrays, cohort):
    mm = out["y_obs"].astype(bool)
    y = out["y_u3"] * CFG["UPDRS_MAX"]
    p = out["gamma"] * CFG["UPDRS_MAX"]
    def _m(mask):
        if mask.sum() < 10: return None
        return dict(R2=float(r2_score(y[mask], p[mask])),
                    MAE=float(mean_absolute_error(y[mask], p[mask])), n=int(mask.sum()))
    pd_mask = cohort == "PD"; pro_mask = cohort == "Prodromal"
    stage_p = out["stage"].argmax(axis=1)
    sbr_m = out["sbr_obs"].astype(bool)
    r_dat, p_dat = pearsonr(out["y_sbr"][sbr_m], out["sbr"][sbr_m]) if sbr_m.sum() >= 10 else (float("nan"), float("nan"))

    u_epi = out["beta"] / (out["nu"] * np.clip(out["alpha"] - 1, 1e-6, None))

    gba = out["GBA"].astype(bool); saa = out["SAA"].astype(bool)

    # ---- Harrell's C-index for NHH (discrete-time hazard) ----
    # Uses cumulative hazard over all bins as the risk score.
    haz_prob = 1.0 / (1.0 + np.exp(-out["hazard"]))                 # [N, HAZARD_BINS]
    risk_score = haz_prob.sum(axis=1)
    c_idx = _harrell_c(out["event_time"], out["event_ind"], risk_score)

    res = dict(
        UPDRS_all=_m(mm), UPDRS_PD=_m(mm & pd_mask), UPDRS_Prodromal=_m(mm & pro_mask),
        UPDRS_GBA_pos=_m(mm & gba), UPDRS_GBA_neg=_m(mm & ~gba),
        UPDRS_SAA_pos=_m(mm & saa), UPDRS_SAA_neg=_m(mm & ~saa),
        NSD_ISS_weighted_F1=float(f1_score(out["y_stage"], stage_p, average="weighted")),
        NSD_ISS_macro_F1   =float(f1_score(out["y_stage"], stage_p, average="macro")),
        DaTscan_Pearson_r=float(r_dat), DaTscan_Pearson_p=float(p_dat),
        NHH_c_index=float(c_idx),
    )
    return res, u_epi, pd_mask, pro_mask

def _harrell_c(event_time, event_ind, risk, max_pairs=2_000_000):
    """Harrell's C (concordance index) for right-censored data.

    A pair (i, j) with i an event is comparable if either
      (a) event_time[j] > event_time[i], or
      (b) event_time[j] == event_time[i] and j is censored (i died, j walked away at same bin).
    The pair is concordant if the event patient has a higher predicted risk.
    Subsampled to max_pairs for tractability.
    """
    n = len(event_time); num = den = 0.0; cnt = 0
    for i in range(n):
        if event_ind[i] != 1: continue
        for j in range(n):
            if i == j: continue
            comparable = (event_time[j] > event_time[i]) or \
                         (event_time[j] == event_time[i] and event_ind[j] == 0)
            if not comparable: continue
            den += 1
            if risk[i] > risk[j]: num += 1
            elif risk[i] == risk[j]: num += 0.5
            cnt += 1
            if cnt > max_pairs: break
        if cnt > max_pairs: break
    return num / max(den, 1)

def make_figures(out, u_epi, cohort, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10})

    def sf(fig, name):
        fig.tight_layout()
        fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    # Fig 3 — scatter coloured by u_epi
    mm = out["y_obs"].astype(bool)
    y = out["y_u3"][mm] * CFG["UPDRS_MAX"]; p = out["gamma"][mm] * CFG["UPDRS_MAX"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(y, p, c=u_epi[mm], cmap="viridis", s=8, alpha=0.6)
    ax.plot([0, 132], [0, 132], "k--", lw=1)
    ax.set_xlabel("True UPDRS-III @ V06"); ax.set_ylabel("Predicted γ")
    plt.colorbar(sc, ax=ax, label="epistemic u"); ax.set_title("UPDRS scatter")
    sf(fig, "fig03_scatter")

    # Fig 13 — mean u_edge atlases by group
    gba = out["GBA"].astype(bool)
    pro_nc = (cohort == "Prodromal") & (out["event_ind"].astype(bool) == False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (lbl, m) in zip(axes, [("GBA+", gba), ("GBA-", ~gba), ("Prodromal NC", pro_nc)]):
        if m.sum() < 5: ax.set_title(f"{lbl} n={m.sum()} (sparse)"); continue
        sns.heatmap(out["u_edge"][m].mean(axis=0),
                    xticklabels=NODE_NAMES, yticklabels=NODE_NAMES, ax=ax, cmap="magma")
        ax.set_title(f"{lbl}  n={m.sum()}")
    fig.suptitle("Edge uncertainty atlas (S-PBGL posterior)")
    sf(fig, "fig13_u_edge_atlas")

    # Fig 12 — training curves
    if history:
        fig, ax = plt.subplots(figsize=(7, 4))
        eps = [h["epoch"] for h in history]
        ls = [h["loss"] for h in history]; rs = [h["r2_val"] for h in history]
        ax.plot(eps, ls, label="train loss", color="steelblue")
        ax2 = ax.twinx(); ax2.plot(eps, rs, label="val R²", color="firebrick")
        ax.axvline(CFG["ughem_warmup"], color="k", ls=":", alpha=0.5, label="UG-HEM on")
        ax.set_xlabel("epoch"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
        sf(fig, "fig12_training")

# =============================== Main ==================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all","full","eval"], default="all")
    ap.add_argument("--epochs", type=int, default=CFG["train_epochs"])
    ap.add_argument("--pretrain_epochs", type=int, default=CFG["pretrain_epochs"])
    ap.add_argument("--batch", type=int, default=CFG["batch_size"])
    args = ap.parse_args()
    CFG["train_epochs"] = args.epochs; CFG["pretrain_epochs"] = args.pretrain_epochs
    CFG["batch_size"] = args.batch
    set_seed()
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}  "
              f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    t0 = time.time()
    master, (tr, va, te), A_braak, extra_cols = load_and_build()
    arrs = build_feature_matrices(master, extra_cols)
    print(f"[data] built feature tensors in {time.time()-t0:.1f}s")

    patnos = arrs["patno"]
    tr_mask = np.isin(patnos, tr); va_mask = np.isin(patnos, va); te_mask = np.isin(patnos, te)
    idx_tr = np.where(tr_mask)[0]; idx_va = np.where(va_mask)[0]; idx_te = np.where(te_mask)[0]
    print(f"[data] tr/va/te = {len(idx_tr)}/{len(idx_va)}/{len(idx_te)}")

    # scale
    for k in ["X_clin","X_gen","X_img","X_bio","X_ltp"]:
        arrs[k] = fit_transform(k.replace("X_",""), arrs[k], tr_mask)
    # scale visit per-feature using train rows
    V = arrs["X_visit"].copy()
    for f in range(V.shape[-1]):
        vals = V[tr_mask, :, f].reshape(-1)
        vals = vals[vals != 0]
        if len(vals) < 10 or vals.std() < 1e-6: continue
        mu, sd = vals.mean(), vals.std()
        V[:, :, f] = (V[:, :, f] - mu) / sd
    arrs["X_visit"] = np.nan_to_num(V, nan=0.0).astype(np.float32)

    # dims
    D_CLIN = arrs["X_clin"].shape[1]; D_GEN = arrs["X_gen"].shape[1]
    D_IMG  = arrs["X_img"].shape[1];  D_BIO = arrs["X_bio"].shape[1]
    D_LTP  = arrs["X_ltp"].shape[1]
    N_STAGES = int(max(int(arrs["y_stage"].max()) + 1, 7))
    print(f"[data] dims clin={D_CLIN} gen={D_GEN} img={D_IMG} bio={D_BIO} ltp={D_LTP} stages={N_STAGES}")

    # class weights
    y_stg_tr = arrs["y_stage"][tr_mask]
    classes = np.unique(y_stg_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_stg_tr)
    class_weights = torch.ones(N_STAGES, device=DEVICE)
    for c, w in zip(classes, cw): class_weights[int(c)] = float(w)

    # datasets
    ds_tr = PPMIDataset(idx_tr, arrs); ds_va = PPMIDataset(idx_va, arrs); ds_te = PPMIDataset(idx_te, arrs)
    loader_tr = DataLoader(ds_tr, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    loader_te = DataLoader(ds_te, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    # model
    A_t = torch.tensor(A_braak, device=DEVICE)
    model = BrainFormerPD(D_CLIN, D_GEN, D_IMG, D_BIO, D_LTP, A_t, n_stages=N_STAGES).to(DEVICE)
    print(f"[model] params = {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    if args.phase in ("all",):
        print("\n=== PHASE 1: GBA-BCSD pretraining ===")
        pretrain(model, loader_tr, DEVICE, CFG["pretrain_epochs"])

    if args.phase in ("all", "full"):
        print("\n=== PHASE 2: full multi-task training ===")
        best = train_full(model, loader_tr, loader_va, DEVICE, class_weights, CFG["train_epochs"])
        print(f"[train] best val R² = {best:.4f}")
        # restore best
        best_ckpt = CHECKPOINTS / "best.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE)["model_state"])
            print("[train] loaded best-val checkpoint")

    # evaluation
    print("\n=== EVALUATION on TEST ===")
    out = run_eval(model, loader_te, DEVICE)
    np.savez(RESULTS / "test_outputs.npz", **out)
    res, u_epi, pd_mask, pro_mask = compute_metrics(out, arrs, arrs["cohort"][idx_te])
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))

    # figures
    history = []
    hist_path = RESULTS / "training_history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text())
    make_figures(out, u_epi, arrs["cohort"][idx_te], history)
    print(f"\nAll results in: {RESULTS}")
    print(f"Total wall-clock: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
