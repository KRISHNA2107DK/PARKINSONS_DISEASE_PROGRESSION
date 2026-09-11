"""Post-hoc analyses on the BrainFormer-PD v2.1 local run.

Reads RESULTS_LOCAL/test_outputs.npz produced by train_local.py and computes:
- Block 08  Uncertainty calibration + clinical monitoring odds ratio.
- Block 09  Ablation B0 (XGBoost+LTP). B1-B4 are trained here, B5 is reused.
- Block 10  Wilcoxon + bootstrap CI (BF vs XGBoost).
- Block 11  GBA-stratified analysis with Mann-Whitney on u_epi.
- Block 12  SAA-stratified analysis + edge-weight MW at SNc / LC.
- Block 13  Graph topology phenotyping (KMeans(k=3) on upper triangle of A_i).
- Block 14  Two-source uncertainty (u_graph vs u_pred).
- Block 15  Latent subtype clustering on the projection head.
- All figures.

Usage:
    python analyze_local.py                     # everything except ablations
    python analyze_local.py --ablation          # also train B1..B4 (slow)
    python analyze_local.py --ablation --ablation_epochs 40
"""
from __future__ import annotations
import argparse, json, math, os, pickle, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score, f1_score
from scipy.stats import pearsonr, mannwhitneyu, wilcoxon, fisher_exact, chi2_contingency

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_local as tl   # reuse: CFG, DEVICE, model/dataset classes

RESULTS = HERE / "RESULTS_LOCAL"
FIGURES = RESULTS / "figures"
ANALYSIS = RESULTS / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

NODE_NAMES = tl.NODE_NAMES

# ============================ Data loading ============================
def load_test_outputs():
    npz = np.load(RESULTS / "test_outputs.npz", allow_pickle=True)
    return {k: npz[k] for k in npz.files}

def load_master_and_idx():
    """Return (master_df, idx_te, cohort_test) — aligned with test_outputs order."""
    master, splits, A, extra = tl.load_and_build()
    te = splits[2]
    # test_outputs was written in DataLoader order over idx_te, which is the order
    # of np.where(np.isin(patnos_all, te))[0] — same as we reconstruct here.
    patnos = master["PATNO"].values
    idx_te = np.where(np.isin(patnos, te))[0]
    cohort_test = master.iloc[idx_te]["COHORT_LABEL"].values
    return master, splits, A, extra, idx_te, cohort_test

# ========================= Block 08 — calibration =====================
def block_08_calibration(out, master, idx_te, cfg):
    u_epi = out["beta"] / (out["nu"] * np.clip(out["alpha"] - 1, 1e-6, None))
    u_ale = out["beta"] / np.clip(out["alpha"] - 1, 1e-6, None)
    obs_m = out["y_obs"].astype(bool)

    y_abs = np.abs(out["y_u3"] * cfg["UPDRS_MAX"] - out["gamma"] * cfg["UPDRS_MAX"])
    pred_std = np.sqrt(u_epi + u_ale) * cfg["UPDRS_MAX"]

    # ECE over deciles of predicted std
    ece = 0.0
    if obs_m.sum() >= 30:
        bins = np.quantile(pred_std[obs_m], np.linspace(0, 1, 11))
        for i in range(10):
            hi = bins[i+1] + (1e-9 if i == 9 else 0.0)
            m = obs_m & (pred_std >= bins[i]) & (pred_std < hi)
            if m.sum() < 5: continue
            frac = m.sum() / obs_m.sum()
            ece += frac * abs(pred_std[m].mean() - y_abs[m].mean()) / cfg["UPDRS_MAX"]

    # Quintile MAE
    mae_q = []
    if obs_m.sum() >= 25:
        qs = np.quantile(u_epi[obs_m], [0.2, 0.4, 0.6, 0.8])
        for lo, hi in zip([-np.inf] + list(qs), list(qs) + [np.inf]):
            m = obs_m & (u_epi >= lo) & (u_epi < hi)
            mae_q.append(float(y_abs[m].mean()) if m.sum() >= 5 else float("nan"))
    unc_ratio = (mae_q[-1] / max(mae_q[0], 1e-6)) if len(mae_q) >= 2 else float("nan")

    # Clinical monitoring OR
    bl_map = dict(zip(master["PATNO"].values, master["UPDRS3_BL"].values))
    v6_map = dict(zip(master["PATNO"].values, master["UPDRS3_V06"].values))
    BL = np.array([bl_map.get(int(p), np.nan) for p in out["patno"]])
    V6 = np.array([v6_map.get(int(p), np.nan) for p in out["patno"]])
    dlt = np.abs(V6 - BL)
    dlt_obs = ~np.isnan(dlt)
    big = (dlt > 15).astype(int)
    q75, q25 = np.quantile(u_epi[obs_m], [0.75, 0.25]) if obs_m.sum() >= 30 else (np.inf, -np.inf)
    top = (u_epi >= q75) & obs_m & dlt_obs
    bot = (u_epi <= q25) & obs_m & dlt_obs
    a = int((top & (big == 1)).sum()); b = int((top & (big == 0)).sum())
    c = int((bot & (big == 1)).sum()); d = int((bot & (big == 0)).sum())

    # ---- Haldane–Anscombe-corrected OR (+0.5 in every cell) ----
    # Handles perfect separation (c == 0) cleanly and gives a reportable finite OR.
    ah, bh, ch, dh = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_h = (ah * dh) / (bh * ch)
    log_or_h = math.log(or_h)
    se_h = math.sqrt(1/ah + 1/bh + 1/ch + 1/dh)
    ci_lo_h = math.exp(log_or_h - 1.96 * se_h)
    ci_hi_h = math.exp(log_or_h + 1.96 * se_h)

    # Fisher exact + chi-squared using the RAW (uncorrected) table for p-value.
    table = np.array([[a, b], [c, d]])
    try:
        from scipy.stats import fisher_exact as _fisher
        fisher_or, fisher_p = _fisher(table, alternative="greater")
    except Exception:
        fisher_or, fisher_p = float("nan"), float("nan")
    if table.sum() > 0 and (table > 0).all():
        chi2_stat, chi2_p, _, _ = chi2_contingency(table)
    else:
        chi2_stat, chi2_p = float("nan"), float("nan")

    res = dict(
        ECE=float(ece), MAE_quintiles=mae_q, uncertainty_ratio=float(unc_ratio),
        clinical_table=dict(a=a, b=b, c=c, d=d),
        clinical_OR_haldane=float(or_h),
        clinical_OR_CI_haldane=[float(ci_lo_h), float(ci_hi_h)],
        fisher_OR=float(fisher_or), fisher_p=float(fisher_p),
        chi2=float(chi2_stat), chi2_p=float(chi2_p),
    )
    (ANALYSIS / "block_08_calibration.json").write_text(json.dumps(res, indent=2))
    print("\n=== Block 08 — calibration ===")
    for k, v in res.items(): print(f"  {k}: {v}")
    return u_epi, u_ale, res

# ===================== Block 11 / 12 — stratified =====================
def block_1112_strata(out, u_epi, cfg):
    gba = out["GBA"].astype(bool); saa = out["SAA"].astype(bool)
    mm = out["y_obs"].astype(bool)
    y = out["y_u3"] * cfg["UPDRS_MAX"]; p = out["gamma"] * cfg["UPDRS_MAX"]

    def _m(mask):
        if mask.sum() < 10: return None
        return dict(R2=float(r2_score(y[mask], p[mask])),
                    MAE=float(mean_absolute_error(y[mask], p[mask])),
                    n=int(mask.sum()))

    gba_res = dict(GBA_pos=_m(mm & gba), GBA_neg=_m(mm & ~gba))
    if (mm & gba).sum() >= 5 and (mm & ~gba).sum() >= 5:
        s, p_mw = mannwhitneyu(u_epi[mm & gba], u_epi[mm & ~gba], alternative="two-sided")
        gba_res["u_epi_MW"] = dict(U=float(s), p=float(p_mw),
                                   mean_pos=float(u_epi[mm & gba].mean()),
                                   mean_neg=float(u_epi[mm & ~gba].mean()))

    saa_res = dict(SAA_pos=_m(mm & saa), SAA_neg=_m(mm & ~saa))
    if (mm & saa).sum() >= 5 and (mm & ~saa).sum() >= 5:
        s, p_mw = mannwhitneyu(u_epi[mm & saa], u_epi[mm & ~saa], alternative="two-sided")
        saa_res["u_epi_MW"] = dict(U=float(s), p=float(p_mw),
                                   mean_pos=float(u_epi[mm & saa].mean()),
                                   mean_neg=float(u_epi[mm & ~saa].mean()))
    # SAA edge-weight MW at SNc / LC
    A_te = out["A"]                                   # [N, 10, 10]
    snc_row = A_te[:, 4, :].mean(axis=1)              # SNc is index 4 in NODE_NAMES
    lc_row  = A_te[:, 2, :].mean(axis=1)              # LC is index 2
    if saa.sum() >= 10 and (~saa).sum() >= 10:
        u, p = mannwhitneyu(snc_row[saa], snc_row[~saa], alternative="two-sided")
        saa_res["edge_MW_SNc"] = dict(U=float(u), p=float(p),
                                      mean_pos=float(snc_row[saa].mean()),
                                      mean_neg=float(snc_row[~saa].mean()))
        u, p = mannwhitneyu(lc_row[saa], lc_row[~saa], alternative="two-sided")
        saa_res["edge_MW_LC"]  = dict(U=float(u), p=float(p),
                                      mean_pos=float(lc_row[saa].mean()),
                                      mean_neg=float(lc_row[~saa].mean()))

    (ANALYSIS / "block_11_gba.json").write_text(json.dumps(gba_res, indent=2, default=str))
    (ANALYSIS / "block_12_saa.json").write_text(json.dumps(saa_res, indent=2, default=str))
    print("\n=== Block 11 — GBA stratified ===")
    print(json.dumps(gba_res, indent=2, default=str))
    print("\n=== Block 12 — SAA stratified ===")
    print(json.dumps(saa_res, indent=2, default=str))
    return gba_res, saa_res, snc_row, lc_row

# ===================== Block 13 — topology phenotyping ================
def block_13_topology(out, cfg):
    tl.set_seed()
    A = out["A"]
    n = cfg["n_nodes"]
    triu = np.triu_indices(n, k=1)
    A_vec = A[:, triu[0], triu[1]]                    # [N, 45]
    km = KMeans(n_clusters=3, n_init=10, random_state=tl.SEED).fit(A_vec)
    cluster = km.labels_

    brainstem = [0, 1, 2, 3, 4, 5]  # DMV, OlfBulb, LC, Raphe, SNc, PPN
    limbic    = [6, 7]              # Amygdala, Hippocampus
    cortical  = [8, 9]              # TempMeso, PFC

    def _label(meanA):
        score = dict(
            brainstem=meanA[np.ix_(brainstem, brainstem)].mean(),
            limbic=meanA[np.ix_(limbic, limbic)].mean(),
            cortical=meanA[np.ix_(cortical, cortical)].mean(),
        )
        return max(score, key=score.get), score

    gba = out["GBA"].astype(bool)
    info = {}
    for cid in range(3):
        m = cluster == cid
        meanA = A[m].mean(axis=0)
        lab, sc = _label(meanA)
        a_c = int((m & gba).sum()); b_c = int((m & ~gba).sum())
        c_c = int((~m & gba).sum()); d_c = int((~m & ~gba).sum())
        try:
            odds, p = fisher_exact([[a_c, b_c], [c_c, d_c]])
        except Exception:
            odds, p = float("nan"), float("nan")
        info[f"cluster_{cid}"] = dict(
            label=lab, score=sc, n=int(m.sum()),
            gba_pos=a_c, gba_neg=b_c,
            fisher_odds=float(odds), fisher_p=float(p),
            mean_A=meanA.tolist(),
        )
    np.save(ANALYSIS / "block_13_cluster_ids.npy", cluster)
    (ANALYSIS / "block_13_topology.json").write_text(json.dumps(info, indent=2, default=str))
    print("\n=== Block 13 — topology phenotypes ===")
    for k, v in info.items():
        print(f"  {k}: label={v['label']}  n={v['n']}  GBA+%={100*v['gba_pos']/max(v['n'],1):.1f}  Fisher_p={v['fisher_p']:.4f}")
    return cluster, info

# ===================== Block 14 — two-source uncertainty ==============
def block_14_two_source(out, u_epi, cfg):
    u_graph = out["u_edge"].mean(axis=(1, 2))
    u_pred  = u_epi
    r, p = pearsonr(u_graph, u_pred)

    q_g_hi = np.quantile(u_graph, 0.75)
    q_p_lo = np.quantile(u_pred, 0.25)
    phen = (u_graph >= q_g_hi) & (u_pred <= q_p_lo)

    gba = out["GBA"].astype(bool)
    n_phen = int(phen.sum())
    gba_in = int((phen & gba).sum())
    gba_out = int((~phen & gba).sum())
    rest_n = int((~phen).sum())
    enrich = (gba_in / max(n_phen, 1)) / (gba_out / max(rest_n, 1) + 1e-9)

    updrs = out["y_u3"] * cfg["UPDRS_MAX"]
    res = dict(
        pearson_r=float(r), pearson_p=float(p),
        phenotype_n=n_phen,
        phenotype_gba_frac=float(gba_in / max(n_phen, 1)),
        phenotype_gba_enrichment=float(enrich),
        phenotype_updrs_mean=float(updrs[phen].mean()) if n_phen else float("nan"),
        phenotype_updrs_std=float(updrs[phen].std()) if n_phen else float("nan"),
        rest_updrs_mean=float(updrs[~phen].mean()),
    )
    np.save(ANALYSIS / "block_14_u_graph.npy", u_graph)
    np.save(ANALYSIS / "block_14_u_pred.npy",  u_pred)
    (ANALYSIS / "block_14_two_source.json").write_text(json.dumps(res, indent=2, default=str))
    print("\n=== Block 14 — two-source uncertainty ===")
    print(json.dumps(res, indent=2, default=str))
    return u_graph, u_pred, res

# ===================== Block 14b — restricted C-index =====================
def block_14b_restricted_cindex(out, master, cfg, thresh=40.0):
    """Harrell's C-index restricted to patients whose observed input UPDRS3 (BL/V02/V04)
    was strictly below the event threshold. This removes the 'event already happened in
    the input window' trivial-prediction cases and measures true forward survival skill.
    """
    bl_map  = dict(zip(master["PATNO"].values, master["UPDRS3_BL"].values))
    v02_map = dict(zip(master["PATNO"].values, master["UPDRS3_V02"].values)) if "UPDRS3_V02" in master.columns else {}
    v04_map = dict(zip(master["PATNO"].values, master["UPDRS3_V04"].values)) if "UPDRS3_V04" in master.columns else {}

    def _sub_thresh(p, m):
        v = m.get(int(p), np.nan)
        return np.isnan(v) or v < thresh

    kept = np.array([
        _sub_thresh(p, bl_map) and _sub_thresh(p, v02_map) and _sub_thresh(p, v04_map)
        for p in out["patno"]
    ])
    haz_prob = 1.0 / (1.0 + np.exp(-out["hazard"]))
    risk = haz_prob.sum(axis=1)

    c_all = tl._harrell_c(out["event_time"], out["event_ind"], risk)
    c_res = tl._harrell_c(out["event_time"][kept], out["event_ind"][kept], risk[kept]) if kept.sum() > 10 else float("nan")

    n_kept = int(kept.sum())
    n_events_kept = int(out["event_ind"][kept].sum())
    res = dict(
        C_index_all=float(c_all),
        C_index_restricted_BL_V02_V04_below_40=float(c_res),
        n_restricted=n_kept,
        n_events_restricted=n_events_kept,
        event_rate_restricted=float(n_events_kept / max(n_kept, 1)),
    )
    (ANALYSIS / "block_14b_restricted_cindex.json").write_text(json.dumps(res, indent=2))
    print("\n=== Block 14b — restricted C-index ===")
    print(json.dumps(res, indent=2))
    return res

# ===================== Block 15 — latent subtypes =====================
def block_15_subtypes(out, cfg):
    tl.set_seed()
    Z = out["proj"]
    km = KMeans(n_clusters=4, n_init=20, random_state=tl.SEED).fit(Z)
    sil = silhouette_score(Z, km.labels_) if len(set(km.labels_)) > 1 else float("nan")

    u_epi = out["beta"] / (out["nu"] * np.clip(out["alpha"] - 1, 1e-6, None))
    sub = {}
    for cid in range(4):
        m = km.labels_ == cid
        sub[f"cluster_{cid}"] = dict(
            n=int(m.sum()),
            gba_frac=float(out["GBA"][m].mean()) if m.sum() else float("nan"),
            saa_frac=float(out["SAA"][m].mean()) if m.sum() else float("nan"),
            u_epi_mean=float(u_epi[m].mean()) if m.sum() else float("nan"),
            updrs_mean=float((out["y_u3"][m] * cfg["UPDRS_MAX"]).mean()) if m.sum() else float("nan"),
        )
    res = dict(silhouette=float(sil), clusters=sub)
    np.save(ANALYSIS / "block_15_cluster_ids.npy", km.labels_)
    (ANALYSIS / "block_15_subtyping.json").write_text(json.dumps(res, indent=2, default=str))
    print("\n=== Block 15 — latent subtypes ===")
    print(json.dumps(res, indent=2, default=str))
    return km.labels_, res

# ===================== Block 09 — ablation ============================
def block_09_ablation(master, splits, A_braak_np, extra_cols, out, cfg,
                      epochs=40, run_xgb=True, run_bf=True):
    """B0 XGBoost+LTP, B1..B4 BF variants, B5 = full (reuse test_outputs)."""
    from xgboost import XGBRegressor

    arrs = tl.build_feature_matrices(master, extra_cols)
    patnos = arrs["patno"]
    tr, va, te = splits
    tr_mask = np.isin(patnos, tr); te_mask = np.isin(patnos, te)

    for k in ["X_clin","X_gen","X_img","X_bio","X_ltp"]:
        arrs[k] = tl.fit_transform(k.replace("X_",""), arrs[k], tr_mask)
    V = arrs["X_visit"].copy()
    for f in range(V.shape[-1]):
        vals = V[tr_mask, :, f].reshape(-1); vals = vals[vals != 0]
        if len(vals) < 10 or vals.std() < 1e-6: continue
        mu, sd = vals.mean(), vals.std(); V[:, :, f] = (V[:, :, f] - mu) / sd
    arrs["X_visit"] = np.nan_to_num(V, nan=0.0).astype(np.float32)

    idx_tr = np.where(tr_mask)[0]; idx_te = np.where(te_mask)[0]
    pd_mask_te = master.iloc[idx_te]["COHORT_LABEL"].values == "PD"

    results = {}

    if run_xgb:
        X_all = np.concatenate([arrs["X_clin"], arrs["X_gen"], arrs["X_img"],
                                arrs["X_bio"], arrs["X_ltp"]], axis=1)
        y_all = arrs["y_u3_norm"] * cfg["UPDRS_MAX"]
        obs   = arrs["y_obs"].astype(bool)

        xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                           random_state=tl.SEED, n_jobs=4, tree_method="hist")
        xgb.fit(X_all[idx_tr][obs[idx_tr]], y_all[idx_tr][obs[idx_tr]])
        pred_te = xgb.predict(X_all[idx_te])
        obs_te  = obs[idx_te]
        r2_all  = float(r2_score(y_all[idx_te][obs_te], pred_te[obs_te]))
        r2_pd   = float(r2_score(y_all[idx_te][obs_te & pd_mask_te],
                                 pred_te[obs_te & pd_mask_te])) if (obs_te & pd_mask_te).sum() >= 10 else float("nan")
        results["B0"] = dict(name="XGBoost+LTP", R2_all=r2_all, R2_PD=r2_pd)
        with open(ANALYSIS / "xgb_b0_preds.npz", "wb") as f:
            np.savez(f, y=y_all[idx_te], p=pred_te, obs=obs_te, pd_mask=pd_mask_te)
        with open(ANALYSIS / "xgb_b0.pkl", "wb") as f:
            pickle.dump(xgb, f)
        print(f"B0 XGBoost+LTP: R2_all={r2_all:.4f}  R2_PD={r2_pd:.4f}")

    if run_bf:
        A_t = torch.tensor(A_braak_np, device=tl.DEVICE)
        ds_tr = tl.PPMIDataset(idx_tr, arrs); ds_te = tl.PPMIDataset(idx_te, arrs)
        loader_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0, drop_last=True)
        loader_te = DataLoader(ds_te, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

        class BFAblation(nn.Module):
            def __init__(self, d_clin, d_gen, d_img, d_bio, d_ltp, A_braak,
                         use_graph, use_gcn, use_ltp, n_stages=7):
                super().__init__()
                self.use_graph, self.use_gcn, self.use_ltp = use_graph, use_gcn, use_ltp
                self.mod = tl.MultiModalEncoder(d_clin, d_gen, d_img, d_bio, cfg["d_model"])
                if use_ltp: self.ltp = tl.LTPEncoder(d_ltp, cfg["d_model"])
                self.register_buffer("A_fixed", A_braak.float())
                if use_graph == "spbgl":
                    self.graph = tl.StochasticBraakGraph(
                        A_braak, cfg["d_model"], cfg["n_nodes"], cfg["d_node"],
                        cfg["braak_eps"], cfg["graph_scale_init"], cfg["kappa_init"],
                        cfg["logvar_clip"])
                elif use_graph == "psbgl":
                    class _Det(nn.Module):
                        def __init__(self, A_braak, d_model, n_nodes, d_node, eps):
                            super().__init__()
                            self.register_buffer("A_braak", A_braak.float())
                            self.n_nodes = n_nodes; self.d_node = d_node; self.eps = eps
                            self.node_proj = nn.Sequential(nn.Linear(d_model, n_nodes*d_node), nn.Tanh())
                            self.scale = nn.Parameter(torch.ones(1) * 0.1)
                        def forward(self, z, node_miss=None):
                            B = z.shape[0]
                            F_i = self.node_proj(z).reshape(B, self.n_nodes, self.d_node)
                            S = torch.bmm(F_i, F_i.transpose(1,2)) / (self.d_node**0.5)
                            A_p = (self.A_braak + self.eps).unsqueeze(0)
                            A_r = torch.sigmoid(self.scale * S) * A_p
                            d = A_r.sum(-1, keepdim=True).clamp(min=1e-6)
                            d_is = d.pow(-0.5); A_i = d_is * A_r * d_is.transpose(1, 2)
                            return A_i, torch.zeros_like(A_i), torch.tensor(0.0, device=z.device)
                        def reg_loss(self, A_i):
                            return F.mse_loss(A_i, self.A_braak.unsqueeze(0).expand_as(A_i))
                    self.graph = _Det(A_braak, cfg["d_model"], cfg["n_nodes"], cfg["d_node"], cfg["braak_eps"])
                if use_gcn and use_graph is not None:
                    self.gcn = tl.BraakGraphConv(d_img, cfg["d_model"], cfg["n_nodes"])
                self.tmp = tl.BSMTA(cfg["d_model"], 5, cfg["n_heads"], cfg["n_nodes"],
                                    cfg["VISIT_T"], cfg["n_tf_layers"])
                self.h_edp = tl.EDPHead(cfg["d_model"])
                self.h_stg = tl.StageHead(cfg["d_model"], n_stages)

            def forward(self, b):
                z = self.mod(b["clin"], b["gen"], b["img"], b["bio"])
                if self.use_ltp: z = z + self.ltp(b["ltp"])
                if self.use_graph == "fixed":
                    A_i = self.A_fixed.unsqueeze(0).expand(z.shape[0], -1, -1)
                    d = A_i.sum(-1, keepdim=True).clamp(min=1e-6)
                    d_is = d.pow(-0.5); A_i = d_is * A_i * d_is.transpose(1, 2)
                elif self.use_graph in ("psbgl", "spbgl"):
                    A_i, _, _ = self.graph(z, b.get("node_miss"))
                else:
                    A_i = torch.eye(cfg["n_nodes"], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1)
                if self.use_gcn and self.use_graph is not None:
                    z = z + self.gcn(b["img_raw"], A_i)
                h = self.tmp(b["visit"], A_i, z_ctx=z, visit_mask=b.get("visit_mask"))
                g, nu, al, be = self.h_edp(h)
                return dict(gamma=g, nu=nu, alpha=al, beta=be,
                            stage_logits=self.h_stg(h), A_i=A_i)

        def _train(tag, use_graph, use_gcn, use_ltp, ep):
            tl.set_seed()
            m = BFAblation(arrs["X_clin"].shape[1], arrs["X_gen"].shape[1],
                           arrs["X_img"].shape[1], arrs["X_bio"].shape[1],
                           arrs["X_ltp"].shape[1], A_t,
                           use_graph=use_graph, use_gcn=use_gcn, use_ltp=use_ltp,
                           n_stages=max(int(arrs["y_stage"].max())+1, 7)).to(tl.DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            for e in range(ep):
                m.train()
                for b in loader_tr:
                    b = tl.move(b, tl.DEVICE); out = m(b)
                    l_each = tl.nig_loss_each(b["y_updrs"], out["gamma"], out["nu"],
                                              out["alpha"], out["beta"])
                    L = (l_each * b["y_obs"]).sum() / b["y_obs"].sum().clamp(min=1)
                    opt.zero_grad(); L.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), cfg["grad_clip"]); opt.step()
            # eval
            m.eval(); ys, ps, obs, patnos_te = [], [], [], []
            with torch.no_grad():
                for b in loader_te:
                    b = tl.move(b, tl.DEVICE); o = m(b)
                    ys.append(b["y_updrs"].cpu().numpy())
                    ps.append(o["gamma"].cpu().numpy())
                    obs.append(b["y_obs"].cpu().numpy())
                    patnos_te.append(b["patno"].cpu().numpy())
            ys = np.concatenate(ys); ps = np.concatenate(ps); obs_ar = np.concatenate(obs)
            patnos_ar = np.concatenate(patnos_te)
            cohort_ar = master.set_index("PATNO").loc[patnos_ar]["COHORT_LABEL"].values
            mm = obs_ar.astype(bool)
            r2_all = float(r2_score(ys[mm]*cfg["UPDRS_MAX"], ps[mm]*cfg["UPDRS_MAX"]))
            pd_m = mm & (cohort_ar == "PD")
            r2_pd = float(r2_score(ys[pd_m]*cfg["UPDRS_MAX"], ps[pd_m]*cfg["UPDRS_MAX"])) \
                    if pd_m.sum() >= 10 else float("nan")
            print(f"{tag}: R2_all={r2_all:.4f}  R2_PD={r2_pd:.4f}")
            return dict(R2_all=r2_all, R2_PD=r2_pd)

        results["B1"] = {**_train("B1 transformer only", None,    False, False, epochs), "name":"Transformer only"}
        results["B2"] = {**_train("B2 + fixed Braak",    "fixed", True,  False, epochs), "name":"+ fixed Braak"}
        results["B3"] = {**_train("B3 + PSBGL det",      "psbgl", False, False, epochs), "name":"+ PSBGL det"}
        results["B4"] = {**_train("B4 + GMP-PBG",        "psbgl", True,  True,  epochs), "name":"+ GMP-PBG"}

        # B5: full S-PBGL — use metrics from test_outputs
        mm = out["y_obs"].astype(bool)
        y = out["y_u3"] * cfg["UPDRS_MAX"]; p = out["gamma"] * cfg["UPDRS_MAX"]
        r2_all = float(r2_score(y[mm], p[mm]))
        # align cohort to the test_outputs order
        cohort_te = master.set_index("PATNO").loc[out["patno"]]["COHORT_LABEL"].values
        pd_m = mm & (cohort_te == "PD")
        r2_pd = float(r2_score(y[pd_m], p[pd_m])) if pd_m.sum() >= 10 else float("nan")
        results["B5"] = dict(name="Full v2.1 (S-PBGL)", R2_all=r2_all, R2_PD=r2_pd)

    (ANALYSIS / "block_09_ablation.json").write_text(json.dumps(results, indent=2, default=str))
    print("\n=== Block 09 — ablation ===")
    for k in sorted(results.keys()):
        v = results[k]
        print(f"  {k} {v['name']:25s}  R²_all={v['R2_all']:.4f}  R²_PD={v['R2_PD']:.4f}")
    return results

# ===================== Block 10 — Wilcoxon + bootstrap ================
def block_10_stats(out, xgb_pred_file, cfg):
    out_file = ANALYSIS / "block_10_stats.json"
    if not xgb_pred_file.exists():
        print("[block 10] skipped (no XGB preds yet — run --ablation)")
        return None
    xgb_d = np.load(xgb_pred_file, allow_pickle=True)
    y_all = xgb_d["y"]; p_xgb = xgb_d["p"]; obs_te = xgb_d["obs"]

    # Align test_outputs rows to xgb prediction rows via patno order
    # train_local writes test_outputs in loader_te order → same ordering as idx_te.
    # xgb preds are written in the same idx_te order → directly comparable.
    y_bf = out["y_u3"] * cfg["UPDRS_MAX"]
    p_bf = out["gamma"] * cfg["UPDRS_MAX"]
    mm = obs_te & out["y_obs"].astype(bool)

    err_bf  = np.abs(y_bf[mm] - p_bf[mm])
    err_xgb = np.abs(y_all[mm] - p_xgb[mm])
    if len(err_bf) < 30:
        print("[block 10] skipped (too few observed)"); return None
    W, p_w = wilcoxon(err_bf, err_xgb)

    rng = np.random.default_rng(tl.SEED)
    dr2 = []
    for _ in range(1000):
        idx = rng.choice(mm.sum(), size=mm.sum(), replace=True)
        dr2.append(r2_score(y_bf[mm][idx], p_bf[mm][idx])
                 - r2_score(y_all[mm][idx], p_xgb[mm][idx]))
    ci = np.percentile(dr2, [2.5, 50, 97.5])

    # DaTscan r bootstrap
    sbr_obs = out["sbr_obs"].astype(bool)
    if sbr_obs.sum() >= 30:
        rs = []
        x, y2 = out["y_sbr"][sbr_obs], out["sbr"][sbr_obs]
        for _ in range(1000):
            idx = rng.choice(len(x), size=len(x), replace=True)
            rs.append(pearsonr(x[idx], y2[idx])[0])
        ci_r = np.percentile(rs, [2.5, 50, 97.5])
    else:
        ci_r = [float("nan")] * 3

    res = dict(wilcoxon_W=float(W), wilcoxon_p=float(p_w),
               delta_R2_ci=[float(x) for x in ci],
               DaTscan_r_ci=[float(x) for x in ci_r])
    out_file.write_text(json.dumps(res, indent=2))
    print("\n=== Block 10 — stats ===")
    print(json.dumps(res, indent=2))
    return res

# ============================== Figures ==============================
def make_figures(out, u_epi, u_graph, u_pred, cluster_topo, cluster_topo_info,
                 subtype_labels, snc_row, lc_row, ablation, cfg,
                 master=None, A_braak=None, xgb_model=None, xgb_feat_names=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10})

    def sf(fig, name):
        fig.tight_layout()
        fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    gba = out["GBA"].astype(bool); saa = out["SAA"].astype(bool)
    mm = out["y_obs"].astype(bool)

    # -------- Fig 1 — cohort overview --------
    if master is not None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        master["COHORT_LABEL"].value_counts().plot.bar(ax=axes[0], color="steelblue")
        axes[0].set_title("Cohort composition"); axes[0].set_ylabel("subjects")
        sns.histplot(pd.to_numeric(master["UPDRS3_V06"], errors="coerce").dropna(),
                     ax=axes[1], bins=30, color="teal")
        axes[1].set_title("UPDRS-III @ V06 distribution"); axes[1].set_xlabel("UPDRS-III")
        sns.countplot(x="NSD_ISS_stage", data=master, ax=axes[2], color="darkorange")
        axes[2].set_title("NSD-ISS stage distribution")
        sf(fig, "fig01_cohort")

    # -------- Fig 2 — canonical Braak adjacency --------
    if A_braak is not None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        sns.heatmap(A_braak, xticklabels=NODE_NAMES, yticklabels=NODE_NAMES,
                    cmap="Blues", ax=ax, cbar_kws={"label": "edge"})
        ax.set_title("Canonical Braak adjacency (A_braak)")
        sf(fig, "fig02_Abraak")

    # -------- Fig 3 — main UPDRS scatter (remade with honest results) --------
    y = out["y_u3"] * cfg["UPDRS_MAX"]; p = out["gamma"] * cfg["UPDRS_MAX"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(y[mm], p[mm], c=u_epi[mm], cmap="viridis", s=10, alpha=0.6)
    lim = [0, cfg["UPDRS_MAX"]]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("True UPDRS-III @ V06"); ax.set_ylabel("Predicted γ")
    plt.colorbar(sc, ax=ax, label="epistemic u"); ax.set_title("UPDRS scatter")
    sf(fig, "fig03_scatter")

    # -------- Fig 5 — latent PCA: by subtype + by GBA --------
    try:
        pca = PCA(n_components=2).fit_transform(out["z"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        if subtype_labels is not None:
            for cid in sorted(set(subtype_labels)):
                ms = subtype_labels == cid
                axes[0].scatter(pca[ms, 0], pca[ms, 1], s=8, label=f"c{cid}", alpha=0.6)
            axes[0].legend(); axes[0].set_title("PCA of latent z by cluster")
        axes[1].scatter(pca[~gba, 0], pca[~gba, 1], s=8, c="#888", alpha=0.4, label="GBA-")
        axes[1].scatter(pca[gba, 0], pca[gba, 1], s=14, c="crimson", alpha=0.8, label="GBA+")
        axes[1].legend(); axes[1].set_title("PCA of latent z by GBA status")
        sf(fig, "fig05_pca_latent")
    except Exception as e:
        print(f"[fig5] skipped: {e}")

    # -------- Fig 6 — NSD-ISS confusion matrix --------
    try:
        stage_pred = out["stage"].argmax(axis=1)
        n_stg = int(max(out["y_stage"].max(), stage_pred.max())) + 1
        cm = confusion_matrix(out["y_stage"], stage_pred, labels=list(range(n_stg)))
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("NSD-ISS confusion matrix")
        sf(fig, "fig06_nsdiss_cm")
    except Exception as e:
        print(f"[fig6] skipped: {e}")

    # Fig 4 — MAE quintiles
    y_abs = np.abs(out["y_u3"] * cfg["UPDRS_MAX"] - out["gamma"] * cfg["UPDRS_MAX"])
    qs = np.quantile(u_epi[mm], [0.2, 0.4, 0.6, 0.8])
    mae_q = []
    for lo, hi in zip([-np.inf] + list(qs), list(qs) + [np.inf]):
        m = mm & (u_epi >= lo) & (u_epi < hi)
        mae_q.append(y_abs[m].mean() if m.sum() >= 5 else np.nan)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(5), mae_q, color="firebrick")
    ax.set_xticks(range(5)); ax.set_xticklabels(["Q1","Q2","Q3","Q4","Q5"])
    ax.set_ylabel("MAE (UPDRS units)")
    ax.set_title("MAE by epistemic quintile")
    sf(fig, "fig04_calibration")

    # Fig 7 — GBA stratified scatter + box
    y = out["y_u3"] * cfg["UPDRS_MAX"]; p = out["gamma"] * cfg["UPDRS_MAX"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y[mm & ~gba], p[mm & ~gba], s=10, c="#888", label="GBA-", alpha=0.5)
    axes[0].scatter(y[mm &  gba], p[mm &  gba], s=18, c="crimson", label="GBA+", alpha=0.8)
    axes[0].plot([0, 132], [0, 132], "k--", lw=1); axes[0].legend()
    axes[0].set_xlabel("true UPDRS-III"); axes[0].set_ylabel("pred γ")
    axes[0].set_title("UPDRS by GBA status")
    axes[1].boxplot([u_epi[~gba & mm], u_epi[gba & mm]], tick_labels=["GBA-", "GBA+"])
    axes[1].set_title("Epistemic uncertainty by GBA"); axes[1].set_ylabel("u_epi")
    sf(fig, "fig07_gba_stratified")

    # Fig 14 — SAA edges
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot([snc_row[~saa], snc_row[saa]], tick_labels=["SAA-", "SAA+"])
    axes[0].set_title("A_i at SNc by SAA"); axes[0].set_ylabel("mean edge weight")
    axes[1].boxplot([lc_row[~saa], lc_row[saa]], tick_labels=["SAA-", "SAA+"])
    axes[1].set_title("A_i at LC by SAA")
    sf(fig, "fig14_saa_edges")

    # Fig 15 — topology phenotype heatmaps
    if cluster_topo_info is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for cid in range(3):
            info = cluster_topo_info[f"cluster_{cid}"]
            sns.heatmap(np.array(info["mean_A"]),
                        xticklabels=NODE_NAMES, yticklabels=NODE_NAMES,
                        ax=axes[cid], cmap="Blues")
            axes[cid].set_title(f"cluster {cid}: {info['label']}  n={info['n']}")
        fig.suptitle("Topology phenotype mean adjacencies")
        sf(fig, "fig15_topology_phenotypes")

    # Fig 16 — u_graph vs u_pred by GBA
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(u_graph[~gba], u_pred[~gba], s=10, c="#888", alpha=0.5, label="GBA-")
    ax.scatter(u_graph[ gba], u_pred[ gba], s=18, c="crimson", alpha=0.8, label="GBA+")
    ax.set_xlabel("u_graph"); ax.set_ylabel("u_pred (NIG epistemic)")
    r, p = pearsonr(u_graph, u_pred)
    ax.legend(); ax.set_title(f"Two-source uncertainty  r={r:.2f}  p={p:.2e}")
    sf(fig, "fig16_two_source_uncertainty")

    # Fig 8 — ablation bars
    if ablation:
        order = [k for k in ["B0","B1","B2","B3","B4","B5"] if k in ablation]
        names = [ablation[k]["name"]  for k in order]
        r2s   = [ablation[k]["R2_PD"] for k in order]
        fig, ax = plt.subplots(figsize=(8, 4))
        cols = ["#bbb","#aab","#99b","#88b","#66a","#264"][:len(order)]
        bars = ax.bar(order, r2s, color=cols)
        ax.set_ylabel("R² (PD cohort)"); ax.set_title("Ablation")
        for b, n, r in zip(bars, names, r2s):
            ax.text(b.get_x() + b.get_width()/2, r + 0.005, f"{r:.3f}", ha="center", fontsize=8)
            ax.text(b.get_x() + b.get_width()/2, -0.05, n, ha="center", fontsize=7, rotation=20)
        sf(fig, "fig08_ablation")

    # Fig 13 — edge uncertainty atlas (already made by train_local but remake in case)
    pro_nc = (out["event_ind"].astype(bool) == False)  # non-event subjects
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (lbl, m) in zip(axes, [("GBA+", gba), ("GBA-", ~gba), ("Non-event", pro_nc)]):
        if m.sum() < 5:
            ax.set_title(f"{lbl}: n={m.sum()} sparse"); continue
        sns.heatmap(out["u_edge"][m].mean(axis=0),
                    xticklabels=NODE_NAMES, yticklabels=NODE_NAMES, ax=ax, cmap="magma")
        ax.set_title(f"{lbl}  n={m.sum()}")
    fig.suptitle("Edge uncertainty atlas (S-PBGL posterior)")
    sf(fig, "fig13_u_edge_atlas")

    # -------- Fig 9 — BSMTA per-head temporal-node queries --------
    try:
        best_ckpt = tl.CHECKPOINTS / "best.pt"
        final_ckpt = tl.CHECKPOINTS / "final.pt"
        ckpt_path = best_ckpt if best_ckpt.exists() else final_ckpt
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"]
            # First transformer layer's temporal-node queries (shape [H, T, n_nodes]).
            tnq = state.get("tmp.layers.0.tnq")
            if tnq is not None:
                Q = tnq.cpu().numpy()
                H, T, N_n = Q.shape
                visits = ["BL", "V02", "V04", "V06"][:T]
                fig, axes = plt.subplots(2, (H + 1) // 2, figsize=(3.5 * ((H + 1) // 2), 6))
                axes = axes.flatten()
                for h in range(H):
                    sns.heatmap(Q[h], yticklabels=visits, xticklabels=NODE_NAMES,
                                ax=axes[h], cmap="coolwarm", center=0, cbar=False)
                    axes[h].set_title(f"head {h}")
                for h in range(H, len(axes)): axes[h].axis("off")
                fig.suptitle("BSMTA temporal-node query heatmaps per head")
                sf(fig, "fig09_bsmta_heads")
    except Exception as e:
        print(f"[fig9] skipped: {e}")

    # -------- Fig 10 — XGBoost top-20 feature importance --------
    try:
        if xgb_model is not None:
            imp = xgb_model.feature_importances_
            if xgb_feat_names is None or len(xgb_feat_names) != len(imp):
                xgb_feat_names = [f"f{i}" for i in range(len(imp))]
            order20 = np.argsort(imp)[::-1][:20]
            fig, ax = plt.subplots(figsize=(7, 6))
            y_positions = np.arange(20)
            ax.barh(y_positions[::-1], imp[order20], color="teal")
            ax.set_yticks(y_positions[::-1])
            ax.set_yticklabels([xgb_feat_names[i] for i in order20], fontsize=8)
            ax.set_title("XGBoost top-20 feature importance")
            sf(fig, "fig10_xgb_importance")
    except Exception as e:
        print(f"[fig10] skipped: {e}")

    # -------- Fig 11 — NHH survival curves by GBA × baseline severity --------
    try:
        haz_prob = 1.0 / (1.0 + np.exp(-out["hazard"]))
        S = np.cumprod(1.0 - haz_prob, axis=1)                                # [N, HAZARD_BINS]
        t_axis = np.array([0, 12, 24, 36])[: S.shape[1]]
        if master is not None:
            bl_map = dict(zip(master["PATNO"].values, master["UPDRS3_BL"].values))
            BL = np.array([bl_map.get(int(p), np.nan) for p in out["patno"]])
        else:
            BL = np.full(len(out["patno"]), np.nan)
        BL_hi = BL > np.nanmedian(BL)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for label, m in [
            ("GBA- / BL-low",  ~gba & ~BL_hi),
            ("GBA- / BL-high", ~gba &  BL_hi),
            ("GBA+ / BL-low",   gba & ~BL_hi),
            ("GBA+ / BL-high",  gba &  BL_hi),
        ]:
            if m.sum() < 5: continue
            curve = S[m].mean(axis=0)
            ax.plot(t_axis, curve, label=f"{label}  (n={int(m.sum())})", marker="o")
        ax.set_xlabel("Months from baseline")
        ax.set_ylabel("S(t)  —  P(not reached UPDRS-III ≥ 40)")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
        ax.set_title("NHH survival curves by GBA × baseline UPDRS severity")
        sf(fig, "fig11_survival")
    except Exception as e:
        print(f"[fig11] skipped: {e}")

    print(f"\nFigures saved in {FIGURES}")

# ================================= Main ================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true", help="also train B0..B4")
    ap.add_argument("--ablation_epochs", type=int, default=40)
    args = ap.parse_args()

    print(f"Device: {tl.DEVICE}")
    t0 = time.time()
    out = load_test_outputs()
    print(f"Loaded test_outputs: {len(out['patno'])} patients")

    master, splits, A_braak, extra = tl.load_and_build()
    idx_te = np.where(np.isin(master["PATNO"].values, splits[2]))[0]

    u_epi, u_ale, cal = block_08_calibration(out, master, idx_te, tl.CFG)
    gba_res, saa_res, snc_row, lc_row = block_1112_strata(out, u_epi, tl.CFG)
    cluster_topo, cluster_topo_info = block_13_topology(out, tl.CFG)
    u_graph, u_pred, ts_res = block_14_two_source(out, u_epi, tl.CFG)
    restricted_c            = block_14b_restricted_cindex(out, master, tl.CFG)
    subtypes, subtype_res   = block_15_subtypes(out, tl.CFG)

    ablation_res = {}
    if args.ablation:
        ablation_res = block_09_ablation(master, splits, A_braak, extra, out, tl.CFG,
                                         epochs=args.ablation_epochs)
        block_10_stats(out, ANALYSIS / "xgb_b0_preds.npz", tl.CFG)

    # Reload XGBoost artefacts for Fig 10 (feature importance).
    xgb_model = None; xgb_feat_names = None
    xgb_pkl = ANALYSIS / "xgb_b0.pkl"
    if xgb_pkl.exists():
        try:
            with open(xgb_pkl, "rb") as f: xgb_model = pickle.load(f)
            cth_cols, aseg_cols, sbr_sub_cols = extra
            CLINICAL = ["AGE","SEX_binary","UPDRS1_BL","UPDRS2_BL","UPDRS3_BL","MOCA_BL"]
            GENETIC  = ["GBA_binary","SAA_positive"]
            IMAGING  = sbr_sub_cols + ["SBR_BL","CAUDATE_mean","PUTAMEN_mean"] + cth_cols + aseg_cols
            BIO      = []
            def _miss_names(names): return [f"{n}__miss" for n in names]
            feat_names = (CLINICAL + _miss_names(CLINICAL)
                        + GENETIC  + _miss_names(GENETIC)
                        + IMAGING  + _miss_names(IMAGING)
                        + BIO      + _miss_names(BIO)
                        + [f"ltp_{i}" for i in range(15)])
            xgb_feat_names = feat_names
        except Exception as e:
            print(f"[xgb reload] {e}")

    make_figures(out, u_epi, u_graph, u_pred, cluster_topo, cluster_topo_info,
                 subtypes, snc_row, lc_row, ablation_res, tl.CFG,
                 master=master, A_braak=A_braak,
                 xgb_model=xgb_model, xgb_feat_names=xgb_feat_names)

    summary = {}
    for p in sorted(ANALYSIS.glob("*.json")):
        try: summary[p.stem] = json.loads(p.read_text())
        except Exception: pass
    metrics_file = RESULTS / "metrics.json"
    if metrics_file.exists():
        summary["train_local_metrics"] = json.loads(metrics_file.read_text())
    (RESULTS / "main_results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nAll analyses in {ANALYSIS}, consolidated in {RESULTS / 'main_results.json'}")
    print(f"Wall-clock: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
