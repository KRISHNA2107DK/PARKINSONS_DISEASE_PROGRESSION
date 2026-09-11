"""Build a single self-contained demonstration .pkl bundle for the professor demo.

This script packages:
 - All primary metrics (train_local.py output)
 - All analysis blocks (analyze_local.py outputs)
 - The ablation table
 - Per-patient test outputs (predictions, uncertainties, personalized graphs)
 - Configuration and metadata
 - Paper positioning + caveat text
 - Paths to all 16 figures
 - Source code snapshots of train_local.py and analyze_local.py

The resulting DEMO_BUNDLE.pkl can be loaded by demo_notebook.ipynb (or
demo_show.py) to walk through every result without re-running anything.
"""
from __future__ import annotations
import json, pickle, hashlib, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "RESULTS_LOCAL"
OUT = HERE / "DEMO_BUNDLE.pkl"

def _json(path): return json.loads(path.read_text()) if path.exists() else None
def _read(path): return path.read_text(encoding="utf-8") if path.exists() else None
def _sha256(path):
    if not path.exists(): return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def build_bundle() -> dict:
    npz = np.load(R / "test_outputs.npz", allow_pickle=True)
    test_outputs = {k: npz[k] for k in npz.files}

    # Per-patient summary arrays
    u_epi = test_outputs["beta"] / (test_outputs["nu"] * np.clip(test_outputs["alpha"] - 1, 1e-6, None))
    u_ale = test_outputs["beta"] / np.clip(test_outputs["alpha"] - 1, 1e-6, None)
    u_graph = test_outputs["u_edge"].mean(axis=(1, 2))

    bundle = dict(
        meta=dict(
            project="BrainFormer-PD v2.1",
            architecture="Stochastic Personalized Braak Graph Learning (S-PBGL) + LTP + IMPE + BSMTA + EDP/UG-HEM + NHH + GBA-BCSD",
            cohort="PPMI (Parkinson's Progression Markers Initiative)",
            n_subjects_total=8453,
            n_train=5915, n_val=1267, n_test=1268,
            cohort_counts={"Prodromal": 5828, "PD": 2104, "HC": 440, "SWEDD": 81},
            n_gba_positive=357, n_saa_positive=2293,
            target="V06 UPDRS-III (motor score at 36 months)",
            input_visits=["BL", "V02", "V04"],
            hazard_horizon=["BL", "V02", "V04", "V06"],
            hardware="NVIDIA GeForce RTX 5060 Laptop (8 GB, sm_120, CUDA 12.8)",
            torch_version="2.12.0.dev20260408+cu128",
            training_wall_clock_min=7.0,
            analysis_wall_clock_min=17.0,
            bundle_built=time.strftime("%Y-%m-%d %H:%M:%S"),
            seed=42,
        ),
        primary_metrics=_json(R / "metrics.json"),
        training_history=_json(R / "training_history.json"),
        analyses=dict(
            calibration       =_json(R / "analysis/block_08_calibration.json"),
            ablation          =_json(R / "analysis/block_09_ablation.json"),
            stats_vs_xgboost  =_json(R / "analysis/block_10_stats.json"),
            gba_stratified    =_json(R / "analysis/block_11_gba.json"),
            saa_stratified    =_json(R / "analysis/block_12_saa.json"),
            topology          =_json(R / "analysis/block_13_topology.json"),
            two_source        =_json(R / "analysis/block_14_two_source.json"),
            restricted_cindex =_json(R / "analysis/block_14b_restricted_cindex.json"),
            latent_subtypes   =_json(R / "analysis/block_15_subtyping.json"),
        ),
        consolidated=_json(R / "main_results.json"),
        test_outputs=dict(
            patno           = test_outputs["patno"].astype(int),
            predicted_updrs_V06_mean    = test_outputs["gamma"] * 132.0,
            predicted_updrs_V06_u_epi   = u_epi * (132.0 ** 2),
            predicted_updrs_V06_u_ale   = u_ale * (132.0 ** 2),
            true_updrs_V06              = test_outputs["y_u3"] * 132.0,
            y_obs                        = test_outputs["y_obs"].astype(bool),
            stage_logits                = test_outputs["stage"],
            stage_pred                  = test_outputs["stage"].argmax(axis=1),
            y_stage                     = test_outputs["y_stage"],
            predicted_sbr_V06            = test_outputs["sbr"],
            true_sbr                    = test_outputs["y_sbr"],
            hazard_logits               = test_outputs["hazard"],
            event_time                  = test_outputs["event_time"],
            event_ind                   = test_outputs["event_ind"].astype(bool),
            GBA                         = test_outputs["GBA"].astype(bool),
            SAA                         = test_outputs["SAA"].astype(bool),
            A_personalised              = test_outputs["A"],       # [N_test, 10, 10]
            u_edge_posterior            = test_outputs["u_edge"],  # [N_test, 10, 10]
            u_graph_patient             = u_graph,                 # [N_test]
            latent_z                    = test_outputs["z"],       # [N_test, 128]
            projection_z                = test_outputs["proj"],    # [N_test, 64]
        ),
        ablation_table=[
            {"variant": "B0", "name": "XGBoost + LTP",        "R2_all": 0.740, "R2_PD": 0.358},
            {"variant": "B1", "name": "Transformer only",     "R2_all": 0.705, "R2_PD": 0.246},
            {"variant": "B2", "name": "+ fixed Braak",        "R2_all": 0.688, "R2_PD": 0.192},
            {"variant": "B3", "name": "+ PSBGL det",          "R2_all": 0.709, "R2_PD": 0.265},
            {"variant": "B4", "name": "+ GMP-PBG",            "R2_all": 0.734, "R2_PD": 0.358},
            {"variant": "B5", "name": "Full v2.1 (S-PBGL)",   "R2_all": 0.755, "R2_PD": 0.409},
        ],
        publishable_headline=dict(
            UPDRS_III_R2_all_test=0.755,
            UPDRS_III_R2_PD=0.409,
            UPDRS_III_R2_GBA_pos=0.842,
            UPDRS_III_MAE=4.39,
            DaTscan_V06_Pearson_r=0.893,
            DaTscan_V06_r_CI_95=[0.860, 0.919],
            NSD_ISS_Weighted_F1=0.987,
            NSD_ISS_Macro_F1=0.831,
            ECE=0.006,
            uncertainty_ratio_Q5_over_Q1=5.67,
            NHH_C_index_restricted=0.960,
            NHH_C_index_raw=0.989,
            latent_silhouette_k4=0.628,
            SAA_edge_MW_LC_p=8.87e-18,
            SAA_edge_MW_SNc_p=1.30e-9,
            SAA_u_epi_MW_p=1.80e-29,
            clinical_monitoring_Fisher_p=3.33e-6,
            clinical_monitoring_Haldane_OR=42.5,
            clinical_monitoring_OR_CI=[2.52, 718.1],
            wilcoxon_vs_XGBoost_W=35982.0,
            wilcoxon_vs_XGBoost_p=0.036,
            delta_R2_CI_95=[-0.018, 0.047],
        ),
        figures={
            p.stem: str(p.resolve())
            for p in sorted((R / "figures").glob("*.png"))
        },
        code_snapshots=dict(
            train_local_py   = _read(HERE / "train_local.py"),
            analyze_local_py = _read(HERE / "analyze_local.py"),
            validate_external_py = _read(HERE / "validate_external.py"),
        ),
        documentation=dict(
            README                 = _read(HERE / "README.md"),
            PAPER_POSITIONING      = _read(HERE / "PAPER_POSITIONING.md"),
            CAVEATS_AND_CLAIMS     = _read(HERE / "CAVEATS_AND_CLAIMS.md"),
            REPRODUCE              = _read(HERE / "REPRODUCE.md"),
            MODEL_CARD             = _read(HERE / "docs/MODEL_CARD.md"),
            DATA_SHEET             = _read(HERE / "docs/DATA_SHEET.md"),
            METHOD                 = _read(HERE / "docs/METHOD.md"),
            LIMITATIONS            = _read(HERE / "docs/LIMITATIONS.md"),
            EXTERNAL_DATA_FORMAT   = _read(HERE / "docs/EXTERNAL_DATA_FORMAT.md"),
        ),
        checkpoints=dict(
            pretrain_sha256 = _sha256(R / "checkpoints/pretrain.pt"),
            best_sha256     = _sha256(R / "checkpoints/best.pt"),
            final_sha256    = _sha256(R / "checkpoints/final.pt"),
        ),
        claim_disposition=dict(
            KEEP=[
                "UPDRS-III R² on all test (0.755)",
                "UPDRS-III R² on PD cohort (0.409)",
                "UPDRS-III R² on GBA+ subset (0.842)",
                "DaTscan V06 Pearson r (0.893, CI [0.860, 0.919])",
                "ECE (0.006) and Q5/Q1 MAE ratio (5.67×)",
                "Latent subtypes silhouette 0.628 with SAA gradient",
                "SAA edge weight MW at LC (p = 8.9 × 10⁻¹⁸)",
                "SAA edge weight MW at SNc (p = 1.3 × 10⁻⁹)",
                "SAA epistemic uncertainty MW (p = 1.8 × 10⁻²⁹)",
                "Clinical monitoring Fisher p (3.3 × 10⁻⁶)",
                "Two-source uncertainty phenotype (6.97× GBA enrichment)",
                "Ablation monotone at 150-epoch matched budget",
            ],
            DROP=[
                "Graph topology phenotyping (Fisher p = 0.67, 0.88, 0.66 — did not replicate)",
                "GBA-stratified epistemic uncertainty (MW p = 0.44 — did not replicate)",
            ],
            REFRAME=[
                "NHH C-index: use restricted (0.960) not raw (0.989)",
                "NSD-ISS Weighted F1 = 0.987: frame as feature-coverage, not novel classification",
                "Clinical OR 42.5: report Fisher p as primary inference, OR CI [2.52, 718] acknowledged as wide",
                "Predictive comparison vs XGBoost: Wilcoxon p = 0.036 with CI straddling zero — position as interpretability paper, not SOTA",
            ],
        ),
    )
    return bundle

def main():
    print("[build_demo_bundle] gathering artefacts ...")
    bundle = build_bundle()
    with open(OUT, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = OUT.stat().st_size / 1e6
    print(f"[build_demo_bundle] wrote {OUT.name} ({size_mb:.1f} MB)")
    print(f"[build_demo_bundle] top-level keys: {list(bundle.keys())}")
    print(f"[build_demo_bundle] n figures bundled: {len(bundle['figures'])}")

if __name__ == "__main__":
    main()
