"""evaluate.py — Floor-Wall sweep on Fontaine IIoT19 + Complex I/Q DW3000 & IIoT20 checks.

Evaluates whether injecting a delayed LOS copy at Δt ∈ [10, 60] ns capped at
+0, +3, +6, +9, +12 dB while leaving the LOS first path completely untouched can:
  1. Shift UWB ML ranging-bias regressors by > 0.5 m (indirect distance reduction), AND/OR
  2. Flip UWB NLOS classifiers from LOS to NLOS (stealthy anchor exclusion).

Outputs:
  - results/floor_wall_summary.csv     (Fontaine IIoT19 full sweep: Regimes A & B + analytical)
  - results/dw3000_complex_summary.csv (Santamaria DW3000 complex-IQ 5-phase avg & phi=0)
  - results/bias_shift_vs_power.png
  - results/nlos_flip_rate_vs_power.png
  - results/sample_cir_overlay.png
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (CNN_WIN_LEN, extract_cir_window, load_fontaine_cirs,
                     load_fontaine_iiot20, load_fontaine_scalar,
                     load_santamaria_dw3000)
from features import FEATURE_NAMES, TAP_NS, compute_feature_matrix, compute_features_from_cir
from injector import inject_phantom_nlos, inject_with_phase_averaging
from models import (AECNNRegressor, APS006Classifier, CNN1DClassifier,
                    CNN1DRegressor, CIRAutoencoder, SCALAR_FEATURE_COLS)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

SEED = 42
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
DEVICE = "cpu"

DELTA_T_NS = [10, 20, 35, 50, 60]
POWER_DB   = [0, 3, 6, 9, 12]


# ============================================================
#  Helpers
# ============================================================

def load_pickle(name: str):
    with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def load_torch(cls, name: str, **kwargs):
    model = cls(**kwargs)
    model.load_state_dict(
        torch.load(MODELS_DIR / f"{name}.pt", map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _norm_windows(cirs_mag: np.ndarray, fp_arr: np.ndarray, regime: str, scale_A: float) -> np.ndarray:
    wins = np.stack([
        extract_cir_window(cirs_mag[i], int(fp_arr[i]))
        for i in range(len(cirs_mag))
    ]).astype(np.float32)
    if regime == "A":
        return (wins / np.float32(max(scale_A, 1e-12))).astype(np.float32)
    mx = np.maximum(wins.max(axis=1, keepdims=True), 1e-12)
    return (wins / mx).astype(np.float32)


def simulate_injection_on_features(row: pd.Series, power_db: float) -> dict:
    """Analytical scalar approximation used only as CSV baseline reference."""
    scale = 10.0 ** (power_db / 20.0)
    fp_p = float(row["FP_power"])
    rx_p = float(row["RX_power"])
    cir_p = float(row["CIRpower"])
    fp_lin = 10.0 ** (fp_p / 10.0)
    rx_lin = 10.0 ** (rx_p / 10.0)
    rx_new = rx_lin + fp_lin * (scale ** 2)
    rx_p_new = 10.0 * np.log10(max(rx_new, 1e-300))

    new_row = row.copy()
    new_row["RX_power"] = rx_p_new
    new_row["CIRpower"] = cir_p * (rx_new / max(rx_lin, 1e-300))
    new_row["Pd"]       = rx_p_new - fp_p
    new_row["Pr"]       = float(row["Pr"]) + (rx_p_new - rx_p)
    return {col: float(new_row[col]) for col in SCALAR_FEATURE_COLS}


# ============================================================
#  Santamaria DW3000 & IIoT20 Complex-IQ Evaluation
# ============================================================

def evaluate_dw3000_complex(
    aps006: APS006Classifier,
    lgbm_cls_cir,
    cnn_cls_A,
    cnn_cls_B,
    scale_A: float,
    fontaine_los_med_peak: float,
) -> pd.DataFrame | None:
    """Test real DW3000 complex-I/Q CIRs under 5 random phase draws + phi=0.

    Uses calibration_anc0/anchors (all LOS) + clean APS006-LOS links from
    lab/corridor (RXP_minus_FPP_dB < 6.0). Never touches DW3000 range errors.
    """
    recs = []
    for scen in ["calibration_anc0", "calibration_anchors", "lab", "corridor"]:
        for r in load_santamaria_dw3000(scen):
            r["scenario"] = scen
            recs.append(r)
    if not recs:
        logger.warning("No Santamaria DW3000 records found; skipping complex-IQ DW3000 test.")
        return None

    # Compute baseline clean APS006 features from raw complex magnitude
    clean_los_recs = []
    for r in recs:
        f = compute_features_from_cir(r["cir_mag"], r["fpi"])
        # Calibration captures are physically LOS (<10 dB); lab/corridor filter <6 dB
        thr = 10.0 if r["scenario"].startswith("calibration") else 6.0
        if f["RXP_minus_FPP_dB"] < thr:
            r["clean_feats"] = f
            clean_los_recs.append(r)

    if len(clean_los_recs) == 0:
        return None

    # Deterministic subsample of up to 250 clean DW3000 LOS CIRs across scenarios
    rng = np.random.RandomState(SEED)
    if len(clean_los_recs) > 250:
        idx_sub = np.sort(rng.choice(len(clean_los_recs), size=250, replace=False))
        clean_los_recs = [clean_los_recs[i] for i in idx_sub]

    n_dw = len(clean_los_recs)
    # Match PAC scale ratio between DW3000 (N-divided) and Fontaine Regime A
    dw_pac_med = float(np.median([r["cir_mag_pac"].max() for r in clean_los_recs]))
    dw_to_fontaine = fontaine_los_med_peak / max(dw_pac_med, 1e-6)
    dw_scale_A = scale_A / max(dw_to_fontaine, 1e-6)

    logger.info(
        "=== Santamaria DW3000 Complex-IQ Evaluation (%d clean LOS CIRs across scenarios; "
        "no ranging-error regression!) ===",
        n_dw,
    )

    clean_fmat = np.array([
        [compute_features_from_cir(r["cir_mag_pac"] * dw_to_fontaine, r["fpi"])[k]
         for k in FEATURE_NAMES]
        for r in clean_los_recs
    ], dtype=np.float32)
    clean_pd = clean_fmat[:, FEATURE_NAMES.index("RXP_minus_FPP_dB")]
    clean_aps10 = aps006.predict(clean_pd)
    clean_aps6  = aps006.predict_flagged(clean_pd)

    # Clean CNN-Cls & LGBM-Cls baselines on DW3000 (domain-transferred classifiers)
    clean_lgbm_p = lgbm_cls_cir.predict(clean_fmat) if lgbm_cls_cir is not None else None
    dw_pac_mags = np.stack([r["cir_mag_pac"] for r in clean_los_recs])
    dw_fp_arr   = np.array([r["fpi"] for r in clean_los_recs], dtype=int)
    wA_clean = _norm_windows(dw_pac_mags, dw_fp_arr, "A", dw_scale_A)
    wB_clean = _norm_windows(dw_pac_mags, dw_fp_arr, "B", 1.0)
    with torch.no_grad():
        clean_cnnA_p = _sigmoid(cnn_cls_A(torch.tensor(wA_clean).unsqueeze(1)).numpy()) if cnn_cls_A else None
        clean_cnnB_p = _sigmoid(cnn_cls_B(torch.tensor(wB_clean).unsqueeze(1)).numpy()) if cnn_cls_B else None

    rows = []
    for dt in DELTA_T_NS:
        for pdb in POWER_DB:
            # 5 random phase draws averaged per sample + worst-case constructive phi=0
            draw_aps10 = np.zeros((5, n_dw))
            draw_aps6  = np.zeros((5, n_dw))
            draw_lgbm  = np.zeros((5, n_dw))
            draw_cnnA  = np.zeros((5, n_dw))
            draw_cnnB  = np.zeros((5, n_dw))

            worst_fmat = []
            per_draw_mags = [[] for _ in range(5)]
            for i, r in enumerate(clean_los_recs):
                _, mag_w, mags5 = inject_with_phase_averaging(
                    r["cir_complex"], r["fpi"], dt, pdb, n_phase_draws=5, seed=SEED + i,
                )
                fw = compute_features_from_cir((mag_w / r["N"]) * dw_to_fontaine, r["fpi"])
                worst_fmat.append([fw[k] for k in FEATURE_NAMES])
                for d_idx in range(5):
                    per_draw_mags[d_idx].append(mags5[d_idx])

            for d_idx in range(5):
                mags_d = per_draw_mags[d_idx]
                fmat_d = np.array([
                    [compute_features_from_cir((mags_d[i] / clean_los_recs[i]["N"]) * dw_to_fontaine,
                                               clean_los_recs[i]["fpi"])[k]
                     for k in FEATURE_NAMES]
                    for i in range(n_dw)
                ], dtype=np.float32)
                pd_d = fmat_d[:, FEATURE_NAMES.index("RXP_minus_FPP_dB")]
                draw_aps10[d_idx] = (clean_aps10 == 0) & (aps006.predict(pd_d) == 1)
                draw_aps6[d_idx]  = (clean_aps6 == 0)  & (aps006.predict_flagged(pd_d) == 1)
                if clean_lgbm_p is not None:
                    draw_lgbm[d_idx] = (clean_lgbm_p < 0.5) & (lgbm_cls_cir.predict(fmat_d) >= 0.5)
                pac_d = np.stack([mags_d[i] / clean_los_recs[i]["N"] for i in range(n_dw)])
                if clean_cnnA_p is not None:
                    wA = _norm_windows(pac_d, dw_fp_arr, "A", dw_scale_A)
                    with torch.no_grad():
                        pA = _sigmoid(cnn_cls_A(torch.tensor(wA).unsqueeze(1)).numpy())
                    draw_cnnA[d_idx] = (clean_cnnA_p < 0.5) & (pA >= 0.5)
                if clean_cnnB_p is not None:
                    wB = _norm_windows(pac_d, dw_fp_arr, "B", 1.0)
                    with torch.no_grad():
                        pB = _sigmoid(cnn_cls_B(torch.tensor(wB).unsqueeze(1)).numpy())
                    draw_cnnB[d_idx] = (clean_cnnB_p < 0.5) & (pB >= 0.5)

            worst_fmat_arr = np.array(worst_fmat, dtype=np.float32)
            worst_pd = worst_fmat_arr[:, FEATURE_NAMES.index("RXP_minus_FPP_dB")]
            row = {
                "delta_t_ns": dt,
                "power_dB": pdb,
                "flip_dw3000_aps10_phase5_avg": float(draw_aps10.mean()),
                "flip_dw3000_aps6_phase5_avg":  float(draw_aps6.mean()),
                "flip_dw3000_aps10_phi0":       float(np.mean((clean_aps10 == 0) & (aps006.predict(worst_pd) == 1))),
                "flip_dw3000_aps6_phi0":        float(np.mean((clean_aps6 == 0)  & (aps006.predict_flagged(worst_pd) == 1))),
                "flip_dw3000_lgbm_phase5_avg":  float(draw_lgbm.mean()) if clean_lgbm_p is not None else np.nan,
                "flip_dw3000_cnnA_phase5_avg":  float(draw_cnnA.mean()) if clean_cnnA_p is not None else np.nan,
                "flip_dw3000_cnnB_phase5_avg":  float(draw_cnnB.mean()) if clean_cnnB_p is not None else np.nan,
            }
            rows.append(row)

    df_dw = pd.DataFrame(rows)
    out_p = RESULTS_DIR / "dw3000_complex_summary.csv"
    df_dw.to_csv(out_p, index=False, float_format="%.6f")
    logger.info("Saved %s (%d rows)", out_p, len(df_dw))
    return df_dw


# ============================================================
#  Main evaluation
# ============================================================

def main() -> int:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- Load Fontaine IIoT19 data & splits ----
    logger.info("=== Loading Fontaine IIoT19 data & splits ===")
    df = load_fontaine_scalar()
    splits = load_pickle("split_indices")
    te_idx = splits["test"]
    train_meta = load_pickle("train_meta")
    have_cirs = bool(train_meta.get("have_cirs", False))
    scale_A = float(train_meta.get("scale_A", 1.0))

    te_df = df.iloc[te_idx].copy()
    te_df["_orig_idx"] = te_idx
    los_mask = te_df["label"] == 0
    los_df = te_df[los_mask].copy()
    los_orig_idx = te_df[los_mask]["_orig_idx"].values
    n_los = len(los_df)
    logger.info("Held-out test set: %d total (%d LOS, %d NLOS)",
                len(te_df), n_los, int((~los_mask).sum()))

    # ---- Load CSV baseline models ----
    lgbm_reg_csv = load_pickle("lgbm_reg_csv")
    lgbm_cls_csv = load_pickle("lgbm_cls_csv")
    aps006       = APS006Classifier()

    X_los_csv = los_df[SCALAR_FEATURE_COLS].values.astype(np.float32)
    clean_reg_pred_csv = lgbm_reg_csv.predict(X_los_csv)
    clean_cls_pred_csv = lgbm_cls_csv.predict(X_los_csv)
    clean_aps_csv_10   = aps006.predict(los_df["Pd"].values)
    clean_aps_csv_6    = aps006.predict_flagged(los_df["Pd"].values)

    # ---- Load CIR-based models ----
    cirs, fp_cir_all, is_complex = load_fontaine_cirs(key="IIoT19", n_expected=len(df)) if have_cirs else (None, None, False)
    lgbm_reg_cir = lgbm_cls_cir = None
    cnn_models: dict[str, torch.nn.Module] = {}

    if cirs is not None:
        cir_mags = np.abs(cirs)
        los_cirs = cirs[los_orig_idx]
        los_cir_mags = cir_mags[los_orig_idx]
        los_fp_arr = fp_cir_all[los_orig_idx]
        fontaine_los_med_peak = float(np.median(los_cir_mags.max(axis=1)))

        lgbm_reg_cir = load_pickle("lgbm_reg_cir")
        lgbm_cls_cir = load_pickle("lgbm_cls_cir")
        for regime in ["A", "B"]:
            cnn_models[f"cnn_reg_{regime}"] = load_torch(CNN1DRegressor, f"cnn_reg_{regime}", input_len=CNN_WIN_LEN)
            cnn_models[f"cnn_cls_{regime}"] = load_torch(CNN1DClassifier, f"cnn_cls_{regime}", input_len=CNN_WIN_LEN)
            ae = load_torch(CIRAutoencoder, f"ae_{regime}", input_len=CNN_WIN_LEN)
            ae_reg = AECNNRegressor(ae.encoder)
            ae_reg.load_state_dict(torch.load(MODELS_DIR / f"ae_reg_{regime}.pt", map_location=DEVICE, weights_only=True))
            ae_reg.eval()
            cnn_models[f"ae_reg_{regime}"] = ae_reg

        # Clean baselines on LOS test samples (computed ONCE)
        feat_los_clean = compute_feature_matrix(los_cir_mags, los_fp_arr)
        clean_reg_cir_pred = lgbm_reg_cir.predict(feat_los_clean)
        clean_cls_cir_pred = lgbm_cls_cir.predict(feat_los_clean)
        rxp_fpp_clean = feat_los_clean[:, FEATURE_NAMES.index("RXP_minus_FPP_dB")]
        clean_aps_cir_10 = aps006.predict(rxp_fpp_clean)
        clean_aps_cir_6  = aps006.predict_flagged(rxp_fpp_clean)

        clean_cnn_preds = {}
        for regime in ["A", "B"]:
            w_clean = _norm_windows(los_cir_mags, los_fp_arr, regime, scale_A)
            xt = torch.tensor(w_clean).unsqueeze(1)
            for mname, model in cnn_models.items():
                if mname.endswith(f"_{regime}"):
                    with torch.no_grad():
                        clean_cnn_preds[mname] = model(xt).numpy()
    else:
        fontaine_los_med_peak = 15.0

    # ---- Floor-Wall Sweep on Fontaine IIoT19 ----
    logger.info("=== Running Floor-Wall Sweep on %d clean LOS test CIRs ===", n_los)
    results = []

    for dt in DELTA_T_NS:
        for pdb in POWER_DB:
            # 1) Analytical CSV reference
            X_atk_csv = np.array([
                [simulate_injection_on_features(row, pdb)[c] for c in SCALAR_FEATURE_COLS]
                for _, row in los_df.iterrows()
            ], dtype=np.float32)
            atk_reg_csv = lgbm_reg_csv.predict(X_atk_csv)
            atk_cls_csv = lgbm_cls_csv.predict(X_atk_csv)
            atk_pd_csv  = X_atk_csv[:, SCALAR_FEATURE_COLS.index("Pd")]
            db_csv = (atk_reg_csv - clean_reg_pred_csv) / 1000.0
            results.append({
                "delta_t_ns": dt, "power_dB": pdb, "regime": "analytical",
                "mean_delta_b_m_lgbm_csv":   float(np.mean(db_csv)),
                "median_delta_b_m_lgbm_csv": float(np.median(db_csv)),
                "p90_delta_b_m_lgbm_csv":    float(np.percentile(db_csv, 90)),
                "ASR_020m_lgbm_csv":         float(np.mean(db_csv > 0.20)),
                "ASR_050m_lgbm_csv":         float(np.mean(db_csv > 0.50)),
                "ASR_100m_lgbm_csv":         float(np.mean(db_csv > 1.00)),
                "flip_aps006_10dB":          float(np.mean((clean_aps_csv_10 == 0) & (aps006.predict(atk_pd_csv) == 1))),
                "flip_aps006_6dB":           float(np.mean((clean_aps_csv_6 == 0)  & (aps006.predict_flagged(atk_pd_csv) == 1))),
                "flip_lgbm_cls_csv":         float(np.mean((clean_cls_pred_csv < 0.5) & (atk_cls_csv >= 0.5))),
            })

            if cirs is None:
                continue

            # 2) Physics-accurate waveform injection ONCE per (dt, pdb)
            atk_mags = np.stack([
                np.abs(inject_phantom_nlos(
                    los_cirs[i], int(los_fp_arr[i]), dt, pdb,
                    is_complex=is_complex, phi=0.0 if is_complex else None,
                ))
                for i in range(n_los)
            ])
            atk_fmat = compute_feature_matrix(atk_mags, los_fp_arr)
            atk_reg_cir = lgbm_reg_cir.predict(atk_fmat)
            atk_cls_cir = lgbm_cls_cir.predict(atk_fmat)
            db_lgbm_cir = (atk_reg_cir - clean_reg_cir_pred) / 1000.0
            rxp_fpp_atk = atk_fmat[:, FEATURE_NAMES.index("RXP_minus_FPP_dB")]

            for regime in ["A", "B"]:
                atk_windows = _norm_windows(atk_mags, los_fp_arr, regime, scale_A)
                xt_atk = torch.tensor(atk_windows).unsqueeze(1)

                r = {
                    "delta_t_ns": dt,
                    "power_dB": pdb,
                    "regime": regime,
                    "mean_delta_b_m_lgbm_cir":   float(np.mean(db_lgbm_cir)),
                    "median_delta_b_m_lgbm_cir": float(np.median(db_lgbm_cir)),
                    "p90_delta_b_m_lgbm_cir":    float(np.percentile(db_lgbm_cir, 90)),
                    "ASR_020m_lgbm_cir":         float(np.mean(db_lgbm_cir > 0.20)),
                    "ASR_050m_lgbm_cir":         float(np.mean(db_lgbm_cir > 0.50)),
                    "ASR_100m_lgbm_cir":         float(np.mean(db_lgbm_cir > 1.00)),
                    "flip_lgbm_cls_cir":         float(np.mean((clean_cls_cir_pred < 0.5) & (atk_cls_cir >= 0.5))),
                    "flip_aps006_cir_10dB":      float(np.mean((clean_aps_cir_10 == 0) & (aps006.predict(rxp_fpp_atk) == 1))),
                    "flip_aps006_cir_6dB":       float(np.mean((clean_aps_cir_6 == 0)  & (aps006.predict_flagged(rxp_fpp_atk) == 1))),
                }

                for base_name in ["cnn_reg", "cnn_cls", "ae_reg"]:
                    mname = f"{base_name}_{regime}"
                    model = cnn_models[mname]
                    with torch.no_grad():
                        p_atk = model(xt_atk).numpy()
                    p_clean = clean_cnn_preds[mname]
                    if "reg" in base_name:
                        db = p_atk - p_clean
                        r[f"mean_delta_b_m_{base_name}"]   = float(np.mean(db))
                        r[f"median_delta_b_m_{base_name}"] = float(np.median(db))
                        r[f"p90_delta_b_m_{base_name}"]    = float(np.percentile(db, 90))
                        r[f"ASR_020m_{base_name}"]         = float(np.mean(db > 0.20))
                        r[f"ASR_050m_{base_name}"]         = float(np.mean(db > 0.50))
                        r[f"ASR_100m_{base_name}"]         = float(np.mean(db > 1.00))
                    else:
                        r[f"flip_{base_name}"] = float(np.mean((_sigmoid(p_clean) < 0.5) & (_sigmoid(p_atk) >= 0.5)))

                results.append(r)

    df_res = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "floor_wall_summary.csv"
    df_res.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info("Saved %s (%d rows)", csv_path, len(df_res))

    # ---- Normalization Ablation Summary (Regime B - Regime A) ----
    if cirs is not None:
        _print_regime_ablation(df_res)

    # ---- Santamaria DW3000 Complex-IQ Sweep (plus IIoT20 phase check) ----
    df_dw = None
    if cirs is not None:
        df_dw = evaluate_dw3000_complex(
            aps006, lgbm_cls_cir,
            cnn_models.get("cnn_cls_A"), cnn_models.get("cnn_cls_B"),
            scale_A, fontaine_los_med_peak,
        )
        _verify_iiot20_complex_phase(aps006)

    # ---- Plots ----
    _plot_bias_shift(df_res)
    _plot_nlos_flip(df_res, df_dw)
    _plot_cir_overlay(df, cirs, fp_cir_all, los_orig_idx if cirs is not None else None)

    # ---- Terminal Summary & Go/No-Go Verdict ----
    _print_sweep_table(df_res, df_dw)
    verdict = _compute_verdict(df_res, df_dw)
    print("\n" + "=" * 82)
    print(verdict)
    print("=" * 82 + "\n")
    return 0


def _verify_iiot20_complex_phase(aps006: APS006Classifier) -> None:
    """Quick complex-IQ check on Fontaine IIoT20 (5 random phase draws vs phi=0)."""
    df20, cirs20, fp20 = load_fontaine_iiot20()
    if cirs20 is None:
        return
    sub_idx = np.linspace(0, len(cirs20) - 1, min(300, len(cirs20)), dtype=int)
    clean_pd = np.array([
        compute_features_from_cir(np.abs(cirs20[i]), int(fp20[i]))["RXP_minus_FPP_dB"]
        for i in sub_idx
    ])
    c_10 = aps006.predict(clean_pd)
    c_6  = aps006.predict_flagged(clean_pd)
    flips_5, flips_0, flips6_5 = [], [], []
    for i in sub_idx:
        _, mw, m5 = inject_with_phase_averaging(cirs20[i], int(fp20[i]), 35.0, 6.0, n_phase_draws=5, seed=SEED + int(i))
        flips_0.append(compute_features_from_cir(mw, int(fp20[i]))["RXP_minus_FPP_dB"] > 10.0)
        pds = [compute_features_from_cir(m, int(fp20[i]))["RXP_minus_FPP_dB"] for m in m5]
        flips_5.append(np.mean([p > 10.0 for p in pds]))
        flips6_5.append(np.mean([p > 6.0 for p in pds]))
    logger.info(
        "Fontaine IIoT20 complex128 check (+6 dB, Δt=35 ns, n=%d): "
        "APS006>10dB flip=%.1f%% (5-phase avg) vs %.1f%% (phi=0) | >6dB flag=%.1f%%",
        len(sub_idx),
        100 * np.mean(np.where(c_10 == 0, flips_5, 0)),
        100 * np.mean((c_10 == 0) & np.array(flips_0)),
        100 * np.mean(np.where(c_6 == 0, flips6_5, 0)),
    )


def _print_regime_ablation(df_res: pd.DataFrame) -> None:
    subA = df_res[df_res["regime"] == "A"].set_index(["delta_t_ns", "power_dB"])
    subB = df_res[df_res["regime"] == "B"].set_index(["delta_t_ns", "power_dB"])
    if subA.empty or subB.empty:
        return
    print("\n" + "=" * 82)
    print("NORMALIZATION REGIME ABLATION (Regime B per-sample max − Regime A fixed PAC)")
    print("=" * 82)
    for pdb in POWER_DB:
        for mcol, lbl in [("median_delta_b_m_cnn_reg", "CNN-Reg"), ("median_delta_b_m_ae_reg", "AE-Reg")]:
            if mcol in subA.columns and mcol in subB.columns:
                mA = subA.xs(pdb, level="power_dB")[mcol].median()
                mB = subB.xs(pdb, level="power_dB")[mcol].median()
                print(f"  +{pdb:2d} dB | {lbl:8s}: Regime A={mA:+.3f} m  |  Regime B={mB:+.3f} m  "
                      f"|  Extra shift (B−A)={mB - mA:+.3f} m")


def _print_sweep_table(df_res: pd.DataFrame, df_dw: pd.DataFrame | None) -> None:
    print("\n" + "=" * 82)
    print("FLOOR-WALL SWEEP SUMMARY (aggregated over Δt ∈ {10,20,35,50,60} ns)")
    print("=" * 82)
    for regime in [r for r in ["A", "B", "analytical"] if r in df_res["regime"].values]:
        sub = df_res[df_res["regime"] == regime]
        print(f"\n--- Regime: {regime} ---")
        for pdb, grp in sub.groupby("power_dB"):
            parts = []
            for col, lbl in [
                ("median_delta_b_m_cnn_reg", "CNN Δb"),
                ("median_delta_b_m_ae_reg",  "AE Δb"),
                ("median_delta_b_m_lgbm_cir", "LGBM-CIR Δb"),
                ("median_delta_b_m_lgbm_csv", "LGBM-CSV Δb"),
            ]:
                if col in grp.columns and grp[col].notna().any():
                    parts.append(f"{lbl}={grp[col].median():+.3f}m")
            for col, lbl in [
                ("ASR_050m_cnn_reg", "CNN ASR>0.5"),
                ("flip_cnn_cls",     "CNN Flip"),
                ("flip_lgbm_cls_cir", "LGBM Flip"),
                ("flip_aps006_cir_10dB", "APS>10"),
                ("flip_aps006_cir_6dB",  "APS>6"),
                ("flip_aps006_10dB", "APS>10"),
            ]:
                if col in grp.columns and grp[col].notna().any():
                    parts.append(f"{lbl}={grp[col].mean()*100:.1f}%")
            print(f"  +{pdb:2d} dB | " + " | ".join(parts))

    if df_dw is not None:
        print("\n--- Santamaria DW3000 Complex-IQ (mean over Δt) ---")
        for pdb, grp in df_dw.groupby("power_dB"):
            print(
                f"  +{pdb:2d} dB | APS>10 (5φ)={grp['flip_dw3000_aps10_phase5_avg'].mean()*100:5.1f}% "
                f"(φ=0: {grp['flip_dw3000_aps10_phi0'].mean()*100:5.1f}%) | "
                f"APS>6 (5φ)={grp['flip_dw3000_aps6_phase5_avg'].mean()*100:5.1f}% | "
                f"LGBM={grp['flip_dw3000_lgbm_phase5_avg'].mean()*100:5.1f}% | "
                f"CNN-B={grp['flip_dw3000_cnnB_phase5_avg'].mean()*100:5.1f}%"
            )


# ============================================================
#  Plots
# ============================================================

def _plot_bias_shift(df_res: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(-0.5, 6.5, color="#e3f2fd", alpha=0.45, label="DW3000 Safe DGC Zone (≤ +6 dB)")
    ax.axhline(0.5, color="#c62828", linestyle="--", linewidth=1.5, label="0.5 m target shift")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)

    curves = [
        ("A", "median_delta_b_m_cnn_reg",  "CNN-Reg Regime A (PAC)",      "o", "#2E7D32", "-"),
        ("B", "median_delta_b_m_cnn_reg",  "CNN-Reg Regime B (Max-Peak)", "^", "#E65100", "-"),
        ("A", "median_delta_b_m_ae_reg",   "AE-Reg Regime A (PAC)",       "s", "#66BB6A", "--"),
        ("B", "median_delta_b_m_ae_reg",   "AE-Reg Regime B (Max-Peak)",  "d", "#FB8C00", "--"),
        ("A", "median_delta_b_m_lgbm_cir", "LightGBM-CIR",                "P", "#1565C0", "-."),
        ("analytical", "median_delta_b_m_lgbm_csv", "LightGBM-CSV (analytical)", "x", "#78909C", ":"),
    ]
    for regime, col, label, marker, color, ls in curves:
        sub = df_res[df_res["regime"] == regime]
        if sub.empty or col not in sub.columns or sub[col].isna().all():
            continue
        grp = sub.groupby("power_dB")[col]
        med, q25, q75 = grp.median(), grp.quantile(0.25), grp.quantile(0.75)
        ax.errorbar(
            med.index, med.values,
            yerr=[med.values - q25.values, q75.values - med.values],
            marker=marker, linestyle=ls, color=color, label=label, capsize=3, linewidth=2,
        )

    ax.set_xlabel("Injected Power (dB relative to LOS first path)", fontsize=12)
    ax.set_ylabel("Median Induced Bias Shift Δb̂ (m)", fontsize=12)
    ax.set_title("Floor-Wall Test: Ranging Bias Shift Across Normalization Regimes", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks(POWER_DB)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bias_shift_vs_power.png", dpi=150)
    logger.info("Saved bias_shift_vs_power.png")
    plt.close(fig)


def _plot_nlos_flip(df_res: pd.DataFrame, df_dw: pd.DataFrame | None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(-0.5, 6.5, color="#e3f2fd", alpha=0.45, label="DW3000 Safe DGC Zone (≤ +6 dB)")
    ax.axhline(70.0, color="#c62828", linestyle="--", linewidth=1.3, label="70% anchor exclusion line")

    specs = [
        ("A", "flip_aps006_cir_10dB", "APS006 >10 dB (IIoT19)", "-",  "o", "#1565C0"),
        ("A", "flip_aps006_cir_6dB",  "APS006 >6 dB Flag (IIoT19)", "--", "s", "#42A5F5"),
        ("A", "flip_lgbm_cls_cir",    "LightGBM-Cls (IIoT19 CIR)", "-.", "^", "#6A1B9A"),
        ("A", "flip_cnn_cls",         "CNN-Cls Regime A (IIoT19)", "-",  "d", "#2E7D32"),
        ("B", "flip_cnn_cls",         "CNN-Cls Regime B (IIoT19)", "-",  "v", "#E65100"),
    ]
    for regime, col, lbl, ls, mk, color in specs:
        sub = df_res[df_res["regime"] == regime]
        if not sub.empty and col in sub.columns and sub[col].notna().any():
            g = sub.groupby("power_dB")[col].mean() * 100.0
            ax.plot(g.index, g.values, marker=mk, linestyle=ls, color=color, label=lbl, linewidth=2)

    if df_dw is not None:
        g_dw10 = df_dw.groupby("power_dB")["flip_dw3000_aps10_phase5_avg"].mean() * 100.0
        g_dw6  = df_dw.groupby("power_dB")["flip_dw3000_aps6_phase5_avg"].mean() * 100.0
        g_dwcnnB = df_dw.groupby("power_dB")["flip_dw3000_cnnB_phase5_avg"].mean() * 100.0
        ax.plot(g_dw10.index, g_dw10.values, marker="*", linestyle="-", color="#00838F",
                label="APS006 >10 dB (DW3000 5φ)", linewidth=2)
        ax.plot(g_dw6.index, g_dw6.values, marker="*", linestyle=":", color="#26C6DA",
                label="APS006 >6 dB (DW3000 5φ)", linewidth=1.8)
        if g_dwcnnB.notna().any():
            ax.plot(g_dwcnnB.index, g_dwcnnB.values, marker="P", linestyle="--", color="#AD1457",
                    label="CNN-Cls Reg B (DW3000 5φ)", linewidth=1.8)

    ax.set_xlabel("Injected Power (dB relative to LOS first path)", fontsize=12)
    ax.set_ylabel("Clean LOS → NLOS Flip Rate (%)", fontsize=12)
    ax.set_title("NLOS Classifier Flip Rate vs. Attacker Power", fontsize=13)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xticks(POWER_DB)
    ax.set_ylim(-4, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "nlos_flip_rate_vs_power.png", dpi=150)
    logger.info("Saved nlos_flip_rate_vs_power.png")
    plt.close(fig)


def _plot_cir_overlay(
    df: pd.DataFrame,
    cirs: np.ndarray | None,
    fp_cir_all: np.ndarray | None,
    los_orig_idx: np.ndarray | None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    if cirs is not None and fp_cir_all is not None and los_orig_idx is not None and len(los_orig_idx) > 0:
        i_los = int(los_orig_idx[min(10, len(los_orig_idx) - 1)])
        fp_los = int(fp_cir_all[i_los])
        cir_los = np.abs(cirs[i_los])

        nlos_idx = df.index[df["label"] == 1].values
        i_nlos = int(nlos_idx[0]) if len(nlos_idx) > 0 else i_los
        fp_nlos = int(fp_cir_all[i_nlos])
        cir_nlos = np.abs(cirs[i_nlos])

        cir_atk = np.abs(inject_phantom_nlos(cirs[i_los], fp_los, 35.0, 6.0, is_complex=False))

        ws, we = max(0, fp_los - 20), min(len(cir_los), fp_los + 95)
        t = (np.arange(ws, we) - fp_los) * TAP_NS
        axes[0].plot(t, cir_los[ws:we], color="#1565C0", linewidth=1.4)
        axes[0].axvline(0.0, color="green", linestyle="--", alpha=0.6, label="fp_idx")
        axes[0].set_title("Real Clean LOS (Fontaine IIoT19)", fontsize=11)
        axes[0].legend(fontsize=8)

        ws2, we2 = max(0, fp_nlos - 20), min(len(cir_nlos), fp_nlos + 95)
        t2 = (np.arange(ws2, we2) - fp_nlos) * TAP_NS
        axes[1].plot(t2, cir_nlos[ws2:we2], color="#E65100", linewidth=1.4)
        axes[1].axvline(0.0, color="green", linestyle="--", alpha=0.6, label="fp_idx")
        axes[1].set_title("Real Clean NLOS (Fontaine IIoT19)", fontsize=11)
        axes[1].legend(fontsize=8)

        axes[2].plot(t, cir_atk[ws:we], color="#B71C1C", linewidth=1.4, label="+6 dB Attacked")
        axes[2].plot(t, cir_los[ws:we], color="#1565C0", linewidth=1.0, alpha=0.45, label="Original LOS")
        axes[2].axvline(0.0, color="green", linestyle="--", alpha=0.6, label="Untouched FP")
        axes[2].axvline(35.0, color="red", linestyle=":", alpha=0.7, label="Injected Δt=35 ns")
        axes[2].set_title("+6 dB Phantom-NLOS (Δt=35 ns)", fontsize=11)
        axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("Delay relative to First Path (ns)", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("CIR Magnitude (PAC-normalized)", fontsize=10)
    fig.suptitle("CIR Morphology Comparison: Real LOS vs. Real NLOS vs. +6 dB Phantom-NLOS", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_cir_overlay.png", dpi=150)
    logger.info("Saved sample_cir_overlay.png")
    plt.close(fig)


# ============================================================
#  Verdict
# ============================================================

def _compute_verdict(df_res: pd.DataFrame, df_dw: pd.DataFrame | None) -> str:
    safe = df_res[df_res["power_dB"] <= 6].copy()
    if safe.empty:
        return "VERDICT: INCONCLUSIVE — no results at ≤ +6 dB."

    reg_cols = [c for c in safe.columns if c.startswith("median_delta_b_m_")]
    asr_cols = [c for c in safe.columns if c.startswith("ASR_050m_")]
    hard_flip_cols = [c for c in safe.columns if c.startswith("flip_") and "6dB" not in c]
    flag_flip_cols = [c for c in safe.columns if c.startswith("flip_") and "6dB" in c]

    max_med_shift, best_reg = 0.0, "none"
    for c in reg_cols:
        v = safe[c].dropna()
        if not v.empty and v.max() > max_med_shift:
            max_med_shift, best_reg = float(v.max()), c

    max_asr_050, best_asr = 0.0, "none"
    for c in asr_cols:
        v = safe[c].dropna()
        if not v.empty and v.max() > max_asr_050:
            max_asr_050, best_asr = float(v.max()), c

    max_hard_flip, best_hard_cls = 0.0, "none"
    for c in hard_flip_cols:
        v = safe[c].dropna()
        if not v.empty and v.max() > max_hard_flip:
            max_hard_flip, best_hard_cls = float(v.max()), c

    max_flag_flip, best_flag_cls = 0.0, "none"
    for c in flag_flip_cols:
        v = safe[c].dropna()
        if not v.empty and v.max() > max_flag_flip:
            max_flag_flip, best_flag_cls = float(v.max()), c

    if df_dw is not None:
        dw_safe = df_dw[df_dw["power_dB"] <= 6]
        for c in [c for c in dw_safe.columns if c.startswith("flip_") and "aps6" not in c]:
            v = dw_safe[c].dropna()
            if not v.empty and v.max() > max_hard_flip:
                max_hard_flip, best_hard_cls = float(v.max()), c
        for c in [c for c in dw_safe.columns if "aps6" in c]:
            v = dw_safe[c].dropna()
            if not v.empty and v.max() > max_flag_flip:
                max_flag_flip, best_flag_cls = float(v.max()), c

    lines = [
        "VERDICT ANALYSIS (≤ +6 dB DW3000 Safe DGC Zone):",
        f"  Max median regressor shift:       {max_med_shift:+.3f} m ({best_reg})",
        f"  Max ASR (> 0.50 m):               {max_asr_050*100:5.1f}% ({best_asr})",
        f"  Max hard NLOS flip (>10 dB / ML): {max_hard_flip*100:5.1f}% ({best_hard_cls})",
        f"  Max uncertain flag (>6 dB):       {max_flag_flip*100:5.1f}% ({best_flag_cls})",
        "",
    ]

    if max_med_shift > 0.50 and max_asr_050 > 0.50:
        lines.append("★ VERDICT: PASS (Full Thesis)")
        lines.append("  At ≤+6 dB, at least one realistic regressor shifts by >0.5 m on >50% of LOS samples.")
    elif max_med_shift < 0.20 and max_hard_flip > 0.70:
        lines.append("★ VERDICT: PARTIAL (Anchor Exclusion Only)")
        lines.append("  At ≤+6 dB, regressors shift <0.2 m, but hard NLOS classifiers flip >70% of LOS links.")
    elif max_hard_flip > 0.70:
        lines.append("★ VERDICT: PARTIAL (Anchor Exclusion + Moderate Bias Shift)")
        lines.append(f"  At ≤+6 dB, regressor shift is {max_med_shift:.3f} m (<0.5 m), while hard NLOS classifiers flip >70%.")
    elif max_flag_flip > 0.70:
        lines.append("★ VERDICT: PARTIAL (Uncertain-Link Flagging Only)")
        lines.append("  Hard NLOS classifiers flip <70% and regressor shifts <0.5 m at ≤+6 dB,")
        lines.append("  but APS006 >6 dB uncertainty flagging exceeds 70%.")
    elif max_med_shift >= 0.20 or max_hard_flip >= 0.30:
        lines.append("★ VERDICT: FAIL / WEAK PARTIAL (Below Anchor-Exclusion & Bias Targets)")
        lines.append("  Neither >0.5 m bias shift (>50% ASR) nor >70% hard NLOS flip is reached at ≤+6 dB.")
    else:
        lines.append("★ VERDICT: FAIL (Floor Too High)")
        lines.append("  Neither regressors nor classifiers move meaningfully at ≤+6 dB.")

    return "\n".join(lines)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    raise SystemExit(main())
