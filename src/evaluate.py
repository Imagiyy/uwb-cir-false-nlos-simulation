"""evaluate.py — Floor-Wall sweep: inject Phantom-NLOS into clean LOS CIRs,
measure bias shift and NLOS classifier flip rates.

Operates in two modes:
1. **Full CIR mode** (CIRsV2.hdf5 available): inject into actual CIR waveforms,
   recompute features, run ALL models (LightGBM-CSV, LightGBM-CIR, CNN, AE, APS006).
2. **Analytical mode** (no CIR waveforms): simulate the effect of injection on
   CIR-derived scalar features analytically and evaluate LightGBM-CSV + APS006.

Sweep parameters:
   delta_t_ns  ∈ {10, 20, 35, 50, 60}
   power_dB    ∈ {0, 3, 6, 9, 12}
   regimes     ∈ {A, B} (PAC-normalized, per-sample max-normalized)
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (CNN_WIN_LEN, extract_cir_window, grouped_split,
                     load_fontaine_cirs, load_fontaine_scalar)
from features import FEATURE_NAMES, compute_features_from_cir
from injector import inject_phantom_nlos
from models import (AECNNRegressor, APS006Classifier, CNN1DClassifier,
                    CNN1DRegressor, CIR_DERIVED_COLS, CIRAutoencoder,
                    SCALAR_FEATURE_COLS)

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
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ============================================================
#  Analytical feature simulation (fallback when no CIR)
# ============================================================

def simulate_injection_on_features(
    row: pd.Series,
    delta_t_ns: float,
    power_db: float,
) -> dict:
    """Estimate how injecting a delayed pulse changes CSV scalar features.

    Physics:  The injected pulse adds power but does NOT change the first
    path.  Therefore:
      - FP_power (dBm) unchanged.
      - RX_power (dBm) increases: RX_lin_new = RX_lin_old + fp_lin · scale²
      - Pd (= RX_power - FP_power, dB) increases.
      - CIRpower (raw register ∝ total energy) increases proportionally.
      - fp_ampl1/2/3, std_noise, RXPACC unchanged.
      - Pr (channel parameter) increases with RX_power.

    Note: RX_power and FP_power are already in dBm (~-80 to -115).
          CIRpower is a raw register value (~10k), NOT in dB.
    """
    scale = 10.0 ** (power_db / 20.0)

    # Current values (RX_power, FP_power are in dBm)
    FP_power = float(row["FP_power"])   # dBm
    RX_power = float(row["RX_power"])   # dBm
    CIRpower = float(row["CIRpower"])   # raw register (linear proxy)

    # Convert dBm to linear milliwatts for addition
    fp_lin = 10.0 ** (FP_power / 10.0)  # milliwatts
    rx_lin = 10.0 ** (RX_power / 10.0)  # milliwatts
    inj_lin = fp_lin * (scale ** 2)      # injected power (milliwatts)

    rx_lin_new = rx_lin + inj_lin
    RX_power_new = 10.0 * np.log10(max(rx_lin_new, 1e-300))
    Pd_new = RX_power_new - FP_power

    # CIRpower is a raw register proportional to total energy.
    # Injection adds energy proportional to fp_lin * scale^2 / rx_lin * CIRpower
    CIRpower_new = CIRpower * (rx_lin_new / max(rx_lin, 1e-300))

    new_row = row.copy()
    new_row["RX_power"]  = RX_power_new
    new_row["CIRpower"]  = CIRpower_new
    new_row["Pd"]        = Pd_new
    new_row["Pr"]        = float(row["Pr"]) + (RX_power_new - RX_power)

    return {col: float(new_row[col]) for col in SCALAR_FEATURE_COLS}


# ============================================================
#  Main evaluation
# ============================================================

def main() -> int:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- Load data & splits ----
    logger.info("=== Loading data ===")
    df = load_fontaine_scalar()
    splits = load_pickle("split_indices")
    te_idx = splits["test"]
    train_meta = load_pickle("train_meta")
    have_cirs = train_meta["have_cirs"]

    # Select clean LOS test samples only
    te_df = df.iloc[te_idx].copy()
    te_df["_orig_idx"] = te_idx  # keep track of original index
    los_mask = te_df["label"] == 0
    los_df = te_df[los_mask].copy()
    los_orig_idx = te_df[los_mask]["_orig_idx"].values

    logger.info("Test set: %d samples (%d LOS, %d NLOS)",
                len(te_df), los_mask.sum(), (~los_mask).sum())

    if len(los_df) == 0:
        logger.error("No LOS samples in test set! Cannot run floor-wall evaluation.")
        return 1

    # ---- Load models ----
    logger.info("=== Loading trained models ===")
    lgbm_reg = load_pickle("lgbm_reg_csv")
    lgbm_cls = load_pickle("lgbm_cls_csv")
    aps006   = APS006Classifier()

    # Compute clean predictions on LOS test samples (CSV features)
    X_los_csv = los_df[SCALAR_FEATURE_COLS].values.astype(np.float32)
    clean_reg_pred_csv = lgbm_reg.predict(X_los_csv)  # mm
    clean_cls_pred_csv = lgbm_cls.predict(X_los_csv)  # probability
    clean_aps_pred     = aps006.predict(los_df["Pd"].values)
    clean_aps_flag     = aps006.predict_flagged(los_df["Pd"].values)

    # CIR-based models
    cirs = None
    lgbm_reg_cir = lgbm_cls_cir = None
    cnn_models = {}  # {f"cnn_reg_{regime}": model, ...}

    if have_cirs:
        cirs = load_fontaine_cirs(n_expected=len(df))

    if cirs is not None:
        cir_mags = np.abs(cirs)
        fp_indices = df["fpi"].values

        try:
            lgbm_reg_cir = load_pickle("lgbm_reg_cir")
            lgbm_cls_cir = load_pickle("lgbm_cls_cir")
        except FileNotFoundError:
            pass

        for regime in ["A", "B"]:
            for mtype, cls_obj, kw in [
                ("cnn_reg", CNN1DRegressor, {"input_len": CNN_WIN_LEN}),
                ("cnn_cls", CNN1DClassifier, {"input_len": CNN_WIN_LEN}),
            ]:
                name = f"{mtype}_{regime}"
                try:
                    cnn_models[name] = load_torch(cls_obj, name, **kw)
                except FileNotFoundError:
                    logger.warning("Model %s not found, skipping.", name)

            # AE+Reg
            ae_name = f"ae_{regime}"
            ae_reg_name = f"ae_reg_{regime}"
            try:
                ae = CIRAutoencoder(input_len=CNN_WIN_LEN)
                ae.load_state_dict(torch.load(MODELS_DIR / f"{ae_name}.pt",
                                              map_location=DEVICE, weights_only=True))
                ae.eval()
                ae_reg = AECNNRegressor(ae.encoder)
                ae_reg.load_state_dict(torch.load(MODELS_DIR / f"{ae_reg_name}.pt",
                                                  map_location=DEVICE, weights_only=True))
                ae_reg.eval()
                cnn_models[ae_reg_name] = ae_reg
            except FileNotFoundError:
                logger.warning("AE model %s not found, skipping.", ae_reg_name)

        # Clean CIR-derived predictions
        los_cir_mags = cir_mags[los_orig_idx]
        los_fp_idx   = fp_indices[los_orig_idx]

        if lgbm_reg_cir is not None:
            feat_los = np.stack([
                list(compute_features_from_cir(los_cir_mags[i], int(los_fp_idx[i])).values())
                for i in range(len(los_cir_mags))
            ]).astype(np.float32)
            clean_reg_cir_pred = lgbm_reg_cir.predict(feat_los)
            clean_cls_cir_pred = lgbm_cls_cir.predict(feat_los)
        else:
            clean_reg_cir_pred = clean_cls_cir_pred = None

        # Clean CNN predictions per regime
        clean_cnn_preds = {}
        for regime in ["A", "B"]:
            windows = np.stack([
                extract_cir_window(los_cir_mags[i], int(los_fp_idx[i]))
                for i in range(len(los_cir_mags))
            ]).astype(np.float32)
            if regime == "B":
                mx = windows.max(axis=1, keepdims=True)
                mx = np.maximum(mx, 1e-12)
                windows = windows / mx
            xt = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)

            for mname, model in cnn_models.items():
                if mname.endswith(f"_{regime}"):
                    with torch.no_grad():
                        p = model(xt).numpy()
                    clean_cnn_preds[mname] = p

        logger.info("Loaded %d CNN/AE models for CIR evaluation.", len(cnn_models))
    else:
        logger.info("No CIR waveforms — using analytical feature simulation only.")

    # ---- Floor-Wall Sweep ----
    logger.info("=== Starting Floor-Wall Sweep ===")
    results = []
    n_los = len(los_df)

    for dt in DELTA_T_NS:
        for pdb in POWER_DB:
            logger.info("--- delta_t=%d ns, power=%+d dB ---", dt, pdb)

            # -- Mode 1: Analytical (always available) --
            atk_features_csv = []
            for _, row in los_df.iterrows():
                sim = simulate_injection_on_features(row, dt, pdb)
                atk_features_csv.append([sim[c] for c in SCALAR_FEATURE_COLS])
            X_atk_csv = np.array(atk_features_csv, dtype=np.float32)

            atk_reg_csv = lgbm_reg.predict(X_atk_csv)
            atk_cls_csv = lgbm_cls.predict(X_atk_csv)
            # Simulated Pd for APS006
            atk_Pd = X_atk_csv[:, SCALAR_FEATURE_COLS.index("Pd")]
            atk_aps  = aps006.predict(atk_Pd)
            atk_aps_flag = aps006.predict_flagged(atk_Pd)

            # Bias shift (meters)
            delta_b_csv = (atk_reg_csv - clean_reg_pred_csv) / 1000.0

            row_base = {
                "delta_t_ns": dt, "power_dB": pdb, "regime": "analytical",
            }
            # Regressor metrics
            row_base["mean_delta_b_m_lgbm_csv"] = float(np.mean(delta_b_csv))
            row_base["median_delta_b_m_lgbm_csv"] = float(np.median(delta_b_csv))
            row_base["p90_delta_b_m_lgbm_csv"] = float(np.percentile(delta_b_csv, 90))
            row_base["ASR_020m_lgbm_csv"] = float(np.mean(delta_b_csv > 0.20))
            row_base["ASR_050m_lgbm_csv"] = float(np.mean(delta_b_csv > 0.50))
            row_base["ASR_100m_lgbm_csv"] = float(np.mean(delta_b_csv > 1.00))
            # Classifier metrics
            row_base["flip_aps006_10dB"] = float(np.mean(
                (clean_aps_pred == 0) & (atk_aps == 1)))
            row_base["flip_aps006_6dB"] = float(np.mean(
                (clean_aps_flag == 0) & (atk_aps_flag == 1)))
            row_base["flip_lgbm_cls_csv"] = float(np.mean(
                (clean_cls_pred_csv < 0.5) & (atk_cls_csv >= 0.5)))

            results.append(row_base)

            # -- Mode 2: CIR-based (if available) --
            if cirs is not None:
                los_cirs_complex = cirs[los_orig_idx]
                los_cir_mags_arr = np.abs(los_cirs_complex)
                los_fp_arr = df["fpi"].values[los_orig_idx]

                for regime in ["A", "B"]:
                    # Inject into each LOS CIR
                    atk_mags = []
                    atk_feats = []
                    for i in range(n_los):
                        fp_i = int(los_fp_arr[i])
                        h_atk = inject_phantom_nlos(
                            los_cirs_complex[i], fp_i, dt, pdb,
                            is_complex=True, phi=0.0,  # worst-case constructive
                        )
                        mag_atk = np.abs(h_atk)
                        atk_mags.append(mag_atk)
                        f_dict = compute_features_from_cir(mag_atk, fp_i)
                        atk_feats.append([f_dict[k] for k in FEATURE_NAMES])

                    atk_mags = np.array(atk_mags)
                    atk_feat_matrix = np.array(atk_feats, dtype=np.float32)

                    # Windows for CNN
                    atk_windows = np.stack([
                        extract_cir_window(atk_mags[i], int(los_fp_arr[i]))
                        for i in range(n_los)
                    ]).astype(np.float32)
                    if regime == "B":
                        mx = atk_windows.max(axis=1, keepdims=True)
                        mx = np.maximum(mx, 1e-12)
                        atk_windows = atk_windows / mx

                    r = {"delta_t_ns": dt, "power_dB": pdb, "regime": regime}

                    # LightGBM CIR-derived regressor
                    if lgbm_reg_cir is not None:
                        atk_reg_cir = lgbm_reg_cir.predict(atk_feat_matrix)
                        delta_b_cir = (atk_reg_cir - clean_reg_cir_pred) / 1000.0
                        r["mean_delta_b_m_lgbm_cir"] = float(np.mean(delta_b_cir))
                        r["median_delta_b_m_lgbm_cir"] = float(np.median(delta_b_cir))
                        r["p90_delta_b_m_lgbm_cir"] = float(np.percentile(delta_b_cir, 90))
                        r["ASR_020m_lgbm_cir"] = float(np.mean(delta_b_cir > 0.20))
                        r["ASR_050m_lgbm_cir"] = float(np.mean(delta_b_cir > 0.50))
                        r["ASR_100m_lgbm_cir"] = float(np.mean(delta_b_cir > 1.00))

                    if lgbm_cls_cir is not None:
                        atk_cls_cir = lgbm_cls_cir.predict(atk_feat_matrix)
                        r["flip_lgbm_cls_cir"] = float(np.mean(
                            (clean_cls_cir_pred < 0.5) & (atk_cls_cir >= 0.5)))

                    # APS006 from CIR-derived RXP-FPP
                    rxp_fpp_atk = atk_feat_matrix[:, FEATURE_NAMES.index("RXP_minus_FPP_dB")]
                    feat_los_clean = np.stack([
                        list(compute_features_from_cir(
                            los_cir_mags_arr[i], int(los_fp_arr[i])).values())
                        for i in range(n_los)
                    ]).astype(np.float32)
                    rxp_fpp_clean = feat_los_clean[:, FEATURE_NAMES.index("RXP_minus_FPP_dB")]
                    clean_aps_cir = (rxp_fpp_clean > 10.0).astype(int)
                    atk_aps_cir   = (rxp_fpp_atk > 10.0).astype(int)
                    clean_aps_cir_6 = (rxp_fpp_clean > 6.0).astype(int)
                    atk_aps_cir_6   = (rxp_fpp_atk > 6.0).astype(int)
                    r["flip_aps006_cir_10dB"] = float(np.mean(
                        (clean_aps_cir == 0) & (atk_aps_cir == 1)))
                    r["flip_aps006_cir_6dB"] = float(np.mean(
                        (clean_aps_cir_6 == 0) & (atk_aps_cir_6 == 1)))

                    # CNN models
                    xt_atk = torch.tensor(atk_windows, dtype=torch.float32).unsqueeze(1)
                    for mname, model in cnn_models.items():
                        if not mname.endswith(f"_{regime}"):
                            continue
                        with torch.no_grad():
                            p_atk = model(xt_atk).numpy()
                        p_clean = clean_cnn_preds.get(mname)
                        if p_clean is None:
                            continue
                        if "reg" in mname:
                            # Predictions are in meters
                            delta_b = p_atk - p_clean
                            r[f"mean_delta_b_m_{mname}"] = float(np.mean(delta_b))
                            r[f"median_delta_b_m_{mname}"] = float(np.median(delta_b))
                            r[f"p90_delta_b_m_{mname}"] = float(np.percentile(delta_b, 90))
                            r[f"ASR_050m_{mname}"] = float(np.mean(delta_b > 0.50))
                        elif "cls" in mname:
                            prob_clean = _sigmoid(p_clean)
                            prob_atk   = _sigmoid(p_atk)
                            r[f"flip_{mname}"] = float(np.mean(
                                (prob_clean < 0.5) & (prob_atk >= 0.5)))

                    results.append(r)

    # ---- Save CSV ----
    df_res = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "floor_wall_summary.csv"
    df_res.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info("Saved %s (%d rows)", csv_path, len(df_res))

    # ---- Print key results table ----
    print("\n" + "=" * 80)
    print("FLOOR-WALL SWEEP RESULTS SUMMARY")
    print("=" * 80)
    for _, row in df_res.iterrows():
        regime = row.get("regime", "?")
        dt = int(row["delta_t_ns"])
        pdb = int(row["power_dB"])
        # Pick best available regressor metric
        med_key = None
        for k in ["median_delta_b_m_lgbm_cir", "median_delta_b_m_lgbm_csv"]:
            if k in row and not pd.isna(row[k]):
                med_key = k
                break
        med_val = f"{row[med_key]:.3f} m" if med_key else "N/A"
        flip_key = None
        for k in ["flip_aps006_cir_10dB", "flip_aps006_10dB"]:
            if k in row and not pd.isna(row[k]):
                flip_key = k
                break
        flip_val = f"{row[flip_key]*100:.1f}%" if flip_key else "N/A"
        print(f"  Δt={dt:2d}ns  +{pdb:2d}dB  regime={regime:>10s}  "
              f"median_Δb={med_val:>10s}  APS006_flip={flip_val:>7s}")

    # ---- Generate Plots ----
    _plot_bias_shift(df_res)
    _plot_nlos_flip(df_res)
    _plot_cir_overlay(df, cirs, los_orig_idx if cirs is not None else None)

    # ---- Verdict ----
    verdict = _compute_verdict(df_res)
    print("\n" + "=" * 80)
    print(verdict)
    print("=" * 80 + "\n")

    return 0


# ============================================================
#  Plots
# ============================================================

def _plot_bias_shift(df_res: pd.DataFrame) -> None:
    """Plot median + IQR of Δb_hat vs power dB."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Shade DGC safe zone
    ax.axhspan(-0.1, 0.1, xmin=0, xmax=1, color="#e8f5e9", alpha=0.5, label=None)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="0.5 m target")
    ax.axvspan(-0.5, 6.5, color="#e3f2fd", alpha=0.3, label="DGC safe zone (≤+6 dB)")

    regimes_in_data = df_res["regime"].unique()
    markers = {"analytical": "s", "A": "o", "B": "^"}
    colors  = {"analytical": "#1565C0", "A": "#2E7D32", "B": "#E65100"}

    for regime in regimes_in_data:
        sub = df_res[df_res["regime"] == regime]
        # Find best available median column
        med_col = None
        for c in ["median_delta_b_m_lgbm_cir", "median_delta_b_m_cnn_reg_" + regime,
                   "median_delta_b_m_lgbm_csv"]:
            if c in sub.columns and sub[c].notna().any():
                med_col = c
                break
        if med_col is None:
            continue

        # Aggregate across delta_t for each power_dB
        grouped = sub.groupby("power_dB")[med_col]
        medians = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)

        ax.errorbar(medians.index, medians.values,
                    yerr=[medians.values - q25.values, q75.values - medians.values],
                    marker=markers.get(regime, "x"), label=f"Regime {regime}",
                    color=colors.get(regime, "gray"), capsize=4, linewidth=2)

    ax.set_xlabel("Injected Power (dB relative to LOS first path)", fontsize=12)
    ax.set_ylabel("Median Induced Bias Shift Δb̂ (m)", fontsize=12)
    ax.set_title("Floor-Wall Test: Ranging Bias Shift vs. Attacker Power", fontsize=14)
    ax.legend(loc="upper left")
    ax.set_xticks(POWER_DB)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bias_shift_vs_power.png", dpi=150)
    logger.info("Saved bias_shift_vs_power.png")
    plt.close(fig)


def _plot_nlos_flip(df_res: pd.DataFrame) -> None:
    """Plot NLOS classifier flip rate vs power dB."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(-0.5, 6.5, color="#e3f2fd", alpha=0.3, label="DGC safe zone (≤+6 dB)")
    ax.axhline(0.70, color="red", linestyle="--", linewidth=1, label="70% threshold")

    flip_cols = [c for c in df_res.columns if c.startswith("flip_")]
    if not flip_cols:
        logger.warning("No flip columns found in results.")
        return

    regimes = df_res["regime"].unique()
    style_map = {
        "flip_aps006_10dB": ("APS006 >10dB", "-"),
        "flip_aps006_6dB":  ("APS006 >6dB", "--"),
        "flip_lgbm_cls_csv": ("LightGBM-Cls (CSV)", "-."),
        "flip_aps006_cir_10dB": ("APS006-CIR >10dB", "-"),
        "flip_aps006_cir_6dB":  ("APS006-CIR >6dB", "--"),
        "flip_lgbm_cls_cir":    ("LightGBM-Cls (CIR)", "-."),
    }
    color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))

    ci = 0
    for regime in regimes:
        sub = df_res[df_res["regime"] == regime]
        for fc in flip_cols:
            if fc not in sub.columns or sub[fc].isna().all():
                continue
            lbl, ls = style_map.get(fc, (fc, "-"))
            grouped = sub.groupby("power_dB")[fc].mean()
            ax.plot(grouped.index, grouped.values * 100,
                    marker="o", linestyle=ls, color=color_cycle[ci % 10],
                    label=f"{lbl} ({regime})", linewidth=1.5)
            ci += 1

    ax.set_xlabel("Injected Power (dB)", fontsize=12)
    ax.set_ylabel("LOS → NLOS Flip Rate (%)", fontsize=12)
    ax.set_title("NLOS Classifier Flip Rate vs. Attacker Power", fontsize=14)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xticks(POWER_DB)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "nlos_flip_rate_vs_power.png", dpi=150)
    logger.info("Saved nlos_flip_rate_vs_power.png")
    plt.close(fig)


def _plot_cir_overlay(
    df: pd.DataFrame,
    cirs: np.ndarray | None,
    los_orig_idx: np.ndarray | None,
) -> None:
    """Plot side-by-side: clean LOS, clean NLOS, +6 dB Phantom-NLOS CIR."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    if cirs is not None and los_orig_idx is not None and len(los_orig_idx) > 0:
        # Pick a representative LOS sample
        i_los = los_orig_idx[0]
        fp_los = int(df["fpi"].iloc[i_los])
        cir_los = np.abs(cirs[i_los])

        # Pick a NLOS sample
        nlos_mask = df["label"] == 1
        nlos_idx = df.index[nlos_mask]
        if len(nlos_idx) > 0:
            i_nlos = nlos_idx[0]
            fp_nlos = int(df["fpi"].iloc[i_nlos])
            cir_nlos = np.abs(cirs[i_nlos])
        else:
            i_nlos = i_los
            fp_nlos = fp_los
            cir_nlos = cir_los

        # +6 dB attack
        h_atk = inject_phantom_nlos(cirs[i_los], fp_los, 35.0, 6.0,
                                     is_complex=True, phi=0.0)
        cir_atk = np.abs(h_atk)

        # Plot windows around first path
        ws = max(0, fp_los - 20)
        we = min(len(cir_los), fp_los + 100)
        t = np.arange(ws, we) * 1.0016

        axes[0].plot(t, cir_los[ws:we], color="#1565C0", linewidth=1)
        axes[0].axvline(fp_los * 1.0016, color="green", linestyle="--", alpha=0.5)
        axes[0].set_title("Clean LOS", fontsize=11)

        ws2 = max(0, fp_nlos - 20)
        we2 = min(len(cir_nlos), fp_nlos + 100)
        t2 = np.arange(ws2, we2) * 1.0016
        axes[1].plot(t2, cir_nlos[ws2:we2], color="#E65100", linewidth=1)
        axes[1].axvline(fp_nlos * 1.0016, color="green", linestyle="--", alpha=0.5)
        axes[1].set_title("Clean NLOS", fontsize=11)

        axes[2].plot(t, cir_atk[ws:we], color="#B71C1C", linewidth=1, label="Attacked")
        axes[2].plot(t, cir_los[ws:we], color="#1565C0", linewidth=0.8,
                     alpha=0.5, label="Original")
        axes[2].axvline(fp_los * 1.0016, color="green", linestyle="--", alpha=0.5)
        axes[2].axvline((fp_los + round(35.0 / 1.0016)) * 1.0016,
                        color="red", linestyle=":", alpha=0.5, label="Injection point")
        axes[2].set_title("+6 dB Phantom-NLOS (Δt=35ns)", fontsize=11)
        axes[2].legend(fontsize=8)
    else:
        # Synthetic illustration when no CIR available
        t = np.arange(0, 150) * 1.0016
        # Fake LOS pulse
        pulse = np.zeros(150)
        pulse[20:25] = [0.2, 0.8, 1.0, 0.6, 0.15]
        noise = np.random.RandomState(42).normal(0, 0.05, 150)
        noise = np.abs(noise)
        los = pulse + noise

        # Fake NLOS
        nlos = los.copy()
        nlos[50:60] = [0.3, 0.5, 0.7, 0.9, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]

        # +6 dB attack
        atk = los.copy()
        scale = 10 ** (6 / 20)
        atk[55:60] += scale * np.array([0.2, 0.8, 1.0, 0.6, 0.15])

        axes[0].plot(t, los, color="#1565C0"); axes[0].set_title("Clean LOS (synthetic)")
        axes[1].plot(t, nlos, color="#E65100"); axes[1].set_title("Clean NLOS (synthetic)")
        axes[2].plot(t, atk, color="#B71C1C", label="Attacked")
        axes[2].plot(t, los, color="#1565C0", alpha=0.5, label="Original")
        axes[2].set_title("+6 dB Phantom-NLOS (synthetic)")
        axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("Time (ns)", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("CIR Magnitude", fontsize=10)
    fig.suptitle("Channel Impulse Response Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_cir_overlay.png", dpi=150, bbox_inches="tight")
    logger.info("Saved sample_cir_overlay.png")
    plt.close(fig)


# ============================================================
#  Verdict
# ============================================================

def _compute_verdict(df_res: pd.DataFrame) -> str:
    """Automated Go/No-Go decision based on ≤+6 dB results."""
    safe = df_res[df_res["power_dB"] <= 6].copy()
    if len(safe) == 0:
        return "VERDICT: INCONCLUSIVE — no results at ≤+6 dB."

    # Check regressor shift
    reg_cols = [c for c in safe.columns if "median_delta_b_m" in c]
    max_median_shift = 0.0
    for c in reg_cols:
        vals = safe[c].dropna()
        if len(vals) > 0:
            max_median_shift = max(max_median_shift, vals.max())

    asr_cols = [c for c in safe.columns if "ASR_050m" in c]
    max_asr_050 = 0.0
    for c in asr_cols:
        vals = safe[c].dropna()
        if len(vals) > 0:
            max_asr_050 = max(max_asr_050, vals.max())

    # Check classifier flip — distinguish hard NLOS (>10dB) from flagged (>6dB)
    # For the verdict, use the >10 dB and ML classifier flip rates (hard NLOS decision)
    hard_flip_cols = [c for c in safe.columns
                      if "flip_" in c and "6dB" not in c]
    max_hard_flip = 0.0
    for c in hard_flip_cols:
        vals = safe[c].dropna()
        if len(vals) > 0:
            max_hard_flip = max(max_hard_flip, vals.max())

    # Also track the flagged (>6 dB) rate separately
    flag_flip_cols = [c for c in safe.columns if "flip_" in c and "6dB" in c]
    max_flag_flip = 0.0
    for c in flag_flip_cols:
        vals = safe[c].dropna()
        if len(vals) > 0:
            max_flag_flip = max(max_flag_flip, vals.max())

    lines = [
        "VERDICT ANALYSIS (at ≤ +6 dB, DGC safe zone):",
        f"  Max median regressor shift:        {max_median_shift:.3f} m",
        f"  Max ASR (>0.5 m):                  {max_asr_050*100:.1f}%",
        f"  Max hard NLOS flip (>10dB / ML):   {max_hard_flip*100:.1f}%",
        f"  Max flagged flip (>6dB uncertain):  {max_flag_flip*100:.1f}%",
        "",
    ]

    if max_median_shift > 0.5 and max_asr_050 > 0.50:
        lines.append("★ VERDICT: PASS (Full Thesis)")
        lines.append("  At ≤+6 dB, at least one regressor shifts by >0.5 m on >50% of LOS samples.")
        lines.append("  The Phantom-NLOS attack is effective for indirect distance reduction")
        lines.append("  within the DW3000 DGC safe power envelope.")
    elif max_hard_flip > 0.70:
        lines.append("★ VERDICT: PARTIAL (Anchor Exclusion Only)")
        lines.append("  At ≤+6 dB, regressors shift <0.5 m median, but hard NLOS classifiers")
        lines.append(f"  flip >{max_hard_flip*100:.0f}% of LOS links → stealthy anchor exclusion is viable.")
    elif max_flag_flip > 0.70:
        lines.append("★ VERDICT: PARTIAL (Anchor Flagging)")
        lines.append("  At ≤+6 dB, regressors shift <0.5 m and hard NLOS classifiers don't flip,")
        lines.append(f"  but >{max_flag_flip*100:.0f}% of LOS links are flagged as uncertain (>6 dB).")
        lines.append("  Systems that exclude flagged/uncertain links are vulnerable to anchor exclusion.")
    elif max_median_shift > 0.2 or max_hard_flip > 0.30:
        lines.append("★ VERDICT: PARTIAL (Weak Effect)")
        lines.append("  Some measurable bias shift and/or classifier confusion at ≤+6 dB,")
        lines.append("  but below the full-thesis thresholds. Attack has limited practical impact.")
    else:
        lines.append("★ VERDICT: FAIL (Floor Too High)")
        lines.append("  Neither regressors nor classifiers move meaningfully at ≤+6 dB.")
        lines.append("  The DW3000 DGC power constraint effectively prevents Phantom-NLOS.")

    return "\n".join(lines)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    raise SystemExit(main())

