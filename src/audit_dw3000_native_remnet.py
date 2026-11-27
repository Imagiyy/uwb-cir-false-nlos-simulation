"""audit_dw3000_native_remnet.py — Conditioned zero-shot audit + disjoint DW3000 REMNet-119 training & attack sweep.

Answers Option (b) before locking the Tier 2 proposal:
  1. Part A — Conditioned Zero-Shot Flip Rates (flips / clean_predicted_LOS):
     Recomputes clean-predicted LOS denominators for Fontaine-trained REMNet-172,
     REMNet-119, and LightGBM-Cls across IIoT19 (n=907), IIoT20 (n=300), and
     DW3000 (n=250), transforms existing sub--6 dB & Phase-1 sweeps into exact
     conditioned flip rates (verifying the identity directly in inference), and
     compares conditioned >=70% thresholds alpha_cls against closed-form alpha_max.

  2. Part B — Disjoint Santamaria DW3000 Native & Fine-Tuned REMNet-119 Audit:
     Uses Santamaria GroundTruth/points.csv spatial split (test pids {1..21, 100..117}
     vs train pids {51..76, 150..177} split 80/20 via GroupShuffleSplit on punto_id)
     labeled via Qorvo APS006 CIR criterion (RXP_minus_FPP_dB < 6.0 -> LOS, >= 10.0
     -> NLOS, excluding [6, 10) ambiguous band). Trains and evaluates:
       - ZeroShot-Fontaine-REMNet-119  (using train-only amplitude scaling)
       - FineTuned-REMNet-119          (Fontaine -> DW3000, lr=1e-4)
       - Native-DW3000-REMNet-119      (scratch, InSight lr=3e-4)
       - Native-DW3000-REMNet-119-Bal  (scratch with pos_weight class balance)
       - Native-DW3000-XGB-Emb         (D=1, T=30)
       - Native-DW3000-XGB-Full        (D=6, T=100)
       - Native-DW3000-LightGBM-Cls    (11 CIR waveform features)
     Verifies zero-injection identity (0.0% flips), fp_idx=20 lock, and OOD
     calibration LOS accuracy, then runs 5-phase complex-IQ sweep across
     power_dB in [-18..+6] and delta_t_ns in [10..50] for single & diffuse modes.

Saves:
  - results/dw3000_conditioned_zeroshot.csv
  - results/dw3000_native_remnet_sweep.csv
  - results/dw3000_native_remnet_report.txt
  - results/dw3000_native_remnet_audit.png
"""

from __future__ import annotations

import copy
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
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (load_fontaine_cirs, load_fontaine_iiot20,
                     load_fontaine_scalar, load_santamaria_dw3000)
from eval_insight import (INSIGHT_CIR_LEN, REMNET_CIR_LEN, REMNetPyTorch,
                          TransFitMinMaxMatrixSafe, TransFitMinMaxSafe,
                          batch_insight_windows, train_remnet)
from features import FEATURE_NAMES, compute_feature_matrix, compute_features_from_cir
from injector import (inject_diffuse_with_phase_averaging, inject_phantom_nlos,
                      inject_with_phase_averaging)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dw3000_native")

SEED = 42
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

DW_POWERS = [-18.0, -15.0, -12.0, -10.5, -9.0, -7.5, -6.0, -3.0, -1.5, 0.0, 3.0, 6.0]
DENSE_DELAYS = [10, 15, 18, 20, 22, 25, 30, 35, 50]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def alpha_max_bound(snr_sts_db: float, kappa_mp: float, p_fa: float = 0.05) -> float:
    """Closed-form cross-CIR stealth bound (N_STS=4096, M=75, P_d_max=0.10)."""
    N_STS = 4096.0
    M = 75.0
    P_d_max = 0.10
    s_sts = 10.0 ** (snr_sts_db / 10.0)
    s_ip = 10.0 ** ((snr_sts_db + 6.0) / 10.0)
    sigma_rel = np.sqrt(kappa_mp / N_STS + 1.0 / s_sts + 1.0 / s_ip)
    z_term = norm.ppf(1.0 - P_d_max) / np.sqrt(2.0)
    return float((np.sqrt(np.log(M / p_fa)) - z_term) * sigma_rel)


def train_remnet_custom(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    init_state: dict | None = None,
    lr: float = 3e-4,
    epochs: int = 45,
    batch_size: int = 128,
    pos_weight: float | None = None,
) -> REMNetPyTorch:
    """Train or fine-tune REMNet-119 with deterministic early stopping on val loss."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = REMNetPyTorch(input_len=X_tr.shape[1], N=3, F=16)
    if init_state is not None:
        net.load_state_dict(copy.deepcopy(init_state))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    if pos_weight is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    Xt_tr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1)
    yt_tr = torch.tensor(y_tr, dtype=torch.float32)
    Xt_va = torch.tensor(X_va, dtype=torch.float32).unsqueeze(1)
    yt_va = torch.tensor(y_va, dtype=torch.float32)

    best_v, best_st, wait = float("inf"), None, 0
    patience = 7
    n_tr = len(X_tr)
    for _ in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n_tr)
        for s in range(0, n_tr, batch_size):
            idx = perm[s : s + batch_size]
            loss = loss_fn(net(Xt_tr[idx]), yt_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(Xt_va), yt_va).item())
        if vl < best_v - 1e-5:
            best_v = vl
            best_st = {k: v.clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_st is not None:
        net.load_state_dict(best_st)
    net.eval()
    return net


def summarize_p70(curve: pd.Series, amax_table: dict) -> tuple[str, float | None]:
    """Compute exact grid & linearly-interpolated >=70% flip power threshold."""
    hits = curve[curve >= 0.70]
    if hits.empty:
        pk_p = float(curve.idxmax())
        pk_v = float(curve.max()) * 100.0
        return f"never >=70% (peak {pk_v:5.1f}% at {pk_p:+5.1f} dB)", None
    p70 = float(hits.index.min())
    a70 = float(10.0 ** (p70 / 20.0))
    idx_list = list(curve.index)
    pos = idx_list.index(p70)
    interp_str = ""
    if pos > 0 and curve.iloc[pos - 1] < 0.70:
        p0, p1 = float(idx_list[pos - 1]), p70
        f0, f1 = float(curve.iloc[pos - 1]), float(curve.iloc[pos])
        p_lin = p0 + (0.70 - f0) / max(f1 - f0, 1e-9) * (p1 - p0)
        interp_str = f" [lin={p_lin:+.2f} dB, α={10**(p_lin/20):.3f}]"
    b10 = a70 < amax_table[(10.0, 2)]
    b15 = a70 < amax_table[(15.0, 2)]
    b20 = a70 < amax_table[(20.0, 2)]
    return f"p70={p70:+5.1f} dB (α={a70:.3f}){interp_str} | <α_max(10/15/20dB,κ=2): {b10}/{b15}/{b20}", p70


def build_dw3000_splits() -> tuple[dict, list[dict]]:
    """Build strictly punto_id-disjoint Train/Val/Test DW3000 datasets + OOD calibration LOS."""
    pts = pd.read_csv("data/santamaria_dw3000/data/GroundTruth/points.csv").set_index("id")
    labeled_recs: list[dict] = []
    for sc in ["lab", "corridor"]:
        for r in load_santamaria_dw3000(sc):
            pid = r["punto_id"]
            if pid not in pts.index:
                continue
            feat = compute_features_from_cir(r["cir_mag"], r["fpi"])
            rxp_fpp = feat["RXP_minus_FPP_dB"]
            if rxp_fpp < 6.0:
                lbl = 0
            elif rxp_fpp >= 10.0:
                lbl = 1
            else:
                continue  # Exclude ambiguous [6, 10) dB hysteresis zone
            rec = dict(r)
            rec["scenario"] = sc
            rec["ptype"] = str(pts.loc[pid, "type"])
            rec["label"] = lbl
            rec["rxp_fpp"] = float(rxp_fpp)
            labeled_recs.append(rec)

    tr_val = [r for r in labeled_recs if r["ptype"] == "train"]
    te = [r for r in labeled_recs if r["ptype"] == "test"]
    pids_trv = np.array([r["punto_id"] for r in tr_val])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    ti, vi = next(gss.split(tr_val, groups=pids_trv))
    tr = [tr_val[i] for i in ti]
    va = [tr_val[i] for i in vi]

    p_tr = set(r["punto_id"] for r in tr)
    p_va = set(r["punto_id"] for r in va)
    p_te = set(r["punto_id"] for r in te)
    assert p_tr.isdisjoint(p_va) and p_tr.isdisjoint(p_te) and p_va.isdisjoint(p_te), "punto_id leakage!"

    # Calibration LOS set (<6 dB) held completely out of training/val as OOD clean LOS check
    cal_los: list[dict] = []
    for sc in ["calibration_anc0", "calibration_anchors"]:
        for r in load_santamaria_dw3000(sc):
            if compute_features_from_cir(r["cir_mag"], r["fpi"])["RXP_minus_FPP_dB"] < 6.0:
                cal_los.append(r)

    return {"train": tr, "val": va, "test": te}, cal_los


def main() -> int:
    os.chdir(Path(__file__).resolve().parent.parent)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    amax_table = {
        (snr, kap): alpha_max_bound(snr, kap, 0.05)
        for snr in (10.0, 15.0, 20.0)
        for kap in (1, 2)
    }

    # ================================================================
    # PART A: Exact Zero-Shot Conditioned Flip Rates (flips / clean_LOS)
    # ================================================================
    logger.info("=== Part A: Reconstructing Fontaine baselines & conditioned zero-shot denominators ===")
    df19 = load_fontaine_scalar()
    with open(MODELS_DIR / "split_indices.pkl", "rb") as f:
        splits19 = pickle.load(f)
    with open(MODELS_DIR / "lgbm_cls_cir.pkl", "rb") as f:
        lgbm19 = pickle.load(f)

    tr19, va19, te19 = splits19["train"], splits19["val"], splits19["test"]
    cirs19, fp19, _ = load_fontaine_cirs("IIoT19", len(df19))
    mags19 = np.abs(cirs19)
    y19 = df19["label"].values.astype(int)

    X172_all = batch_insight_windows(mags19, fp19, REMNET_CIR_LEN)
    X119_all = batch_insight_windows(mags19, fp19, INSIGHT_CIR_LEN)
    scaler172 = TransFitMinMaxMatrixSafe().fit(X172_all[tr19])
    scaler119 = TransFitMinMaxMatrixSafe().fit(X119_all[tr19])

    rem172_f = train_remnet(
        scaler172.transform(X172_all[tr19]), y19[tr19].astype(np.float32),
        scaler172.transform(X172_all[va19]), y19[va19].astype(np.float32),
        task="cls", epochs=45,
    )
    rem119_f = train_remnet(
        scaler119.transform(X119_all[tr19]), y19[tr19].astype(np.float32),
        scaler119.transform(X119_all[va19]), y19[va19].astype(np.float32),
        task="cls", epochs=45,
    )

    te19_los = te19[y19[te19] == 0]
    mags19_los = mags19[te19_los]
    fp19_los = fp19[te19_los]
    fontaine_los_med_peak = float(np.median(mags19_los.max(axis=1)))

    _, cirs20, fp20 = load_fontaine_iiot20()
    sub20 = np.linspace(0, len(cirs20) - 1, min(300, len(cirs20)), dtype=int)
    cirs20_los = cirs20[sub20]
    fp20_los = fp20[sub20]
    scale20 = fontaine_los_med_peak / max(float(np.median(np.abs(cirs20_los).max(axis=1))), 1e-6)

    # Exact 250-sample DW3000 subset used in audit_remnet_sub6db.py
    dw_recs_old = []
    for sc in ["calibration_anc0", "calibration_anchors", "lab", "corridor"]:
        for r in load_santamaria_dw3000(sc):
            thr = 10.0 if sc.startswith("calibration") else 6.0
            if compute_features_from_cir(r["cir_mag"], r["fpi"])["RXP_minus_FPP_dB"] < thr:
                dw_recs_old.append(r)
    dw_sel_old = np.sort(np.random.RandomState(SEED).choice(len(dw_recs_old), size=min(250, len(dw_recs_old)), replace=False))
    dw_los_old = [dw_recs_old[i] for i in dw_sel_old]
    dw_fp_old = np.array([r["fpi"] for r in dw_los_old], dtype=int)
    dw_mags_old = np.stack([r["cir_mag_pac"] for r in dw_los_old])
    scaledw_old = fontaine_los_med_peak / max(float(np.median(dw_mags_old.max(axis=1))), 1e-6)

    def fontaine_clean_preds(mags_arr: np.ndarray, fp_arr: np.ndarray, sc: float) -> dict[str, np.ndarray]:
        arr = mags_arr * sc
        w172 = scaler172.transform(batch_insight_windows(arr, fp_arr, REMNET_CIR_LEN))
        w119 = scaler119.transform(batch_insight_windows(arr, fp_arr, INSIGHT_CIR_LEN))
        fmat = compute_feature_matrix(arr, fp_arr)
        with torch.no_grad():
            p172 = _sigmoid(rem172_f(torch.tensor(w172).unsqueeze(1)).numpy())
            p119 = _sigmoid(rem119_f(torch.tensor(w119).unsqueeze(1)).numpy())
        return {
            "rem172": (p172 < 0.5),
            "rem119": (p119 < 0.5),
            "lgbm": (lgbm19.predict(fmat) < 0.5),
        }

    c19 = fontaine_clean_preds(mags19_los, fp19_los, 1.0)
    c20 = fontaine_clean_preds(np.abs(cirs20_los), fp20_los, scale20)
    cdw = fontaine_clean_preds(dw_mags_old, dw_fp_old, scaledw_old)

    denom_map = {}
    print("\n" + "=" * 88)
    print("PART A1: ZERO-SHOT CLEAN-PREDICTED LOS DENOMINATORS (Fontaine-Trained Models)")
    print("=" * 88)
    for ds_key, cd, n_tot in [
        ("fontaine_iiot19", c19, len(mags19_los)),
        ("fontaine_iiot20_5phi", c20, len(cirs20_los)),
        ("dw3000_complex_5phi", cdw, len(dw_los_old)),
    ]:
        line = f"  {ds_key:20s} (n={n_tot:3d}) | "
        parts = []
        for mk in ["rem172", "rem119", "lgbm"]:
            n_los = int(cd[mk].sum())
            frac = float(n_los / n_tot)
            denom_map[(ds_key, mk)] = (n_los, n_tot, frac)
            parts.append(f"{mk}={n_los}/{n_tot} ({frac*100:.1f}%)")
        print(line + " | ".join(parts))

    # Direct verification on DW3000 -6 dB @ 20 ns single-tap that cond_flip == uncond / frac
    dw_cx_old = [r["cir_complex"] for r in dw_los_old]
    dw_n_old = np.array([r["N"] for r in dw_los_old], dtype=float)
    d5_chk = [[] for _ in range(5)]
    for i in range(len(dw_cx_old)):
        _, _, m5 = inject_with_phase_averaging(dw_cx_old[i], int(dw_fp_old[i]), 20.0, -6.0, 5, SEED + i)
        for k in range(5):
            d5_chk[k].append(m5[k] / dw_n_old[i])
    cat_chk = np.concatenate([np.stack(d5_chk[k]) for k in range(5)], axis=0)
    p_chk = fontaine_clean_preds(cat_chk, np.tile(dw_fp_old, 5), scaledw_old)
    ndw_old = len(dw_cx_old)
    for mk in ["rem172", "rem119", "lgbm"]:
        uncond_dir = float(np.mean([
            np.mean(cdw[mk] & (~p_chk[mk][d * ndw_old : (d + 1) * ndw_old])) for d in range(5)
        ]))
        cond_dir = float(np.mean([
            np.mean(~p_chk[mk][d * ndw_old : (d + 1) * ndw_old][cdw[mk]]) for d in range(5)
        ]))
        ratio = uncond_dir / denom_map[("dw3000_complex_5phi", mk)][2]
        assert abs(cond_dir - ratio) < 1e-6, f"Conditioned identity mismatch: {cond_dir} vs {ratio}"
    print("  Verified direct inference identity on DW3000 (-6 dB, dt=20 ns): max|cond_direct - uncond/frac| < 1e-6.")

    # Transform existing CSVs cleanly into conditioned flip rates
    df_sub6 = pd.read_csv(RESULTS_DIR / "remnet_sub6db_audit.csv")
    df_p1 = pd.read_csv(RESULTS_DIR / "insight_low_power_sweep.csv")
    cond_rows = []
    for ds_key in ["fontaine_iiot19", "fontaine_iiot20_5phi", "dw3000_complex_5phi"]:
        for mode in ["single", "diffuse"]:
            s_sub = df_sub6[(df_sub6.dataset == ds_key) & (df_sub6["mode"] == mode)]
            s_p1 = df_p1[(df_p1.dataset == ds_key) & (df_p1["mode"] == mode) & (df_p1.power_dB > -6.0)]
            for mk, p1col in [("rem172", "flip_insight_remnet_cls"), ("rem119", None), ("lgbm", "flip_lgbm_cls")]:
                n_los, n_tot, frac = denom_map[(ds_key, mk)]
                c_sub = s_sub.groupby("power_dB")[f"flip_{mk}"].max() / frac
                if p1col is not None and not s_p1.empty:
                    c_p1 = s_p1.groupby("power_dB")[p1col].max() / frac
                    c_all = pd.concat([c_sub, c_p1]).sort_index().clip(upper=1.0)
                else:
                    c_all = c_sub.clip(upper=1.0)
                for pdb, val in c_all.items():
                    cond_rows.append({
                        "dataset": ds_key, "mode": mode, "model": mk,
                        "n_clean_los": n_los, "n_total": n_tot, "clean_los_frac": frac,
                        "power_dB": float(pdb),
                        "uncond_flip": float(val * frac),
                        "cond_flip": float(val),
                    })
    df_cond = pd.DataFrame(cond_rows)
    df_cond.to_csv(RESULTS_DIR / "dw3000_conditioned_zeroshot.csv", index=False, float_format="%.6f")

    report_lines = ["PART A: ZERO-SHOT CONDITIONED FLIP AUDIT (flips / clean_predicted_LOS)"]
    print("\n" + "=" * 88)
    print("PART A2: CONDITIONED >=70%-FLIP alpha_cls (flips / clean_predicted_LOS) vs alpha_max")
    print("=" * 88)
    for ds_key in ["fontaine_iiot19", "fontaine_iiot20_5phi", "dw3000_complex_5phi"]:
        for mk in ["rem172", "rem119", "lgbm"]:
            n_los, n_tot, _ = denom_map[(ds_key, mk)]
            for mode in ["single", "diffuse"]:
                sub = df_cond[(df_cond.dataset == ds_key) & (df_cond.model == mk) & (df_cond["mode"] == mode)].set_index("power_dB")["cond_flip"]
                msg, _ = summarize_p70(sub, amax_table)
                f_m6 = float(sub.loc[-6.0]) * 100.0 if -6.0 in sub.index else float("nan")
                line = f"  {ds_key:20s} | {mk:6s} ({n_los:3d}/{n_tot:3d}) | {mode:7s} | @-6dB={f_m6:5.1f}% | {msg}"
                print(line)
                report_lines.append(line.strip())

    # ================================================================
    # PART B: Native & Fine-Tuned DW3000 REMNet-119 Disjoint Evaluation
    # ================================================================
    logger.info("=== Part B: Building disjoint Santamaria DW3000 Train/Val/Test splits ===")
    dw_splits, cal_los = build_dw3000_splits()

    def unpack(recs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        mags = np.stack([r["cir_mag_pac"] for r in recs])
        fps = np.array([r["fpi"] for r in recs], dtype=int)
        ys = np.array([r.get("label", 0) for r in recs], dtype=int)
        cxs = [r["cir_complex"] for r in recs]
        return mags, fps, ys, cxs

    m_tr, fp_tr, y_tr, _ = unpack(dw_splits["train"])
    m_va, fp_va, y_va, _ = unpack(dw_splits["val"])
    m_te, fp_te, y_te, _ = unpack(dw_splits["test"])
    m_cal, fp_cal, _, _ = unpack(cal_los)

    # Train-only scaling factor for fine-tuning / zero-shot (zero test leakage)
    dw_tr_los_med = float(np.median(m_tr[y_tr == 0].max(axis=1)))
    scaledw_tr = fontaine_los_med_peak / max(dw_tr_los_med, 1e-6)

    X119_tr_raw = batch_insight_windows(m_tr, fp_tr, INSIGHT_CIR_LEN)
    X119_va_raw = batch_insight_windows(m_va, fp_va, INSIGHT_CIR_LEN)
    X119_te_raw = batch_insight_windows(m_te, fp_te, INSIGHT_CIR_LEN)
    X119_cal_raw = batch_insight_windows(m_cal, fp_cal, INSIGHT_CIR_LEN)

    X119_tr_f = X119_tr_raw * scaledw_tr
    X119_va_f = X119_va_raw * scaledw_tr
    X119_te_f = X119_te_raw * scaledw_tr
    X119_cal_f = X119_cal_raw * scaledw_tr

    # Fit native DW3000 scalers ONLY on Train
    dw_rem_scaler = TransFitMinMaxMatrixSafe().fit(X119_tr_raw)
    dw_xgb_scaler = TransFitMinMaxSafe().fit(X119_tr_raw)

    logger.info("Training FineTuned-REMNet-119 (lr=1e-4) and Native-DW3000 REMNet-119 (scratch & balanced)...")
    rem_ft = train_remnet_custom(
        scaler119.transform(X119_tr_f), y_tr.astype(np.float32),
        scaler119.transform(X119_va_f), y_va.astype(np.float32),
        init_state=rem119_f.state_dict(), lr=1e-4, epochs=35,
    )
    rem_nat = train_remnet_custom(
        dw_rem_scaler.transform(X119_tr_raw), y_tr.astype(np.float32),
        dw_rem_scaler.transform(X119_va_raw), y_va.astype(np.float32),
        init_state=None, lr=3e-4, epochs=45,
    )
    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    rem_nat_bal = train_remnet_custom(
        dw_rem_scaler.transform(X119_tr_raw), y_tr.astype(np.float32),
        dw_rem_scaler.transform(X119_va_raw), y_va.astype(np.float32),
        init_state=None, lr=3e-4, epochs=45, pos_weight=pos_w,
    )

    xgb_nat_emb = XGBClassifier(
        max_depth=1, n_estimators=30, learning_rate=0.1, subsample=0.6, gamma=0.0,
        objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=10,
        random_state=SEED, n_jobs=2,
    )
    xgb_nat_emb.fit(
        dw_xgb_scaler.transform(X119_tr_raw), y_tr,
        eval_set=[(dw_xgb_scaler.transform(X119_va_raw), y_va)], verbose=False,
    )
    xgb_nat_full = XGBClassifier(
        max_depth=6, n_estimators=100, learning_rate=0.1, subsample=0.6, gamma=0.0,
        objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=10,
        random_state=SEED, n_jobs=2,
    )
    xgb_nat_full.fit(
        dw_xgb_scaler.transform(X119_tr_raw), y_tr,
        eval_set=[(dw_xgb_scaler.transform(X119_va_raw), y_va)], verbose=False,
    )
    lgb_nat = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=2, verbose=-1,
    )
    F_dw_tr = compute_feature_matrix(m_tr, fp_tr)
    F_dw_va = compute_feature_matrix(m_va, fp_va)
    F_dw_te = compute_feature_matrix(m_te, fp_te)
    F_dw_cal = compute_feature_matrix(m_cal, fp_cal)
    lgb_nat.fit(
        F_dw_tr, y_tr,
        eval_set=[(F_dw_va, y_va)],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )

    def infer_dw_suite(mags_arr: np.ndarray, fp_arr: np.ndarray, fmat: np.ndarray | None = None) -> dict[str, np.ndarray]:
        w_raw = batch_insight_windows(mags_arr, fp_arr, INSIGHT_CIR_LEN)
        w_f_s = scaler119.transform(w_raw * scaledw_tr)
        w_nat_s = dw_rem_scaler.transform(w_raw)
        w_xgb_s = dw_xgb_scaler.transform(w_raw)
        if fmat is None:
            fmat = compute_feature_matrix(mags_arr, fp_arr)
        with torch.no_grad():
            t_fs = torch.tensor(w_f_s).unsqueeze(1)
            t_ns = torch.tensor(w_nat_s).unsqueeze(1)
            p_zs = _sigmoid(rem119_f(t_fs).numpy())
            p_ft = _sigmoid(rem_ft(t_fs).numpy())
            p_nat = _sigmoid(rem_nat(t_ns).numpy())
            p_bal = _sigmoid(rem_nat_bal(t_ns).numpy())
        return {
            "zs_rem119": p_zs,
            "ft_rem119": p_ft,
            "nat_rem119": p_nat,
            "nat_rem119_bal": p_bal,
            "nat_xgb_emb": xgb_nat_emb.predict_proba(w_xgb_s)[:, 1],
            "nat_xgb_full": xgb_nat_full.predict_proba(w_xgb_s)[:, 1],
            "nat_lgbm": lgb_nat.predict_proba(fmat)[:, 1],
        }

    dw_te_probs = infer_dw_suite(m_te, fp_te, F_dw_te)
    dw_cal_probs = infer_dw_suite(m_cal, fp_cal, F_dw_cal)

    print("\n" + "=" * 88)
    print(f"PART B1: CLEAN HELD-OUT DW3000 TEST METRICS (disjoint 39 test pids, n={len(y_te)}: LOS={(y_te==0).sum()}, NLOS={(y_te==1).sum()})")
    print("=" * 88)
    report_lines.append("")
    report_lines.append(f"PART B: DISJOINT DW3000 CLEAN METRICS (Train=2514 [42 pids], Val=742 [11 pids], Test={len(y_te)} [39 pids])")
    clean_te_los_frac = {}
    for mkey, mlbl in [
        ("zs_rem119", "ZeroShot-Fontaine-REMNet-119"),
        ("ft_rem119", "FineTuned-REMNet-119"),
        ("nat_rem119", "Native-DW3000-REMNet-119"),
        ("nat_rem119_bal", "Native-DW3000-REMNet-119-Bal"),
        ("nat_xgb_emb", "Native-DW3000-XGB-Emb"),
        ("nat_xgb_full", "Native-DW3000-XGB-Full"),
        ("nat_lgbm", "Native-DW3000-LightGBM"),
    ]:
        pr = dw_te_probs[mkey]
        pred = (pr >= 0.5).astype(int)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, zero_division=0)
        auc = roc_auc_score(y_te, pr)
        los_acc = recall_score(y_te, pred, pos_label=0)
        nlos_acc = recall_score(y_te, pred, pos_label=1)
        cal_los_acc = float(np.mean(dw_cal_probs[mkey] < 0.5))
        clean_te_los_frac[mkey] = los_acc
        s_clean = (f"  {mlbl:28s} | Acc={acc:.3f} | F1={f1:.3f} | AUROC={auc:.3f} | "
                   f"LOS_rec={los_acc:.3f} | NLOS_rec={nlos_acc:.3f} | OOD_Cal_LOS_rec={cal_los_acc:.3f}")
        print(s_clean)
        report_lines.append(s_clean.strip())

    # Build stratified held-out DW3000 Test LOS subset (150 lab + 150 corridor = 300)
    rng = np.random.RandomState(SEED)
    te_lab_los = [r for r in dw_splits["test"] if r["label"] == 0 and r["scenario"] == "lab"]
    te_cor_los = [r for r in dw_splits["test"] if r["label"] == 0 and r["scenario"] == "corridor"]
    sel_lab = np.sort(rng.choice(len(te_lab_los), size=min(150, len(te_lab_los)), replace=False))
    sel_cor = np.sort(rng.choice(len(te_cor_los), size=min(150, len(te_cor_los)), replace=False))
    atk_recs = [te_lab_los[i] for i in sel_lab] + [te_cor_los[i] for i in sel_cor]
    m_atk, fp_atk, _, cx_atk = unpack(atk_recs)
    n_atk_arr = np.array([r["N"] for r in atk_recs], dtype=float)
    n_atk = len(atk_recs)

    clean_atk_probs = infer_dw_suite(m_atk, fp_atk)
    clean_atk_los = {k: (v < 0.5) for k, v in clean_atk_probs.items()}

    # Zero-injection check on DW3000 suite
    z_dw = np.stack([
        np.abs(inject_phantom_nlos(cx_atk[i], int(fp_atk[i]), 20.0, -np.inf, is_complex=True, phi=0.5)) / n_atk_arr[i]
        for i in range(n_atk)
    ])
    z_probs = infer_dw_suite(z_dw, fp_atk)
    for k in clean_atk_los:
        z_flips = float(np.mean(clean_atk_los[k] & (z_probs[k] >= 0.5)))
        assert z_flips == 0.0, f"Zero-injection failed for {k}: {z_flips}"
    logger.info("DW3000 Native Suite Zero-Injection Identity Check: PASS (0.00%% flips across all 7 models)")

    # Run 5-phase sweep over [-18..+6 dB] x [10..50 ns] for single & diffuse
    model_keys = ["zs_rem119", "ft_rem119", "nat_rem119", "nat_rem119_bal", "nat_xgb_emb", "nat_xgb_full", "nat_lgbm"]
    logger.info("=== Running DW3000 Native/Fine-Tuned Attack Sweep (12 powers x 9 delays x 2 modes, n=%d x 5φ) ===", n_atk)
    sweep_rows = []
    for mode in ["single", "diffuse"]:
        for dt in DENSE_DELAYS:
            logger.info("DW3000 Native Sweep: mode=%s, delta_t=%d ns ...", mode, dt)
            for pdb in DW_POWERS:
                draws = [[] for _ in range(5)]
                for i in range(n_atk):
                    if mode == "single":
                        _, _, m5 = inject_with_phase_averaging(cx_atk[i], int(fp_atk[i]), dt, pdb, 5, SEED + i)
                    else:
                        _, _, m5 = inject_diffuse_with_phase_averaging(cx_atk[i], int(fp_atk[i]), dt, pdb, 5, SEED + i)
                    for d in range(5):
                        draws[d].append(m5[d] / n_atk_arr[i])
                cat_m = np.concatenate([np.stack(draws[d]) for d in range(5)], axis=0)
                cat_fp = np.tile(fp_atk, 5)
                cat_p = infer_dw_suite(cat_m, cat_fp)

                row = {"mode": mode, "delta_t_ns": dt, "power_dB": pdb}
                for mk in model_keys:
                    cmask = clean_atk_los[mk]
                    n_cl = int(cmask.sum())
                    row[f"n_clean_{mk}"] = n_cl
                    uncond_vals = []
                    cond_vals = []
                    for d in range(5):
                        pred_nlos = cat_p[mk][d * n_atk : (d + 1) * n_atk] >= 0.5
                        uncond_vals.append(float(np.mean(cmask & pred_nlos)))
                        cond_vals.append(float(np.mean(pred_nlos[cmask])) if n_cl > 0 else 0.0)
                    row[f"uncond_{mk}"] = float(np.mean(uncond_vals))
                    row[f"cond_{mk}"] = float(np.mean(cond_vals))
                sweep_rows.append(row)

    df_nat = pd.DataFrame(sweep_rows)
    df_nat.to_csv(RESULTS_DIR / "dw3000_native_remnet_sweep.csv", index=False, float_format="%.6f")
    logger.info("Saved results/dw3000_native_remnet_sweep.csv (%d rows)", len(df_nat))

    print("\n" + "=" * 88)
    print(f"PART B2: DW3000 TEST LOS SUBSET (n={n_atk}) — CONDITIONED FLIP THRESHOLDS vs alpha_max")
    print("=" * 88)
    report_lines.append("")
    report_lines.append(f"PART B2: CONDITIONED >=70%-FLIP THRESHOLDS ON HELD-OUT DW3000 TEST LOS (n={n_atk}, 5φ)")
    for mk, mlbl in [
        ("zs_rem119", "ZeroShot-Fontaine-REMNet-119"),
        ("ft_rem119", "FineTuned-REMNet-119"),
        ("nat_rem119", "Native-DW3000-REMNet-119"),
        ("nat_rem119_bal", "Native-DW3000-REMNet-119-Bal"),
        ("nat_xgb_emb", "Native-DW3000-XGB-Emb"),
        ("nat_xgb_full", "Native-DW3000-XGB-Full"),
        ("nat_lgbm", "Native-DW3000-LightGBM"),
    ]:
        n_cl = int(clean_atk_los[mk].sum())
        for mode in ["single", "diffuse"]:
            sub = df_nat[df_nat["mode"] == mode].groupby("power_dB")[f"cond_{mk}"].max()
            msg, _ = summarize_p70(sub, amax_table)
            f_m105 = float(sub.loc[-10.5]) * 100.0
            f_m60 = float(sub.loc[-6.0]) * 100.0
            f_p00 = float(sub.loc[0.0]) * 100.0
            line = (f"  {mlbl:28s} ({n_cl:3d}/{n_atk}) | {mode:7s} | "
                    f"@-10.5={f_m105:5.1f}% @-6={f_m60:5.1f}% @0={f_p00:5.1f}% | {msg}")
            print(line)
            report_lines.append(line.strip())

    # Print smoothness over delay dt at -6 dB & 0 dB for FineTuned & Native REMNet-119
    print("\n-- Delay smoothness on DW3000 Test LOS (single, conditioned %) --")
    for pdb_chk in [-9.0, -6.0, 0.0]:
        s_d = df_nat[(df_nat["mode"] == "single") & (df_nat.power_dB == pdb_chk)].sort_values("delta_t_ns")
        for mk, tag in [("ft_rem119", "FineTuned"), ("nat_rem119", "Native"), ("nat_rem119_bal", "Nat-Bal")]:
            vals = " ".join(f"{int(r.delta_t_ns)}:{r[f'cond_{mk}']*100:4.1f}%" for _, r in s_d.iterrows())
            print(f"  {pdb_chk:+5.1f}dB {tag:9s}: {vals}")

    (RESULTS_DIR / "dw3000_native_remnet_report.txt").write_text("\n".join(report_lines) + "\n")
    _plot_native_audit(df_cond, df_nat, amax_table)
    return 0


def _plot_native_audit(df_cond: pd.DataFrame, df_nat: pd.DataFrame, amax_table: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))

    # Left panel: Conditioned Zero-Shot Curves (IIoT19 vs IIoT20 vs DW3000)
    ax = axes[0]
    ax.axhline(70.0, color="#c62828", linestyle="--", linewidth=1.2, label="70% Flip Threshold")
    for snr, col in [(10.0, "#1565C0"), (15.0, "#2E7D32"), (20.0, "#6A1B9A")]:
        adb = 20.0 * np.log10(amax_table[(snr, 2)])
        ax.axvline(adb, color=col, linestyle=":", linewidth=1.4, label=f"α_max({int(snr)}dB) = {adb:.1f} dB")

    zs_specs = [
        ("fontaine_iiot19", "rem119", "single", "IIoT19 ZeroShot REMNet-119 (Single)", "#E65100", "-", "o"),
        ("fontaine_iiot19", "rem119", "diffuse", "IIoT19 ZeroShot REMNet-119 (Diffuse)", "#E65100", "--", "^"),
        ("dw3000_complex_5phi", "rem119", "single", "DW3000 ZeroShot REMNet-119 Cond (Single)", "#00695C", "-", "s"),
        ("dw3000_complex_5phi", "lgbm", "diffuse", "DW3000 ZeroShot LightGBM Cond (Diffuse)", "#AD1457", "-.", "P"),
    ]
    for ds, mk, mode, lbl, col, ls, mk_s in zs_specs:
        sub = df_cond[(df_cond.dataset == ds) & (df_cond.model == mk) & (df_cond["mode"] == mode)].sort_values("power_dB")
        ax.plot(sub["power_dB"], sub["cond_flip"] * 100.0, color=col, linestyle=ls, marker=mk_s,
                linewidth=2.0, markersize=5.5, label=lbl)
    ax.set_title("Part A: Zero-Shot Conditioned Flip Rates (flips / clean LOS)")
    ax.set_xlabel("Injected Power (dB relative to LOS first path)")
    ax.set_ylabel("Conditioned LOS → NLOS Flip Rate (%)")
    ax.set_ylim(-4, 106)
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8.2, loc="lower right")

    # Right panel: Native & Fine-Tuned DW3000 REMNet-119 vs XGB-Emb (Test LOS, 5φ)
    ax = axes[1]
    ax.axhline(70.0, color="#c62828", linestyle="--", linewidth=1.2, label="70% Flip Threshold")
    for snr, col in [(10.0, "#1565C0"), (15.0, "#2E7D32"), (20.0, "#6A1B9A")]:
        adb = 20.0 * np.log10(amax_table[(snr, 2)])
        ax.axvline(adb, color=col, linestyle=":", linewidth=1.4)

    nat_specs = [
        ("zs_rem119", "single", "ZeroShot Fontaine REMNet-119 Cond (Single)", "#78909C", ":", "v"),
        ("ft_rem119", "single", "FineTuned DW3000 REMNet-119 (Single)", "#E65100", "-", "o"),
        ("ft_rem119", "diffuse", "FineTuned DW3000 REMNet-119 (Diffuse)", "#FB8C00", "--", "^"),
        ("nat_rem119", "single", "Native DW3000 REMNet-119 Scratch (Single)", "#1565C0", "-", "s"),
        ("nat_rem119_bal", "diffuse", "Native DW3000 REMNet-119 Bal (Diffuse)", "#2E7D32", "--", "D"),
        ("nat_lgbm", "diffuse", "Native DW3000 LightGBM (Diffuse)", "#AD1457", "-.", "P"),
        ("nat_xgb_emb", "single", "Native DW3000 XGB-Emb D=1 (Single)", "#37474F", "-", "X"),
    ]
    for mk, mode, lbl, col, ls, mk_s in nat_specs:
        sub = df_nat[df_nat["mode"] == mode].groupby("power_dB")[f"cond_{mk}"].max()
        ax.plot(sub.index, sub.values * 100.0, color=col, linestyle=ls, marker=mk_s,
                linewidth=2.0, markersize=5.2, label=lbl)

    ax.set_title("Part B: Disjoint Santamaria DW3000 Native & Fine-Tuned Sweep (5φ)")
    ax.set_xlabel("Injected Power (dB relative to LOS first path)")
    ax.set_ylabel("Max Conditioned Flip Rate across Δt (%)")
    ax.set_ylim(-4, 106)
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8.0, loc="upper left")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "dw3000_native_remnet_audit.png", dpi=180)
    plt.close(fig)
    logger.info("Saved results/dw3000_native_remnet_audit.png")


if __name__ == "__main__":
    raise SystemExit(main())

