"""audit_remnet_sub6db.py — Move 1 audit: verify REMNet & diffuse LightGBM below -6 dB.

Mandatory checks:
  1. Zero-injection identity test (scale = 0.0 -> flip_rate == 0.0% on IIoT19, IIoT20, DW3000).
  2. Sub--6 dB & dense-delay sweep:
       power_dB   in [-18, -15, -12, -10.5, -9, -7.5, -6]
       delta_t_ns in [10, 15, 18, 20, 22, 25, 30, 35, 50]
     for InSight-REMNet-Cls and diffuse LightGBM-Cls (plus single LightGBM-Cls).
  3. Alignment & padding ablation:
     - Verify ROLL_CIR=20 fp_idx stays locked at tap 20 after injection (direct taps 0..23
       untouched AND simulated leading-edge LDE detector remains locked at tap 20).
     - Retrain & evaluate REMNet on native 119-tap window (NO [119:172] zero-padding).
  4. Print exact 70%-flip alpha_cls (in linear & dB) vs closed-form alpha_max at
     SNR_STS = 10, 15, and 20 dB (kappa_mp in {1, 2}; P_fa = 0.05).

Saves:
  - results/remnet_sub6db_audit.csv
  - results/remnet_sub6db_audit.png
  - results/remnet_sub6db_report.txt
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
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (load_fontaine_cirs, load_fontaine_iiot20,
                     load_fontaine_scalar, load_santamaria_dw3000)
from eval_insight import (INSIGHT_CIR_LEN, REMNET_CIR_LEN,
                          TransFitMinMaxMatrixSafe, batch_insight_windows,
                          train_remnet)
from features import FEATURE_NAMES, compute_feature_matrix, compute_features_from_cir
from injector import (inject_diffuse_nlos, inject_diffuse_with_phase_averaging,
                      inject_phantom_nlos, inject_with_phase_averaging)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audit_remnet")

SEED = 42
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

SUB6_POWERS = [-18.0, -15.0, -12.0, -10.5, -9.0, -7.5, -6.0]
DENSE_DELAYS = [10, 15, 18, 20, 22, 25, 30, 35, 50]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def alpha_max_bound(snr_sts_db: float, kappa_mp: float, p_fa: float = 0.05) -> float:
    N_STS = 4096.0
    M = 75.0
    P_d_max = 0.10
    s_sts = 10.0 ** (snr_sts_db / 10.0)
    s_ip  = 10.0 ** ((snr_sts_db + 6.0) / 10.0)
    sigma_rel = np.sqrt(kappa_mp / N_STS + 1.0 / s_sts + 1.0 / s_ip)
    z_term = norm.ppf(1.0 - P_d_max) / np.sqrt(2.0)
    return float((np.sqrt(np.log(M / p_fa)) - z_term) * sigma_rel)


def detect_lde_fp_roll20(win119: np.ndarray) -> np.ndarray:
    """Estimate leading-edge first-path tap index on ROLL_CIR=20 aligned window.

    Uses pre-FP noise baseline [0:15] (mu + 6*sigma) and first rising crossing
    relative to main-lobe search around [16:25]. Confirms delayed pulse never
    shifts early leading-edge detection off tap 20.
    """
    detected = np.empty(len(win119), dtype=int)
    for i, c in enumerate(win119):
        noise = c[:15]
        thr = float(np.mean(noise) + 6.0 * np.std(noise))
        # Also include local fraction of direct peak (10% of peak in [18:24])
        fp_peak = float(np.max(c[18:24]))
        thr = max(thr, 0.20 * fp_peak)
        hits = np.flatnonzero(c[16:] > thr)
        detected[i] = 16 + int(hits[0]) if len(hits) > 0 else 20
    return detected


def main() -> int:
    os.chdir(Path(__file__).resolve().parent.parent)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- Load datasets & exact disjoint split ----
    df = load_fontaine_scalar()
    with open(MODELS_DIR / "split_indices.pkl", "rb") as f:
        splits = pickle.load(f)
    with open(MODELS_DIR / "lgbm_cls_cir.pkl", "rb") as f:
        lgbm_cls = pickle.load(f)

    tr_idx, va_idx, te_idx = splits["train"], splits["val"], splits["test"]
    assert set(df.iloc[te_idx]["location_ID"]) == {16, 18}

    cirs19, fp19, is_cx19 = load_fontaine_cirs("IIoT19", len(df))
    assert cirs19 is not None and not is_cx19
    mags19 = np.abs(cirs19)
    y_cls = df["label"].values.astype(int)

    # ---- Train REMNet-172 (padded) and REMNet-119 (native unpadded) ----
    logger.info("Training InSight-REMNet-Cls (172-tap padded) and (119-tap native unpadded)...")
    X172_all = batch_insight_windows(mags19, fp19, REMNET_CIR_LEN)
    X119_all = batch_insight_windows(mags19, fp19, INSIGHT_CIR_LEN)

    scaler172 = TransFitMinMaxMatrixSafe().fit(X172_all[tr_idx])
    scaler119 = TransFitMinMaxMatrixSafe().fit(X119_all[tr_idx])

    rem172 = train_remnet(
        scaler172.transform(X172_all[tr_idx]), y_cls[tr_idx].astype(np.float32),
        scaler172.transform(X172_all[va_idx]), y_cls[va_idx].astype(np.float32),
        task="cls", epochs=45,
    )
    rem119 = train_remnet(
        scaler119.transform(X119_all[tr_idx]), y_cls[tr_idx].astype(np.float32),
        scaler119.transform(X119_all[va_idx]), y_cls[va_idx].astype(np.float32),
        task="cls", epochs=45,
    )

    print("\n" + "=" * 82)
    print("CHECK 3a: CLEAN HELD-OUT TEST METRICS — REMNet-172 (Padded) vs REMNet-119 (Unpadded)")
    print("=" * 82)
    with torch.no_grad():
        p172 = _sigmoid(rem172(torch.tensor(scaler172.transform(X172_all[te_idx])).unsqueeze(1)).numpy())
        p119 = _sigmoid(rem119(torch.tensor(scaler119.transform(X119_all[te_idx])).unsqueeze(1)).numpy())
    for name, p in [("InSight-REMNet-172 (padded)", p172), ("InSight-REMNet-119 (native)", p119)]:
        pred = (p >= 0.5).astype(int)
        print(f"  {name:28s} | Acc={accuracy_score(y_cls[te_idx], pred):.3f} | "
              f"F1={f1_score(y_cls[te_idx], pred):.3f} | AUROC={roc_auc_score(y_cls[te_idx], p):.3f}")

    # ---- Prepare evaluation LOS subsets (IIoT19, IIoT20, DW3000) ----
    te_los_idx = te_idx[y_cls[te_idx] == 0]
    mags19_los = mags19[te_los_idx]
    fp19_los = fp19[te_los_idx]
    fontaine_los_med_peak = float(np.median(mags19_los.max(axis=1)))

    _, cirs20, fp20 = load_fontaine_iiot20()
    sub20 = np.linspace(0, len(cirs20) - 1, min(300, len(cirs20)), dtype=int)
    cirs20_los = cirs20[sub20]
    fp20_los = fp20[sub20]
    scale20 = fontaine_los_med_peak / max(float(np.median(np.abs(cirs20_los).max(axis=1))), 1e-6)

    dw_recs = []
    for scen in ["calibration_anc0", "calibration_anchors", "lab", "corridor"]:
        for r in load_santamaria_dw3000(scen):
            f = compute_features_from_cir(r["cir_mag"], r["fpi"])
            thr = 10.0 if scen.startswith("calibration") else 6.0
            if f["RXP_minus_FPP_dB"] < thr:
                dw_recs.append(r)
    rng_dw = np.random.RandomState(SEED)
    dw_sel = np.sort(rng_dw.choice(len(dw_recs), size=min(250, len(dw_recs)), replace=False))
    dw_los = [dw_recs[i] for i in dw_sel]
    dw_cx = [r["cir_complex"] for r in dw_los]
    dw_fp = np.array([r["fpi"] for r in dw_los], dtype=int)
    dw_n = np.array([r["N"] for r in dw_los], dtype=float)
    dw_pac_mags = np.stack([r["cir_mag_pac"] for r in dw_los])
    scaledw = fontaine_los_med_peak / max(float(np.median(dw_pac_mags.max(axis=1))), 1e-6)

    def infer_models(mags_arr: np.ndarray, fp_arr: np.ndarray, pac_scale: float) -> dict[str, np.ndarray]:
        scaled = mags_arr * pac_scale
        w172 = scaler172.transform(batch_insight_windows(scaled, fp_arr, REMNET_CIR_LEN))
        w119 = scaler119.transform(batch_insight_windows(scaled, fp_arr, INSIGHT_CIR_LEN))
        fmat = compute_feature_matrix(scaled, fp_arr)
        with torch.no_grad():
            l172 = rem172(torch.tensor(w172).unsqueeze(1)).numpy()
            l119 = rem119(torch.tensor(w119).unsqueeze(1)).numpy()
        p_lgbm = lgbm_cls.predict(fmat)
        return {
            "rem172": (_sigmoid(l172) >= 0.5).astype(float),
            "rem119": (_sigmoid(l119) >= 0.5).astype(float),
            "lgbm":   (p_lgbm >= 0.5).astype(float),
            "logit172": l172,
            "logit119": l119,
            "w119": batch_insight_windows(scaled, fp_arr, INSIGHT_CIR_LEN),
        }

    clean19 = infer_models(mags19_los, fp19_los, 1.0)
    clean20 = infer_models(np.abs(cirs20_los), fp20_los, scale20)
    cleandw = infer_models(dw_pac_mags, dw_fp, scaledw)

    # ---- CHECK 1: Zero-Injection Identity Test (amplitude = 0.0 -> flip == 0.0%) ----
    print("\n" + "=" * 82)
    print("CHECK 1: ZERO-INJECTION IDENTITY TEST (injected amplitude = 0.0)")
    print("=" * 82)
    zero_19 = np.stack([inject_phantom_nlos(mags19_los[i], int(fp19_los[i]), 20.0, -np.inf, is_complex=False)
                        for i in range(len(mags19_los))])
    zero_20 = np.stack([np.abs(inject_phantom_nlos(cirs20_los[i], int(fp20_los[i]), 20.0, -np.inf, is_complex=True, phi=0.5))
                        for i in range(len(cirs20_los))])
    zero_dw = np.stack([np.abs(inject_phantom_nlos(dw_cx[i], int(dw_fp[i]), 20.0, -np.inf, is_complex=True, phi=0.5)) / dw_n[i]
                        for i in range(len(dw_cx))])

    for ds_name, c_dict, z_mags, fp_arr, sc in [
        ("IIoT19", clean19, zero_19, fp19_los, 1.0),
        ("IIoT20", clean20, zero_20, fp20_los, scale20),
        ("DW3000", cleandw, zero_dw, dw_fp, scaledw),
    ]:
        z_pred = infer_models(z_mags, fp_arr, sc)
        max_w_diff = float(np.max(np.abs(z_pred["w119"] - c_dict["w119"])))
        flips = {
            k: float(np.mean((c_dict[k] == 0.0) & (z_pred[k] == 1.0))) * 100.0
            for k in ["rem172", "rem119", "lgbm"]
        }
        print(f"  {ds_name:6s} | max|w_zero - w_clean|={max_w_diff:.2e} | "
              f"flips: REMNet-172={flips['rem172']:.2f}%, REMNet-119={flips['rem119']:.2f}%, LGBM={flips['lgbm']:.2f}%")
        assert max_w_diff == 0.0 and all(v == 0.0 for v in flips.values()), f"Zero-injection failed on {ds_name}!"
    print("  PASS: All zero-injection flip rates == 0.0% across IIoT19, IIoT20, and DW3000.")

    # ---- CHECK 3b: First-Path Lock Verification under Attack ----
    print("\n" + "=" * 82)
    print("CHECK 3b: ALIGNMENT & FP LOCK VERIFICATION (ROLL_CIR=20)")
    print("=" * 82)
    clean_lde_19 = detect_lde_fp_roll20(clean19["w119"])
    clean_lde_dw = detect_lde_fp_roll20(cleandw["w119"])
    for mode_name, fn_s in [("single -6dB dt=10ns", lambda c, f: inject_phantom_nlos(c, f, 10.0, -6.0, is_complex=False)),
                            ("diffuse -6dB dt=10ns", lambda c, f: inject_diffuse_nlos(c, f, 10.0, -6.0, is_complex=False))]:
        atk_w = np.stack([
            batch_insight_windows(np.array([fn_s(mags19_los[i], int(fp19_los[i]))]), np.array([fp19_los[i]]), INSIGHT_CIR_LEN)[0]
            for i in range(len(mags19_los))
        ])
        fp_region_diff = float(np.max(np.abs(atk_w[:, :24] - clean19["w119"][:, :24])))
        atk_lde = detect_lde_fp_roll20(atk_w)
        same_lde = float(np.mean(atk_lde == clean_lde_19)) * 100.0
        print(f"  IIoT19 {mode_name:22s}: max|taps[0:24] diff|={fp_region_diff:.2e} | LDE fp unchanged={same_lde:.1f}% (clean med fp={int(np.median(clean_lde_19))})")
        assert fp_region_diff == 0.0 and same_lde == 100.0
    # Check DW3000 at +0 dB, dt=10 ns
    atk_dw10 = np.stack([
        np.abs(inject_phantom_nlos(dw_cx[i], int(dw_fp[i]), 10.0, 0.0, is_complex=True, phi=0.0)) / dw_n[i]
        for i in range(len(dw_cx))
    ]) * scaledw
    atk_dw10_w = batch_insight_windows(atk_dw10, dw_fp, INSIGHT_CIR_LEN)
    dw_fp_diff = float(np.max(np.abs(atk_dw10_w[:, :24] - cleandw["w119"][:, :24])))
    dw_lde_same = float(np.mean(detect_lde_fp_roll20(atk_dw10_w) == clean_lde_dw)) * 100.0
    print(f"  DW3000 single  0dB dt=10ns: max|taps[0:24] diff|={dw_fp_diff:.2e} | LDE fp unchanged={dw_lde_same:.1f}% (clean med fp={int(np.median(clean_lde_dw))})")
    assert dw_fp_diff < 1e-4 and dw_lde_same == 100.0
    print("  PASS: fp_idx locked at index 20 and taps [0:24] untouched across all modes.")

    # ---- CHECK 2: Sub--6 dB & Dense-Delay Sweep ----
    logger.info("=== Check 2: Running sub--6 dB & dense-delay sweep (7 powers x 9 delays x 2 modes) ===")
    rows = []

    def add_rows(ds_name, mode_name, dt, pdb, clean_d, atk_list):
        r = {"dataset": ds_name, "mode": mode_name, "delta_t_ns": dt, "power_dB": pdb}
        for k in ["rem172", "rem119", "lgbm"]:
            r[f"flip_{k}"] = float(np.mean([
                np.mean((clean_d[k] == 0.0) & (ad[k] == 1.0)) for ad in atk_list
            ]))
        r["dlogit_rem172"] = float(np.mean([ad["logit172"] - clean_d["logit172"] for ad in atk_list]))
        r["dlogit_rem119"] = float(np.mean([ad["logit119"] - clean_d["logit119"] for ad in atk_list]))
        rows.append(r)

    for mode_name in ["single", "diffuse"]:
        for dt in DENSE_DELAYS:
            logger.info("Sub--6dB sweep: mode=%s, delta_t=%d ns ...", mode_name, dt)
            for pdb in SUB6_POWERS:
                # IIoT19
                if mode_name == "single":
                    a19 = np.stack([inject_phantom_nlos(mags19_los[i], int(fp19_los[i]), dt, pdb, is_complex=False)
                                    for i in range(len(mags19_los))])
                else:
                    a19 = np.stack([inject_diffuse_nlos(mags19_los[i], int(fp19_los[i]), dt, pdb, is_complex=False)
                                    for i in range(len(mags19_los))])
                p19 = infer_models(a19, fp19_los, 1.0)
                add_rows("fontaine_iiot19", mode_name, dt, pdb, clean19, [p19])

                # IIoT20 (5 random phase draws)
                d20 = [[] for _ in range(5)]
                for i in range(len(cirs20_los)):
                    if mode_name == "single":
                        _, _, m5 = inject_with_phase_averaging(cirs20_los[i], int(fp20_los[i]), dt, pdb, 5, SEED + i)
                    else:
                        _, _, m5 = inject_diffuse_with_phase_averaging(cirs20_los[i], int(fp20_los[i]), dt, pdb, 5, SEED + i)
                    for k in range(5):
                        d20[k].append(m5[k])
                # Batch all 5 phase draws cleanly, then split back into 5 chunks for per-draw flip mean
                cat20 = np.concatenate([np.stack(d20[k]) for k in range(5)], axis=0)
                fp20_cat = np.tile(fp20_los, 5)
                p20_cat = infer_models(cat20, fp20_cat, scale20)
                n20 = len(cirs20_los)
                p20_list = [{k: v[d * n20:(d + 1) * n20] for k, v in p20_cat.items()} for d in range(5)]
                add_rows("fontaine_iiot20_5phi", mode_name, dt, pdb, clean20, p20_list)

                # DW3000 (5 random phase draws)
                ddw = [[] for _ in range(5)]
                for i in range(len(dw_cx)):
                    if mode_name == "single":
                        _, _, m5 = inject_with_phase_averaging(dw_cx[i], int(dw_fp[i]), dt, pdb, 5, SEED + i)
                    else:
                        _, _, m5 = inject_diffuse_with_phase_averaging(dw_cx[i], int(dw_fp[i]), dt, pdb, 5, SEED + i)
                    for k in range(5):
                        ddw[k].append(m5[k] / dw_n[i])
                catdw = np.concatenate([np.stack(ddw[k]) for k in range(5)], axis=0)
                fpdw_cat = np.tile(dw_fp, 5)
                pdw_cat = infer_models(catdw, fpdw_cat, scaledw)
                ndw = len(dw_cx)
                pdw_list = [{k: v[d * ndw:(d + 1) * ndw] for k, v in pdw_cat.items()} for d in range(5)]
                add_rows("dw3000_complex_5phi", mode_name, dt, pdb, cleandw, pdw_list)

    df_sub = pd.DataFrame(rows)
    csv_p = RESULTS_DIR / "remnet_sub6db_audit.csv"
    df_sub.to_csv(csv_p, index=False, float_format="%.6f")
    logger.info("Saved %s (%d rows)", csv_p, len(df_sub))

    # Print Check 2 & Check 3c summary tables
    print("\n" + "=" * 82)
    print("CHECK 2 & 3c: SUB--6 dB FLIP RATES (% across dense delays)")
    print("=" * 82)
    for ds in ["fontaine_iiot19", "fontaine_iiot20_5phi", "dw3000_complex_5phi"]:
        for mode in ["single", "diffuse"]:
            sub = df_sub[(df_sub.dataset == ds) & (df_sub["mode"] == mode)]
            piv = sub.groupby("power_dB")[["flip_rem172", "flip_rem119", "flip_lgbm"]].max() * 100.0
            print(f"-- {ds} ({mode}) max across dt in [10..50] ns --")
            print(piv.round(1).to_string())

    print("\n-- Smoothness over delay dt at -12 dB and -9 dB and -6 dB (IIoT19 single) --")
    s19 = df_sub[(df_sub.dataset == "fontaine_iiot19") & (df_sub["mode"] == "single") & (df_sub.power_dB.isin([-12.0, -9.0, -6.0]))]
    for pdb, grp in s19.groupby("power_dB"):
        line172 = " ".join(f"{int(r.delta_t_ns)}:{r.flip_rem172*100:4.1f}%" for _, r in grp.iterrows())
        line119 = " ".join(f"{int(r.delta_t_ns)}:{r.flip_rem119*100:4.1f}%" for _, r in grp.iterrows())
        print(f"  {pdb:+5.1f}dB REMNet-172: {line172}")
        print(f"  {pdb:+5.1f}dB REMNet-119: {line119}")

    # Combine with previous [-6..+6 dB] sweep if available so 70% thresholds above -6 dB also resolve cleanly
    prev_csv = RESULTS_DIR / "insight_low_power_sweep.csv"
    df_prev = pd.read_csv(prev_csv) if prev_csv.exists() else None

    # ---- CHECK 4: Exact 70%-flip alpha_cls vs alpha_max ----
    print("\n" + "=" * 82)
    print("CHECK 4: EMPIRICAL >=70%-FLIP alpha_cls (dB & linear) vs CLOSED-FORM alpha_max (P_fa=0.05)")
    print("=" * 82)
    print("  Analytical alpha_max bounds (N_STS=4096, M=75, P_d_max=0.10, P_fa=0.05, SNR_IP=SNR_STS+6dB):")
    amax_table = {}
    for snr in [10.0, 15.0, 20.0]:
        a1 = alpha_max_bound(snr, 1.0, 0.05)
        a2 = alpha_max_bound(snr, 2.0, 0.05)
        amax_table[(snr, 1)] = a1
        amax_table[(snr, 2)] = a2
        print(f"    SNR_STS={snr:4.0f} dB | kappa_mp=1: alpha_max={a1:.4f} ({20*np.log10(a1):+.2f} dB) | "
              f"kappa_mp=2: alpha_max={a2:.4f} ({20*np.log10(a2):+.2f} dB)")

    print("\n  Empirical >=70% Flip Thresholds alpha_cls:")
    report_lines = [
        "MOVE 1 AUDIT SUMMARY: SUB--6 dB REMNet & DIFFUSE LIGHTGBM CHECK",
        f"Check 1 (Zero-injection identity): PASS (0.0% flips, 0.0 diff across IIoT19/IIoT20/DW3000)",
        f"Check 3 (fp_idx=20 lock & 119-tap unpadded ablation): PASS (fp_idx 100% locked; REMNet-119 clean Acc={accuracy_score(y_cls[te_idx], (p119>=0.5).astype(int)):.3f})",
    ]

    for ds_label, ds_key in [("Fontaine IIoT19", "fontaine_iiot19"), ("Fontaine IIoT20 (5φ)", "fontaine_iiot20_5phi"), ("Santamaria DW3000 (5φ)", "dw3000_complex_5phi")]:
        for mkey, mlbl in [("rem172", "REMNet-172 (padded)"), ("rem119", "REMNet-119 (unpadded)"), ("lgbm", "LightGBM-Cls (CIR)")]:
            for mode in ["single", "diffuse"]:
                sub = df_sub[(df_sub.dataset == ds_key) & (df_sub["mode"] == mode)].groupby("power_dB")[f"flip_{mkey}"].max()
                # Also merge > -6 dB from Phase 1 CSV if available for rem172 / lgbm
                if df_prev is not None and mkey in ["rem172", "lgbm"]:
                    pcol = "flip_insight_remnet_cls" if mkey == "rem172" else "flip_lgbm_cls"
                    sp = df_prev[(df_prev.dataset == ds_key) & (df_prev["mode"] == mode) & (df_prev.power_dB > -6.0)].groupby("power_dB")[pcol].max()
                    sub = pd.concat([sub, sp]).sort_index()
                hits = sub[sub >= 0.70]
                if hits.empty:
                    # Also print peak flip reached
                    pk_p = float(sub.idxmax())
                    pk_v = float(sub.max()) * 100.0
                    print(f"    {ds_label:22s} | {mlbl:22s} | {mode:7s} | never >=70% (peak {pk_v:4.1f}% at {pk_p:+.1f} dB)")
                else:
                    p70 = float(hits.index.min())
                    a70 = float(10.0 ** (p70 / 20.0))
                    # Interpolate crossing if strictly between -18 and hits.index.min()
                    idx_list = list(sub.index)
                    pos = idx_list.index(p70)
                    interp_str = ""
                    if pos > 0 and sub.iloc[pos - 1] < 0.70:
                        p0, p1 = float(idx_list[pos - 1]), p70
                        f0, f1 = float(sub.iloc[pos - 1]), float(sub.iloc[pos])
                        p_lin = p0 + (0.70 - f0) / max(f1 - f0, 1e-9) * (p1 - p0)
                        interp_str = f" [lin_interp={p_lin:+.2f} dB, α={10**(p_lin/20):.3f}]"
                    beats10 = a70 < amax_table[(10.0, 2)]
                    beats15 = a70 < amax_table[(15.0, 2)]
                    beats20 = a70 < amax_table[(20.0, 2)]
                    s_line = (f"    {ds_label:22s} | {mlbl:22s} | {mode:7s} | grid_p70={p70:+5.1f} dB (α={a70:.3f}){interp_str} "
                              f"| <α_max(10/15/20dB,κ=2): {beats10}/{beats15}/{beats20}")
                    print(s_line)
                    if ds_key in ["fontaine_iiot19", "dw3000_complex_5phi"] and mkey in ["rem119", "lgbm"]:
                        report_lines.append(s_line.strip())

    out_txt = RESULTS_DIR / "remnet_sub6db_report.txt"
    out_txt.write_text("\n".join(report_lines) + "\n")

    # ---- Plot ----
    _plot_sub6db_audit(df_sub, amax_table)
    return 0


def _plot_sub6db_audit(df_sub: pd.DataFrame, amax_table: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    # Left panel: max flip across dt vs power_dB (-18 to -6 dB) with alpha_max vertical lines
    ax = axes[0]
    ax.axhline(70.0, color="#c62828", linestyle="--", linewidth=1.3, label="70% Flip Threshold")
    for snr, col in [(10.0, "#1565C0"), (15.0, "#2E7D32"), (20.0, "#6A1B9A")]:
        db_max = 20.0 * np.log10(amax_table[(snr, 2)])
        ax.axvline(db_max, color=col, linestyle=":", linewidth=1.5,
                   label=f"α_max(SNR={snr:.0f}dB, κ=2) = {db_max:+.1f} dB")

    curves = [
        ("fontaine_iiot19",     "single",  "flip_rem172", "IIoT19 REMNet-172 (Single)",  "-",  "o", "#EF6C00"),
        ("fontaine_iiot19",     "single",  "flip_rem119", "IIoT19 REMNet-119 (Single)",  "--", "s", "#E65100"),
        ("fontaine_iiot19",     "diffuse", "flip_lgbm",   "IIoT19 LightGBM (Diffuse)",   "-",  "^", "#8E24AA"),
        ("dw3000_complex_5phi", "single",  "flip_rem172", "DW3000 REMNet-172 (Single)",  "-.", "d", "#00897B"),
        ("dw3000_complex_5phi", "single",  "flip_rem119", "DW3000 REMNet-119 (Single)",  ":",  "v", "#004D40"),
        ("dw3000_complex_5phi", "diffuse", "flip_lgbm",   "DW3000 LightGBM (Diffuse)",   "-.", "P", "#AD1457"),
    ]
    for ds, mode, col, lbl, ls, mk, color in curves:
        sub = df_sub[(df_sub.dataset == ds) & (df_sub["mode"] == mode)].groupby("power_dB")[col].max() * 100.0
        ax.plot(sub.index, sub.values, marker=mk, linestyle=ls, color=color, label=lbl, linewidth=1.8)
    ax.set_xlabel("Injected Power (dB relative to LOS first path)", fontsize=11)
    ax.set_ylabel("Max LOS → NLOS Flip Rate (%)", fontsize=11)
    ax.set_title("Sub--6 dB Audit vs. Closed-Form α_max Thresholds", fontsize=12)
    ax.set_xticks(SUB6_POWERS)
    ax.set_ylim(-4, 105)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # Right panel: Delay sensitivity profile (10..50 ns) at -12, -9, -6 dB for REMNet-172 & 119 on IIoT19
    ax2 = axes[1]
    s19 = df_sub[(df_sub.dataset == "fontaine_iiot19") & (df_sub["mode"] == "single")]
    for pdb, color in [(-15.0, "#90CAF9"), (-12.0, "#1E88E5"), (-9.0, "#FB8C00"), (-6.0, "#D32F2F")]:
        g = s19[s19.power_dB == pdb]
        ax2.plot(g["delta_t_ns"], g["flip_rem172"] * 100.0, marker="o", linestyle="-", color=color,
                 linewidth=2.0, label=f"REMNet-172 ({pdb:+.0f} dB)")
        ax2.plot(g["delta_t_ns"], g["flip_rem119"] * 100.0, marker="s", linestyle="--", color=color,
                 linewidth=1.5, alpha=0.85, label=f"REMNet-119 ({pdb:+.0f} dB)")
    ax2.axhline(70.0, color="#c62828", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Injected Pulse Delay Δt (ns)", fontsize=11)
    ax2.set_ylabel("LOS → NLOS Flip Rate on IIoT19 (%)", fontsize=11)
    ax2.set_title("Delay Profile (10–50 ns): Padded (172) vs. Unpadded (119)", fontsize=12)
    ax2.set_xticks(DENSE_DELAYS)
    ax2.set_ylim(-4, 105)
    ax2.legend(loc="upper right", fontsize=7.5, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_p = RESULTS_DIR / "remnet_sub6db_audit.png"
    fig.savefig(out_p, dpi=150)
    logger.info("Saved %s", out_p)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

