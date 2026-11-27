"""ablate_remnet_recipe.py — Narrow recipe-sensitivity ablation:
Our PyTorch training recipe (batch=128, epochs=45, patience=7, shuffle=True, Adam eps=1e-8)
vs. external/InSight's published TF/Keras classifier recipe (batch=800, epochs=150,
patience=5, shuffle=False in DataFrame order, Adam eps=1e-7).

Evaluates paired clean accuracy and sub--6 dB >=70%-flip thresholds (alpha_cls, both
unconditioned and conditioned on clean-predicted LOS) across REMNet-172 & REMNet-119
on IIoT19, IIoT20 (5φ), and zero-shot DW3000 (5φ).

Saves:
  - results/remnet_recipe_ablation.csv
  - results/remnet_recipe_ablation_report.txt
  - results/remnet_recipe_ablation.png
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
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (load_fontaine_cirs, load_fontaine_iiot20,
                     load_fontaine_scalar, load_santamaria_dw3000)
from eval_insight import (INSIGHT_CIR_LEN, REMNET_CIR_LEN, REMNetPyTorch,
                          TransFitMinMaxMatrixSafe, batch_insight_windows)
from features import compute_features_from_cir
from injector import (inject_diffuse_nlos, inject_diffuse_with_phase_averaging,
                      inject_phantom_nlos, inject_with_phase_averaging)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("recipe_ablation")

SEED = 42
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

POWERS = [-18.0, -15.0, -12.0, -10.5, -9.0, -7.5, -6.0, -3.0]
DELAYS = [10, 15, 18, 20, 22, 25, 30, 35]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def alpha_max_bound(snr_sts_db: float, kappa_mp: float, p_fa: float = 0.05) -> float:
    N_STS, M, P_d_max = 4096.0, 75.0, 0.10
    s_sts = 10.0 ** (snr_sts_db / 10.0)
    s_ip = 10.0 ** ((snr_sts_db + 6.0) / 10.0)
    sigma_rel = np.sqrt(kappa_mp / N_STS + 1.0 / s_sts + 1.0 / s_ip)
    return float((np.sqrt(np.log(M / p_fa)) - norm.ppf(1.0 - P_d_max) / np.sqrt(2.0)) * sigma_rel)


def train_remnet_recipe(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    patience: int,
    shuffle: bool,
    adam_eps: float,
) -> tuple[REMNetPyTorch, int, float]:
    """Train REMNet with exact recipe parameters, returning (model, stop_ep, best_val)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = REMNetPyTorch(input_len=X_tr.shape[1], N=3, F=16)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=adam_eps)
    loss_fn = nn.BCEWithLogitsLoss()

    Xt_tr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1)
    yt_tr = torch.tensor(y_tr, dtype=torch.float32)
    Xt_va = torch.tensor(X_va, dtype=torch.float32).unsqueeze(1)
    yt_va = torch.tensor(y_va, dtype=torch.float32)

    best_v, best_ep, best_st, wait = float("inf"), 0, None, 0
    n_tr = len(X_tr)
    ep_ran = 0
    for ep in range(1, epochs + 1):
        ep_ran = ep
        net.train()
        order = torch.randperm(n_tr) if shuffle else torch.arange(n_tr)
        for s in range(0, n_tr, batch_size):
            idx = order[s : s + batch_size]
            loss = loss_fn(net(Xt_tr[idx]), yt_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(Xt_va), yt_va).item())
        if vl < best_v - 1e-5:
            best_v = vl
            best_ep = ep
            best_st = {k: v.clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_st is not None:
        net.load_state_dict(best_st)
    net.eval()
    return net, ep_ran, best_v


def interp_p70(curve: pd.Series) -> tuple[float | None, float | None]:
    """Return (grid_p70_dB, lin_interp_p70_dB)."""
    hits = curve[curve >= 0.70]
    if hits.empty:
        return None, None
    p70 = float(hits.index.min())
    idx_list = list(curve.index)
    pos = idx_list.index(p70)
    p_lin = p70
    if pos > 0 and curve.iloc[pos - 1] < 0.70:
        p0, p1 = float(idx_list[pos - 1]), p70
        f0, f1 = float(curve.iloc[pos - 1]), float(curve.iloc[pos])
        p_lin = p0 + (0.70 - f0) / max(f1 - f0, 1e-9) * (p1 - p0)
    return p70, float(p_lin)


def main() -> int:
    os.chdir(Path(__file__).resolve().parent.parent)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_fontaine_scalar()
    with open(MODELS_DIR / "split_indices.pkl", "rb") as f:
        splits = pickle.load(f)
    tr_idx, va_idx, te_idx = splits["train"], splits["val"], splits["test"]
    # DataFrame-sorted index order matches external/InSight data_split.py `_df[_df.isin(...)]`
    tr_df_order = np.sort(tr_idx)

    cirs19, fp19, _ = load_fontaine_cirs("IIoT19", len(df))
    mags19 = np.abs(cirs19)
    y_cls = df["label"].values.astype(int)

    X172 = batch_insight_windows(mags19, fp19, REMNET_CIR_LEN)
    X119 = batch_insight_windows(mags19, fp19, INSIGHT_CIR_LEN)
    sc172 = TransFitMinMaxMatrixSafe().fit(X172[tr_idx])
    sc119 = TransFitMinMaxMatrixSafe().fit(X119[tr_idx])

    logger.info("Training 4 REMNet models: Our Recipe vs InSight Exact Recipe (172-tap & 119-tap)...")
    m_base172, ep_b172, vl_b172 = train_remnet_recipe(
        sc172.transform(X172[tr_idx]), y_cls[tr_idx].astype(np.float32),
        sc172.transform(X172[va_idx]), y_cls[va_idx].astype(np.float32),
        batch_size=128, epochs=45, patience=7, shuffle=True, adam_eps=1e-8,
    )
    m_ins172, ep_i172, vl_i172 = train_remnet_recipe(
        sc172.transform(X172[tr_df_order]), y_cls[tr_df_order].astype(np.float32),
        sc172.transform(X172[va_idx]), y_cls[va_idx].astype(np.float32),
        batch_size=800, epochs=150, patience=5, shuffle=False, adam_eps=1e-7,
    )
    m_base119, ep_b119, vl_b119 = train_remnet_recipe(
        sc119.transform(X119[tr_idx]), y_cls[tr_idx].astype(np.float32),
        sc119.transform(X119[va_idx]), y_cls[va_idx].astype(np.float32),
        batch_size=128, epochs=45, patience=7, shuffle=True, adam_eps=1e-8,
    )
    m_ins119, ep_i119, vl_i119 = train_remnet_recipe(
        sc119.transform(X119[tr_df_order]), y_cls[tr_df_order].astype(np.float32),
        sc119.transform(X119[va_idx]), y_cls[va_idx].astype(np.float32),
        batch_size=800, epochs=150, patience=5, shuffle=False, adam_eps=1e-7,
    )

    models_info = [
        ("base172", "OurRecipe-REMNet-172", m_base172, REMNET_CIR_LEN, sc172, ep_b172, vl_b172),
        ("ins172",  "InSightExact-REMNet-172", m_ins172, REMNET_CIR_LEN, sc172, ep_i172, vl_i172),
        ("base119", "OurRecipe-REMNet-119", m_base119, INSIGHT_CIR_LEN, sc119, ep_b119, vl_b119),
        ("ins119",  "InSightExact-REMNet-119", m_ins119, INSIGHT_CIR_LEN, sc119, ep_i119, vl_i119),
    ]

    report_lines = [
        "RECIPE-SENSITIVITY ABLATION: OUR PYTORCH RECIPE vs. INSIGHT PUBLISHED RECIPE",
        "Our Recipe   : batch=128, max_epochs=45,  patience=7, shuffle=True,  Adam(lr=3e-4, eps=1e-8)",
        "InSight Exact: batch=800, max_epochs=150, patience=5, shuffle=False, Adam(lr=3e-4, eps=1e-7)",
        "",
        "1. CLEAN HELD-OUT TEST METRICS (disjoint location_ID={16, 18}, n=2874):",
    ]
    print("\n" + "=" * 88)
    print("1. CLEAN HELD-OUT TEST METRICS & CONVERGENCE")
    print("=" * 88)
    with torch.no_grad():
        for k, lbl, net, wlen, sc, ep, vl in models_info:
            Xte = sc.transform(X172[te_idx] if wlen == 172 else X119[te_idx])
            pr = _sigmoid(net(torch.tensor(Xte).unsqueeze(1)).numpy())
            pred = (pr >= 0.5).astype(int)
            acc = accuracy_score(y_cls[te_idx], pred)
            f1 = f1_score(y_cls[te_idx], pred)
            auc = roc_auc_score(y_cls[te_idx], pr)
            s = f"  {lbl:25s} | stop_ep={ep:3d} val_bce={vl:.4f} | Acc={acc:.4f} | F1={f1:.4f} | AUROC={auc:.4f}"
            print(s)
            report_lines.append(s.strip())

    # Prepare evaluation LOS sets matching audit_remnet_sub6db.py
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
            thr = 10.0 if scen.startswith("calibration") else 6.0
            if compute_features_from_cir(r["cir_mag"], r["fpi"])["RXP_minus_FPP_dB"] < thr:
                dw_recs.append(r)
    dw_sel = np.sort(np.random.RandomState(SEED).choice(len(dw_recs), size=min(250, len(dw_recs)), replace=False))
    dw_los = [dw_recs[i] for i in dw_sel]
    dw_cx = [r["cir_complex"] for r in dw_los]
    dw_fp = np.array([r["fpi"] for r in dw_los], dtype=int)
    dw_n = np.array([r["N"] for r in dw_los], dtype=float)
    dw_mags = np.stack([r["cir_mag_pac"] for r in dw_los])
    scaledw = fontaine_los_med_peak / max(float(np.median(dw_mags.max(axis=1))), 1e-6)

    def infer_all4(mags_arr: np.ndarray, fp_arr: np.ndarray, gain: float) -> dict[str, np.ndarray]:
        arr = mags_arr * gain
        w172 = sc172.transform(batch_insight_windows(arr, fp_arr, REMNET_CIR_LEN))
        w119 = sc119.transform(batch_insight_windows(arr, fp_arr, INSIGHT_CIR_LEN))
        t172 = torch.tensor(w172).unsqueeze(1)
        t119 = torch.tensor(w119).unsqueeze(1)
        with torch.no_grad():
            return {
                "base172": (_sigmoid(m_base172(t172).numpy()) >= 0.5),
                "ins172":  (_sigmoid(m_ins172(t172).numpy()) >= 0.5),
                "base119": (_sigmoid(m_base119(t119).numpy()) >= 0.5),
                "ins119":  (_sigmoid(m_ins119(t119).numpy()) >= 0.5),
            }

    clean19 = infer_all4(mags19_los, fp19_los, 1.0)
    clean20 = infer_all4(np.abs(cirs20_los), fp20_los, scale20)
    cleandw = infer_all4(dw_mags, dw_fp, scaledw)

    # Zero-injection check
    z19 = np.stack([inject_phantom_nlos(mags19_los[i], int(fp19_los[i]), 20.0, -np.inf, is_complex=False)
                    for i in range(len(mags19_los))])
    pz19 = infer_all4(z19, fp19_los, 1.0)
    for k in clean19:
        assert float(np.mean((~clean19[k]) & pz19[k])) == 0.0
    logger.info("Zero-Injection Identity Check: PASS (0.00%% flips across all 4 models)")

    report_lines.append("")
    report_lines.append("2. CLEAN-PREDICTED LOS DENOMINATORS:")
    print("\n" + "=" * 88)
    print("2. CLEAN-PREDICTED LOS DENOMINATORS")
    print("=" * 88)
    for ds_name, cd, n_tot in [
        ("IIoT19", clean19, len(mags19_los)),
        ("IIoT20", clean20, len(cirs20_los)),
        ("DW3000", cleandw, len(dw_cx)),
    ]:
        parts = [f"{k}={int((~cd[k]).sum())}/{n_tot} ({float((~cd[k]).mean())*100:.1f}%)"
                 for k in ["base172", "ins172", "base119", "ins119"]]
        line = f"  {ds_name:6s} | " + " | ".join(parts)
        print(line)
        report_lines.append(line.strip())

    # Paired attack sweep on exact identical waveforms
    logger.info("=== Running Paired Sub--6 dB Sweep (8 powers x 8 delays x 2 modes across 4 models) ===")
    rows = []
    mkeys = ["base172", "ins172", "base119", "ins119"]

    def add_row(ds_name, mode, dt, pdb, cl_d, atk_list):
        r = {"dataset": ds_name, "mode": mode, "delta_t_ns": dt, "power_dB": pdb}
        for k in mkeys:
            cl_los = ~cl_d[k]
            n_cl = int(cl_los.sum())
            r[f"uncond_{k}"] = float(np.mean([np.mean(cl_los & ad[k]) for ad in atk_list]))
            r[f"cond_{k}"] = float(np.mean([np.mean(ad[k][cl_los]) for ad in atk_list])) if n_cl > 0 else 0.0
        rows.append(r)

    for mode in ["single", "diffuse"]:
        for dt in DELAYS:
            logger.info("Paired sweep: mode=%s, delta_t=%d ns ...", mode, dt)
            for pdb in POWERS:
                if mode == "single":
                    a19 = np.stack([inject_phantom_nlos(mags19_los[i], int(fp19_los[i]), dt, pdb, is_complex=False)
                                    for i in range(len(mags19_los))])
                else:
                    a19 = np.stack([inject_diffuse_nlos(mags19_los[i], int(fp19_los[i]), dt, pdb, is_complex=False)
                                    for i in range(len(mags19_los))])
                add_row("fontaine_iiot19", mode, dt, pdb, clean19, [infer_all4(a19, fp19_los, 1.0)])

                d20 = [[] for _ in range(5)]
                for i in range(len(cirs20_los)):
                    fn = inject_with_phase_averaging if mode == "single" else inject_diffuse_with_phase_averaging
                    _, _, m5 = fn(cirs20_los[i], int(fp20_los[i]), dt, pdb, 5, SEED + i)
                    for d in range(5):
                        d20[d].append(m5[d])
                cat20 = np.concatenate([np.stack(d20[d]) for d in range(5)], axis=0)
                p20 = infer_all4(cat20, np.tile(fp20_los, 5), scale20)
                n20 = len(cirs20_los)
                add_row("fontaine_iiot20_5phi", mode, dt, pdb, clean20,
                        [{k: p20[k][d * n20 : (d + 1) * n20] for k in mkeys} for d in range(5)])

                ddw = [[] for _ in range(5)]
                for i in range(len(dw_cx)):
                    fn = inject_with_phase_averaging if mode == "single" else inject_diffuse_with_phase_averaging
                    _, _, m5 = fn(dw_cx[i], int(dw_fp[i]), dt, pdb, 5, SEED + i)
                    for d in range(5):
                        ddw[d].append(m5[d] / dw_n[i])
                catdw = np.concatenate([np.stack(ddw[d]) for d in range(5)], axis=0)
                pdw = infer_all4(catdw, np.tile(dw_fp, 5), scaledw)
                ndw = len(dw_cx)
                add_row("dw3000_complex_5phi", mode, dt, pdb, cleandw,
                        [{k: pdw[k][d * ndw : (d + 1) * ndw] for k in mkeys} for d in range(5)])

    df_res = pd.DataFrame(rows)
    df_res.to_csv(RESULTS_DIR / "remnet_recipe_ablation.csv", index=False, float_format="%.6f")

    a10 = alpha_max_bound(10.0, 2.0, 0.05)
    a15 = alpha_max_bound(15.0, 2.0, 0.05)
    a20 = alpha_max_bound(20.0, 2.0, 0.05)

    report_lines.append("")
    report_lines.append(f"3. PAIRED >=70% alpha_cls COMPARISON (alpha_max κ=2: 10dB={20*np.log10(a10):.2f} dB, 15dB={20*np.log10(a15):.2f} dB, 20dB={20*np.log10(a20):.2f} dB):")
    print("\n" + "=" * 92)
    print("3. PAIRED RECIPE-SENSITIVITY CHECK: alpha_cls (dB, >=70% flip) — OUR RECIPE vs INSIGHT EXACT")
    print("=" * 92)
    for metric_kind in ["cond", "uncond"]:
        print(f"--- Metric: {metric_kind.upper()} ---")
        report_lines.append(f"--- {metric_kind.upper()} ---")
        for ds in ["fontaine_iiot19", "fontaine_iiot20_5phi", "dw3000_complex_5phi"]:
            for mode in ["single", "diffuse"]:
                sub = df_res[(df_res.dataset == ds) & (df_res["mode"] == mode)]
                for wlen, k_b, k_i in [("172", "base172", "ins172"), ("119", "base119", "ins119")]:
                    cb = sub.groupby("power_dB")[f"{metric_kind}_{k_b}"].max()
                    ci = sub.groupby("power_dB")[f"{metric_kind}_{k_i}"].max()
                    gb, lb = interp_p70(cb)
                    gi, li = interp_p70(ci)
                    sb = f"grid={gb:+5.1f} lin={lb:+6.2f}" if gb is not None else f"pk={cb.max()*100:4.1f}%"
                    si = f"grid={gi:+5.1f} lin={li:+6.2f}" if gi is not None else f"pk={ci.max()*100:4.1f}%"
                    delta = f"{li - lb:+5.2f} dB" if (lb is not None and li is not None) else "n/a"
                    tiers_i = f"{10**(li/20)<a10}/{10**(li/20)<a15}/{10**(li/20)<a20}" if li is not None else "False/False/False"
                    ln = f"  {ds:20s} | REMNet-{wlen} | {mode:7s} | Our: {sb:22s} | InSightExact: {si:22s} | Δ={delta:9s} | <α_max(10/15/20): {tiers_i}"
                    print(ln)
                    report_lines.append(ln.strip())

    (RESULTS_DIR / "remnet_recipe_ablation_report.txt").write_text("\n".join(report_lines) + "\n")
    _plot_recipe_ablation(df_res, a10, a15, a20)
    return 0


def _plot_recipe_ablation(df_res: pd.DataFrame, a10: float, a15: float, a20: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, ds, title in [
        (axes[0], "fontaine_iiot19", "Fontaine IIoT19 (Conditioned Flip %)"),
        (axes[1], "dw3000_complex_5phi", "Zero-Shot DW3000 5φ (Conditioned Flip %)"),
    ]:
        ax.axhline(70.0, color="#c62828", linestyle="--", linewidth=1.2, label="70% Threshold")
        for snr, av, col in [(10, a10, "#1565C0"), (15, a15, "#2E7D32"), (20, a20, "#6A1B9A")]:
            ax.axvline(20 * np.log10(av), color=col, linestyle=":", linewidth=1.4, label=f"α_max({snr}dB)={20*np.log10(av):.1f}dB")

        specs = [
            ("single",  "cond_base172", "OurRecipe-172 (Single)",     "#E65100", "-",  "o"),
            ("single",  "cond_ins172",  "InSightExact-172 (Single)",  "#1565C0", "--", "s"),
            ("diffuse", "cond_base172", "OurRecipe-172 (Diffuse)",    "#FB8C00", "-",  "^"),
            ("diffuse", "cond_ins172",  "InSightExact-172 (Diffuse)", "#42A5F5", "--", "d"),
            ("single",  "cond_base119", "OurRecipe-119 (Single)",     "#2E7D32", "-",  "P"),
            ("single",  "cond_ins119",  "InSightExact-119 (Single)",  "#AD1457", "--", "X"),
        ]
        for mode, col, lbl, c, ls, mk in specs:
            g = df_res[(df_res.dataset == ds) & (df_res["mode"] == mode)].groupby("power_dB")[col].max() * 100.0
            ax.plot(g.index, g.values, color=c, linestyle=ls, marker=mk, linewidth=1.9, markersize=5.2, label=lbl)
        ax.set_title(title)
        ax.set_xlabel("Injected Power (dB relative to LOS first path)")
        ax.set_xticks(POWERS)
        ax.set_ylim(-4, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.6, loc="lower right")
    axes[0].set_ylabel("Max Conditioned LOS → NLOS Flip Rate (%)")
    fig.suptitle("Recipe-Sensitivity Check: PyTorch Training Recipe vs. InSight Exact (batch=800, shuffle=False, patience=5)", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "remnet_recipe_ablation.png", dpi=160)
    plt.close(fig)
    logger.info("Saved results/remnet_recipe_ablation.png")


if __name__ == "__main__":
    raise SystemExit(main())

