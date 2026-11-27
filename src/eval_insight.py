"""eval_insight.py — Faithful LENS-TUGraz/InSight evaluation + fine low-power & diffuse sweep
+ closed-form stealth window alpha_max overlay.

Implements verbatim preprocessing and model definitions from external/InSight/:
  - ROLL_CIR = 20 alignment (fp_idx placed at tap index 20).
  - TransFitMinMax (per-tap column min-max) for InSight-XGBoost (Embedded D=1,T=30
    & Full D=6,T=100; plus Regressor).
  - InSight-SVM 8-feature extractor (calc_features: f1..f6, f8, f11) + MinMaxScaler
    + RBF SVC(C=50, gamma=1/(D*var)).
  - TransFitMinMaxMatrix (global min-max) + PyTorch twin of REMNet (N=3, F=16, r=8,
    len=172, Adam lr=3e-4, L1 regression / BCE classification).

Runs fine sweep over power_dB in [-6, -3, -1.5, 0, +1.5, +3, +4.5, +6] and
delta_t_ns in [10, 20, 35, 50] for BOTH single-tap and K=3 diffuse injection on:
  - Fontaine IIoT19 (907 clean held-out LOS test CIRs)
  - Fontaine IIoT20 (300 complex128 LOS CIRs, 5-phase avg & phi=0)
  - Santamaria DW3000 (250 clean LOS complex-IQ CIRs, 5-phase avg & phi=0)

Generates:
  - results/insight_low_power_sweep.csv
  - results/insight_flip_vs_power.png
  - results/stealth_window_alpha_max.png
and prints the Step 5 Go/No-Go verdict.
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
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (CNN_WIN_LEN, extract_cir_window, load_fontaine_cirs,
                     load_fontaine_iiot20, load_fontaine_scalar,
                     load_santamaria_dw3000)
from features import FEATURE_NAMES, compute_feature_matrix, compute_features_from_cir
from injector import (inject_diffuse_nlos, inject_diffuse_with_phase_averaging,
                      inject_phantom_nlos, inject_with_phase_averaging)
from models import APS006Classifier, CNN1DClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_insight")

SEED = 42
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

POWER_GRID = [-6.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0]
DELTA_GRID = [10, 20, 35, 50]

# InSight preprocessing constants (verified from external/InSight/)
INSIGHT_ROLL_FP = 20
INSIGHT_CIR_LEN = 119   # [fp-20 : fp+99] avoids trailing dead zeros on IIoT19 (150 taps, fp=51)
REMNET_CIR_LEN  = 172   # exact cir_start=0, cir_end=172 from hyper_parameter_optimization_config.yaml


# ============================================================
#  1. InSight Alignment & Feature Extractors (Verbatim Mirror)
# ============================================================

def align_insight_window(cir_mag: np.ndarray, fp_idx: int, out_len: int) -> np.ndarray:
    """Mirror InSight ROLL_CIR=20: place fp_idx at index 20, slice [0:out_len]."""
    return extract_cir_window(cir_mag, int(fp_idx), pre=INSIGHT_ROLL_FP, post=out_len - INSIGHT_ROLL_FP)


def batch_insight_windows(cirs_mag: np.ndarray | list[np.ndarray], fp_arr: np.ndarray, out_len: int) -> np.ndarray:
    return np.stack([align_insight_window(cirs_mag[i], int(fp_arr[i]), out_len)
                     for i in range(len(cirs_mag))]).astype(np.float32)


def calc_insight_svm_features_single(cir_full: np.ndarray, fp_idx: int) -> list[float]:
    """Verbatim mirror of external/InSight/Model Generator/ML_models/SVM/features.py.

    On ROLL_CIR=20 aligned window [0:INSIGHT_CIR_LEN]:
      std_noise = std(cir_bak[-10:])
      f1 = sum(c^2)
      f2 = max(c)
      f3 = argfirst(c > 0.6*f2) - argfirst(c > 6*std_noise)
      f4, f5 = power-weighted mean excess delay & RMS delay spread
      f6 = sum((|c| - mean(c))^4) / (var(c)^2 * T)   [Pearson kurtosis]
      f8 = f4 - argfirst(c > 6*std_noise)
      f11 = std(cir_bak[fp_idx-10 : fp_idx-2])
    """
    c = align_insight_window(cir_full, fp_idx, INSIGHT_CIR_LEN).astype(np.float64)
    T = len(c)
    std_noise = float(np.std(c[-10:]))
    f1 = float(np.sum(c ** 2))
    f2 = float(np.max(c))
    thL = 6.0 * std_noise
    thH = 0.6 * f2
    idxL = np.flatnonzero(c > thL)
    idxH = np.flatnonzero(c > thH)
    iL = int(idxL[0]) if len(idxL) > 0 else 0
    iH = int(idxH[0]) if len(idxH) > 0 else 0
    f3 = float(iH - iL)

    rs = c ** 2
    s_rs = float(np.sum(rs))
    if s_rs > 1e-18:
        phi = rs / s_rs
        t = np.arange(T, dtype=np.float64)
        f4 = float(np.sum(t * phi))
        f5 = float(np.sqrt(np.sum(((t - f4) ** 2) * phi)))
    else:
        f4 = f5 = 0.0

    mu_r = float(np.mean(c))
    sig_r = float(np.var(c))
    if sig_r > 1e-18:
        f6 = float(np.sum((np.abs(c) - mu_r) ** 4) / ((sig_r ** 2) * T))
    else:
        f6 = 0.0

    f8 = float(f4 - iL) if len(idxL) > 0 else float(f4)
    pre_s = max(0, INSIGHT_ROLL_FP - 10)
    pre_e = max(pre_s + 1, INSIGHT_ROLL_FP - 2)
    f11 = float(np.std(c[pre_s:pre_e]))
    return [f1, f2, f3, f4, f5, f6, f8, f11]


def batch_insight_svm_features(cirs_mag: np.ndarray | list[np.ndarray], fp_arr: np.ndarray) -> np.ndarray:
    return np.array([
        calc_insight_svm_features_single(cirs_mag[i], int(fp_arr[i]))
        for i in range(len(cirs_mag))
    ], dtype=np.float32)


class TransFitMinMaxSafe:
    """Mirror of InSight TransFitMinMaxModelBased (per-tap column MinMax)."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        self.x_min = np.min(X, axis=0)
        self.x_max = np.max(X, axis=0)
        self.x_span = np.where((self.x_max - self.x_min) > 1e-12, self.x_max - self.x_min, 1.0)
        if y is not None:
            self.y_min = float(np.min(y))
            self.y_max = float(np.max(y))
            self.y_span = max(self.y_max - self.y_min, 1e-12)
        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None):
        Xs = (X - self.x_min) / self.x_span
        if y is not None:
            return Xs, (y - self.y_min) / self.y_span
        return Xs

    def inverse_y(self, ys: np.ndarray) -> np.ndarray:
        return ys * self.y_span + self.y_min


class TransFitMinMaxMatrixSafe:
    """Mirror of InSight TransFitMinMaxMatrix (global scalar MinMax across whole array)."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        self.x_min = float(np.min(X))
        self.x_max = float(np.max(X))
        self.x_span = max(self.x_max - self.x_min, 1e-12)
        if y is not None:
            self.y_min = float(np.min(y))
            self.y_max = float(np.max(y))
            self.y_span = max(self.y_max - self.y_min, 1e-12)
        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None):
        Xs = ((X - self.x_min) / self.x_span).astype(np.float32)
        if y is not None:
            return Xs, ((y - self.y_min) / self.y_span).astype(np.float32)
        return Xs

    def inverse_y(self, ys: np.ndarray) -> np.ndarray:
        return ys * self.y_span + self.y_min


# ============================================================
#  2. InSight REMNet Architecture (Exact PyTorch Mirror)
# ============================================================

class REMNetBlock(nn.Module):
    """Exact mirror of InSight REMNet ResNet+SE+Stride-2 Reduction block."""

    def __init__(self, F: int = 16, r: int = 8):
        super().__init__()
        self.conv2 = nn.Conv1d(F, F, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(F, max(1, F // r))
        self.fc2 = nn.Linear(max(1, F // r), F)
        self.red1 = nn.Conv1d(F, F, kernel_size=3, stride=2, padding=1)
        self.red2 = nn.Conv1d(F, F, kernel_size=1, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, identity: torch.Tensor) -> torch.Tensor:
        conv_2 = self.relu(self.conv2(identity))
        gap = conv_2.mean(dim=-1)
        ex = self.sig(self.fc2(self.relu(self.fc1(gap)))).unsqueeze(-1)
        add_se = conv_2 * ex + identity
        r1 = self.relu(self.red1(add_se))
        r2 = self.relu(self.red2(add_se))
        return r1 + r2


class REMNetPyTorch(nn.Module):
    """Exact mirror of ML_models/REMNet/model.py (N=3, F=16, r=8, L=172)."""

    def __init__(self, input_len: int = REMNET_CIR_LEN, N: int = 3, F: int = 16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, F, kernel_size=7, padding=3), nn.ReLU())
        self.blocks = nn.ModuleList([REMNetBlock(F=F, r=max(1, F // 2)) for _ in range(N)])
        # Compute flattened length via N stride-2 steps
        L = input_len
        for _ in range(N):
            L = (L + 1) // 2
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(F * L, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)


def train_remnet(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    task: str,
    epochs: int = 45,
    batch_size: int = 128,
) -> REMNetPyTorch:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = REMNetPyTorch(input_len=X_tr.shape[1], N=3, F=16)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)  # exact lr=3e-4 from ml_models.py
    loss_fn = nn.BCEWithLogitsLoss() if task == "cls" else nn.L1Loss()

    Xt_tr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1)
    yt_tr = torch.tensor(y_tr, dtype=torch.float32)
    Xt_va = torch.tensor(X_va, dtype=torch.float32).unsqueeze(1)
    yt_va = torch.tensor(y_va, dtype=torch.float32)

    best_v, best_st, wait = float("inf"), None, 0
    patience = 7 if task == "cls" else 10
    n_tr = len(X_tr)
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n_tr)
        for s in range(0, n_tr, batch_size):
            idx = perm[s : s + batch_size]
            loss = loss_fn(net(Xt_tr[idx]), yt_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
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


# ============================================================
#  3. Helper Utilities
# ============================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _cls_metrics(y_true, proba):
    pred = (np.asarray(proba) >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = float("nan")
    return float(accuracy_score(y_true, pred)), float(f1_score(y_true, pred, zero_division=0)), auc


def _reg_metrics(y_true_mm, y_pred_mm):
    raw_mae = float(np.mean(np.abs(y_true_mm)))
    raw_rmse = float(np.sqrt(np.mean(y_true_mm ** 2)))
    mae = float(mean_absolute_error(y_true_mm, y_pred_mm))
    rmse = float(np.sqrt(mean_squared_error(y_true_mm, y_pred_mm)))
    return mae, rmse, (1.0 - mae / raw_mae) * 100, (1.0 - rmse / raw_rmse) * 100


# ============================================================
#  4. Main Evaluation Pipeline
# ============================================================

def main() -> int:
    os.chdir(Path(__file__).resolve().parent.parent)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- Load data & exact disjoint location_ID split from Phase 0 ----
    df = load_fontaine_scalar()
    with open(MODELS_DIR / "split_indices.pkl", "rb") as f:
        splits = pickle.load(f)
    with open(MODELS_DIR / "train_meta.pkl", "rb") as f:
        train_meta = pickle.load(f)

    tr_idx, va_idx, te_idx = splits["train"], splits["val"], splits["test"]
    assert set(df.iloc[tr_idx]["location_ID"]) == {12, 13, 14, 15, 19, 20, 21, 22, 23}
    assert set(df.iloc[va_idx]["location_ID"]) == {10, 11, 17}
    assert set(df.iloc[te_idx]["location_ID"]) == {16, 18}
    logger.info("Verified exact Phase 0 disjoint location_ID split: test=%s",
                sorted(set(df.iloc[te_idx]["location_ID"])))

    cirs19, fp19, is_cx19 = load_fontaine_cirs("IIoT19", len(df))
    assert cirs19 is not None and not is_cx19
    mags19 = np.abs(cirs19)

    y_cls = df["label"].values.astype(int)
    y_err_m = (df["error_mm"].values / 1000.0).astype(np.float32)

    # ---- Step 3: Train Clean InSight Models ----
    logger.info("=== Step 3: Training Clean LENS-TUGraz/InSight Models ===")

    # Aligned CIRs for XGBoost (ROLL_CIR=20, len=119) and REMNet (ROLL_CIR=20, len=172)
    X_xgb_all = batch_insight_windows(mags19, fp19, INSIGHT_CIR_LEN)
    xgb_cls_scaler = TransFitMinMaxSafe().fit(X_xgb_all[tr_idx])
    X_tr_xgb = xgb_cls_scaler.transform(X_xgb_all[tr_idx])
    X_va_xgb = xgb_cls_scaler.transform(X_xgb_all[va_idx])
    X_te_xgb = xgb_cls_scaler.transform(X_xgb_all[te_idx])

    # 3a. InSight-XGB Embedded Classifier (exact C header config: D=1, T=30)
    xgb_emb = XGBClassifier(
        max_depth=1, n_estimators=30, learning_rate=0.1, subsample=0.6, gamma=0.0,
        objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=10,
        random_state=SEED, n_jobs=2,
    )
    xgb_emb.fit(X_tr_xgb, y_cls[tr_idx], eval_set=[(X_va_xgb, y_cls[va_idx])], verbose=False)

    # 3b. InSight-XGB Full Classifier (HPO config: D=6, T=100)
    xgb_full = XGBClassifier(
        max_depth=6, n_estimators=100, learning_rate=0.1, subsample=0.6, gamma=0.0,
        objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=10,
        random_state=SEED, n_jobs=2,
    )
    xgb_full.fit(X_tr_xgb, y_cls[tr_idx], eval_set=[(X_va_xgb, y_cls[va_idx])], verbose=False)

    # 3c. InSight-XGB Full Regressor (HPO config: D=6, T=100, gamma=0.5)
    xgb_reg_scaler = TransFitMinMaxSafe().fit(X_xgb_all[tr_idx], y_err_m[tr_idx])
    X_tr_xr, y_tr_xr = xgb_reg_scaler.transform(X_xgb_all[tr_idx], y_err_m[tr_idx])
    X_va_xr, y_va_xr = xgb_reg_scaler.transform(X_xgb_all[va_idx], y_err_m[va_idx])
    X_te_xr = xgb_reg_scaler.transform(X_xgb_all[te_idx])
    xgb_reg = XGBRegressor(
        max_depth=6, n_estimators=100, learning_rate=0.1, subsample=0.6, gamma=0.5,
        objective="reg:squarederror", eval_metric="rmse", early_stopping_rounds=10,
        random_state=SEED, n_jobs=2,
    )
    xgb_reg.fit(X_tr_xr, y_tr_xr, eval_set=[(X_va_xr, y_va_xr)], verbose=False)

    # 3d. InSight-SVM Classifier (8 waveform features + MinMaxScaler + SVC C=50, RBF)
    # InSight libsvm uses `-s 0 -t 2 -c 50 -g gamma` (no Platt calibration)
    F_svm_all = batch_insight_svm_features(mags19, fp19)
    svm_scaler = MinMaxScaler().fit(F_svm_all[tr_idx])
    F_tr_s = svm_scaler.transform(F_svm_all[tr_idx])
    F_te_s = svm_scaler.transform(F_svm_all[te_idx])
    svm_gamma = 1.0 / (F_tr_s.shape[1] * max(float(F_tr_s.var()), 1e-12))
    svm_cls = SVC(C=50.0, kernel="rbf", gamma=svm_gamma, random_state=SEED)
    svm_cls.fit(F_tr_s, y_cls[tr_idx])

    # 3e. InSight-REMNet Classifier & Error Corrector (len=172, TransFitMinMaxMatrix)
    X_rem_all = batch_insight_windows(mags19, fp19, REMNET_CIR_LEN)
    rem_cls_scaler = TransFitMinMaxMatrixSafe().fit(X_rem_all[tr_idx])
    rem_cls = train_remnet(
        rem_cls_scaler.transform(X_rem_all[tr_idx]), y_cls[tr_idx].astype(np.float32),
        rem_cls_scaler.transform(X_rem_all[va_idx]), y_cls[va_idx].astype(np.float32),
        task="cls", epochs=45,
    )

    rem_reg_scaler = TransFitMinMaxMatrixSafe().fit(X_rem_all[tr_idx], y_err_m[tr_idx])
    Xr_tr_m, yr_tr_m = rem_reg_scaler.transform(X_rem_all[tr_idx], y_err_m[tr_idx])
    Xr_va_m, yr_va_m = rem_reg_scaler.transform(X_rem_all[va_idx], y_err_m[va_idx])
    rem_reg = train_remnet(Xr_tr_m, yr_tr_m, Xr_va_m, yr_va_m, task="reg", epochs=45)

    # ---- Print Clean InSight Baseline Metrics ----
    print("\n" + "=" * 82)
    print("INSIGHT CLEAN HELD-OUT TEST METRICS (disjoint location_ID={16, 18}, n=2874)")
    print("=" * 82)
    for cname, scores, thr in [
        ("InSight-XGB-Emb (D=1,T=30)", xgb_emb.predict_proba(X_te_xgb)[:, 1], 0.5),
        ("InSight-XGB-Full (D=6,T=100)", xgb_full.predict_proba(X_te_xgb)[:, 1], 0.5),
        ("InSight-SVM (C=50 RBF)", svm_cls.decision_function(F_te_s), 0.0),
        ("InSight-REMNet-Cls (N=3,F=16)", _sigmoid(rem_cls(torch.tensor(rem_cls_scaler.transform(X_rem_all[te_idx])).unsqueeze(1)).detach().numpy()), 0.5),
    ]:
        pred = (np.asarray(scores) >= thr).astype(int)
        acc = float(accuracy_score(y_cls[te_idx], pred))
        f1 = float(f1_score(y_cls[te_idx], pred, zero_division=0))
        auc = float(roc_auc_score(y_cls[te_idx], scores))
        print(f"  {cname:30s} | Acc={acc:.3f} | F1={f1:.3f} | AUROC={auc:.3f}")

    y_te_mm = y_err_m[te_idx] * 1000.0
    pred_xgbr_mm = xgb_reg_scaler.inverse_y(xgb_reg.predict(X_te_xr)) * 1000.0
    with torch.no_grad():
        pred_remr_s = rem_reg(torch.tensor(rem_reg_scaler.transform(X_rem_all[te_idx])).unsqueeze(1)).numpy()
    pred_remr_mm = rem_reg_scaler.inverse_y(pred_remr_s) * 1000.0
    for rname, pmm in [
        ("InSight-XGB-Reg (D=6,T=100)", pred_xgbr_mm),
        ("InSight-REMNet-Reg (N=3,F=16)", pred_remr_mm),
    ]:
        mae, rmse, mi, ri = _reg_metrics(y_te_mm, pmm)
        print(f"  {rname:30s} | MAE={mae:6.1f}mm ({mi:+5.1f}%) | RMSE={rmse:6.1f}mm ({ri:+5.1f}%)")
    print("=" * 82)

    # ---- Load Phase 0 baseline models for side-by-side fine sweep ----
    with open(MODELS_DIR / "lgbm_cls_cir.pkl", "rb") as f:
        lgbm_cls_cir = pickle.load(f)
    scale_A = float(train_meta["scale_A"])
    cnn_cls_A = CNN1DClassifier(CNN_WIN_LEN)
    cnn_cls_A.load_state_dict(torch.load(MODELS_DIR / "cnn_cls_A.pt", map_location="cpu", weights_only=True))
    cnn_cls_A.eval()
    cnn_cls_B = CNN1DClassifier(CNN_WIN_LEN)
    cnn_cls_B.load_state_dict(torch.load(MODELS_DIR / "cnn_cls_B.pt", map_location="cpu", weights_only=True))
    cnn_cls_B.eval()
    aps006 = APS006Classifier()

    # Helper to run ALL classifiers + bias regressors on any list/array of magnitude CIRs
    def predict_all_models(mags_arr: np.ndarray, fp_arr: np.ndarray, pac_scale_to_fontaine: float = 1.0) -> dict[str, np.ndarray]:
        scaled_mags = mags_arr * pac_scale_to_fontaine
        # 1. CIR-derived features (LGBM-CIR & APS006)
        fmat = compute_feature_matrix(scaled_mags, fp_arr)
        pd_val = fmat[:, FEATURE_NAMES.index("RXP_minus_FPP_dB")]
        # 2. CNN Regime A / B
        w_cnn = np.stack([extract_cir_window(scaled_mags[i], int(fp_arr[i])) for i in range(len(scaled_mags))]).astype(np.float32)
        wA = (w_cnn / np.float32(scale_A)).astype(np.float32)
        wB = (w_cnn / np.maximum(w_cnn.max(axis=1, keepdims=True), 1e-12)).astype(np.float32)
        # 3. InSight XGB & SVM & REMNet
        w_xgb = xgb_cls_scaler.transform(batch_insight_windows(scaled_mags, fp_arr, INSIGHT_CIR_LEN))
        w_xr  = xgb_reg_scaler.transform(batch_insight_windows(scaled_mags, fp_arr, INSIGHT_CIR_LEN))
        f_svm = svm_scaler.transform(batch_insight_svm_features(scaled_mags, fp_arr))
        w_rem_c = rem_cls_scaler.transform(batch_insight_windows(scaled_mags, fp_arr, REMNET_CIR_LEN))
        w_rem_r = rem_reg_scaler.transform(batch_insight_windows(scaled_mags, fp_arr, REMNET_CIR_LEN))

        with torch.no_grad():
            p_cnnA = _sigmoid(cnn_cls_A(torch.tensor(wA).unsqueeze(1)).numpy())
            p_cnnB = _sigmoid(cnn_cls_B(torch.tensor(wB).unsqueeze(1)).numpy())
            p_remc = _sigmoid(rem_cls(torch.tensor(w_rem_c).unsqueeze(1)).numpy())
            b_remr = rem_reg_scaler.inverse_y(rem_reg(torch.tensor(w_rem_r).unsqueeze(1)).numpy())

        return {
            "aps10": (pd_val > 10.0).astype(float),
            "aps6":  (pd_val > 6.0).astype(float),
            "lgbm_cls": (lgbm_cls_cir.predict(fmat) >= 0.5).astype(float),
            "cnn_cls_A": (p_cnnA >= 0.5).astype(float),
            "cnn_cls_B": (p_cnnB >= 0.5).astype(float),
            "insight_xgb_emb":  (xgb_emb.predict_proba(w_xgb)[:, 1] >= 0.5).astype(float),
            "insight_xgb_full": (xgb_full.predict_proba(w_xgb)[:, 1] >= 0.5).astype(float),
            "insight_svm":      svm_cls.predict(f_svm).astype(float),
            "insight_remnet_cls": (p_remc >= 0.5).astype(float),
            "insight_xgb_reg_m":  xgb_reg_scaler.inverse_y(xgb_reg.predict(w_xr)),
            "insight_remnet_reg_m": b_remr,
            "_RXP_dB": fmat[:, FEATURE_NAMES.index("RXP_dB")],
            "_Pd_dB":  pd_val,
            "_kurtosis": fmat[:, FEATURE_NAMES.index("kurtosis")],
            "_energy_ratio": fmat[:, FEATURE_NAMES.index("energy_ratio")],
        }

    # ---- Step 4: Prepare Held-Out Clean LOS Test Datasets ----
    # 4a. Fontaine IIoT19 (907 LOS test CIRs)
    te_los_idx = te_idx[y_cls[te_idx] == 0]
    mags19_los = mags19[te_los_idx]
    fp19_los = fp19[te_los_idx]
    fontaine_los_med_peak = float(np.median(mags19_los.max(axis=1)))

    # 4b. Fontaine IIoT20 (300 complex128 LOS CIRs)
    _, cirs20, fp20 = load_fontaine_iiot20()
    iiot20_sub = np.linspace(0, len(cirs20) - 1, min(300, len(cirs20)), dtype=int)
    cirs20_los = cirs20[iiot20_sub]
    fp20_los = fp20[iiot20_sub]
    iiot20_pac_scale = fontaine_los_med_peak / max(float(np.median(np.abs(cirs20_los).max(axis=1))), 1e-6)

    # 4c. Santamaria DW3000 (250 clean LOS complex-IQ CIRs)
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
    dw_cirs_cx = [r["cir_complex"] for r in dw_los]
    dw_fp = np.array([r["fpi"] for r in dw_los], dtype=int)
    dw_n = np.array([r["N"] for r in dw_los], dtype=float)
    dw_pac_mags = np.stack([r["cir_mag_pac"] for r in dw_los])
    dw_pac_scale = fontaine_los_med_peak / max(float(np.median(dw_pac_mags.max(axis=1))), 1e-6)

    # Clean baselines for all three datasets
    clean19 = predict_all_models(mags19_los, fp19_los, 1.0)
    clean20 = predict_all_models(np.abs(cirs20_los), fp20_los, iiot20_pac_scale)
    cleandw = predict_all_models(dw_pac_mags, dw_fp, dw_pac_scale)

    cls_keys = [
        "aps10", "aps6", "lgbm_cls", "cnn_cls_A", "cnn_cls_B",
        "insight_xgb_emb", "insight_xgb_full", "insight_svm", "insight_remnet_cls",
    ]
    reg_keys = ["insight_xgb_reg_m", "insight_remnet_reg_m"]

    # Log feature-distribution diagnostic explaining why LightGBM-Cls peaks at ~0 dB
    nlos_fmat = compute_feature_matrix(mags19[tr_idx][y_cls[tr_idx] == 1], fp19[tr_idx][y_cls[tr_idx] == 1])
    logger.info(
        "Training NLOS feature medians: RXP_dB=%.2f, RXP-FPP=%.2f dB, kurtosis=%.2f, energy_ratio=%.3f | "
        "Clean Test LOS medians: RXP_dB=%.2f, RXP-FPP=%.2f dB, kurtosis=%.2f, energy_ratio=%.3f",
        float(np.median(nlos_fmat[:, FEATURE_NAMES.index("RXP_dB")])),
        float(np.median(nlos_fmat[:, FEATURE_NAMES.index("RXP_minus_FPP_dB")])),
        float(np.median(nlos_fmat[:, FEATURE_NAMES.index("kurtosis")])),
        float(np.median(nlos_fmat[:, FEATURE_NAMES.index("energy_ratio")])),
        float(np.median(clean19["_RXP_dB"])),
        float(np.median(clean19["_Pd_dB"])),
        float(np.median(clean19["_kurtosis"])),
        float(np.median(clean19["_energy_ratio"])),
    )

    # ---- Run Fine Low-Power + Diffuse Sweep ----
    logger.info("=== Step 4: Fine Low-Power & Diffuse Sweep (IIoT19, IIoT20, DW3000) ===")
    sweep_rows = []

    def record_sweep_row(dataset_name, mode_name, dt, pdb, clean_d, atk_draws_list):
        row = {"dataset": dataset_name, "mode": mode_name, "delta_t_ns": dt, "power_dB": pdb}
        # Average flip indicators & bias shifts across phase draws (1 draw for IIoT19, 5 draws for complex IQ)
        for ck in cls_keys:
            flips = [
                np.mean((clean_d[ck] == 0.0) & (ad[ck] == 1.0))
                for ad in atk_draws_list
            ]
            row[f"flip_{ck}"] = float(np.mean(flips))
        for rk in reg_keys:
            dbs = np.mean([ad[rk] - clean_d[rk] for ad in atk_draws_list], axis=0)
            row[f"median_db_{rk}"] = float(np.median(dbs))
            row[f"asr050_{rk}"] = float(np.mean(dbs > 0.50))
        row["med_RXP_dB"] = float(np.mean([np.median(ad["_RXP_dB"]) for ad in atk_draws_list]))
        row["med_Pd_dB"]  = float(np.mean([np.median(ad["_Pd_dB"]) for ad in atk_draws_list]))
        sweep_rows.append(row)

    for mode_name in ["single", "diffuse"]:
        for dt in DELTA_GRID:
            logger.info("Sweep progress: mode=%s, delta_t=%d ns ...", mode_name, dt)
            for pdb in POWER_GRID:
                # 1. Fontaine IIoT19 (magnitude-only)
                if mode_name == "single":
                    atk19 = np.stack([inject_phantom_nlos(mags19_los[i], int(fp19_los[i]), dt, pdb, is_complex=False)
                                      for i in range(len(mags19_los))])
                else:
                    atk19 = np.stack([inject_diffuse_nlos(mags19_los[i], int(fp19_los[i]), dt, pdb, is_complex=False)
                                      for i in range(len(mags19_los))])
                pred19 = predict_all_models(atk19, fp19_los, 1.0)
                record_sweep_row("fontaine_iiot19", mode_name, dt, pdb, clean19, [pred19])

                # 2. Fontaine IIoT20 (complex128, 5 random phase draws + worst-case phi=0)
                draws20_mags = [[] for _ in range(5)]
                worst20_mags = []
                for i in range(len(cirs20_los)):
                    if mode_name == "single":
                        _, mw, m5 = inject_with_phase_averaging(cirs20_los[i], int(fp20_los[i]), dt, pdb, 5, SEED + i)
                    else:
                        _, mw, m5 = inject_diffuse_with_phase_averaging(cirs20_los[i], int(fp20_los[i]), dt, pdb, 5, SEED + i)
                    worst20_mags.append(mw)
                    for d in range(5):
                        draws20_mags[d].append(m5[d])
                preds20_5 = [predict_all_models(np.stack(draws20_mags[d]), fp20_los, iiot20_pac_scale) for d in range(5)]
                pred20_0  = predict_all_models(np.stack(worst20_mags), fp20_los, iiot20_pac_scale)
                record_sweep_row("fontaine_iiot20_5phi", mode_name, dt, pdb, clean20, preds20_5)
                record_sweep_row("fontaine_iiot20_phi0", mode_name, dt, pdb, clean20, [pred20_0])

                # 3. Santamaria DW3000 (complex I/Q, 5 random phase draws + worst-case phi=0)
                drawsdw_mags = [[] for _ in range(5)]
                worstdw_mags = []
                for i in range(len(dw_cirs_cx)):
                    if mode_name == "single":
                        _, mw, m5 = inject_with_phase_averaging(dw_cirs_cx[i], int(dw_fp[i]), dt, pdb, 5, SEED + i)
                    else:
                        _, mw, m5 = inject_diffuse_with_phase_averaging(dw_cirs_cx[i], int(dw_fp[i]), dt, pdb, 5, SEED + i)
                    worstdw_mags.append(mw / dw_n[i])
                    for d in range(5):
                        drawsdw_mags[d].append(m5[d] / dw_n[i])
                predsdw_5 = [predict_all_models(np.stack(drawsdw_mags[d]), dw_fp, dw_pac_scale) for d in range(5)]
                preddw_0  = predict_all_models(np.stack(worstdw_mags), dw_fp, dw_pac_scale)
                record_sweep_row("dw3000_complex_5phi", mode_name, dt, pdb, cleandw, predsdw_5)
                record_sweep_row("dw3000_complex_phi0", mode_name, dt, pdb, cleandw, [preddw_0])

    df_sweep = pd.DataFrame(sweep_rows)
    sweep_csv = RESULTS_DIR / "insight_low_power_sweep.csv"
    df_sweep.to_csv(sweep_csv, index=False, float_format="%.6f")
    logger.info("Saved %s (%d rows)", sweep_csv, len(df_sweep))

    # Print fine sweep tables
    _print_fine_summary(df_sweep)

    # ---- Step 5: Plots + Closed-Form alpha_max Stealth Overlay & Verdict ----
    _plot_insight_flip_vs_power(df_sweep)
    alpha_cls_map = _plot_stealth_alpha_max(df_sweep)
    verdict = _compute_insight_verdict(df_sweep, alpha_cls_map)
    print("\n" + "=" * 82)
    print(verdict)
    print("=" * 82 + "\n")
    return 0


def _print_fine_summary(df_sweep: pd.DataFrame) -> None:
    print("\n" + "=" * 82)
    print("FINE LOW-POWER SWEEP (-6 to +6 dB): MAX FLIP RATE ACROSS Δt ∈ {10,20,35,50} ns")
    print("=" * 82)
    for ds in ["fontaine_iiot19", "fontaine_iiot20_5phi", "dw3000_complex_5phi"]:
        for mode in ["single", "diffuse"]:
            sub = df_sweep[(df_sweep["dataset"] == ds) & (df_sweep["mode"] == mode)]
            print(f"\n--- Dataset={ds} | Mode={mode} ---")
            for pdb, grp in sub.groupby("power_dB"):
                print(
                    f"  {pdb:+5.1f}dB | XGB-Emb={grp['flip_insight_xgb_emb'].max()*100:5.1f}% | "
                    f"XGB-Full={grp['flip_insight_xgb_full'].max()*100:5.1f}% | "
                    f"SVM={grp['flip_insight_svm'].max()*100:5.1f}% | "
                    f"REMNet={grp['flip_insight_remnet_cls'].max()*100:5.1f}% | "
                    f"LGBM={grp['flip_lgbm_cls'].max()*100:5.1f}% | "
                    f"APS>6={grp['flip_aps6'].max()*100:5.1f}% | "
                    f"CNN-B={grp['flip_cnn_cls_B'].max()*100:5.1f}%"
                )


def _plot_insight_flip_vs_power(df_sweep: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, ds, title in [
        (axes[0], "fontaine_iiot19", "Fontaine IIoT19 (907 LOS Test CIRs)"),
        (axes[1], "dw3000_complex_5phi", "Santamaria DW3000 Complex-IQ (250 LOS, 5φ avg)"),
    ]:
        ax.axvspan(-6.3, 6.3, color="#e3f2fd", alpha=0.4, label="DW3000 DGC-Safe Zone (≤ +6 dB)")
        ax.axhline(70.0, color="#c62828", linestyle="--", linewidth=1.3, label="70% Anchor-Exclusion Target")
        ax.axhline(20.0, color="gray", linestyle=":", linewidth=1.0, label="20% Pivot Threshold")

        curves = [
            ("single",  "flip_insight_xgb_emb",    "InSight-XGB-Emb (Single)",  "-",  "o", "#1565C0"),
            ("single",  "flip_insight_xgb_full",   "InSight-XGB-Full (Single)", "-",  "s", "#0D47A1"),
            ("single",  "flip_insight_svm",        "InSight-SVM (Single)",      "-",  "^", "#2E7D32"),
            ("single",  "flip_insight_remnet_cls", "InSight-REMNet (Single)",   "-",  "d", "#EF6C00"),
            ("single",  "flip_lgbm_cls",           "LightGBM-Cls (Single)",     "--", "P", "#6A1B9A"),
            ("single",  "flip_aps6",               "APS006 >6 dB (Single)",     ":",  "x", "#00838F"),
            ("single",  "flip_cnn_cls_B",          "CNN-Cls Reg B (Single)",    "--", "v", "#546E7A"),
            ("diffuse", "flip_insight_xgb_full",   "InSight-XGB-Full (Diffuse)", "-.", "s", "#42A5F5"),
            ("diffuse", "flip_insight_svm",        "InSight-SVM (Diffuse)",     "-.", "^", "#66BB6A"),
            ("diffuse", "flip_insight_remnet_cls", "InSight-REMNet (Diffuse)",  "-.", "d", "#FFA726"),
        ]
        for mode, col, lbl, ls, mk, color in curves:
            sub = df_sweep[(df_sweep["dataset"] == ds) & (df_sweep["mode"] == mode)]
            if sub.empty or col not in sub.columns:
                continue
            # Plot max flip across Δt at each power_dB (best attack delay)
            g = sub.groupby("power_dB")[col].max() * 100.0
            ax.plot(g.index, g.values, marker=mk, linestyle=ls, color=color, label=lbl, linewidth=1.8)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Injected Power (dB relative to LOS first path)", fontsize=11)
        ax.set_xticks(POWER_GRID)
        ax.set_ylim(-4, 105)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Max Clean LOS → NLOS Flip Rate (% across Δt)", fontsize=11)
    axes[1].legend(loc="upper left", fontsize=8, ncol=1)
    axes[0].legend(loc="upper left", fontsize=8, ncol=1)
    fig.suptitle("InSight Embedded ML & Baselines: Low-Power Single vs. Diffuse Anchor Exclusion", fontsize=13)
    fig.tight_layout()
    out_p = RESULTS_DIR / "insight_flip_vs_power.png"
    fig.savefig(out_p, dpi=150)
    logger.info("Saved %s", out_p)
    plt.close(fig)


def _min_alpha_for_threshold(df_sweep: pd.DataFrame, col: str, target_flip: float = 0.70) -> tuple[float | None, float | None]:
    """Return (min power_dB, linear alpha_cls = 10^(p/20)) achieving >= target_flip across IIoT19 & DW3000."""
    sub = df_sweep[df_sweep["dataset"].isin(["fontaine_iiot19", "dw3000_complex_5phi"])]
    by_p = sub.groupby(["dataset", "mode", "power_dB"])[col].max().reset_index()
    hits = by_p[by_p[col] >= target_flip]
    if hits.empty:
        return None, None
    best_p = float(hits["power_dB"].min())
    return best_p, float(10.0 ** (best_p / 20.0))


def _plot_stealth_alpha_max(df_sweep: pd.DataFrame) -> dict[str, tuple[float | None, float | None]]:
    """Plot closed-form cross-CIR stealth window alpha_max(SNR_STS) vs empirical alpha_cls."""
    N_STS = 4096.0
    M = 75.0
    P_d_max = 0.10
    snr_sts_db = np.linspace(5.0, 25.0, 200)
    snr_sts = 10.0 ** (snr_sts_db / 10.0)
    snr_ip  = 10.0 ** ((snr_sts_db + 6.0) / 10.0)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axvspan(5.0, 15.0, color="#e8f5e9", alpha=0.45, label="Realistic Indoor Low-SNR Regime (SNR_STS ≤ 15 dB)")

    kappas = [1.0, 2.0, 4.0]
    pfas = [0.05, 0.01]
    colors_k = {1.0: "#1565C0", 2.0: "#2E7D32", 4.0: "#E65100"}
    styles_p = {0.05: "-", 0.01: "--"}

    z_term = norm.ppf(1.0 - P_d_max) / np.sqrt(2.0)
    for kap in kappas:
        sigma_rel = np.sqrt(kap / N_STS + 1.0 / snr_sts + 1.0 / snr_ip)
        for pfa in pfas:
            alpha_max = (np.sqrt(np.log(M / pfa)) - z_term) * sigma_rel
            ax.plot(
                snr_sts_db, alpha_max,
                color=colors_k[kap], linestyle=styles_p[pfa], linewidth=2.0,
                label=f"α_max (κ_mp={kap:.0f}, P_fa={pfa})",
            )

    # Overlay empirical >= 70% flip thresholds (alpha_cls = 10^(power_dB/20))
    model_specs = [
        ("InSight-XGB-Emb",  "flip_insight_xgb_emb",  "#D81B60", "-."),
        ("InSight-XGB-Full", "flip_insight_xgb_full", "#880E4F", ":"),
        ("InSight-SVM",      "flip_insight_svm",      "#00897B", "-."),
        ("LightGBM-Cls",     "flip_lgbm_cls",         "#6A1B9A", "--"),
        ("APS006 (>6 dB)",   "flip_aps6",             "#37474F", ":"),
    ]
    alpha_cls_map: dict[str, tuple[float | None, float | None]] = {}
    for lbl, col, c, ls in model_specs:
        pdb_min, alpha_cls = _min_alpha_for_threshold(df_sweep, col, target_flip=0.70)
        alpha_cls_map[lbl] = (pdb_min, alpha_cls)
        if alpha_cls is not None:
            ax.axhline(
                alpha_cls, color=c, linestyle=ls, linewidth=1.8,
                label=f"α_cls {lbl}: {alpha_cls:.2f} ({pdb_min:+.1f} dB)",
            )

    ax.set_xlabel("STS Signal-to-Noise Ratio SNR_STS (dB)  [SNR_IP = SNR_STS + 6 dB]", fontsize=11)
    ax.set_ylabel("Linear Amplitude Ratio α (relative to LOS first path)", fontsize=11)
    ax.set_title("Cross-CIR STS Stealth Window α_max vs. Empirical ≥70% Anchor-Exclusion α_cls", fontsize=13)
    ax.set_xlim(5.0, 25.0)
    ax.set_ylim(0.0, 2.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_p = RESULTS_DIR / "stealth_window_alpha_max.png"
    fig.savefig(out_p, dpi=150)
    logger.info("Saved %s", out_p)
    plt.close(fig)
    return alpha_cls_map


def _compute_insight_verdict(
    df_sweep: pd.DataFrame,
    alpha_cls_map: dict[str, tuple[float | None, float | None]],
) -> str:
    # Closed-form alpha_max benchmark at SNR_STS = 15 dB (and 10 dB), kappa_mp=2, P_fa=0.05
    def calc_amax(snr_db: float, kap: float = 2.0, pfa: float = 0.05) -> float:
        s_sts = 10.0 ** (snr_db / 10.0)
        s_ip  = 10.0 ** ((snr_db + 6.0) / 10.0)
        sig = np.sqrt(kap / 4096.0 + 1.0 / s_sts + 1.0 / s_ip)
        return float((np.sqrt(np.log(75.0 / pfa)) - norm.ppf(0.90) / np.sqrt(2.0)) * sig)

    amax_15 = calc_amax(15.0, kap=2.0, pfa=0.05)
    amax_10 = calc_amax(10.0, kap=2.0, pfa=0.05)
    amax_5  = calc_amax(5.0,  kap=4.0, pfa=0.05)

    insight_cols = [
        ("InSight-XGB-Emb",  "flip_insight_xgb_emb"),
        ("InSight-XGB-Full", "flip_insight_xgb_full"),
        ("InSight-SVM",      "flip_insight_svm"),
        ("InSight-REMNet",   "flip_insight_remnet_cls"),
    ]
    core = df_sweep[df_sweep["dataset"].isin(["fontaine_iiot19", "dw3000_complex_5phi"])]

    max_ins_6 = max(float(core[core["power_dB"] <= 6.0][c].max()) for _, c in insight_cols)
    max_ins_3 = max(float(core[core["power_dB"] <= 3.0][c].max()) for _, c in insight_cols)
    max_ins_0 = max(float(core[core["power_dB"] <= 0.0][c].max()) for _, c in insight_cols)

    # Check if any InSight model (XGB or SVM) hits >=70% at <= +3 dB and < alpha_max(SNR<=15 dB)
    xgb_svm_pass_3 = any(
        alpha_cls_map.get(k, (None, None))[0] is not None and alpha_cls_map[k][0] <= 3.0
        for k in ["InSight-XGB-Emb", "InSight-XGB-Full", "InSight-SVM"]
    )
    xgb_svm_pass_6 = any(
        alpha_cls_map.get(k, (None, None))[0] is not None and alpha_cls_map[k][0] <= 6.0
        for k in ["InSight-XGB-Emb", "InSight-XGB-Full", "InSight-SVM"]
    )
    min_ins_alpha = min(
        [alpha_cls_map[k][1] for k in ["InSight-XGB-Emb", "InSight-XGB-Full", "InSight-SVM"]
         if alpha_cls_map.get(k, (None, None))[1] is not None] or [float("inf")]
    )

    lines = [
        "STEP 5 STEALTHINESS & INSIGHT GO / NO-GO VERDICT:",
        f"  Closed-form α_max (κ_mp=2, P_fa=0.05): SNR_STS=10dB → {amax_10:.3f} ({20*np.log10(amax_10):+.2f} dB) | SNR_STS=15dB → {amax_15:.3f} ({20*np.log10(amax_15):+.2f} dB) | SNR=5dB,κ=4 → {amax_5:.3f} ({20*np.log10(amax_5):+.2f} dB)",
        f"  InSight max flip across models (single/diffuse): ≤0dB={max_ins_0*100:.1f}% | ≤+3dB={max_ins_3*100:.1f}% | ≤+6dB={max_ins_6*100:.1f}%",
        "  Empirical α_cls thresholds (≥70% flip): " + ", ".join(
            f"{k}={f'{v[1]:.3f} ({v[0]:+.1f}dB)' if v[1] is not None else 'never(<70%)'}"
            for k, v in alpha_cls_map.items()
        ),
        "",
    ]

    if xgb_svm_pass_3 and min_ins_alpha < amax_15:
        lines.append("★ VERDICT: GO — TIER 1/2 ANCHOR EXCLUSION")
        lines.append(f"  InSight flips ≥70% at ≤+3 dB AND α_cls ({min_ins_alpha:.3f}) < α_max(15 dB) ({amax_15:.3f}).")
    elif xgb_svm_pass_3 and min_ins_alpha < amax_10:
        lines.append("★ VERDICT: GO — CONDITIONAL TIER 1/2 (LOW-SNR INDOOR WINDOW ≤ 10 dB)")
        lines.append(f"  InSight flips ≥70% at ≤+3 dB; α_cls ({min_ins_alpha:.3f}) beats α_max at SNR_STS≤10 dB ({amax_10:.3f}) but exceeds α_max at 15 dB ({amax_15:.3f}).")
    elif xgb_svm_pass_6:
        lines.append("★ VERDICT: GO — TIER 3 UNHARDENED STACK")
        lines.append(f"  InSight flips ≥70% at ≤+6 dB, but α_cls ({min_ins_alpha:.3f}) exceeds α_max across normal indoor SNRs (≤15 dB).")
    elif max_ins_6 < 0.20:
        lines.append("★ VERDICT: PIVOT PROJECT (NO-GO)")
        lines.append("  InSight remains immune (<20% flip at ≤+6 dB across both single-tap and diffuse injection).")
    else:
        lines.append("★ VERDICT: NO-GO / BORDERLINE WEAK EFFECT")
        lines.append(f"  InSight max flip reached {max_ins_6*100:.1f}% at ≤+6 dB (between 20% and 70%), failing the ≥70% anchor-exclusion bar.")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
