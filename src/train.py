"""train.py — Train all UWB ranging-bias regressors & NLOS classifiers.

Trains ONLY on clean (un-attacked) Fontaine IIoT19 data using an anti-leakage
grouped split by ``location_ID`` (no spatial coordinates shared across
train/val/test).  Models are saved to ./models/.

If CIRsV2.hdf5 is available:
  - Trains LightGBM-Reg and LightGBM-Cls on recomputed CIR-derived features
    (CIR_DERIVED_COLS) so post-injection features remain physically consistent.
  - Trains DeepUWB-style 1D-CNN Regressor, 1D-CNN Classifier, and Undercomplete
    AE + CNN Regressor under BOTH normalization regimes:
      * Regime A: PAC-normalized CIR scaled by fixed training constant ``scale_A``
        (original LOS first-path peak stays physically unchanged after injection).
      * Regime B: Per-sample max-peak normalized CIR (``cir_win / max(cir_win)``).

Sanity check: prints clean held-out test metrics and verifies that clean
regressors beat raw uncorrected ranging error by >30 %.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)

# Restrict CPU threads to P-cores on hybrid CPUs to avoid OpenMP barrier stalls.
torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (CNN_WIN_LEN, extract_cir_window, grouped_split,
                     load_fontaine_cirs, load_fontaine_scalar)
from features import FEATURE_NAMES, compute_feature_matrix
from models import (AECNNRegressor, APS006Classifier, CNN1DClassifier,
                    CNN1DRegressor, CIRAutoencoder, LGBM_CLS_PARAMS,
                    LGBM_REG_PARAMS, SCALAR_FEATURE_COLS)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

SEED = 42
MODELS_DIR = Path("models")
DEVICE = "cpu"
EPOCHS_CNN = 35
EPOCHS_AE  = 25
BATCH = 128


# ============================================================
#  Helpers
# ============================================================

def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_pickle(obj, name: str) -> None:
    p = MODELS_DIR / f"{name}.pkl"
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved %s (%d bytes)", p, p.stat().st_size)


def save_torch(model: nn.Module, name: str) -> None:
    p = MODELS_DIR / f"{name}.pt"
    torch.save(model.state_dict(), p)
    logger.info("Saved %s (%d bytes)", p, p.stat().st_size)


def regression_metrics(
    y_true_mm: np.ndarray,
    y_pred_mm: np.ndarray,
    label: str = "",
) -> dict:
    raw_mae = float(np.mean(np.abs(y_true_mm)))
    raw_rmse = float(np.sqrt(np.mean(y_true_mm ** 2)))
    mae = float(mean_absolute_error(y_true_mm, y_pred_mm))
    rmse = float(np.sqrt(mean_squared_error(y_true_mm, y_pred_mm)))
    mae_imp = 1.0 - mae / max(raw_mae, 1e-6)
    rmse_imp = 1.0 - rmse / max(raw_rmse, 1e-6)
    logger.info(
        "  %-28s MAE=%6.1f mm (%+5.1f%% vs raw) | RMSE=%6.1f mm (%+5.1f%% vs raw)",
        label, mae, mae_imp * 100, rmse, rmse_imp * 100,
    )
    return {
        "mae_mm": mae,
        "rmse_mm": rmse,
        "mae_imp": mae_imp,
        "rmse_imp": rmse_imp,
    }


def classification_metrics(y_true, y_proba, label: str = "") -> dict:
    y_pred = (np.asarray(y_proba) >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = float("nan")
    logger.info("  %-28s Acc=%.3f  F1=%.3f  AUROC=%.3f", label, acc, f1, auc)
    return {"accuracy": acc, "f1": f1, "auroc": auc}


def make_cir_dataset(
    cir_mags: np.ndarray,
    fp_indices: np.ndarray,
    regime: str,
    scale_A: float | None = None,
) -> np.ndarray:
    """Extract 157-tap windows and apply normalization regime.

    regime = 'A' → PAC-normalized (Fontaine already /RXPACC) divided by a
                   fixed training constant ``scale_A`` so LOS first-path
                   amplitude stays physically invariant after injection.
    regime = 'B' → per-sample max-peak normalized (``win / max(win)``).
    """
    windows = np.stack([
        extract_cir_window(cir_mags[i], int(fp_indices[i]))
        for i in range(len(cir_mags))
    ]).astype(np.float32)

    if regime == "A":
        const = scale_A if (scale_A is not None and scale_A > 0) else 1.0
        windows = windows / np.float32(const)
    elif regime == "B":
        mx = np.maximum(windows.max(axis=1, keepdims=True), 1e-12)
        windows = windows / mx
    else:
        raise ValueError(f"Unknown normalization regime: {regime}")

    return windows.astype(np.float32)


# ============================================================
#  PyTorch training loops
# ============================================================

def _train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_fn,
    epochs: int,
    lr: float = 1.5e-3,
    task: str = "reg",
) -> nn.Module:
    """Train 1D-CNN or head on (N, 157) windows with cosine annealing."""
    set_seeds(SEED)
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_va_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_va_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    best_val = float("inf")
    best_state = None
    n_tr = len(X_train)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        ep_loss = 0.0
        for s in range(0, n_tr, BATCH):
            idx  = perm[s : s + BATCH]
            pred = model(X_tr_t[idx])
            loss = loss_fn(pred, y_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n_tr
        sched.step()

        model.eval()
        with torch.no_grad():
            vp = model(X_va_t)
            vl = loss_fn(vp, y_va_t).item()

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            logger.info("  Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f", ep, epochs, ep_loss, vl)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _train_autoencoder(
    ae: CIRAutoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = EPOCHS_AE,
) -> CIRAutoencoder:
    """Stage 1: Train undercomplete AE with MSE reconstruction loss."""
    set_seeds(SEED)
    ae = ae.to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1.5e-3, weight_decay=1e-5)
    mse = nn.MSELoss()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_va_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    n_tr = len(X_train)
    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        ae.train()
        perm = torch.randperm(n_tr)
        ep_loss = 0.0
        for s in range(0, n_tr, BATCH):
            idx = perm[s : s + BATCH]
            xb  = X_tr_t[idx]
            recon, _ = ae(xb)
            loss = mse(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n_tr

        ae.eval()
        with torch.no_grad():
            rv, _ = ae(X_va_t)
            vl = mse(rv, X_va_t).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in ae.state_dict().items()}
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            logger.info("  AE Epoch %2d/%d  train_recon=%.6f  val_recon=%.6f", ep, epochs, ep_loss, vl)

    if best_state is not None:
        ae.load_state_dict(best_state)
    ae.eval()
    return ae


# ============================================================
#  Main training pipeline
# ============================================================

def main() -> int:
    set_seeds()
    MODELS_DIR.mkdir(exist_ok=True)

    # ---- 1. Load scalar data & anti-leakage grouped split ----
    logger.info("=== Loading Fontaine scalar data ===")
    df = load_fontaine_scalar()
    tr_idx, va_idx, te_idx = grouped_split(df, group_col="location_ID", strategy="groupkfold")

    tr_grp = set(df.iloc[tr_idx]["location_ID"])
    va_grp = set(df.iloc[va_idx]["location_ID"])
    te_grp = set(df.iloc[te_idx]["location_ID"])
    assert tr_grp.isdisjoint(va_grp) and tr_grp.isdisjoint(te_grp) and va_grp.isdisjoint(te_grp), \
        f"Group overlap detected! tr={tr_grp} va={va_grp} te={te_grp}"
    logger.info("Anti-leakage grouped split verified: train=%s  val=%s  test=%s",
                sorted(tr_grp), sorted(va_grp), sorted(te_grp))

    save_pickle({"train": tr_idx, "val": va_idx, "test": te_idx}, "split_indices")

    X_tr = df.iloc[tr_idx][SCALAR_FEATURE_COLS].values.astype(np.float32)
    X_va = df.iloc[va_idx][SCALAR_FEATURE_COLS].values.astype(np.float32)
    X_te = df.iloc[te_idx][SCALAR_FEATURE_COLS].values.astype(np.float32)
    y_reg_tr = df.iloc[tr_idx]["error_mm"].values.astype(np.float32)
    y_reg_va = df.iloc[va_idx]["error_mm"].values.astype(np.float32)
    y_reg_te = df.iloc[te_idx]["error_mm"].values.astype(np.float32)
    y_cls_tr = df.iloc[tr_idx]["label"].values.astype(int)
    y_cls_va = df.iloc[va_idx]["label"].values.astype(int)
    y_cls_te = df.iloc[te_idx]["label"].values.astype(int)

    raw_mae = float(np.mean(np.abs(y_reg_te)))
    raw_rmse = float(np.sqrt(np.mean(y_reg_te ** 2)))
    logger.info("Raw uncorrected held-out test error: MAE=%.1f mm | RMSE=%.1f mm", raw_mae, raw_rmse)

    reg_clean_summary: dict[str, dict] = {}
    cls_clean_summary: dict[str, dict] = {}

    # ---- 2. LightGBM (CSV features) ----
    logger.info("=== Training LightGBM Regressor & Classifier (CSV features) ===")
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]

    dtrain = lgb.Dataset(X_tr, label=y_reg_tr, feature_name=SCALAR_FEATURE_COLS)
    dval   = lgb.Dataset(X_va, label=y_reg_va, reference=dtrain)
    lgbm_reg_csv = lgb.train(
        {k: v for k, v in LGBM_REG_PARAMS.items() if k != "n_estimators"},
        dtrain, num_boost_round=LGBM_REG_PARAMS["n_estimators"],
        valid_sets=[dval], callbacks=callbacks,
    )
    save_pickle(lgbm_reg_csv, "lgbm_reg_csv")
    reg_clean_summary["LightGBM-Reg (CSV)"] = regression_metrics(
        y_reg_te, lgbm_reg_csv.predict(X_te), "LightGBM-Reg (CSV)")

    dtrain_c = lgb.Dataset(X_tr, label=y_cls_tr, feature_name=SCALAR_FEATURE_COLS)
    dval_c   = lgb.Dataset(X_va, label=y_cls_va, reference=dtrain_c)
    lgbm_cls_csv = lgb.train(
        {k: v for k, v in LGBM_CLS_PARAMS.items() if k != "n_estimators"},
        dtrain_c, num_boost_round=LGBM_CLS_PARAMS["n_estimators"],
        valid_sets=[dval_c], callbacks=callbacks,
    )
    save_pickle(lgbm_cls_csv, "lgbm_cls_csv")
    cls_clean_summary["LightGBM-Cls (CSV)"] = classification_metrics(
        y_cls_te, lgbm_cls_csv.predict(X_te), "LightGBM-Cls (CSV)")

    # ---- 3. APS006 baseline ----
    aps = APS006Classifier()
    Pd_te = df.iloc[te_idx]["Pd"].values
    cls_clean_summary["APS006 (>10 dB CSV)"] = classification_metrics(
        y_cls_te, aps.predict(Pd_te).astype(float), "APS006 (>10 dB CSV)")
    logger.info("  APS006 (>6 dB CSV) clean test flag rate: %.1f%%",
                100.0 * float(np.mean(aps.predict_flagged(Pd_te))))

    # ---- 4. CIR waveforms & Neural / CIR-derived Models ----
    logger.info("=== Loading CIR waveforms from CIRsV2.hdf5 ===")
    cirs, fp_indices_cir, is_complex = load_fontaine_cirs(key="IIoT19", n_expected=len(df))
    have_cirs = cirs is not None

    if have_cirs:
        cir_mags = np.abs(cirs)
        logger.info("Computing CIR-derived feature matrix (local fp_idx=%d)...",
                    int(fp_indices_cir[0]))
        feat_matrix = compute_feature_matrix(cir_mags, fp_indices_cir)
        save_pickle({"feat_matrix": feat_matrix, "feature_names": FEATURE_NAMES}, "cir_features")

        # 4a. LightGBM on CIR-derived features
        Xd_tr, Xd_va, Xd_te = feat_matrix[tr_idx], feat_matrix[va_idx], feat_matrix[te_idx]
        dtrain_d = lgb.Dataset(Xd_tr, label=y_reg_tr, feature_name=FEATURE_NAMES)
        dval_d   = lgb.Dataset(Xd_va, label=y_reg_va, reference=dtrain_d)
        lgbm_reg_cir = lgb.train(
            {k: v for k, v in LGBM_REG_PARAMS.items() if k != "n_estimators"},
            dtrain_d, num_boost_round=LGBM_REG_PARAMS["n_estimators"],
            valid_sets=[dval_d], callbacks=callbacks,
        )
        save_pickle(lgbm_reg_cir, "lgbm_reg_cir")
        reg_clean_summary["LightGBM-Reg (CIR)"] = regression_metrics(
            y_reg_te, lgbm_reg_cir.predict(Xd_te), "LightGBM-Reg (CIR)")

        dtrain_dc = lgb.Dataset(Xd_tr, label=y_cls_tr, feature_name=FEATURE_NAMES)
        dval_dc   = lgb.Dataset(Xd_va, label=y_cls_va, reference=dtrain_dc)
        lgbm_cls_cir = lgb.train(
            {k: v for k, v in LGBM_CLS_PARAMS.items() if k != "n_estimators"},
            dtrain_dc, num_boost_round=LGBM_CLS_PARAMS["n_estimators"],
            valid_sets=[dval_dc], callbacks=callbacks,
        )
        save_pickle(lgbm_cls_cir, "lgbm_cls_cir")
        cls_clean_summary["LightGBM-Cls (CIR)"] = classification_metrics(
            y_cls_te, lgbm_cls_cir.predict(Xd_te), "LightGBM-Cls (CIR)")

        pd_cir_te = Xd_te[:, FEATURE_NAMES.index("RXP_minus_FPP_dB")]
        cls_clean_summary["APS006 (>10 dB CIR)"] = classification_metrics(
            y_cls_te, aps.predict(pd_cir_te).astype(float), "APS006 (>10 dB CIR)")

        # Fixed Regime A scale derived strictly from clean training split
        raw_win_tr = make_cir_dataset(cir_mags[tr_idx], fp_indices_cir[tr_idx], "A", scale_A=1.0)
        scale_A = float(np.percentile(raw_win_tr, 99))
        logger.info("Fixed Regime A training scaling constant (p99): %.4f", scale_A)

        # 4b. Neural models under Regime A (PAC) and Regime B (max-peak)
        y_m_tr = y_reg_tr / 1000.0
        y_m_va = y_reg_va / 1000.0

        for regime in ["A", "B"]:
            logger.info("=== Normalization Regime %s ===", regime)
            windows = make_cir_dataset(cir_mags, fp_indices_cir, regime, scale_A=scale_A)
            W_tr, W_va, W_te = windows[tr_idx], windows[va_idx], windows[te_idx]

            # 1D-CNN Regressor
            logger.info("--- CNN Regressor (Regime %s) ---", regime)
            cnn_reg = _train_torch_model(
                CNN1DRegressor(input_len=CNN_WIN_LEN),
                W_tr, y_m_tr, W_va, y_m_va,
                nn.HuberLoss(delta=0.3), EPOCHS_CNN, lr=1.5e-3, task="reg",
            )
            save_torch(cnn_reg, f"cnn_reg_{regime}")
            with torch.no_grad():
                pred_cnn_mm = cnn_reg(torch.tensor(W_te).unsqueeze(1)).numpy() * 1000.0
            reg_clean_summary[f"CNN-Reg (Regime {regime})"] = regression_metrics(
                y_reg_te, pred_cnn_mm, f"CNN-Reg (Regime {regime})")

            # 1D-CNN Classifier
            logger.info("--- CNN Classifier (Regime %s) ---", regime)
            cnn_cls = _train_torch_model(
                CNN1DClassifier(input_len=CNN_WIN_LEN),
                W_tr, y_cls_tr.astype(np.float32),
                W_va, y_cls_va.astype(np.float32),
                nn.BCEWithLogitsLoss(), EPOCHS_CNN, lr=1.5e-3, task="cls",
            )
            save_torch(cnn_cls, f"cnn_cls_{regime}")
            with torch.no_grad():
                logits = cnn_cls(torch.tensor(W_te).unsqueeze(1)).numpy()
                probs  = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            cls_clean_summary[f"CNN-Cls (Regime {regime})"] = classification_metrics(
                y_cls_te, probs, f"CNN-Cls (Regime {regime})")

            # Undercomplete Autoencoder + Frozen-Encoder Regressor
            logger.info("--- Undercomplete Autoencoder (Regime %s) ---", regime)
            ae = _train_autoencoder(CIRAutoencoder(input_len=CNN_WIN_LEN), W_tr, W_va, epochs=EPOCHS_AE)
            save_torch(ae, f"ae_{regime}")

            logger.info("--- AE+CNN Regressor (Regime %s) ---", regime)
            ae_reg = _train_torch_model(
                AECNNRegressor(ae.encoder),
                W_tr, y_m_tr, W_va, y_m_va,
                nn.HuberLoss(delta=0.3), EPOCHS_CNN, lr=1.5e-3, task="reg",
            )
            save_torch(ae_reg, f"ae_reg_{regime}")
            with torch.no_grad():
                pred_ae_mm = ae_reg(torch.tensor(W_te).unsqueeze(1)).numpy() * 1000.0
            reg_clean_summary[f"AE-Reg (Regime {regime})"] = regression_metrics(
                y_reg_te, pred_ae_mm, f"AE-Reg (Regime {regime})")

        save_pickle({
            "have_cirs": True,
            "scale_A": scale_A,
            "is_complex": is_complex,
            "reg_clean_summary": reg_clean_summary,
            "cls_clean_summary": cls_clean_summary,
        }, "train_meta")
    else:
        logger.warning("CIRs unavailable — trained scalar-feature models only.")
        save_pickle({
            "have_cirs": False,
            "scale_A": 1.0,
            "is_complex": False,
            "reg_clean_summary": reg_clean_summary,
            "cls_clean_summary": cls_clean_summary,
        }, "train_meta")

    # ---- 5. Clean Sanity Check Gate (>30% improvement over raw error) ----
    print("\n" + "=" * 78)
    print(f"CLEAN HELD-OUT TEST SANITY CHECK (raw MAE={raw_mae:.1f} mm, raw RMSE={raw_rmse:.1f} mm)")
    print("=" * 78)
    best_imp = -1e9
    for name, m in reg_clean_summary.items():
        imp = max(m["mae_imp"], m["rmse_imp"])
        best_imp = max(best_imp, imp)
        print(f"  {name:24s} | MAE={m['mae_mm']:6.1f} mm ({m['mae_imp']*100:+5.1f}%) | "
              f"RMSE={m['rmse_mm']:6.1f} mm ({m['rmse_imp']*100:+5.1f}%)")
    for name, m in cls_clean_summary.items():
        print(f"  {name:24s} | Acc={m['accuracy']:.3f} | F1={m['f1']:.3f} | AUROC={m['auroc']:.3f}")
    print("=" * 78)

    if best_imp <= 0.30:
        logger.error(
            "ABORT: No clean regressor beat raw uncorrected error by >30%% (best=%.1f%%).",
            best_imp * 100,
        )
        return 1

    logger.info("Sanity check PASSED (best clean error reduction = %.1f%% > 30%%).", best_imp * 100)
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    raise SystemExit(main())
