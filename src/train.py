"""train.py — Train all UWB ranging-bias regressors & NLOS classifiers.

Trains ONLY on clean (un-attacked) data.  Models are saved to ./models/.
If CIRsV2.hdf5 is available, CNN/AE models are trained under two
normalization regimes (A: PAC-normalized, B: per-sample max-peak).
Otherwise only scalar-feature models (LightGBM + APS006) are trained.

Sanity check: clean held-out test MAE must beat raw uncorrected error
MAE by >30 %, else the script prints a warning and continues.
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
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)

# ----- project modules -----
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (CNN_WIN_LEN, extract_cir_window, grouped_split,
                     load_fontaine_cirs, load_fontaine_scalar)
from features import FEATURE_NAMES, compute_feature_matrix
from models import (AECNNRegressor, APS006Classifier, CNN1DClassifier,
                    CNN1DRegressor, CIR_DERIVED_COLS, CIRAutoencoder,
                    LGBM_CLS_PARAMS, LGBM_REG_PARAMS, SCALAR_FEATURE_COLS)

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
EPOCHS_CNN = 50
EPOCHS_AE  = 30
BATCH = 64


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


def regression_metrics(y_true, y_pred, label: str = "") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    logger.info("  %s  MAE=%.1f mm  RMSE=%.1f mm", label, mae, rmse)
    return {"mae_mm": mae, "rmse_mm": rmse}


def classification_metrics(y_true, y_proba, label: str = "") -> dict:
    y_pred = (np.asarray(y_proba) > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")
    logger.info("  %s  Acc=%.3f  F1=%.3f  AUROC=%.3f", label, acc, f1, auc)
    return {"accuracy": acc, "f1": f1, "auroc": auc}


def make_cir_dataset(
    cir_mags: np.ndarray,
    fp_indices: np.ndarray,
    regime: str,
) -> np.ndarray:
    """Extract 157-tap windows and apply normalization regime.

    regime = 'A'  → PAC-normalized (identity, Fontaine data already divided by RXPACC)
    regime = 'B'  → per-sample max-peak normalized
    """
    windows = np.stack([
        extract_cir_window(cir_mags[i], int(fp_indices[i]))
        for i in range(len(cir_mags))
    ])  # (N, 157)

    if regime == "B":
        mx = windows.max(axis=1, keepdims=True)
        mx = np.maximum(mx, 1e-12)
        windows = windows / mx

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
    lr: float = 1e-3,
    task: str = "reg",
) -> nn.Module:
    """Generic train loop for 1D-CNN input (N,157) → scalar."""
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(X_train))
        ep_loss = 0.0
        for s in range(0, len(perm), BATCH):
            idx   = perm[s : s + BATCH]
            xb    = torch.tensor(X_train[idx], dtype=torch.float32).unsqueeze(1).to(DEVICE)
            yb    = torch.tensor(y_train[idx], dtype=torch.float32).to(DEVICE)
            pred  = model(xb)
            loss  = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= len(perm)

        # Validation
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            yv = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
            vp = model(xv)
            vl = loss_fn(vp, yv).item()

        sched.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            logger.info("  Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f", ep, epochs, ep_loss, vl)

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _train_autoencoder(
    ae: CIRAutoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = EPOCHS_AE,
) -> CIRAutoencoder:
    """Stage 1: Train AE with MSE reconstruction loss."""
    ae = ae.to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        ae.train()
        perm = np.random.permutation(len(X_train))
        ep_loss = 0.0
        for s in range(0, len(perm), BATCH):
            idx = perm[s : s + BATCH]
            xb  = torch.tensor(X_train[idx], dtype=torch.float32).unsqueeze(1).to(DEVICE)
            recon, _ = ae(xb)
            loss = mse(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= len(perm)
        if ep % 10 == 0 or ep == 1:
            ae.eval()
            with torch.no_grad():
                xv = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)
                rv, _ = ae(xv)
                vl = mse(rv, xv).item()
            logger.info("  AE Epoch %3d/%d  train_recon=%.6f  val_recon=%.6f", ep, epochs, ep_loss, vl)
    ae.eval()
    return ae


# ============================================================
#  Main training pipeline
# ============================================================

def main() -> int:
    set_seeds()
    MODELS_DIR.mkdir(exist_ok=True)

    # ---- 1. Load scalar data ----
    logger.info("=== Loading Fontaine scalar data ===")
    df = load_fontaine_scalar()
    tr_idx, va_idx, te_idx = grouped_split(df, group_col="location_ID")

    # Verify no group overlap
    tr_grp = set(df.iloc[tr_idx]["location_ID"])
    va_grp = set(df.iloc[va_idx]["location_ID"])
    te_grp = set(df.iloc[te_idx]["location_ID"])
    assert tr_grp.isdisjoint(va_grp) and tr_grp.isdisjoint(te_grp) and va_grp.isdisjoint(te_grp), \
        f"Group overlap detected! tr={tr_grp} va={va_grp} te={te_grp}"
    logger.info("Group split verified: train=%s  val=%s  test=%s", tr_grp, va_grp, te_grp)

    # Save split indices for evaluate.py
    save_pickle({"train": tr_idx, "val": va_idx, "test": te_idx}, "split_indices")

    # Feature & target matrices
    X_tr = df.iloc[tr_idx][SCALAR_FEATURE_COLS].values.astype(np.float32)
    X_va = df.iloc[va_idx][SCALAR_FEATURE_COLS].values.astype(np.float32)
    X_te = df.iloc[te_idx][SCALAR_FEATURE_COLS].values.astype(np.float32)
    y_reg_tr = df.iloc[tr_idx]["error_mm"].values.astype(np.float32)
    y_reg_va = df.iloc[va_idx]["error_mm"].values.astype(np.float32)
    y_reg_te = df.iloc[te_idx]["error_mm"].values.astype(np.float32)
    y_cls_tr = df.iloc[tr_idx]["label"].values.astype(int)
    y_cls_va = df.iloc[va_idx]["label"].values.astype(int)
    y_cls_te = df.iloc[te_idx]["label"].values.astype(int)

    # ---- 2. LightGBM Regressor (CSV features) ----
    logger.info("=== Training LightGBM Regressor (CSV features) ===")
    dtrain = lgb.Dataset(X_tr, label=y_reg_tr, feature_name=SCALAR_FEATURE_COLS)
    dval   = lgb.Dataset(X_va, label=y_reg_va, reference=dtrain)
    callbacks = [lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)]
    lgbm_reg = lgb.train(
        {k: v for k, v in LGBM_REG_PARAMS.items() if k != "n_estimators"},
        dtrain, num_boost_round=LGBM_REG_PARAMS["n_estimators"],
        valid_sets=[dval], callbacks=callbacks,
    )
    save_pickle(lgbm_reg, "lgbm_reg_csv")

    pred_reg_te = lgbm_reg.predict(X_te)
    m_reg = regression_metrics(y_reg_te, pred_reg_te, "LightGBM-Reg (CSV)")

    # Sanity check: the regressor predicts error_mm (= estimated - true).
    # "Correcting" means: corrected_range = estimated - b_hat, so
    # residual = true_error - predicted_error.  The raw (uncorrected)
    # residual is just the error itself (i.e., using b_hat=0).
    raw_mae = float(np.mean(np.abs(y_reg_te)))  # MAE of uncorrected error
    residual_after = np.abs(y_reg_te - pred_reg_te)
    corrected_mae = float(np.mean(residual_after))  # MAE after correction
    improvement = 1.0 - corrected_mae / max(raw_mae, 1e-6)
    logger.info("  Raw uncorrected MAE=%.1f mm → Corrected residual MAE=%.1f mm → improvement %.1f%%",
                raw_mae, corrected_mae, improvement * 100)
    if improvement < 0.30:
        logger.warning("⚠ LightGBM regressor did NOT improve >30%% over raw error (%.1f%%). "
                       "Continuing anyway.", improvement * 100)

    # ---- 3. LightGBM Classifier (CSV features) ----
    logger.info("=== Training LightGBM Classifier (CSV features) ===")
    dtrain_c = lgb.Dataset(X_tr, label=y_cls_tr, feature_name=SCALAR_FEATURE_COLS)
    dval_c   = lgb.Dataset(X_va, label=y_cls_va, reference=dtrain_c)
    lgbm_cls = lgb.train(
        {k: v for k, v in LGBM_CLS_PARAMS.items() if k != "n_estimators"},
        dtrain_c, num_boost_round=LGBM_CLS_PARAMS["n_estimators"],
        valid_sets=[dval_c], callbacks=callbacks,
    )
    save_pickle(lgbm_cls, "lgbm_cls_csv")

    pred_cls_te = lgbm_cls.predict(X_te)
    classification_metrics(y_cls_te, pred_cls_te, "LightGBM-Cls (CSV)")

    # ---- 4. APS006 (no training) ----
    logger.info("=== APS006 Threshold Rule ===")
    aps = APS006Classifier()
    Pd_te = df.iloc[te_idx]["Pd"].values
    aps_pred = aps.predict(Pd_te)
    aps_flag = aps.predict_flagged(Pd_te)
    classification_metrics(y_cls_te, aps_pred.astype(float), "APS006 (>10 dB)")
    logger.info("  APS006 flagged (>6 dB) rate: %.1f%%",
                100 * np.mean(aps_flag))

    # ---- 5. CIR-derived feature LightGBM (for post-injection evaluation) ----
    # We need CIR waveforms for this.  Try loading them.
    logger.info("=== Attempting to load CIR waveforms ===")
    cirs = load_fontaine_cirs(n_expected=len(df))
    have_cirs = cirs is not None

    if have_cirs:
        logger.info("CIRs loaded. Computing CIR-derived features for LightGBM training.")
        cir_mags = np.abs(cirs)
        fp_indices = df["fpi"].values

        # Compute CIR-derived features for all samples
        feat_matrix = compute_feature_matrix(cir_mags, fp_indices)
        logger.info("CIR-derived feature matrix: shape=%s", feat_matrix.shape)

        # Train LightGBM Regressor on CIR-derived features
        logger.info("=== Training LightGBM Regressor (CIR-derived) ===")
        Xd_tr = feat_matrix[tr_idx]
        Xd_va = feat_matrix[va_idx]
        Xd_te = feat_matrix[te_idx]
        dtrain_d = lgb.Dataset(Xd_tr, label=y_reg_tr, feature_name=FEATURE_NAMES)
        dval_d   = lgb.Dataset(Xd_va, label=y_reg_va, reference=dtrain_d)
        lgbm_reg_cir = lgb.train(
            {k: v for k, v in LGBM_REG_PARAMS.items() if k != "n_estimators"},
            dtrain_d, num_boost_round=LGBM_REG_PARAMS["n_estimators"],
            valid_sets=[dval_d], callbacks=callbacks,
        )
        save_pickle(lgbm_reg_cir, "lgbm_reg_cir")
        pred_cir_te = lgbm_reg_cir.predict(Xd_te)
        regression_metrics(y_reg_te, pred_cir_te, "LightGBM-Reg (CIR-derived)")

        # Train LightGBM Classifier on CIR-derived features
        logger.info("=== Training LightGBM Classifier (CIR-derived) ===")
        dtrain_dc = lgb.Dataset(Xd_tr, label=y_cls_tr, feature_name=FEATURE_NAMES)
        dval_dc   = lgb.Dataset(Xd_va, label=y_cls_va, reference=dtrain_dc)
        lgbm_cls_cir = lgb.train(
            {k: v for k, v in LGBM_CLS_PARAMS.items() if k != "n_estimators"},
            dtrain_dc, num_boost_round=LGBM_CLS_PARAMS["n_estimators"],
            valid_sets=[dval_dc], callbacks=callbacks,
        )
        save_pickle(lgbm_cls_cir, "lgbm_cls_cir")
        pred_cls_cir_te = lgbm_cls_cir.predict(Xd_te)
        classification_metrics(y_cls_te, pred_cls_cir_te, "LightGBM-Cls (CIR-derived)")

        # Save CIR-derived feature matrix for evaluate.py
        save_pickle({"feat_matrix": feat_matrix, "feature_names": FEATURE_NAMES}, "cir_features")

        # ---- 6. CNN/AE models under two normalization regimes ----
        for regime in ["A", "B"]:
            logger.info("=== Normalization Regime %s ===", regime)
            windows = make_cir_dataset(cir_mags, fp_indices, regime)
            W_tr, W_va, W_te = windows[tr_idx], windows[va_idx], windows[te_idx]

            # 6a. 1D-CNN Regressor
            logger.info("--- CNN Regressor (Regime %s) ---", regime)
            cnn_reg = CNN1DRegressor(input_len=CNN_WIN_LEN)
            # Target in meters for CNN (more natural for ranging)
            y_m_tr = y_reg_tr / 1000.0
            y_m_va = y_reg_va / 1000.0
            y_m_te = y_reg_te / 1000.0
            cnn_reg = _train_torch_model(
                cnn_reg, W_tr, y_m_tr, W_va, y_m_va,
                nn.HuberLoss(delta=0.5), EPOCHS_CNN, task="reg",
            )
            save_torch(cnn_reg, f"cnn_reg_{regime}")
            with torch.no_grad():
                xt = torch.tensor(W_te, dtype=torch.float32).unsqueeze(1)
                pred_cnn_te = cnn_reg(xt).numpy()
            regression_metrics(y_reg_te, pred_cnn_te * 1000, f"CNN-Reg Regime {regime}")

            # 6b. 1D-CNN Classifier
            logger.info("--- CNN Classifier (Regime %s) ---", regime)
            cnn_cls = CNN1DClassifier(input_len=CNN_WIN_LEN)
            cnn_cls = _train_torch_model(
                cnn_cls, W_tr, y_cls_tr.astype(np.float32),
                W_va, y_cls_va.astype(np.float32),
                nn.BCEWithLogitsLoss(), EPOCHS_CNN, task="cls",
            )
            save_torch(cnn_cls, f"cnn_cls_{regime}")
            with torch.no_grad():
                xt = torch.tensor(W_te, dtype=torch.float32).unsqueeze(1)
                logits = cnn_cls(xt).numpy()
                probs  = 1.0 / (1.0 + np.exp(-logits))
            classification_metrics(y_cls_te, probs, f"CNN-Cls Regime {regime}")

            # 6c. Autoencoder + CNN Regressor
            logger.info("--- Autoencoder (Regime %s) ---", regime)
            ae = CIRAutoencoder(input_len=CNN_WIN_LEN)
            ae = _train_autoencoder(ae, W_tr, W_va, epochs=EPOCHS_AE)
            save_torch(ae, f"ae_{regime}")

            logger.info("--- AE+CNN Regressor (Regime %s) ---", regime)
            ae_reg = AECNNRegressor(ae.encoder)
            ae_reg = _train_torch_model(
                ae_reg, W_tr, y_m_tr, W_va, y_m_va,
                nn.HuberLoss(delta=0.5), EPOCHS_CNN, task="reg",
            )
            save_torch(ae_reg, f"ae_reg_{regime}")
            with torch.no_grad():
                xt = torch.tensor(W_te, dtype=torch.float32).unsqueeze(1)
                pred_ae_te = ae_reg(xt).numpy()
            regression_metrics(y_reg_te, pred_ae_te * 1000, f"AE-Reg Regime {regime}")

        # Save normalization stats for evaluate.py
        save_pickle({"have_cirs": True}, "train_meta")

    else:
        logger.warning("CIRs unavailable — training ONLY scalar-feature models.")
        logger.warning("CNN, AE, and CIR-derived LightGBM models will be SKIPPED.")
        logger.info("The floor-wall evaluation will use analytical CIR feature simulation.")
        save_pickle({"have_cirs": False}, "train_meta")

    logger.info("=== Training complete. Models saved to %s ===", MODELS_DIR.resolve())
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    raise SystemExit(main())

