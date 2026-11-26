"""models.py — Model definitions for UWB ranging-bias regression & NLOS classification.

Models
------
1. LightGBM Regressor   (scalar CIR features → ranging error mm)
2. LightGBM Classifier  (scalar CIR features → LOS / NLOS)
3. APS006 Threshold Rule (Qorvo APS006 App Note)
4. 1D-CNN Regressor      (157-tap CIR window  → ranging error mm)
5. 1D-CNN Classifier     (157-tap CIR window  → LOS / NLOS)
6. AE + CNN Regressor    (autoencoder latent   → ranging error mm)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ============================================================
#  Scalar feature column sets
# ============================================================

# CSV-native features (from features_IIoT_19.csv)
SCALAR_FEATURE_COLS = [
    "fp_ampl1", "fp_ampl2", "fp_ampl3", "std_noise",
    "CIRpower", "RXPACC", "RX_power", "FP_power", "Pd", "Pr",
]

# CIR-derived features (recomputed from waveform after injection)
CIR_DERIVED_COLS = [
    "FPP_dB", "RXP_dB", "RXP_minus_FPP_dB", "PPR_dB",
    "rise_time_ns", "mean_excess_delay_ns", "rms_delay_spread_ns",
    "kurtosis", "skewness", "energy_ratio",
]

# LightGBM hyper-parameters
LGBM_REG_PARAMS: dict = dict(
    objective="regression", metric="rmse", boosting_type="gbdt",
    num_leaves=63, learning_rate=0.05, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, verbose=-1, seed=42,
    n_estimators=500,
)
LGBM_CLS_PARAMS: dict = dict(
    objective="binary", metric="binary_logloss", boosting_type="gbdt",
    num_leaves=63, learning_rate=0.05, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, verbose=-1, seed=42,
    n_estimators=500, is_unbalance=True,
)


# ============================================================
#  APS006 Threshold Rule
# ============================================================

class APS006Classifier:
    """Qorvo APS006: flag NLOS when RXP − FPP exceeds threshold (dB).

    Default: >10 dB → NLOS,  >6 dB → uncertain / flagged.
    """

    def __init__(self, thr_nlos: float = 10.0, thr_flag: float = 6.0):
        self.thr_nlos = thr_nlos
        self.thr_flag = thr_flag

    def predict(self, rxp_minus_fpp: np.ndarray) -> np.ndarray:
        return (np.asarray(rxp_minus_fpp) > self.thr_nlos).astype(int)

    def predict_flagged(self, rxp_minus_fpp: np.ndarray) -> np.ndarray:
        return (np.asarray(rxp_minus_fpp) > self.thr_flag).astype(int)


# ============================================================
#  1D-CNN Regressor
# ============================================================

class CNN1DRegressor(nn.Module):
    """DeepUWB-style 1D-CNN: 157-tap CIR window → ranging bias (m)."""

    def __init__(self, input_len: int = 157):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x).squeeze(-1)).squeeze(-1)


# ============================================================
#  1D-CNN Classifier
# ============================================================

class CNN1DClassifier(nn.Module):
    """1D-CNN for binary LOS / NLOS classification."""

    def __init__(self, input_len: int = 157):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x).squeeze(-1)).squeeze(-1)


# ============================================================
#  Convolutional Autoencoder
# ============================================================

class CIRAutoencoder(nn.Module):
    """Undercomplete 1D-convolutional autoencoder for CIR windows."""

    def __init__(self, input_len: int = 157, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 64, 4, stride=2, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 5, padding=2),
        )
        self._adapt = nn.AdaptiveAvgPool1d(input_len)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self._adapt(self.decoder(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class AECNNRegressor(nn.Module):
    """Frozen AE encoder + MLP regression head for ranging bias."""

    def __init__(self, encoder: nn.Module, latent_ch: int = 64, latent_dim: int = 32):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_ch * latent_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)
        return self.head(z).squeeze(-1)

