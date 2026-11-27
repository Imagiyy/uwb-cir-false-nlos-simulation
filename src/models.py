"""models.py — Model definitions for UWB ranging-bias regression & NLOS classification.

Models
------
1. LightGBM Regressor   (scalar CIR features → ranging bias mm)
2. LightGBM Classifier  (scalar CIR features → LOS / NLOS)
3. APS006 Threshold Rule (Qorvo APS006 App Note)
4. 1D-CNN Regressor      (157-tap CIR window  → ranging bias m)
5. 1D-CNN Classifier     (157-tap CIR window  → LOS / NLOS)
6. AE + CNN Regressor    (undercomplete autoencoder latent → ranging bias m)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from features import FEATURE_NAMES


# ============================================================
#  Scalar feature column sets
# ============================================================

# CSV-native features (from features_IIoT_19.csv)
SCALAR_FEATURE_COLS = [
    "fp_ampl1", "fp_ampl2", "fp_ampl3", "std_noise",
    "CIRpower", "RXPACC", "RX_power", "FP_power", "Pd", "Pr",
]

# CIR-derived features (recomputed from waveform after injection)
CIR_DERIVED_COLS = list(FEATURE_NAMES)

# LightGBM hyper-parameters
LGBM_REG_PARAMS: dict = dict(
    objective="regression", metric="rmse", boosting_type="gbdt",
    num_leaves=31, learning_rate=0.03, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, min_child_samples=50,
    verbose=-1, seed=42, n_estimators=500,
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
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1),
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
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x).squeeze(-1)).squeeze(-1)


# ============================================================
#  Undercomplete Convolutional Autoencoder + Regressor Head
# ============================================================

class CIRAutoencoder(nn.Module):
    """Undercomplete 1D-convolutional autoencoder for 157-tap CIR windows.

    Bottleneck shape is (B, 8, 8) = 64 floats (< 157 input taps, ~2.45x
    compression), ensuring the encoder learns a compact latent representation.
    """

    def __init__(self, input_len: int = 157, latent_ch: int = 8, latent_len: int = 8):
        super().__init__()
        self.latent_ch = latent_ch
        self.latent_len = latent_len
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, latent_ch, 3, padding=1), nn.BatchNorm1d(latent_ch), nn.ReLU(),
            nn.AdaptiveAvgPool1d(latent_len),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_ch, 32, 4, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 32, 4, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 1, 5, padding=2),
        )
        self._adapt = nn.AdaptiveAvgPool1d(input_len)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._adapt(self.decoder(z))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


class AECNNRegressor(nn.Module):
    """Frozen undercomplete AE encoder + lightweight CNN/MLP head for bias (m)."""

    def __init__(self, encoder: nn.Module, latent_ch: int = 8, latent_len: int = 8):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Conv1d(latent_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * latent_len, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(x)
        return self.head(z).squeeze(-1)
