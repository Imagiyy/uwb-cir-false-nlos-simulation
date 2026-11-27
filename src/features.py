"""features.py — Recompute scalar UWB CIR diagnostic features from waveform.

All features are computed directly from the CIR magnitude envelope,
so they can be used consistently before AND after phantom-NLOS injection.
This avoids stale-CSV-feature leakage: after modifying the CIR waveform,
hardware-register values (fp_ampl1/2/3 etc.) are no longer valid.

Unit discipline
---------------
- 1 CIR tap ≈ 1.0016 ns  (DW1000 / DW3000, fs = 499.2 MHz × 2 = ~1 GHz)
- Power is in *linear* units (sum of squared amplitudes); dB via 10·log10.
- Amplitude scaling for +X dB *power* boost  → ×10^(X/20).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as _sp_kurtosis, skew as _sp_skew

TAP_NS = 1.0016  # nanoseconds per CIR tap


def compute_features_from_cir(
    cir_mag: np.ndarray,
    fp_idx: int,
    signal_window: int = 150,
) -> dict[str, float]:
    """Compute ALL scalar CIR features from a magnitude envelope.

    Parameters
    ----------
    cir_mag : 1-D float array
        Magnitude envelope of the CIR (already |h| or sqrt(I²+Q²)).
    fp_idx : int
        First-path tap index inside ``cir_mag`` (e.g., 51 for CIRsV2 150-tap
        slices, or round(fp_ind/64) for full 1016-tap captures).
    signal_window : int
        Taps after fp_idx to include in the "signal" region.

    Returns
    -------
    dict mapping feature name → float value.
    """
    N = len(cir_mag)
    fp = int(max(0, min(fp_idx, N - 1)))

    # ---- noise floor from [fp-50 … fp-10] ----
    ns, ne = max(0, fp - 50), max(1, fp - 10)
    noise = cir_mag[ns:ne]
    mu_n = float(np.mean(noise)) if len(noise) > 0 else 0.0
    std_n = float(np.std(noise))  if len(noise) > 0 else 1e-12
    noise_pow_mean = float(np.mean(noise ** 2)) if len(noise) > 0 else 1e-12

    # ---- signal window [fp-3 … fp+window) ----
    ss = max(0, fp - 3)
    se = min(N, fp + signal_window)
    signal = cir_mag[ss:se]

    # ---- First-Path Power  (3 consecutive taps fp, fp+1, fp+2) ----
    fp3 = cir_mag[fp : min(N, fp + 3)]
    fpp_lin = float(np.sum(fp3 ** 2)) if len(fp3) > 0 else 1e-12
    FPP_dB = 10.0 * np.log10(max(fpp_lin, 1e-12))
    FP_SNR_dB = 10.0 * np.log10(max(fpp_lin, 1e-12) / max(3.0 * noise_pow_mean, 1e-12))

    # ---- Total Received Power over signal window ----
    rxp_lin = float(np.sum(signal ** 2)) if len(signal) > 0 else 1e-12
    RXP_dB = 10.0 * np.log10(max(rxp_lin, 1e-12))

    # ---- APS006 metric ----
    RXP_minus_FPP_dB = RXP_dB - FPP_dB

    # ---- Peak-to-First-Path Ratio ----
    if len(signal) > 0:
        pk_local = int(np.argmax(signal))
        pk_global = ss + pk_local
    else:
        pk_global = fp
    pk_amp  = float(cir_mag[pk_global]) if pk_global < N else 1e-12
    fp_amp  = float(cir_mag[fp])        if fp < N       else 1e-12
    PPR_dB  = 20.0 * np.log10(max(pk_amp, 1e-12) / max(fp_amp, 1e-12))

    # ---- Rise time  (peak – first path) ----
    rise_time_ns = float(pk_global - fp) * TAP_NS

    # ---- Mean-excess delay, RMS delay spread, kurtosis, skewness ----
    threshold = mu_n + 3.0 * std_n
    above = signal > threshold
    if np.sum(above) >= 3:
        t_ns     = np.arange(len(signal), dtype=float) * TAP_NS
        amp_cl   = signal[above]
        pow_cl   = amp_cl ** 2
        t_cl     = t_ns[above]
        t_fp_ns  = float(fp - ss) * TAP_NS
        total_p  = float(np.sum(pow_cl))

        if total_p > 1e-18:
            med = float(np.sum((t_cl - t_fp_ns) * pow_cl) / total_p)
            rds = float(np.sqrt(
                np.sum(pow_cl * (t_cl - t_fp_ns - med) ** 2) / total_p
            ))
        else:
            med = rds = 0.0
        kurt     = float(_sp_kurtosis(amp_cl, fisher=True))
        skewness = float(_sp_skew(amp_cl))
    else:
        med = rds = kurt = skewness = 0.0

    # ---- Energy ratio  (FP 5-tap energy / total) ----
    fp_e5 = float(np.sum(cir_mag[max(0, fp - 2) : min(N, fp + 3)] ** 2))
    energy_ratio = fp_e5 / max(rxp_lin, 1e-12)

    return {
        "FPP_dB":               FPP_dB,
        "RXP_dB":               RXP_dB,
        "RXP_minus_FPP_dB":     RXP_minus_FPP_dB,
        "FP_SNR_dB":            FP_SNR_dB,
        "PPR_dB":               PPR_dB,
        "rise_time_ns":         rise_time_ns,
        "mean_excess_delay_ns": med,
        "rms_delay_spread_ns":  rds,
        "kurtosis":             kurt,
        "skewness":             skewness,
        "energy_ratio":         energy_ratio,
    }


# Ordered list of CIR-derived feature names (stable across calls)
FEATURE_NAMES: list[str] = list(
    compute_features_from_cir(np.ones(100), 50).keys()
)


def compute_feature_matrix(
    cir_mags: np.ndarray,
    fp_indices: np.ndarray,
    signal_window: int = 150,
) -> np.ndarray:
    """Compute recomputed CIR feature matrix for a batch of waveforms."""
    rows = []
    for i in range(len(cir_mags)):
        d = compute_features_from_cir(cir_mags[i], int(fp_indices[i]),
                                      signal_window=signal_window)
        rows.append([d[k] for k in FEATURE_NAMES])
    return np.array(rows, dtype=np.float32)
