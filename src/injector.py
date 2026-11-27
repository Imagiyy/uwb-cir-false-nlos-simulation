"""injector.py — Physics-accurate Phantom-NLOS injector for UWB CIRs.

Implements modular injection of a delayed, amplitude-scaled copy of the
LOS direct-path pulse into Channel Impulse Response waveforms.

Key physics constraints enforced
---------------------------------
1. **Untouched First Path:** the original LOS peak at fp_index is never
   attenuated, shifted, or duplicated.
2. **Amplitude vs. Power dB:**  +X dB *power* boost  →  linear amplitude
   scale = 10^(X / 20).  E.g. +6 dB → ×1.9953.
3. **Tap period:**  1 CIR tap ≈ 1.0016 ns  (499.2 MHz chip rate).
4. **Pulse template:** extracted from [fp−3 : fp+12] after noise-floor
   subtraction (magnitude) or directly (complex I/Q).
5. **Complex injection:** adds ``scale · exp(jφ) · pulse_shifted`` with
   controllable or random phase φ.
6. **Magnitude injection:** documented assumption that complex tail
   interference is negligible at Δt ≥ 10 ns (UWB pulse ≈ 2 ns wide at
   −3 dB for a 500 MHz bandwidth).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

TAP_NS = 1.0016  # ns per CIR tap

logger = logging.getLogger(__name__)
_LOGGED_MAGNITUDE_ASSUMPTION = False


# ------------------------------------------------------------------
# Pulse-template extraction
# ------------------------------------------------------------------

def extract_pulse_template(
    cir: np.ndarray,
    fp_idx: int,
    is_complex: bool = False,
) -> tuple[np.ndarray, float]:
    """Extract the clean LOS direct-path pulse from a CIR.

    Returns
    -------
    pulse : ndarray  (15 taps, [fp−3 … fp+12))
    mu_n  : float    mean magnitude of the pre-first-path noise floor.
    """
    N   = len(cir)
    fp  = int(max(3, min(fp_idx, N - 12)))
    ns  = max(0, fp - 50)
    ne  = max(ns + 1, fp - 10)

    if is_complex:
        mu_n  = float(np.mean(np.abs(cir[ns:ne])))
        pulse = cir[fp - 3 : fp + 12].copy()
    else:
        mu_n  = float(np.mean(cir[ns:ne]))
        pulse = np.maximum(0.0, cir[fp - 3 : fp + 12].astype(float) - mu_n)

    return pulse, mu_n


# ------------------------------------------------------------------
# Single-shot injection
# ------------------------------------------------------------------

def inject_phantom_nlos(
    cir: np.ndarray,
    fp_idx: int,
    delta_t_ns: float,
    power_db: float,
    *,
    is_complex: bool = False,
    phi: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Inject a delayed copy of the LOS pulse into the CIR.

    Parameters
    ----------
    cir        : Original CIR (complex or magnitude).  NOT modified.
    fp_idx     : First-path tap index.
    delta_t_ns : Delay (ns) of injected pulse relative to first path.
    power_db   : Power boost (dB) relative to LOS first-path peak.
    is_complex : True ⇒ cir is complex I/Q.
    phi        : Phase (rad) for complex injection; None → random.
    seed       : RNG seed for reproducible random phase.

    Returns
    -------
    cir_atk : Modified CIR (same dtype as input).
    """
    global _LOGGED_MAGNITUDE_ASSUMPTION
    if not is_complex and not _LOGGED_MAGNITUDE_ASSUMPTION:
        logger.info(
            "Magnitude-only CIR injection: complex tail interference is negligible "
            "at Δt >= 10 ns since a 500 MHz UWB pulse is ~2 ns wide."
        )
        _LOGGED_MAGNITUDE_ASSUMPTION = True

    N          = len(cir)
    delta_taps = round(delta_t_ns / TAP_NS)
    scale      = 10.0 ** (power_db / 20.0)

    pulse, _ = extract_pulse_template(cir, fp_idx, is_complex=is_complex)

    cir_atk = cir.copy()

    # Where the injection starts/ends
    inj_s = fp_idx - 3 + delta_taps
    inj_e = inj_s + len(pulse)

    # Clip to valid range
    if inj_s < 0 or inj_e > N:
        src_s = max(0, -inj_s)
        src_e = min(len(pulse), N - inj_s)
        inj_s = max(0, inj_s)
        inj_e = min(N, inj_e)
        pulse = pulse[src_s:src_e]

    if len(pulse) == 0:
        return cir_atk

    # Build injection signal
    if is_complex:
        if phi is None:
            rng = np.random.RandomState(seed)
            phi = rng.uniform(0, 2 * np.pi)
        injection = scale * np.exp(1j * phi) * pulse
    else:
        injection = scale * pulse

    cir_atk[inj_s:inj_e] += injection
    return cir_atk


# ------------------------------------------------------------------
# Phase-averaged injection (complex CIR only)
# ------------------------------------------------------------------

def inject_with_phase_averaging(
    cir: np.ndarray,
    fp_idx: int,
    delta_t_ns: float,
    power_db: float,
    n_phase_draws: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Inject with multiple random phases and return magnitudes across draws.

    Returns
    -------
    mag_avg   : Mean |CIR_atk| across phase draws.
    mag_worst : Worst-case constructive (φ = 0) magnitude.
    mags_all  : List of individual |CIR_atk| for each draw.
    """
    rng    = np.random.RandomState(seed)
    phases = rng.uniform(0, 2 * np.pi, size=n_phase_draws)

    mags = [
        np.abs(
            inject_phantom_nlos(
                cir, fp_idx, delta_t_ns, power_db, is_complex=True, phi=float(ph)
            )
        )
        for ph in phases
    ]

    mag_worst = np.abs(
        inject_phantom_nlos(
            cir, fp_idx, delta_t_ns, power_db, is_complex=True, phi=0.0
        )
    )

    mag_avg = np.mean(mags, axis=0)
    return mag_avg, mag_worst, mags


# ------------------------------------------------------------------
# Diffuse multi-tap injection (K closely spaced decaying paths)
# ------------------------------------------------------------------

def diffuse_amplitude_weights(
    offsets_ns: tuple[float, ...] = (0.0, 4.0, 9.0),
    decay_ns: float = 5.0,
) -> np.ndarray:
    """Compute normalized per-tap amplitude multipliers sqrt(p_k), sum(p_k)=1."""
    offs = np.asarray(offsets_ns, dtype=float)
    pow_w = np.exp(-offs / max(decay_ns, 1e-6))
    pow_w /= np.sum(pow_w)
    return np.sqrt(pow_w)


def inject_diffuse_nlos(
    cir: np.ndarray,
    fp_idx: int,
    delta_t_ns: float,
    total_power_db: float,
    *,
    offsets_ns: tuple[float, ...] = (0.0, 4.0, 9.0),
    decay_ns: float = 5.0,
    is_complex: bool = False,
    phis: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Inject K=3 decaying delayed copies whose summed power equals total_power_db.

    Total linear amplitude scale A = 10^(total_power_db / 20); each component k
    uses amplitude scale a_k = sqrt(p_k) * A so sum(a_k^2) = 10^(total_power_db / 10).
    """
    global _LOGGED_MAGNITUDE_ASSUMPTION
    if not is_complex and not _LOGGED_MAGNITUDE_ASSUMPTION:
        logger.info(
            "Magnitude-only CIR injection: complex tail interference is negligible "
            "at Δt >= 10 ns since a 500 MHz UWB pulse is ~2 ns wide."
        )
        _LOGGED_MAGNITUDE_ASSUMPTION = True

    N = len(cir)
    amp_w = diffuse_amplitude_weights(offsets_ns, decay_ns=decay_ns)
    total_scale = 10.0 ** (total_power_db / 20.0)
    pulse, _ = extract_pulse_template(cir, fp_idx, is_complex=is_complex)

    if is_complex and phis is None:
        rng = np.random.RandomState(seed)
        phis = rng.uniform(0.0, 2.0 * np.pi, size=len(offsets_ns))

    cir_atk = cir.copy()
    for k, off_ns in enumerate(offsets_ns):
        dt_k = delta_t_ns + off_ns
        delta_taps = round(dt_k / TAP_NS)
        scale_k = total_scale * float(amp_w[k])

        inj_s = fp_idx - 3 + delta_taps
        inj_e = inj_s + len(pulse)
        p_slice = pulse
        if inj_s < 0 or inj_e > N:
            src_s = max(0, -inj_s)
            src_e = min(len(pulse), N - inj_s)
            inj_s = max(0, inj_s)
            inj_e = min(N, inj_e)
            p_slice = pulse[src_s:src_e]
        if len(p_slice) == 0:
            continue

        if is_complex:
            inj = scale_k * np.exp(1j * float(phis[k])) * p_slice
        else:
            inj = scale_k * p_slice
        cir_atk[inj_s:inj_e] += inj

    return cir_atk


def inject_diffuse_with_phase_averaging(
    cir: np.ndarray,
    fp_idx: int,
    delta_t_ns: float,
    total_power_db: float,
    n_phase_draws: int = 5,
    seed: int = 42,
    offsets_ns: tuple[float, ...] = (0.0, 4.0, 9.0),
    decay_ns: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Complex diffuse injection across independent random phase tuples and phi=0."""
    rng = np.random.RandomState(seed)
    k_paths = len(offsets_ns)
    mags = []
    for _ in range(n_phase_draws):
        phis = rng.uniform(0.0, 2.0 * np.pi, size=k_paths)
        mags.append(np.abs(
            inject_diffuse_nlos(
                cir, fp_idx, delta_t_ns, total_power_db,
                offsets_ns=offsets_ns, decay_ns=decay_ns,
                is_complex=True, phis=phis,
            )
        ))
    mag_worst = np.abs(
        inject_diffuse_nlos(
            cir, fp_idx, delta_t_ns, total_power_db,
            offsets_ns=offsets_ns, decay_ns=decay_ns,
            is_complex=True, phis=np.zeros(k_paths),
        )
    )
    return np.mean(mags, axis=0), mag_worst, mags
