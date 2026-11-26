"""injector.py — Physics-accurate Phantom-NLOS injector for UWB CIRs.

Implements modular injection of a delayed, amplitude-scaled copy of the
LOS direct-path pulse into Channel Impulse Response waveforms.

Key physics constraints enforced
---------------------------------
1. **Untouched First Path:** the original LOS peak at fp_index is never
   attenuated, shifted, or duplicated.
2. **Amplitude vs. Power dB:**  +X dB *power* boost  →  linear amplitude
   scale = 10^(X / 20).  E.g. +6 dB → ×1.995.
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
    """Inject with multiple random phases and return averaged magnitude.

    Returns
    -------
    mag_avg   : Mean |CIR_atk| across phase draws.
    mag_worst : Worst-case constructive (φ = 0) magnitude.
    mags_all  : List of individual |CIR_atk| for each draw.
    """
    rng    = np.random.RandomState(seed)
    phases = rng.uniform(0, 2 * np.pi, size=n_phase_draws)

    mags = []
    for ph in phases:
        mags.append(np.abs(
            inject_phantom_nlos(cir, fp_idx, delta_t_ns, power_db,
                                is_complex=True, phi=ph)
        ))

    mag_worst = np.abs(
        inject_phantom_nlos(cir, fp_idx, delta_t_ns, power_db,
                            is_complex=True, phi=0.0)
    )

    mag_avg = np.mean(mags, axis=0)
    return mag_avg, mag_worst, mags

