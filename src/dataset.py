"""dataset.py — Load Fontaine & Santamaria UWB datasets; grouped splits.

Data sources
------------
- **Fontaine features_IIoT_19.csv** (17 160 rows): scalar CIR features +
  LOS/NLOS label + ranging error (mm).  Columns include Pd = RX−FP (dB).
- **Fontaine meta_IIoT_19.csv**: location_ID for anti-leakage grouping.
- **Fontaine CIRsV2.hdf5**: Complex-128 CIR waveforms keyed "IIoT19".
- **Fontaine exportloc9IQ.csv** (1 102 rows): complex CIR + scalar feats
  (no LOS/NLOS label — dataset 9 is unlabelled).
- **Santamaria DW3000**: Complex I/Q raw CIR captures (lab, corridor,
  calibration).  Used for DW3000 APS006 testing & CIR overlay plots only.
  NEVER for ranging-error regression (known DW3000 timestamp bug).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

DATA_DIR       = Path("data")
FONTAINE_DIR   = DATA_DIR / "fontaine"
SANTAMARIA_DIR = DATA_DIR / "santamaria_dw3000" / "data"

SEED = 42

# CIR window for CNN input: [fp−10 … fp+147)  → 157 taps
CNN_WIN_PRE  = 10
CNN_WIN_POST = 147
CNN_WIN_LEN  = CNN_WIN_PRE + CNN_WIN_POST  # 157


# ==================================================================
# Fontaine scalar features (always available)
# ==================================================================

def load_fontaine_scalar() -> pd.DataFrame:
    """Load features_IIoT_19.csv merged with meta_IIoT_19.csv.

    Returns DataFrame with columns:
      fp_ampl1 … Pr, error_mm, label (0=LOS,1=NLOS), location_ID,
      anchorNumber, loc_tag, fpi (fp tap index).
    """
    df_feat = pd.read_csv(FONTAINE_DIR / "features_IIoT_19.csv")
    df_meta = pd.read_csv(FONTAINE_DIR / "meta_IIoT_19.csv")
    assert len(df_feat) == len(df_meta), "Row-count mismatch features vs meta"

    df = df_feat.copy()
    df["location_ID"]  = df_meta["location_ID"].values
    df["anchorNumber"]  = df_meta["anchorNumber"].values
    df["loc_tag"]       = df_meta["loc_tag"].values
    df["fpi"]           = (df["fpindex"] / 64.0).round().astype(int)
    df["error_mm"]      = df["error"].astype(float)

    logger.info("Fontaine scalar: %d rows, LOS=%d, NLOS=%d, locs=%s",
                len(df), (df.label==0).sum(), (df.label==1).sum(),
                sorted(df.location_ID.unique()))
    return df


# ==================================================================
# Fontaine CIR waveforms (CIRsV2.hdf5)
# ==================================================================

def load_fontaine_cirs(n_expected: int = 17160):
    """Return complex-128 CIR array (N,1016) or None if unavailable."""
    p = FONTAINE_DIR / "CIRsV2.hdf5"
    if not p.exists():
        logger.warning("CIRsV2.hdf5 not found — CNN models will be skipped.")
        return None
    try:
        with h5py.File(p, "r") as f:
            key = "IIoT19" if "IIoT19" in f else list(f.keys())[0]
            cirs = np.array(f[key], dtype=np.complex128)
            logger.info("CIRs loaded: key=%s shape=%s", key, cirs.shape)
            if cirs.shape[0] != n_expected:
                logger.warning("CIR count %d ≠ expected %d", cirs.shape[0], n_expected)
            return cirs
    except OSError as exc:
        logger.warning("Cannot open CIRsV2.hdf5 (truncated?): %s", exc)
        return None


# ==================================================================
# Fontaine exportloc9IQ  (complex CIR fallback)
# ==================================================================

def load_fontaine_loc9():
    """Load exportloc9IQ.csv.  Returns (df, cirs) or (None, None)."""
    p = FONTAINE_DIR / "exportloc9IQ.csv"
    if not p.exists():
        return None, None

    df = pd.read_csv(p)
    cir_cols = sorted(
        [c for c in df.columns if c.startswith("CIR") and c[3:].isdigit()],
        key=lambda c: int(c[3:]),
    )
    cirs = np.array(
        [[complex(str(v).replace(" ", "")) for v in row] for row in df[cir_cols].values],
        dtype=np.complex128,
    )
    df["fpi"]      = (df["fpindex"] / 64.0).round().astype(int)
    df["error_mm"] = df["error"].astype(float)
    logger.info("exportloc9IQ: %d rows, CIR shape %s", len(df), cirs.shape)
    return df, cirs


# ==================================================================
# Santamaria DW3000  (complex I/Q; NO reliable ranging error)
# ==================================================================

def load_santamaria_dw3000(scenario: str = "lab") -> list[dict]:
    """Load raw DW3000 CIR records for a given scenario.

    Each record dict has keys: cir_complex, cir_mag, fpi, fp_ind_raw,
    Pk_ind, fp_pwr, rx_pwr, stdN, N, Anc, punto_id, archivo.
    """
    folder = SANTAMARIA_DIR / "dw3000" / scenario
    if not folder.exists():
        logger.warning("Santamaria folder missing: %s", folder)
        return []

    records: list[dict] = []
    for f in sorted(folder.glob("*.txt")):
        m = re.search(r"id(\d+)", f.stem)
        pid = int(m.group(1)) if m else None
        for line in open(f, errors="ignore"):
            try:
                r = json.loads(line.strip())
            except (json.JSONDecodeError, ValueError):
                continue
            cr, ci = r.get("CIR_R"), r.get("CIR_I")
            if not (cr and ci and len(cr) == 1016 and len(ci) == 1016
                    and r.get("fp_ind") is not None):
                continue
            h   = np.array(cr, dtype=float) + 1j * np.array(ci, dtype=float)
            fpi = int(round(r["fp_ind"] / 64.0))
            records.append(dict(
                cir_complex=h, cir_mag=np.abs(h), fpi=fpi,
                fp_ind_raw=r["fp_ind"], Pk_ind=r.get("Pk_ind"),
                fp_pwr=r.get("fp_pwr"), rx_pwr=r.get("rx_pwr"),
                stdN=r.get("stdN"), N=r.get("N"), Anc=r.get("Anc"),
                punto_id=pid, archivo=f.name,
            ))
    logger.info("Santamaria DW3000/%s: %d valid CIRs", scenario, len(records))
    return records


# ==================================================================
# Anti-leakage grouped split
# ==================================================================

def grouped_split(
    df: pd.DataFrame,
    group_col: str = "location_ID",
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) with no group leakage."""
    groups = df[group_col].values
    test_frac = 1.0 - train_frac - val_frac

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tv_idx, te_idx = next(gss1.split(df, groups=groups))

    rel_val = val_frac / (train_frac + val_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    tr_rel, va_rel = next(gss2.split(df.iloc[tv_idx], groups=groups[tv_idx]))

    tr_idx = tv_idx[tr_rel]
    va_idx = tv_idx[va_rel]

    for name, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        logger.info("  %s: %d samples, groups=%s", name, len(idx),
                    sorted(set(groups[idx])))
    return tr_idx, va_idx, te_idx


# ==================================================================
# CIR window extraction for CNNs
# ==================================================================

def extract_cir_window(
    cir_mag: np.ndarray,
    fp_idx: int,
    pre: int = CNN_WIN_PRE,
    post: int = CNN_WIN_POST,
) -> np.ndarray:
    """Return a fixed-size window [fp−pre … fp+post) zero-padded."""
    N = len(cir_mag)
    win = np.zeros(pre + post, dtype=np.float32)
    ss = max(0, fp_idx - pre)
    se = min(N, fp_idx + post)
    ds = max(0, pre - fp_idx)
    de = ds + (se - ss)
    if se > ss:
        win[ds:de] = cir_mag[ss:se]
    return win

