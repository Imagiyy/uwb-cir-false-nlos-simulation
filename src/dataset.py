"""dataset.py — Load Fontaine & Santamaria UWB datasets; grouped splits.

Data sources & verified schemas
-------------------------------
- **Fontaine features_IIoT_19.csv** (17 160 rows): scalar CIR features +
  LOS/NLOS label + ranging error (mm).  NOTE: features_IIoT_19.csv stores
  ``error = distance_GT - estimated_range`` (negative in NLOS), whereas
  meta_IIoT_19.csv stores ``estimated_range - actual_range`` (positive in NLOS).
  We compute ``error_mm = estimated_range - distance_GT`` so regressors predict
  the physical signed bias ``b = d_measured - d_true`` (>0 in NLOS).
- **Fontaine meta_IIoT_19.csv**: ``location_ID`` (14 locations: 10..23),
  ``anchorNumber``, ``loc_tag`` (248 anchor-tag links) for anti-leakage grouping.
- **Fontaine CIRsV2.hdf5**:
  - ``/IIoT19``: shape (17160, 150), dtype float64 (magnitude-only, already
    divided by RXPACC). Pre-sliced around first path: ``floor(fpindex/64)`` is
    at tap 50 and ``FP_AMPL1..3`` align at taps ``[51, 52, 53]`` (r=0.990 with
    CSV ``FP_power``, r=0.981 with CSV ``Pd``), so local ``fp_idx = 51``.
  - ``/IIoT20``: shape (3925, 150), dtype complex128 (all LOS, aligns at tap 51).
- **Santamaria DW3000**: Real DW3000 complex I/Q 1016-tap CIR captures (lab,
  corridor, calibration_anc0, calibration_anchors). Used ONLY for DW3000
  complex-IQ NLOS classification & APS006 testing — NEVER ranging regression.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

logger = logging.getLogger(__name__)

DATA_DIR       = Path("data")
FONTAINE_DIR   = DATA_DIR / "fontaine"
SANTAMARIA_DIR = DATA_DIR / "santamaria_dw3000" / "data"

SEED = 42

# CIR window for CNN input: [fp−10 … fp+147)  → 157 taps
CNN_WIN_PRE  = 10
CNN_WIN_POST = 147
CNN_WIN_LEN  = CNN_WIN_PRE + CNN_WIN_POST  # 157

# Verified first-path tap index inside pre-sliced 150-tap CIRsV2 waveforms
CIRV2_LOCAL_FP_IDX = 51


# ==================================================================
# Fontaine scalar features
# ==================================================================

def load_fontaine_scalar() -> pd.DataFrame:
    """Load features_IIoT_19.csv merged with meta_IIoT_19.csv.

    Returns DataFrame with columns:
      fp_ampl1 … Pr, error_mm (= d_measured - d_true), label (0=LOS, 1=NLOS),
      location_ID, anchorNumber, loc_tag, fpi (full-CIR tap index).
    """
    df_feat = pd.read_csv(FONTAINE_DIR / "features_IIoT_19.csv")
    df_meta = pd.read_csv(FONTAINE_DIR / "meta_IIoT_19.csv")
    assert len(df_feat) == len(df_meta), "Row-count mismatch features vs meta"

    df = df_feat.copy()
    df["location_ID"]  = df_meta["location_ID"].values
    df["anchorNumber"] = df_meta["anchorNumber"].values
    df["loc_tag"]      = df_meta["loc_tag"].values
    df["fpi"]          = (df["fpindex"] / 64.0).round().astype(int)
    # Signed bias b = d_measured - d_true (positive in NLOS; negates CSV 'error')
    df["error_mm"]     = (df["estimated_range"] - df["distance_GT"]).astype(float)

    logger.info(
        "Fontaine scalar: %d rows, LOS=%d, NLOS=%d, locs=%s | "
        "error_mm=d_measured-d_true (LOS med=%.1f mm, NLOS med=%.1f mm)",
        len(df), (df.label == 0).sum(), (df.label == 1).sum(),
        sorted(df.location_ID.unique()),
        float(np.median(df.loc[df.label == 0, "error_mm"])),
        float(np.median(df.loc[df.label == 1, "error_mm"])),
    )
    return df


# ==================================================================
# Fontaine CIR waveforms (CIRsV2.hdf5)
# ==================================================================

def load_fontaine_cirs(
    key: str = "IIoT19",
    n_expected: int | None = 17160,
) -> tuple[np.ndarray, np.ndarray, bool] | tuple[None, None, bool]:
    """Load CIRsV2.hdf5 waveforms preserving real vs complex dtype.

    Returns
    -------
    cirs       : (N, 150) float64 or complex128 array, or None if unavailable.
    fp_indices : (N,) int array of local first-path tap indices (51 for CIRsV2).
    is_complex : True if CIRs carry complex I/Q, False if magnitude-only.
    """
    p = FONTAINE_DIR / "CIRsV2.hdf5"
    if not p.exists():
        logger.warning("CIRsV2.hdf5 not found — CIR/CNN models will be skipped.")
        return None, None, False
    try:
        with h5py.File(p, "r") as f:
            ds_key = key if key in f else list(f.keys())[0]
            raw = np.array(f[ds_key])
            is_complex = bool(np.iscomplexobj(raw))
            cirs = raw.astype(np.complex128 if is_complex else np.float64)
            fp_indices = np.full(len(cirs), CIRV2_LOCAL_FP_IDX, dtype=int)
            logger.info(
                "CIRs loaded from CIRsV2.hdf5/%s: shape=%s dtype=%s complex=%s local_fp=%d",
                ds_key, cirs.shape, cirs.dtype, is_complex, CIRV2_LOCAL_FP_IDX,
            )
            if n_expected is not None and cirs.shape[0] != n_expected:
                logger.warning("CIR count %d != expected %d", cirs.shape[0], n_expected)
            return cirs, fp_indices, is_complex
    except OSError as exc:
        logger.warning("Cannot open CIRsV2.hdf5 (truncated?): %s", exc)
        return None, None, False


def load_fontaine_iiot20() -> tuple[pd.DataFrame | None, np.ndarray | None, np.ndarray | None]:
    """Load all-LOS complex128 IIoT20 CIRs and metadata for phase-averaging tests."""
    p_meta = FONTAINE_DIR / "meta_IIoT_20.csv"
    cirs, fp_idx, is_cx = load_fontaine_cirs(key="IIoT20", n_expected=3925)
    if cirs is None or not p_meta.exists():
        return None, None, None
    df20 = pd.read_csv(p_meta)
    assert len(df20) == len(cirs), "IIoT20 meta vs CIR count mismatch"
    assert is_cx, "Expected IIoT20 to be complex128"
    return df20, cirs, fp_idx


# ==================================================================
# Fontaine exportloc9IQ  (complex CIR fallback)
# ==================================================================

def load_fontaine_loc9():
    """Load exportloc9IQ.csv. Returns (df, cirs) or (None, None)."""
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
    df["error_mm"] = (df["estimated_range"] - df["actual_range"]).astype(float)
    logger.info("exportloc9IQ: %d rows, CIR shape %s", len(df), cirs.shape)
    return df, cirs


# ==================================================================
# Santamaria DW3000  (complex I/Q; NO reliable ranging error)
# ==================================================================

def _repair_json_line(line: str) -> dict | None:
    """Parse one JSON-lines record with Santamaria CIR_I bracket repair fallback."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r"(-?\d+)\}$", r"\1]}", line)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def load_santamaria_dw3000(scenario: str = "lab") -> list[dict]:
    """Load raw DW3000 CIR records for a given scenario.

    Valid scenarios: 'lab', 'corridor', 'calibration_anc0', 'calibration_anchors'.
    NOTE: Per Zenodo 22690295 notes, ranging timestamps suffer from negative
    outliers; timestamp/range fields are intentionally omitted from training.
    """
    folder = SANTAMARIA_DIR / "dw3000" / scenario
    if not folder.exists():
        logger.warning("Santamaria folder missing: %s", folder)
        return []

    records: list[dict] = []
    for f in sorted(folder.glob("*.txt")):
        m = re.search(r"id(\d+)", f.stem)
        pid = int(m.group(1)) if m else None
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                r = _repair_json_line(s)
                if r is None:
                    continue
                cr, ci = r.get("CIR_R"), r.get("CIR_I")
                if not (cr and ci and len(cr) == 1016 and len(ci) == 1016
                        and r.get("fp_ind") is not None):
                    continue
                h   = np.array(cr, dtype=float) + 1j * np.array(ci, dtype=float)
                fpi = int(round(r["fp_ind"] / 64.0))
                n_pac = max(float(r.get("N") or 1.0), 1.0)
                records.append(dict(
                    cir_complex=h,
                    cir_mag=np.abs(h),
                    cir_complex_pac=h / n_pac,
                    cir_mag_pac=np.abs(h) / n_pac,
                    fpi=fpi,
                    fp_ind_raw=r["fp_ind"],
                    Pk_ind=r.get("Pk_ind"),
                    fp_pwr=r.get("fp_pwr"),
                    rx_pwr=r.get("rx_pwr"),
                    stdN=r.get("stdN"),
                    N=n_pac,
                    Anc=r.get("Anc"),
                    punto_id=pid,
                    archivo=f.name,
                ))
    logger.info("Santamaria DW3000/%s: %d valid 1016-tap complex CIRs", scenario, len(records))
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
    strategy: str = "groupkfold",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) with zero group leakage.

    When ``group_col == 'location_ID'`` and ``strategy == 'groupkfold'``,
    we use deterministic 5-fold GroupKFold folds so the held-out test set
    contains a representative mix of LOS and NLOS links (locs [16, 18]:
    907 LOS / 1967 NLOS; val = locs [10, 11, 17]; train = remaining 9 locs),
    avoiding the anomalous low-error test draw of random GroupShuffleSplit(42)
    on only 14 discrete locations.
    """
    groups = df[group_col].values

    if strategy == "groupkfold" and pd.Series(groups).nunique() >= 5:
        gkf = GroupKFold(n_splits=5)
        folds = [te for _, te in gkf.split(df, groups=groups)]
        te_idx = folds[0]
        va_idx = folds[1]
        tr_idx = np.concatenate([folds[2], folds[3], folds[4]])
    else:
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
    """Return a fixed-size 157-tap window [fp−pre … fp+post) zero-padded."""
    N = len(cir_mag)
    win = np.zeros(pre + post, dtype=np.float32)
    ss = max(0, fp_idx - pre)
    se = min(N, fp_idx + post)
    ds = max(0, pre - fp_idx)
    de = ds + (se - ss)
    if se > ss:
        win[ds:de] = cir_mag[ss:se]
    return win
