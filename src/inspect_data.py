"""Inspect all files under ./data/ and print schema, shapes, dtypes, and sample rows.

Run this before writing any dataset-specific loaders so every column name,
file layout, and data type is verified directly from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np
import pandas as pd


DATA_DIR = Path("data")


def inspect_csv(path: Path, max_rows: int = 3) -> None:
    """Print CSV shape, column names, dtypes, and sample rows."""
    df = pd.read_csv(path, nrows=500)
    # Count total lines cheaply without loading huge files twice if needed.
    with open(path, "rb") as f:
        n_lines = sum(1 for _ in f)
    print(f"  [CSV] {path} | total_lines~={n_lines} | sample_cols={len(df.columns)}")
    print("  Columns & dtypes:")
    for col in df.columns:
        print(f"    - {col}: {df[col].dtype} (non-null in first 500: {df[col].notna().sum()})")
    print("  Sample rows (truncated values > 80 chars):")
    preview = df.head(max_rows).copy()
    for col in preview.columns:
        if preview[col].dtype == object:
            preview[col] = preview[col].astype(str).str.slice(0, 80)
    print(preview.to_string(index=False))


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


def inspect_jsonl_txt(path: Path, max_records: int = 3) -> None:
    """Inspect line-delimited JSON or delimited text UWB capture files."""
    records = []
    raw_lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            if idx < 5:
                raw_lines.append(s)
            rec = _repair_json_line(s)
            if rec is not None and len(records) < max_records:
                records.append(rec)
            if idx >= 50:
                break

    with open(path, "rb") as f:
        n_lines = sum(1 for _ in f)

    print(f"  [TXT] {path} | lines={n_lines}")
    if records:
        print("  JSON-lines detected. Keys & types from first record:")
        for k, v in records[0].items():
            if isinstance(v, list):
                print(f"    - {k}: list(len={len(v)}, elem_type={type(v[0]).__name__ if v else 'empty'}) sample[:5]={v[:5]}")
            else:
                print(f"    - {k}: {type(v).__name__} = {v!r}")
        if len(records) > 1:
            print(f"  Second record summary keys={list(records[1].keys())}, scalar fields="
                  f"{ {k: v for k, v in records[1].items() if not isinstance(v, list)} }")
    else:
        print("  Non-JSON text lines (first 3):")
        for s in raw_lines[:3]:
            print(f"    {s[:160]}")


def inspect_hdf5(path: Path) -> None:
    """Print HDF5 datasets, shapes, dtypes, and sample entries."""
    print(f"  [HDF5] {path}")
    try:
        with h5py.File(path, "r") as f:
            def _visit(name: str, obj: h5py.HLObject) -> None:
                if isinstance(obj, h5py.Dataset):
                    print(f"    - /{name}: shape={obj.shape}, dtype={obj.dtype}")
                    if obj.ndim >= 1 and obj.shape[0] > 0:
                        sample = obj[0]
                        if isinstance(sample, np.ndarray) and sample.size > 8:
                            print(f"      sample[0][:8]={sample.flat[:8]}")
                        else:
                            print(f"      sample[0]={sample!r}")
            f.visititems(_visit)
    except OSError as exc:
        print(f"    ⚠ Cannot open (possibly truncated/downloading): {exc}")


def inspect_numpy(path: Path) -> None:
    """Print .npy / .npz shapes, dtypes, and sample values."""
    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        print(f"  [NPZ] {path} | keys={list(arr.keys())}")
        for k in arr.keys():
            v = arr[k]
            print(f"    - {k}: shape={v.shape}, dtype={v.dtype}, sample={v.flat[:5] if v.size else 'empty'}")
    else:
        print(f"  [NPY] {path} | shape={arr.shape}, dtype={arr.dtype}, sample={arr.flat[:5] if arr.size else 'empty'}")


def main() -> int:
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR.resolve()} does not exist.", file=sys.stderr)
        return 1

    print(f"=== Inspecting directory tree under {DATA_DIR.resolve()} ===")
    all_files = sorted(p for p in DATA_DIR.rglob("*") if p.is_file())
    if not all_files:
        print("No files found under ./data/.")
        return 1

    # Group by parent directory to avoid spamming thousands of identical capture files.
    by_dir: dict[Path, list[Path]] = {}
    for p in all_files:
        by_dir.setdefault(p.parent, []).append(p)

    for folder, files in by_dir.items():
        exts = sorted({f.suffix.lower() for f in files})
        total_mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"\nFolder: {folder} ({len(files)} files, {total_mb:.2f} MB, exts={exts})")

        # Inspect at most 2 representative files per extension in each folder.
        seen_ext_count: dict[str, int] = {}
        for f in files:
            ext = f.suffix.lower()
            if ext in {".zip", ".png", ".jpg", ".pdf", ".docx"}:
                if seen_ext_count.get(ext, 0) == 0:
                    print(f"  [{ext.upper().lstrip('.')}] {f} ({f.stat().st_size / 1e6:.2f} MB)")
                seen_ext_count[ext] = seen_ext_count.get(ext, 0) + 1
                continue

            if seen_ext_count.get(ext, 0) >= 2:
                continue
            seen_ext_count[ext] = seen_ext_count.get(ext, 0) + 1

            if ext == ".csv":
                inspect_csv(f)
            elif ext in {".txt", ".json", ".jsonl"}:
                inspect_jsonl_txt(f)
            elif ext in {".h5", ".hdf5"}:
                inspect_hdf5(f)
            elif ext in {".npy", ".npz"}:
                inspect_numpy(f)
            else:
                print(f"  [OTHER] {f} ({f.stat().st_size} bytes)")
                with open(f, "rb") as fh:
                    print(f"    head(120)={fh.read(120)!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

