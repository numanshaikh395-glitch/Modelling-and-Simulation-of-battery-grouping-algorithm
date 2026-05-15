"""
data_loader.py
Fast parallel loader — reads static params and voltage curves from CSV files.
Uses threads for I/O so 200k files load in minutes instead of hours.
"""

import os
import csv
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import config


def _read_static(fp):
    """Read only the header + first data row (512 bytes) for static params."""
    try:
        with open(fp, "rb") as f:
            chunk = f.read(512).decode("utf-8", errors="replace")
        lines = chunk.splitlines()
        if len(lines) < 2:
            return None
        header = next(csv.reader([lines[0]]))
        values = next(csv.reader([lines[1]]))
        row = dict(zip(header, values))
        return {
            "cell_id" : row.get(config.COL_ID, Path(fp).stem),
            "Q_Ah"    : float(row[config.COL_Q]),
            "R0_mOhm" : float(row[config.COL_R0]),
            "VOCV_V"  : float(row[config.COL_VOCV]),
            "file"    : Path(fp).name,
            "fpath"   : str(fp),
        }
    except Exception:
        return None


def _read_curve(fp):
    """Read full voltage curve for one cell."""
    try:
        df = pd.read_csv(fp, usecols=[config.COL_Q_STEP, config.COL_V_CURVE])
        df = df.sort_values(config.COL_Q_STEP)
        return df[config.COL_V_CURVE].values.astype(np.float32)
    except Exception:
        return None


def load_cells(data_dir=None, n_cells=None, load_curves=False):
    """
    Load cells from CSV folder.
    Returns DataFrame of static params, and optionally a dict of voltage curves.

    Parameters
    ----------
    data_dir   : folder path (defaults to config.DATA_DIR)
    n_cells    : max files to load (None = all)
    load_curves: if True, also load full voltage curves per cell

    Returns
    -------
    cells_df   : DataFrame  [cell_id, Q_Ah, R0_mOhm, VOCV_V, file, fpath]
    curves     : dict {cell_id: np.array} or None
    """
    data_dir = data_dir or config.DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    if n_cells:
        files = files[:n_cells]

    print(f"Loading {len(files):,} files ...")

    n_threads = min(32, (os.cpu_count() or 4) * 4)
    records = [None] * len(files)
    errors  = 0

    # parallel static param reads
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        future_map = {pool.submit(_read_static, fp): i for i, fp in enumerate(files)}
        for future in as_completed(future_map):
            i   = future_map[future]
            res = future.result()
            if res:
                records[i] = res
            else:
                errors += 1

    records  = [r for r in records if r is not None]
    cells_df = pd.DataFrame(records).reset_index(drop=True)
    print(f"  {len(cells_df):,} cells loaded ({errors} skipped)")

    curves = None
    if load_curves:
        print("  Loading voltage curves ...")
        curves = {}
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            future_map = {
                pool.submit(_read_curve, row["fpath"]): row["cell_id"]
                for _, row in cells_df.iterrows()
            }
            for future in as_completed(future_map):
                cid   = future_map[future]
                curve = future.result()
                if curve is not None:
                    curves[cid] = curve
        print(f"  {len(curves):,} voltage curves loaded")

    return cells_df, curves


def get_static_matrix(cells_df):
    """Return (N, 3) numpy array of [Q, R0, VOCV]."""
    return cells_df[["Q_Ah", "R0_mOhm", "VOCV_V"]].values.astype(np.float64)


def get_curve_matrix(cells_df, curves, length=100):
    """
    Resample all voltage curves to fixed length and return (N, length) matrix.
    Cells missing a curve are filled with their row mean.
    """
    from scipy.interpolate import interp1d
    N   = len(cells_df)
    X   = np.zeros((N, length), dtype=np.float32)
    x_u = np.linspace(0, 1, length)

    for i, row in cells_df.iterrows():
        curve = curves.get(row["cell_id"])
        if curve is not None and len(curve) >= 2:
            x_o = np.linspace(0, 1, len(curve))
            fn  = interp1d(x_o, curve, kind="linear", bounds_error=False, fill_value="extrapolate")
            X[i] = fn(x_u)
        else:
            X[i] = np.full(length, row["VOCV_V"])   # fallback: flat line at OCV
    return X
