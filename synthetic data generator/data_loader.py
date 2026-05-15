import os
import re
import numpy as np
import pandas as pd

from config import DATA_ROOT, BOL_CAPACITY_AH, SOC_WINDOW_MAP

# ── Experiment number → SoC window ────────────────────────────────────────────
# From Table 1 in the paper.  The filenames use "Expt 1", "Expt 2", etc.
EXPT_TO_SOC = {
    1: "0-30",
    2: "70-85",
    3: "85-100",
    4: "0-100-DC",
    5: "0-100-CC",
}

# ── Column names ───────────────────────────────────────────────────────────────
# Confirmed from the uploaded sample file — do not change these.
COL_TIME    = "Time (s)"
COL_VOLTAGE = "Voltage (V)"
COL_CURRENT = "Current (mA)"
COL_CHARGE  = "Charge (mA.h)"
COL_TEMP    = "Temperature (degC)"


def _parse_filename(fname):
    """
    Pull experiment number, cell letter, and RPT number out of a filename
    that follows the pattern:

        Expt 1 - cell A - RPT0 - 0.1C discharge data.csv

    Returns (expt_num, cell_letter, rpt_num) or None if it does not match.
    """
    pattern = r"expt\s*(\d+)\s*-\s*cell\s*([A-Za-z]+)\s*-\s*rpt\s*(\d+)"
    m = re.search(pattern, fname, re.IGNORECASE)
    if m is None:
        return None
    return int(m.group(1)), m.group(2).upper(), int(m.group(3))


def _load_single_file(fpath):
    """
    Read one discharge CSV and return the raw arrays we need.
    Returns None if anything looks off.
    """
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"  Could not read {os.path.basename(fpath)}: {e}")
        return None

    for col in [COL_TIME, COL_VOLTAGE, COL_CURRENT, COL_CHARGE, COL_TEMP]:
        if col not in df.columns:
            print(f"  Missing column '{col}' in {os.path.basename(fpath)}")
            return None

    time_s   = df[COL_TIME].values.astype(float)
    voltage  = df[COL_VOLTAGE].values.astype(float)
    current  = df[COL_CURRENT].values.astype(float)    # mA, negative = discharge
    charge   = df[COL_CHARGE].values.astype(float)     # mA.h, grows from ~0
    temp_raw = df[COL_TEMP].values.astype(float)

    # Current is negative for discharge (~-500 mA at C/10 for this cell).
    # Charge column should grow monotonically through the step.
    if current.mean() > 0:
        return None
    if charge[-1] <= charge[0]:
        return None

    # Convert mA.h → Ah, zero-base so Q starts at 0
    Q_Ah = (charge - charge[0]) / 1000.0

    # Zero-base time so every curve starts at t = 0
    t_s = time_s - time_s[0]

    # Read temperature from inside the file, snap to nearest valid setpoint
    temp_mean   = float(np.mean(temp_raw))
    temperature = min([10, 25, 40], key=lambda t: abs(t - temp_mean))

    total_cap = float(Q_Ah[-1])
    if total_cap < 2.0:    # less than 2 Ah from a 5 Ah cell means something went wrong
        return None

    return {
        "Q_Ah":        Q_Ah,
        "V_OCV":       voltage,
        "t_s_raw":     t_s,
        "temperature": temperature,
        "capacity_Ah": total_cap,
        "n_raw_pts":   len(Q_Ah),
    }


def load_dataset():
    """
    Walk DATA_ROOT, parse every file whose name matches
    'Expt X - cell Y - RPTZ - 0.1C discharge data.csv',
    and return a list of record dicts ready for the preprocessor.

    Each dict contains:
        cell_id, rpt_number, temperature, SoC_window,
        capacity_Ah, SOH, charge_throughput_kAh,
        Q_Ah (array), V_OCV (array), t_s_raw (array)
    """
    raw     = []
    skipped = 0

    for fname in sorted(os.listdir(DATA_ROOT)):
        if not fname.lower().endswith(".csv"):
            skipped += 1
            continue

        parsed = _parse_filename(fname)
        if parsed is None:
            skipped += 1
            continue

        expt_num, cell_letter, rpt_num = parsed
        soc_window = EXPT_TO_SOC.get(expt_num)
        if soc_window is None:
            skipped += 1
            continue

        fpath  = os.path.join(DATA_ROOT, fname)
        result = _load_single_file(fpath)
        if result is None:
            skipped += 1
            continue

        # One cell_id per physical cell — groups all its RPTs together
        cell_id = f"Expt{expt_num}_cell{cell_letter}"

        raw.append({
            "cell_id":     cell_id,
            "rpt_number":  rpt_num,
            "temperature": result["temperature"],
            "SoC_window":  soc_window,
            "Q_Ah":        result["Q_Ah"],
            "V_OCV":       result["V_OCV"],
            "t_s_raw":     result["t_s_raw"],
            "capacity_Ah": result["capacity_Ah"],
            "n_raw_pts":   result["n_raw_pts"],
        })

    if len(raw) == 0:
        raise RuntimeError(
            f"No valid files found in:\n  {DATA_ROOT}\n\n"
            "Expected filenames like:\n"
            "  'Expt 1 - cell A - RPT0 - 0.1C discharge data.csv'\n\n"
            "Check that DATA_ROOT in config.py points at the right folder."
        )

    # Sort so every cell's RPTs are in chronological order
    raw.sort(key=lambda r: (r["cell_id"], r["rpt_number"]))

    # ── SOH and cumulative charge throughput ──────────────────────────────────
    # SOH   = current capacity / capacity at RPT0 for that cell
    # kAh   = sets completed × 78 EFC × BOL capacity (from the paper)
    bol_cap   = {}
    rpt_count = {}

    for r in raw:
        cid = r["cell_id"]

        if cid not in bol_cap:
            bol_cap[cid]   = r["capacity_Ah"]
            rpt_count[cid] = 0

        r["SOH"] = r["capacity_Ah"] / bol_cap[cid]
        r["charge_throughput_kAh"] = (
            rpt_count[cid] * 78.0 * BOL_CAPACITY_AH / 1000.0
        )
        rpt_count[cid] += 1

    unique_cells = len(bol_cap)
    print(
        f"Loaded {len(raw)} discharge curves "
        f"from {unique_cells} cells  "
        f"({skipped} files skipped)."
    )

    # Print a quick summary so any parsing mistakes are obvious immediately
    for cid in sorted(bol_cap):
        rpts = sorted(r["rpt_number"] for r in raw if r["cell_id"] == cid)
        print(f"  {cid:30s}  RPTs: {rpts}")

    return raw
