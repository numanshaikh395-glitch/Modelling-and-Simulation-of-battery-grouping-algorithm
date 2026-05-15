import json
import numpy as np

from config import (
    N_STEPS, DVDQ_CLIP_THRESHOLD,
    C10_CURRENT_A, BOL_CAPACITY_AH,
    SOC_WINDOW_MAP, NORM_PARAMS_FILE,
    DYNAMIC_DIM, STATIC_DIM, FEATURE_DIM,
)


# ── Active-region detection ────────────────────────────────────────────────────

def _find_active_region(Q, V, threshold=DVDQ_CLIP_THRESHOLD):
    """
    The C/10 discharge curve has almost flat tails at each end — a shallow
    plateau near 4.2 V at the top and a steep but featureless drop near 2.5 V
    at the bottom.  Both tails compress poorly under interpolation and add
    noise without adding information.

    We clip them by looking at the absolute gradient |dV/dQ|.
    Anything below the threshold at either end gets removed.

    Returns (i_start, i_end) indices into Q and V.
    """
    if len(Q) < 10:
        return 0, len(Q) - 1

    dV = np.abs(np.gradient(V, Q))

    # Scan from the left until the gradient wakes up
    i_start = 0
    for i in range(len(dV)):
        if dV[i] > threshold:
            i_start = i
            break

    # Scan from the right
    i_end = len(dV) - 1
    for i in range(len(dV) - 1, -1, -1):
        if dV[i] > threshold:
            i_end = i
            break

    # Make sure we kept a reasonable chunk of the curve
    if i_end - i_start < 20:
        return 0, len(Q) - 1

    return i_start, i_end


def resample_curve(Q_raw, V_raw, n_steps=N_STEPS):
    """
    Clip the flat tails, then interpolate V onto n_steps evenly spaced
    Q points within the active window.

    Returns Q_resampled, V_resampled — both length n_steps.
    """
    i0, i1 = _find_active_region(Q_raw, V_raw)

    Q_clip = Q_raw[i0 : i1 + 1]
    V_clip = V_raw[i0 : i1 + 1]

    Q_uniform = np.linspace(Q_clip[0], Q_clip[-1], n_steps)
    V_uniform  = np.interp(Q_uniform, Q_clip, V_clip)

    return Q_uniform, V_uniform


def compute_t_s(Q_uniform):
    """
    At C/10, current is constant so time and capacity are proportional.
    t_s[i] = Q[i] / I * 3600   (convert hours to seconds)

    We re-zero each curve at its own first Q point so t always starts at 0.
    """
    Q_zeroed = Q_uniform - Q_uniform[0]
    t_s = Q_zeroed / C10_CURRENT_A * 3600.0
    return t_s


# ── Build the full feature matrix ──────────────────────────────────────────────

def build_feature_matrix(records):
    """
    Turn the list of raw records (from data_loader) into a 3-D numpy array
    of shape (N_sequences, N_STEPS, FEATURE_DIM).

    Feature layout per time step (same order as config.FEATURE_DIM = 9):
        dim 0  : Q_Ah        (dynamic)
        dim 1  : t_s         (dynamic)
        dim 2  : V_OCV       (dynamic)
        dim 3  : SOH         (static, repeated)
        dim 4  : capacity_Ah (static, repeated)
        dim 5  : temperature (static, repeated)
        dim 6  : SoC_window  (static, repeated, encoded as 0-4)
        dim 7  : charge_throughput_kAh (static, repeated)
        dim 8  : rpt_number  (static, repeated)

    Also returns a parallel list of metadata dicts for bookkeeping.
    """
    sequences = []
    meta      = []

    for r in records:
        Q_res, V_res = resample_curve(r["Q_Ah"], r["V_OCV"])
        t_s          = compute_t_s(Q_res)

        soc_label = SOC_WINDOW_MAP.get(r["SoC_window"], 0)

        # Build one (N_STEPS, FEATURE_DIM) block
        seq = np.zeros((N_STEPS, FEATURE_DIM), dtype=np.float32)

        seq[:, 0] = Q_res
        seq[:, 1] = t_s
        seq[:, 2] = V_res
        seq[:, 3] = r["SOH"]
        seq[:, 4] = r["capacity_Ah"]
        seq[:, 5] = float(r["temperature"])
        seq[:, 6] = float(soc_label)
        seq[:, 7] = r["charge_throughput_kAh"]
        seq[:, 8] = float(max(r["rpt_number"], 0))

        sequences.append(seq)
        meta.append({
            "cell_id":               r["cell_id"],
            "rpt_number":            r["rpt_number"],
            "temperature":           r["temperature"],
            "SoC_window":            r["SoC_window"],
            "capacity_Ah":           r["capacity_Ah"],
            "SOH":                   r["SOH"],
            "charge_throughput_kAh": r["charge_throughput_kAh"],
        })

    data = np.stack(sequences, axis=0)    # (N, 100, 9)
    return data, meta


# ── Normalization ──────────────────────────────────────────────────────────────

def compute_norm_params(data):
    """
    Compute per-feature min and max across all sequences and all time steps.
    data shape: (N, T, F)

    We normalise each feature dimension independently to [0, 1].
    Saving min/max (not mean/std) keeps the physical meaning clear when we
    invert the transform later.
    """
    flat = data.reshape(-1, data.shape[-1])    # (N*T, F)

    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)

    # Guard against any feature that is constant (would cause divide-by-zero)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    params = {
        "mins":   mins.tolist(),
        "maxs":   maxs.tolist(),
        "ranges": ranges.tolist(),
        "n_features": int(data.shape[-1]),
    }
    return params


def save_norm_params(params, path=NORM_PARAMS_FILE):
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def load_norm_params(path=NORM_PARAMS_FILE):
    with open(path) as f:
        p = json.load(f)
    p["mins"]   = np.array(p["mins"],   dtype=np.float32)
    p["maxs"]   = np.array(p["maxs"],   dtype=np.float32)
    p["ranges"] = np.array(p["ranges"], dtype=np.float32)
    return p


def normalize(data, params):
    """data: (N, T, F)  →  normalised (N, T, F) in [0, 1]"""
    mins   = np.array(params["mins"],   dtype=np.float32)
    ranges = np.array(params["ranges"], dtype=np.float32)
    return (data - mins) / ranges


def denormalize(data_norm, params):
    """Inverse of normalize.  data_norm: (N, T, F)  →  physical units."""
    mins   = np.array(params["mins"],   dtype=np.float32)
    ranges = np.array(params["ranges"], dtype=np.float32)
    return data_norm * ranges + mins


# ── Full pipeline convenience wrapper ─────────────────────────────────────────

def prepare_training_data(records):
    """
    Goes from raw loaded records all the way to a normalised numpy array
    ready to feed into TimeGAN.

    Saves normalization parameters to disk so generate.py can invert them.

    Returns:
        data_norm  — (N, 100, 9) float32 in [0, 1]
        params     — normalization dict
        meta       — list of metadata dicts (one per sequence)
    """
    data, meta   = build_feature_matrix(records)
    params        = compute_norm_params(data)
    save_norm_params(params)
    data_norm     = normalize(data, params)

    print(
        f"Training tensor: {data_norm.shape}  "
        f"({data_norm.shape[0]} sequences, "
        f"{data_norm.shape[1]} steps, "
        f"{data_norm.shape[2]} features)"
    )
    return data_norm, params, meta
