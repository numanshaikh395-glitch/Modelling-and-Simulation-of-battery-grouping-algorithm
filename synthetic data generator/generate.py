import os
import numpy as np
import torch
import pandas as pd

from config import (
    N_SYNTHETIC, SYNTH_BATCH_SIZE,
    N_STEPS, NOISE_DIM, OUTPUT_DIR,
    SOC_WINDOW_MAP, C10_CURRENT_A,
    FEATURE_DIM,
)
from timegan import build_models
from train import load_checkpoint
from preprocessor import load_norm_params, denormalize


# ── Reverse SoC label lookup ───────────────────────────────────────────────────
# During preprocessing we turned "0-30" into 0, "70-85" into 1, etc.
# We need to go back the other way when writing CSVs.
INV_SOC_MAP = {v: k for k, v in SOC_WINDOW_MAP.items()}


def _label_to_soc_window(val):
    """
    val comes out of denormalization as a float (e.g. 0.97, 2.03).
    Round to nearest integer label and look up the string.
    """
    label = int(round(float(val)))
    label = max(0, min(label, len(INV_SOC_MAP) - 1))
    return INV_SOC_MAP.get(label, "unknown")


def _label_to_temperature(val):
    """
    Temperature was stored as a raw float (10, 25, or 40).
    After denorm it should be close to one of those values.
    Snap to the nearest valid setpoint so the CSVs are clean.
    """
    options = [10, 25, 40]
    return min(options, key=lambda t: abs(t - float(val)))


# ── Single-batch generation ────────────────────────────────────────────────────

def _generate_batch(G, R, S, batch_size, device):
    """
    Sample noise → Generator → Supervisor (optional chaining) → Recovery → data.

    We use the Generator alone (not chained through the Supervisor) for the
    primary latent sequence.  The Supervisor was used only during training to
    enforce temporal consistency; at inference the Generator has already learned
    to produce temporally coherent sequences.
    """
    with torch.no_grad():
        Z     = torch.randn(batch_size, N_STEPS, NOISE_DIM, device=device)
        E_hat = G(Z)          # (B, T, hidden)
        X_hat = R(E_hat)      # (B, T, feature_dim)

    return X_hat.cpu().numpy()    # (B, T, F)  still normalised


# ── Write one synthetic cell to disk ──────────────────────────────────────────

def _save_cell(synth_id, seq_raw, params):
    """
    seq_raw : (N_STEPS, FEATURE_DIM) — values in normalised [0, 1] space.
    params  : normalization dict from preprocessor.

    Steps:
        1. Denormalise back to physical units.
        2. Clip V_OCV to the physical voltage limits (safety).
        3. Build a clean DataFrame with human-readable column names.
        4. Save as CSV named by cell ID.
    """
    seq = denormalize(seq_raw[np.newaxis], params)[0]   # (100, 9)

    Q_Ah       = seq[:, 0]
    t_s        = seq[:, 1]
    V_OCV      = np.clip(seq[:, 2], 2.5, 4.2)    # physical voltage bounds

    # Static context — repeat across all 100 rows but read from a single step
    # (we average over all steps to smooth out any GAN noise in the constants)
    SOH                  = float(np.clip(np.mean(seq[:, 3]), 0.5, 1.05))
    capacity_Ah          = float(np.clip(np.mean(seq[:, 4]), 2.0, 5.5))
    temperature          = _label_to_temperature(np.mean(seq[:, 5]))
    SoC_window           = _label_to_soc_window(np.mean(seq[:, 6]))
    charge_throughput_kAh = float(np.clip(np.mean(seq[:, 7]), 0.0, 100.0))
    rpt_number           = int(round(np.clip(np.mean(seq[:, 8]), 0, 30)))

    cell_id = f"SYNTH_{synth_id:07d}"

    df = pd.DataFrame({
        "cell_id":                cell_id,
        "rpt_number":             rpt_number,
        "temperature":            temperature,
        "SoC_window":             SoC_window,
        "charge_throughput_kAh":  round(charge_throughput_kAh, 4),
        "capacity_Ah":            round(capacity_Ah, 4),
        "SOH":                    round(SOH, 5),
        "q_step":                 np.arange(N_STEPS),
        "Q_Ah":                   np.round(Q_Ah, 5),
        "t_s":                    np.round(t_s, 2),
        "V_OCV":                  np.round(V_OCV, 5),
    })

    out_path = os.path.join(OUTPUT_DIR, f"{cell_id}.csv")
    df.to_csv(out_path, index=False)


# ── Main generation loop ───────────────────────────────────────────────────────

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating on: {device}")

    # Load model weights
    E, R, S, G, D = build_models(device)
    load_checkpoint(E, R, S, G, D, device)

    G.eval()
    R.eval()
    S.eval()

    # Load normalization parameters saved during preprocessing
    params = load_norm_params()

    n_generated = 0
    n_batches   = (N_SYNTHETIC + SYNTH_BATCH_SIZE - 1) // SYNTH_BATCH_SIZE

    for batch_idx in range(n_batches):
        this_batch = min(SYNTH_BATCH_SIZE, N_SYNTHETIC - n_generated)
        if this_batch <= 0:
            break

        batch_norm = _generate_batch(G, R, S, this_batch, device)

        for i in range(this_batch):
            synth_id = n_generated + i + 1
            _save_cell(synth_id, batch_norm[i], params)

        n_generated += this_batch

        # Simple progress line that overwrites itself
        pct = 100.0 * n_generated / N_SYNTHETIC
        print(
            f"\rGenerated {n_generated:>7} / {N_SYNTHETIC}  ({pct:.1f}%)   ",
            end="", flush=True
        )

    print(f"\nDone.  {n_generated} synthetic cells written to '{OUTPUT_DIR}/'.")
