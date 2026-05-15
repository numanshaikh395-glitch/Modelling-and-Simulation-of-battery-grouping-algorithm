import os

# ── Paths ──────────────────────────────────────────────────────────────────────
# Point DATA_ROOT at whatever folder you extracted from Zenodo.
# The loader walks it recursively, so nesting depth does not matter.
DATA_ROOT = (
    r"E:\Thesis\Modelling_Simlulation_of optimization_code"
    r"\Synthetic data generator\Timegansynthetic"
    r"\0.1C Voltage Discharge Curves"
)

PROCESSED_DIR    = r"E:\Thesis\Modelling_Simlulation_of optimization_code\Synthetic data generator\Timegansynthetic\processed"
CHECKPOINT_DIR   = r"E:\Thesis\Modelling_Simlulation_of optimization_code\Synthetic data generator\Timegansynthetic\checkpoints"
OUTPUT_DIR       = r"E:\Thesis\Modelling_Simlulation_of optimization_code\Synthetic data generator\Timegansynthetic\synthetic_cells"
NORM_PARAMS_FILE = r"E:\Thesis\Modelling_Simlulation_of optimization_code\Synthetic data generator\Timegansynthetic\normalization_params.json"

for _d in [PROCESSED_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Cell constants from Table 7 in the paper ───────────────────────────────────
# Mean C/10 capacity across 40 cells at beginning of life.
BOL_CAPACITY_AH = 4.865

# C/10 means one-tenth the capacity drained per hour.
C10_CURRENT_A   = BOL_CAPACITY_AH / 10.0      # 0.4865 A

# Biologic cyclers do not always hit exactly C/10 due to rounding.
# Accept anything within ±10 % of the target current.
C10_CURRENT_TOL = C10_CURRENT_A * 0.10        # ~0.049 A

# Voltage limits used in all experiments (Section 2.3 of the paper).
V_UPPER = 4.2    # V
V_LOWER = 2.5    # V

# ── Curve resampling ───────────────────────────────────────────────────────────
N_STEPS = 100

# Used only during active-region detection to clip the uninformative flat
# tails off each end of the V-Q curve.  Not stored as a feature anywhere.
DVDQ_CLIP_THRESHOLD = 0.008   # V / Ah

# ── TimeGAN architecture ───────────────────────────────────────────────────────
# Dynamic features change at every one of the 100 time steps.
# Static features describe the cell/condition and repeat across every step.

DYNAMIC_DIM = 3    # Q_Ah, t_s, V_OCV
STATIC_DIM  = 6    # SOH, capacity_Ah, temperature, SoC_window,
                   # charge_throughput_kAh, rpt_number

FEATURE_DIM = DYNAMIC_DIM + STATIC_DIM    # 9, total width per time step

HIDDEN_DIM  = 32
NUM_LAYERS  = 2    # keeping this modest given the small training set (~360 seqs)
NOISE_DIM   = 32

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
LR              = 1e-3

# Phase 1 + 2: train the autoencoder and the supervisor before any GAN loss.
PRETRAIN_EPOCHS = 300

# Phase 3: joint adversarial training.
GAN_EPOCHS      = 200

# Print a proper newline (not just overwrite) every this many epochs.
PRINT_EVERY     = 10

# Loss weighting — standard TimeGAN values from the original paper.
LAMBDA_SUPERVISED = 10.0
LAMBDA_MOMENT     = 100.0

# ── Generation ─────────────────────────────────────────────────────────────────
N_SYNTHETIC      = 200000
SYNTH_BATCH_SIZE = 500       # how many sequences to generate at once

# ── Condition label encodings ──────────────────────────────────────────────────
# Temperature is a continuous value; we store it as a float and normalise later.
TEMP_VALUES = [10, 25, 40]

# SoC window is categorical; encode as an integer label 0-4.
SOC_WINDOW_MAP = {
    "0-30":     0,
    "70-85":    1,
    "85-100":   2,
    "0-100-DC": 3,
    "0-100-CC": 4,
}
