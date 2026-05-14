import os

# ============================================================
# CONFIG
# ============================================================

# ── PUT YOUR PROCESSED CSV FOLDER PATH HERE ───────────────────
PROCESSED_FOLDER = r"E:\Thesis\Modelling_Simlulation_of optimization_code\Synthetic data generator\Syntheticdata generator timegan\datagenerator"

# ── OUTPUT FOLDERS — created automatically ────────────────────
OUTPUT_FOLDER  = os.path.join(os.path.dirname(PROCESSED_FOLDER), "synthetic_fresh_cells")
RESULTS_FOLDER = os.path.join(os.path.dirname(PROCESSED_FOLDER), "results")

os.makedirs(OUTPUT_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ── resistance values from Kirkaldy et al. 2024 Table 7 ───────
R01S_MEAN = 26.42
R01S_STD  = 0.78
R10S_MEAN = 32.96
R10S_STD  = 0.79

# ── model ─────────────────────────────────────────────────────
SEQ_LEN    = 500
INPUT_DIM  = 2
COND_DIM   = 1
HIDDEN_DIM = 32
NUM_LAYERS = 2

# ── training ──────────────────────────────────────────────────
BATCH_SIZE = 16
LR         = 1e-3
GAMMA      = 1.0
EPOCHS_AE  = 100
EPOCHS_SUP = 100
EPOCHS_GAN = 200

# ── generation ────────────────────────────────────────────────
N_GENERATE = 100_000
SOH_TARGET = 1.0
CHUNK      = 1000
SEEDS      = [42, 123, 456, 789, 2024]
N_CSV_SAVE = 100_000