import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# flush every print immediately — no buffering
sys.stdout.reconfigure(line_buffering=True)

from config import (
    PROCESSED_FOLDER, OUTPUT_FOLDER, RESULTS_FOLDER,
    SEQ_LEN, INPUT_DIM, COND_DIM, HIDDEN_DIM, NUM_LAYERS,
    BATCH_SIZE, LR, GAMMA,
    EPOCHS_AE, EPOCHS_SUP, EPOCHS_GAN,
    N_GENERATE, SOH_TARGET, CHUNK, SEEDS, N_CSV_SAVE,
    R01S_MEAN, R01S_STD, R10S_MEAN, R10S_STD
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}", flush=True)


# ============================================================
# LOAD PROCESSED CSV FILES
# ============================================================

def load_data():

    pattern   = os.path.join(PROCESSED_FOLDER, "*.csv")
    csv_files = sorted(glob(pattern))

    print("=" * 60, flush=True)
    print("LOADING PROCESSED FILES",  flush=True)
    print("=" * 60, flush=True)
    print(f"Folder : {PROCESSED_FOLDER}", flush=True)
    print(f"Found  : {len(csv_files)} files", flush=True)

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No CSV files found in:\n{PROCESSED_FOLDER}\n"
            "Check PROCESSED_FOLDER in config.py"
        )

    sequences = []
    soh_list  = []
    cell_ids  = []

    for path in csv_files:

        df = pd.read_csv(path)

        if path == csv_files[0]:
            print(f"Columns : {df.columns.tolist()}\n", flush=True)

        required = {"Voltage_norm", "Charge_norm", "SoH", "cell_id"}
        if required - set(df.columns):
            print(f"  SKIP {os.path.basename(path)} — missing columns",
                  flush=True)
            continue

        seq = df[["Voltage_norm", "Charge_norm"]].values.astype(np.float32)

        if seq.shape != (500, 2):
            print(f"  SKIP {os.path.basename(path)} — shape {seq.shape}",
                  flush=True)
            continue

        sequences.append(seq)
        soh_list.append(float(df["SoH"].iloc[0]))
        cell_ids.append(str(df["cell_id"].iloc[0]))

        print(f"  {os.path.basename(path):<60} "
              f"cell={cell_ids[-1]}  SoH={soh_list[-1]:.4f}",
              flush=True)

    X         = np.array(sequences,  dtype=np.float32)
    soh_array = np.array(soh_list,   dtype=np.float32)

    print(f"\nDataset shape : {X.shape}",              flush=True)
    print(f"SoH range     : {soh_array.min():.4f} → {soh_array.max():.4f}",
          flush=True)
    print(f"Unique cells  : {sorted(set(cell_ids))}", flush=True)

    return X, soh_array


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, soh_array):

    indices     = np.arange(len(X))
    soh_buckets = pd.cut(
        soh_array,
        bins=[0.0, 0.92, 0.95, 0.97, 1.01],
        labels=[0, 1, 2, 3]
    ).astype(int)

    min_class = int(np.bincount(soh_buckets).min())

    if min_class < 2:
        print("Splitting without stratify — too few samples per bucket",
              flush=True)
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42)
    else:
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42,
            stratify=soh_buckets)

    print(f"Train : {len(train_idx)}  |  Test : {len(test_idx)}", flush=True)
    return train_idx, test_idx


# ============================================================
# SOH CONDITIONING
# ============================================================

def add_condition(X_seq, soh_vals):
    soh_broad = np.repeat(
        soh_vals[:, np.newaxis, np.newaxis], SEQ_LEN, axis=1)
    return np.concatenate(
        [X_seq, soh_broad], axis=-1).astype(np.float32)


# ============================================================
# NETWORKS
# ============================================================

class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM + COND_DIM, HIDDEN_DIM,
                          NUM_LAYERS, batch_first=True)
        self.fc  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
    def forward(self, x):
        out, _ = self.gru(x)
        return torch.sigmoid(self.fc(out))


class Recovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(HIDDEN_DIM, HIDDEN_DIM,
                          NUM_LAYERS, batch_first=True)
        self.fc  = nn.Linear(HIDDEN_DIM, INPUT_DIM)
    def forward(self, h):
        out, _ = self.gru(h)
        return self.fc(out)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM + COND_DIM, HIDDEN_DIM,
                          NUM_LAYERS, batch_first=True)
        self.fc  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
    def forward(self, z):
        out, _ = self.gru(z)
        return torch.sigmoid(self.fc(out))


class Supervisor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(HIDDEN_DIM, HIDDEN_DIM,
                          NUM_LAYERS, batch_first=True)
        self.fc  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
    def forward(self, h):
        out, _ = self.gru(h)
        return torch.sigmoid(self.fc(out))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(HIDDEN_DIM + COND_DIM, HIDDEN_DIM,
                          NUM_LAYERS, batch_first=True)
        self.fc  = nn.Linear(HIDDEN_DIM, 1)
    def forward(self, h, soh):
        out, _ = self.gru(torch.cat([h, soh], dim=-1))
        return self.fc(out)


# ============================================================
# NOISE HELPER
# ============================================================

def make_noise(batch_size, soh_vals):
    z   = torch.randn(batch_size, SEQ_LEN, INPUT_DIM).to(DEVICE)
    soh = soh_vals[:, None, None].expand(batch_size, SEQ_LEN, COND_DIM)
    return torch.cat([z, soh], dim=-1)


# ============================================================
# TRAINING
# ============================================================

def train_model(train_loader, test_tensor,
                embedder, recovery, generator,
                supervisor, discriminator):

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    opt_er = torch.optim.Adam(
        list(embedder.parameters()) + list(recovery.parameters()), lr=LR)
    opt_gs = torch.optim.Adam(
        list(generator.parameters()) + list(supervisor.parameters()), lr=LR)
    opt_g  = torch.optim.Adam(
        list(generator.parameters()) + list(supervisor.parameters()), lr=LR)
    opt_d  = torch.optim.Adam(discriminator.parameters(), lr=LR)

    # ── Phase 1 : Autoencoder ─────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1 — AUTOENCODER",    flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()

    for epoch in range(EPOCHS_AE):

        epoch_loss = 0.0
        embedder.train(); recovery.train()

        for (batch,) in train_loader:
            H     = embedder(batch)
            X_hat = recovery(H)
            loss  = mse(X_hat, batch[:, :, :INPUT_DIM])
            opt_er.zero_grad()
            loss.backward()
            opt_er.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(train_loader)

        # print every 10 epochs so terminal stays active
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:>4d}/{EPOCHS_AE}  "
                  f"Loss: {avg:.6f}  "
                  f"Time: {elapsed:.1f}s",
                  flush=True)

    print(f"  Phase 1 done in {time.time()-t0:.1f}s", flush=True)

    # ── Phase 2 : Supervisor ──────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("PHASE 2 — SUPERVISOR",     flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()

    for epoch in range(EPOCHS_SUP):

        epoch_loss = 0.0
        supervisor.train()

        for (batch,) in train_loader:
            H     = embedder(batch).detach()
            H_sup = supervisor(H)
            loss  = mse(H_sup[:, :-1, :], H[:, 1:, :])
            opt_gs.zero_grad()
            loss.backward()
            opt_gs.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(train_loader)

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:>4d}/{EPOCHS_SUP}  "
                  f"Loss: {avg:.6f}  "
                  f"Time: {elapsed:.1f}s",
                  flush=True)

    print(f"  Phase 2 done in {time.time()-t0:.1f}s", flush=True)

    # ── Phase 3 : Joint GAN ───────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("PHASE 3 — JOINT GAN",      flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()

    for epoch in range(EPOCHS_GAN):

        g_total = 0.0
        d_total = 0.0
        generator.train(); discriminator.train()

        for (batch,) in train_loader:

            B        = batch.size(0)
            soh_vals = batch[:, 0, -1]
            soh_seq  = batch[:, :, -1:]

            H_real   = embedder(batch).detach()
            Z        = make_noise(B, soh_vals)
            H_fake   = generator(Z)
            H_fake_s = supervisor(H_fake)

            D_real  = discriminator(H_real,            soh_seq)
            D_fake  = discriminator(H_fake_s.detach(), soh_seq)
            loss_d  = (bce(D_real, torch.ones_like(D_real)) +
                       bce(D_fake, torch.zeros_like(D_fake)))
            d_acc   = ((D_real > 0).float().mean() +
                       (D_fake < 0).float().mean()) / 2.0

            if d_acc < 0.85:
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

            H_fake   = generator(Z)
            H_fake_s = supervisor(H_fake)
            D_fake_g = discriminator(H_fake_s, soh_seq)
            loss_g   = (bce(D_fake_g, torch.ones_like(D_fake_g))
                        + GAMMA * mse(H_fake_s[:, :-1, :],
                                      H_fake[:, 1:, :])
                        + 10    * mse(recovery(H_fake_s).mean(0),
                                      recovery(H_real).mean(0)))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            g_total += loss_g.item()
            d_total += loss_d.item()

        avg_g = g_total / len(train_loader)
        avg_d = d_total / len(train_loader)

        # print every 50 epochs
        if (epoch + 1) % 50 == 0:
            embedder.eval(); recovery.eval()
            with torch.no_grad():
                test_recon = mse(
                    recovery(embedder(test_tensor)),
                    test_tensor[:, :, :INPUT_DIM]
                ).item()
            embedder.train(); recovery.train()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:>4d}/{EPOCHS_GAN}  "
                  f"G: {avg_g:.4f}  "
                  f"D: {avg_d:.4f}  "
                  f"TestRecon: {test_recon:.6f}  "
                  f"Time: {elapsed:.1f}s",
                  flush=True)

    print(f"  Phase 3 done in {time.time()-t0:.1f}s", flush=True)

    # save models
    torch.save(generator.state_dict(),
               os.path.join(RESULTS_FOLDER, "generator.pt"))
    torch.save(recovery.state_dict(),
               os.path.join(RESULTS_FOLDER, "recovery.pt"))
    torch.save(supervisor.state_dict(),
               os.path.join(RESULTS_FOLDER, "supervisor.pt"))
    torch.save(embedder.state_dict(),
               os.path.join(RESULTS_FOLDER, "embedder.pt"))
    print("\nModels saved.", flush=True)


# ============================================================
# GENERATE + SAVE
# ============================================================
#
# Fix 1: cell_id is now a simple integer  1, 2, 3 ... N
# Fix 2: each synthetic cell saved as its own CSV
#        with only fresh cell data — no mixed original data
#

def generate(generator, recovery, supervisor):

    generator.eval()
    recovery.eval()
    supervisor.eval()

    rng        = np.random.default_rng(seed=42)
    all_chunks = []
    N_PER_SEED = N_GENERATE // len(SEEDS)

    print("\n" + "=" * 60, flush=True)
    print(f"GENERATING {N_GENERATE:,} SYNTHETIC CELLS", flush=True)
    print("=" * 60, flush=True)

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        seed_batches = []

        with torch.no_grad():
            for i in range(0, N_PER_SEED, CHUNK):
                soh_vals = torch.full((CHUNK,), SOH_TARGET).to(DEVICE)
                Z        = make_noise(CHUNK, soh_vals)
                out      = recovery(
                    supervisor(generator(Z))).cpu().numpy()
                seed_batches.append(out)

        result = np.concatenate(seed_batches, axis=0)
        all_chunks.append(result)
        print(f"  Seed {seed} → {result.shape}", flush=True)

    synthetic = np.concatenate(all_chunks, axis=0)   # (N, 500, 2)
    print(f"\nTotal synthetic shape : {synthetic.shape}", flush=True)

    np.save(os.path.join(RESULTS_FOLDER, "synthetic_100k.npy"), synthetic)

    # resistance from paper Table 7
    r01s = np.clip(
        rng.normal(R01S_MEAN, R01S_STD, N_GENERATE),
        R01S_MEAN - 3 * R01S_STD, R01S_MEAN + 3 * R01S_STD)
    r10s = np.clip(
        rng.normal(R10S_MEAN, R10S_STD, N_GENERATE),
        R10S_MEAN - 3 * R10S_STD, R10S_MEAN + 3 * R10S_STD)

    print(f"R0.1s : {r01s.min():.2f} → {r01s.max():.2f} mΩ", flush=True)
    print(f"R10s  : {r10s.min():.2f} → {r10s.max():.2f} mΩ", flush=True)

    # ── save each synthetic cell as its own CSV ───────────────
    #
    # Fix 1: cell_id is integer starting from 1
    # Fix 2: one CSV per cell — only fresh cell data
    #        no original data mixed in
    #
    # Output CSV columns:
    #   cell_id      → integer  1, 2, 3 ... N
    #   SoH          → 1.0 (fresh)
    #   timestep     → 0 to 499
    #   Voltage_norm → normalised voltage
    #   Charge_norm  → normalised charge
    #   R01s_mOhm    → scalar resistance per cell
    #   R10s_mOhm    → scalar resistance per cell
    #

    print(f"\nSaving {N_CSV_SAVE:,} individual fresh cell CSVs...", flush=True)
    t0 = time.time()

    for i in range(N_CSV_SAVE):

        cell_df = pd.DataFrame({
            "cell_id":      i + 1,           # 1, 2, 3 ... N
            "SoH":          1.0,
            "timestep":     np.arange(SEQ_LEN),
            "Voltage_norm": synthetic[i, :, 0].round(6),
            "Charge_norm":  synthetic[i, :, 1].round(6),
            "R01s_mOhm":    round(float(r01s[i]), 4),
            "R10s_mOhm":    round(float(r10s[i]), 4),
        })

        # filename uses integer cell id
        fname = f"fresh_cell_{i+1:06d}.csv"
        cell_df.to_csv(os.path.join(OUTPUT_FOLDER, fname), index=False)

        if (i + 1) % 5000 == 0:
            print(f"  {i+1:>7,} / {N_CSV_SAVE:,} saved  "
                  f"{time.time()-t0:.1f}s", flush=True)

    print(f"\nAll fresh cell CSVs saved to : {OUTPUT_FOLDER}", flush=True)
    return synthetic


# ============================================================
# METRICS
# ============================================================
#
# Fix 3: KL Divergence, PCA, t-SNE
#

def compute_kl_divergence(real, synthetic, n_bins=50):
    """
    Compute KL divergence between real and synthetic distributions.
    Estimated per feature-timestep dimension using histogram binning.
    Averaged across all dimensions.
    Lower is better. Below 0.5 is acceptable.
    Benchmark from Naaz et al. (2021) on NASA dataset = 0.2317.
    """
    from scipy.stats import entropy

    n_features   = real.shape[2]        # 2 features
    n_timesteps  = real.shape[1]        # 500 timesteps
    kl_scores    = []

    for f in range(n_features):
        for t in range(0, n_timesteps, 50):   # sample every 50 timesteps
            r_vals = real[:, t, f]
            s_vals = synthetic[:, t, f]

            # build histograms on same bin edges
            min_v  = min(r_vals.min(), s_vals.min())
            max_v  = max(r_vals.max(), s_vals.max())
            bins   = np.linspace(min_v, max_v, n_bins + 1)

            r_hist, _ = np.histogram(r_vals, bins=bins, density=True)
            s_hist, _ = np.histogram(s_vals, bins=bins, density=True)

            # add small epsilon to avoid log(0)
            r_hist = r_hist + 1e-10
            s_hist = s_hist + 1e-10

            # normalise
            r_hist = r_hist / r_hist.sum()
            s_hist = s_hist / s_hist.sum()

            kl = entropy(r_hist, s_hist)
            kl_scores.append(kl)

    mean_kl = float(np.mean(kl_scores))
    return mean_kl


# ============================================================
# VALIDATION PLOTS
# ============================================================
#
# Fix 4: working PCA, t-SNE, and overlay plots
#

def validate(synthetic, X_test, soh_test):

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    soc_axis = np.linspace(1.0, 0.0, SEQ_LEN)

    # pick real samples closest to SoH = 1.0
    real_idx = np.argsort(np.abs(soh_test - 1.0))[:200]
    real     = X_test[real_idx]                          # (200, 500, 2)
    synth    = synthetic[np.random.choice(
        len(synthetic), 200, replace=False)]             # (200, 500, 2)

    # ── KL Divergence ─────────────────────────────────────────
    print("\nComputing KL Divergence...", flush=True)
    kl = compute_kl_divergence(real, synth)
    print(f"  KL Divergence : {kl:.4f}", flush=True)
    print(f"  Benchmark     : 0.2317  (Naaz et al. 2021)", flush=True)
    if kl < 0.5:
        print(f"  Result        : ACCEPTABLE (below 0.5)", flush=True)
    else:
        print(f"  Result        : POOR (above 0.5 — distributions differ)",
              flush=True)

    # flatten sequences for PCA and t-SNE
    # each sample (500, 2) → flattened to (1000,)
    real_flat  = real.reshape(len(real),   -1)   # (200, 1000)
    synth_flat = synth.reshape(len(synth), -1)   # (200, 1000)
    combined   = np.concatenate([real_flat, synth_flat], axis=0)  # (400, 1000)
    labels     = np.array([0] * len(real) + [1] * len(synth))
    # 0 = real, 1 = synthetic

    # ── Plot 1: Voltage overlay ───────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i in range(min(50, len(real))):
        axes[0, 0].plot(soc_axis, real[i, :, 0],
                        color="steelblue", lw=0.6, alpha=0.4,
                        label="Real" if i == 0 else "")
        axes[0, 0].plot(soc_axis, synth[i, :, 0],
                        color="tomato", lw=0.6, alpha=0.4,
                        label="Synthetic" if i == 0 else "")
    axes[0, 0].set(title="Voltage Curves — Real vs Synthetic",
                   xlabel="SoC", ylabel="Voltage_norm")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].invert_xaxis()

    # ── Plot 2: Charge overlay ────────────────────────────────
    for i in range(min(50, len(real))):
        axes[0, 1].plot(soc_axis, real[i, :, 1],
                        color="steelblue", lw=0.6, alpha=0.4,
                        label="Real" if i == 0 else "")
        axes[0, 1].plot(soc_axis, synth[i, :, 1],
                        color="tomato", lw=0.6, alpha=0.4,
                        label="Synthetic" if i == 0 else "")
    axes[0, 1].set(title="Charge Curves — Real vs Synthetic",
                   xlabel="SoC", ylabel="Charge_norm")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].invert_xaxis()

    # ── Plot 3: Endpoint voltage distribution ─────────────────
    axes[0, 2].hist(real[:, -1, 0],  bins=20, color="steelblue",
                    alpha=0.6, density=True, label="Real")
    axes[0, 2].hist(synth[:, -1, 0], bins=20, color="tomato",
                    alpha=0.6, density=True, label="Synthetic")
    axes[0, 2].set(title=f"Endpoint Voltage Distribution\nKL = {kl:.4f}",
                   xlabel="Voltage_norm at SoC=0", ylabel="Density")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(alpha=0.3)

    # ── Plot 4: PCA ───────────────────────────────────────────
    print("\nRunning PCA...", flush=True)

    pca      = PCA(n_components=2)
    pca_all  = pca.fit_transform(combined)
    pca_real = pca_all[:len(real)]
    pca_syn  = pca_all[len(real):]

    axes[1, 0].scatter(pca_real[:, 0], pca_real[:, 1],
                       c="steelblue", alpha=0.5, s=15, label="Real")
    axes[1, 0].scatter(pca_syn[:, 0],  pca_syn[:, 1],
                       c="tomato",     alpha=0.5, s=15, label="Synthetic")
    axes[1, 0].set(
        title="PCA — Real vs Synthetic\n"
              "High overlap = good distribution match",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    print(f"  PCA variance explained : "
          f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%", flush=True)

    # ── Plot 5: t-SNE ─────────────────────────────────────────
    print("Running t-SNE (this takes a few minutes)...", flush=True)

    tsne     = TSNE(n_components=2, random_state=42,
                    perplexity=30, n_iter=1000)
    tsne_all = tsne.fit_transform(combined)
    tsne_real = tsne_all[:len(real)]
    tsne_syn  = tsne_all[len(real):]

    axes[1, 1].scatter(tsne_real[:, 0], tsne_real[:, 1],
                       c="steelblue", alpha=0.5, s=15, label="Real")
    axes[1, 1].scatter(tsne_syn[:, 0],  tsne_syn[:, 1],
                       c="tomato",     alpha=0.5, s=15, label="Synthetic")
    axes[1, 1].set(
        title="t-SNE — Real vs Synthetic\n"
              "Interspersed points = good generation",
        xlabel="t-SNE 1", ylabel="t-SNE 2")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    # ── Plot 6: metrics summary ───────────────────────────────
    axes[1, 2].axis("off")
    summary = (
        f"VALIDATION SUMMARY\n"
        f"{'─'*30}\n\n"
        f"KL Divergence  :  {kl:.4f}\n"
        f"Benchmark      :  0.2317\n"
        f"Threshold      :  0.5\n"
        f"Status         :  {'PASS' if kl < 0.5 else 'FAIL'}\n\n"
        f"PCA PC1        :  {pca.explained_variance_ratio_[0]*100:.1f}%\n"
        f"PCA PC2        :  {pca.explained_variance_ratio_[1]*100:.1f}%\n\n"
        f"Real samples   :  {len(real)}\n"
        f"Synth samples  :  {len(synth)}\n"
        f"Seq length     :  {SEQ_LEN}\n"
        f"Features       :  Voltage, Charge\n"
        f"SoH target     :  {SOH_TARGET}"
    )
    axes[1, 2].text(0.05, 0.95, summary,
                    transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat",
                              alpha=0.4))

    plt.suptitle("TimeGAN Validation — Real vs Synthetic  [SoH = 1.0]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "validation.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → validation.png", flush=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    X, soh_array = load_data()

    train_idx, test_idx = split_data(X, soh_array)

    X_train   = X[train_idx]
    X_test    = X[test_idx]
    soh_train = soh_array[train_idx]
    soh_test  = soh_array[test_idx]

    train_tensor = torch.tensor(
        add_condition(X_train, soh_train)).to(DEVICE)
    test_tensor  = torch.tensor(
        add_condition(X_test, soh_test)).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size = BATCH_SIZE,
        shuffle    = True,
        num_workers = 0,
        pin_memory  = True if DEVICE.type == "cuda" else False
    )

    embedder      = Embedder().to(DEVICE)
    recovery      = Recovery().to(DEVICE)
    generator     = Generator().to(DEVICE)
    supervisor    = Supervisor().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    train_model(train_loader, test_tensor,
                embedder, recovery, generator,
                supervisor, discriminator)

    synthetic = generate(generator, recovery, supervisor)

    validate(synthetic, X_test, soh_test)