import os
import sys
import numpy as np
import torch
import torch.nn as nn

from config import (
    BATCH_SIZE, LR,
    PRETRAIN_EPOCHS, GAN_EPOCHS,
    PRINT_EVERY, CHECKPOINT_DIR,
    LAMBDA_SUPERVISED, LAMBDA_MOMENT,
    HIDDEN_DIM, NOISE_DIM, N_STEPS,
)
from timegan import build_models


# ── Loss helpers ───────────────────────────────────────────────────────────────

_bce = nn.BCEWithLogitsLoss()
_mse = nn.MSELoss()


def _real_labels(logits):
    return torch.ones_like(logits)


def _fake_labels(logits):
    return torch.zeros_like(logits)


def _moment_loss(X_real, X_fake):
    """
    Penalise differences in the mean and standard deviation of each feature
    across the batch.  Keeps the synthetic distribution roughly calibrated
    even if the GAN loss alone is not converging perfectly.
    """
    mean_loss = _mse(X_real.mean(0), X_fake.mean(0))
    std_loss  = _mse(X_real.std(0),  X_fake.std(0))
    return mean_loss + std_loss


# ── Noise sampling ─────────────────────────────────────────────────────────────

def _sample_noise(batch_size, device):
    return torch.randn(batch_size, N_STEPS, NOISE_DIM, device=device)


# ── Data batching ──────────────────────────────────────────────────────────────

def _iter_batches(data_tensor, batch_size, device):
    """
    Yield random mini-batches from data_tensor (N, T, F) without replacement
    within one epoch.
    """
    idx = torch.randperm(data_tensor.shape[0])
    for start in range(0, len(idx) - batch_size + 1, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield data_tensor[batch_idx].to(device)


# ── Print helper ───────────────────────────────────────────────────────────────

def _print_line(epoch, total, tag, **losses):
    """
    Overwrite the current terminal line with a compact loss summary.
    On milestone epochs print a proper newline so the history is readable.
    """
    parts = "  ".join(f"{k}: {v:.5f}" for k, v in losses.items())
    line  = f"\r[{tag}] epoch {epoch:>5}/{total}  {parts}   "

    if epoch % PRINT_EVERY == 0 or epoch == total:
        sys.stdout.write(line.lstrip("\r") + "\n")
    else:
        sys.stdout.write(line)
    sys.stdout.flush()


# ── Phase 1: Autoencoder pre-training ─────────────────────────────────────────

def pretrain_autoencoder(E, R, data_tensor, device):
    """
    Train the Embedder-Recovery pair as a plain autoencoder.
    No GAN involvement yet — just get a stable latent space.
    """
    opt = torch.optim.Adam(list(E.parameters()) + list(R.parameters()), lr=LR)

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for X in _iter_batches(data_tensor, BATCH_SIZE, device):
            H     = E(X)
            X_hat = R(H)
            loss  = _mse(X, X_hat)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        _print_line(epoch, PRETRAIN_EPOCHS, "Autoencoder",
                    recon=epoch_loss / max(n_batches, 1))

    print()    # final newline after the buffer line


# ── Phase 2: Supervisor pre-training ─────────────────────────────────────────

def pretrain_supervisor(E, S, data_tensor, device):
    """
    Train the Supervisor to predict the next latent step from the current one,
    using real embeddings from the already-trained Embedder.

    The Generator will later be trained to also fool this Supervisor,
    which enforces temporal coherence in the synthetic sequences.
    """
    opt = torch.optim.Adam(S.parameters(), lr=LR)

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for X in _iter_batches(data_tensor, BATCH_SIZE, device):
            with torch.no_grad():
                H = E(X)

            # Predict steps 1..T-1 from steps 0..T-2
            S_hat = S(H[:, :-1, :])
            loss  = _mse(H[:, 1:, :], S_hat)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        _print_line(epoch, PRETRAIN_EPOCHS, "Supervisor",
                    super=epoch_loss / max(n_batches, 1))

    print()


# ── Phase 3: Joint adversarial training ───────────────────────────────────────

def train_joint(E, R, S, G, D, data_tensor, device):
    """
    The main TimeGAN training loop.  All five components are trained together.

    Generator wants:
        1. The Discriminator to accept its latent sequences as real.
        2. Its sequences to fool the Supervisor (temporal coherence).
        3. The recovered sequences to match the real data distribution
           in mean and variance.

    Discriminator wants:
        1. To correctly label real embeddings as real.
        2. To correctly label Generator output as fake.
        3. To correctly label Supervisor-propagated sequences as fake.

    Embedder keeps refining itself to minimise reconstruction error and
    stay consistent with the Supervisor.
    """
    opt_G = torch.optim.Adam(
        list(G.parameters()) + list(S.parameters()), lr=LR
    )
    opt_D = torch.optim.Adam(D.parameters(), lr=LR)
    opt_E = torch.optim.Adam(
        list(E.parameters()) + list(R.parameters()), lr=LR
    )

    for epoch in range(1, GAN_EPOCHS + 1):
        g_loss_avg = 0.0
        d_loss_avg = 0.0
        e_loss_avg = 0.0
        n_batches  = 0

        for X in _iter_batches(data_tensor, BATCH_SIZE, device):
            batch_size = X.shape[0]
            Z          = _sample_noise(batch_size, device)

            # ── Discriminator step ─────────────────────────────────────────────
            H         = E(X).detach()
            E_hat     = G(Z).detach()
            H_hat_sup = S(E_hat[:, :-1, :]).detach()

            d_real    = _bce(D(H),                    _real_labels(D(H)))
            d_fake_g  = _bce(D(E_hat),                _fake_labels(D(E_hat)))
            d_fake_s  = _bce(D(H_hat_sup),            _fake_labels(D(H_hat_sup)))

            d_loss = d_real + d_fake_g + d_fake_s

            # Only update the discriminator if it is not already dominant —
            # this is a common stability trick for small datasets.
            if d_loss.item() > 0.15:
                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            # ── Generator step ─────────────────────────────────────────────────
            E_hat     = G(Z)
            H_hat_sup = S(E_hat[:, :-1, :])
            X_hat     = R(E_hat)

            g_loss_unsup  = _bce(D(E_hat),     _real_labels(D(E_hat)))
            g_loss_super  = _mse(E(X)[:, 1:, :], H_hat_sup)
            g_loss_moment = _moment_loss(X, X_hat)

            g_loss = (
                g_loss_unsup
                + LAMBDA_SUPERVISED * g_loss_super
                + LAMBDA_MOMENT     * g_loss_moment
            )

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            # ── Embedder refinement step ────────────────────────────────────────
            H     = E(X)
            X_hat = R(H)
            S_hat = S(H[:, :-1, :])

            e_loss_recon = _mse(X, X_hat)
            e_loss_super = _mse(H[:, 1:, :], S_hat)
            e_loss       = e_loss_recon + LAMBDA_SUPERVISED * e_loss_super

            opt_E.zero_grad()
            e_loss.backward()
            opt_E.step()

            g_loss_avg += g_loss.item()
            d_loss_avg += d_loss.item()
            e_loss_avg += e_loss.item()
            n_batches  += 1

        nb = max(n_batches, 1)
        _print_line(epoch, GAN_EPOCHS, "GAN",
                    G=g_loss_avg / nb,
                    D=d_loss_avg / nb,
                    E=e_loss_avg / nb)

    print()


# ── Checkpoint utilities ───────────────────────────────────────────────────────

def save_checkpoint(E, R, S, G, D):
    path = os.path.join(CHECKPOINT_DIR, "timegan.pt")
    torch.save({
        "E": E.state_dict(),
        "R": R.state_dict(),
        "S": S.state_dict(),
        "G": G.state_dict(),
        "D": D.state_dict(),
    }, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(E, R, S, G, D, device):
    path = os.path.join(CHECKPOINT_DIR, "timegan.pt")
    ckpt = torch.load(path, map_location=device)
    E.load_state_dict(ckpt["E"])
    R.load_state_dict(ckpt["R"])
    S.load_state_dict(ckpt["S"])
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    print(f"Checkpoint loaded ← {path}")


# ── Main entry point ───────────────────────────────────────────────────────────

def train(data_norm):
    """
    Run all three training phases in sequence.

    data_norm : np.ndarray of shape (N, N_STEPS, FEATURE_DIM),
                already normalised to [0, 1].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    data_tensor = torch.tensor(data_norm, dtype=torch.float32)

    E, R, S, G, D = build_models(device)

    print(f"\n-- Phase 1: Autoencoder ({PRETRAIN_EPOCHS} epochs) --")
    pretrain_autoencoder(E, R, data_tensor, device)

    print(f"\n-- Phase 2: Supervisor ({PRETRAIN_EPOCHS} epochs) --")
    pretrain_supervisor(E, S, data_tensor, device)

    print(f"\n-- Phase 3: Joint GAN ({GAN_EPOCHS} epochs) --")
    train_joint(E, R, S, G, D, data_tensor, device)

    save_checkpoint(E, R, S, G, D)
    return E, R, S, G, D
