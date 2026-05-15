import torch
import torch.nn as nn

from config import FEATURE_DIM, HIDDEN_DIM, NUM_LAYERS, NOISE_DIM, N_STEPS


# ── Shared building block ──────────────────────────────────────────────────────

class _GRUBlock(nn.Module):
    """
    A GRU stack followed by a linear projection.
    Used as the core inside every TimeGAN component.
    Keeping it separate avoids copy-pasting the same boilerplate four times.
    """

    def __init__(self, in_dim, out_dim, n_layers, activation=None):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc         = nn.Linear(HIDDEN_DIM, out_dim)
        self.activation = activation    # None / sigmoid / relu / tanh

    def forward(self, x):
        out, _ = self.gru(x)           # (B, T, hidden)
        out    = self.fc(out)          # (B, T, out_dim)
        if self.activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.activation == "relu":
            out = torch.relu(out)
        elif self.activation == "tanh":
            out = torch.tanh(out)
        return out


# ── TimeGAN components ─────────────────────────────────────────────────────────

class Embedder(nn.Module):
    """
    Maps real data sequences to a lower-dimensional latent space.
    Real X  →  H   (both shape: B × T × dim)
    """

    def __init__(self):
        super().__init__()
        self.net = _GRUBlock(FEATURE_DIM, HIDDEN_DIM, NUM_LAYERS,
                             activation="sigmoid")

    def forward(self, x):
        return self.net(x)


class Recovery(nn.Module):
    """
    Decodes the latent space back into the data space.
    H  →  X_hat
    Paired with Embedder to form an autoencoder.
    """

    def __init__(self):
        super().__init__()
        self.net = _GRUBlock(HIDDEN_DIM, FEATURE_DIM, NUM_LAYERS,
                             activation=None)    # unbounded; MSE loss handles it

    def forward(self, h):
        return self.net(h)


class Supervisor(nn.Module):
    """
    Predicts the next latent state from the current one.
    Trained on real embeddings to capture temporal dynamics — the generator
    is then asked to produce latent sequences that fool the supervisor too.

    Uses NUM_LAYERS - 1 GRU layers (one fewer than the others) as in the
    original TimeGAN paper.
    """

    def __init__(self):
        super().__init__()
        layers = max(1, NUM_LAYERS - 1)
        self.net = _GRUBlock(HIDDEN_DIM, HIDDEN_DIM, layers,
                             activation="sigmoid")

    def forward(self, h):
        return self.net(h)


class Generator(nn.Module):
    """
    Takes random Gaussian noise and produces a synthetic latent sequence.
    Z  →  E_hat  (same shape as Embedder output)
    """

    def __init__(self):
        super().__init__()
        self.net = _GRUBlock(NOISE_DIM, HIDDEN_DIM, NUM_LAYERS,
                             activation="sigmoid")

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    Works entirely in latent space — it never sees raw data directly.
    Classifies latent sequences as coming from real embeddings or the generator.
    H  →  logit per time step  (B × T × 1)

    No final sigmoid here; we apply it inside the loss so we can use
    BCEWithLogitsLoss which is numerically more stable.
    """

    def __init__(self):
        super().__init__()
        self.net = _GRUBlock(HIDDEN_DIM, 1, NUM_LAYERS, activation=None)

    def forward(self, h):
        return self.net(h)


# ── Convenience: build everything at once ─────────────────────────────────────

def build_models(device):
    """
    Instantiate all five components and move them to the target device.
    Returns them in the order (E, R, S, G, D).
    """
    E = Embedder().to(device)
    R = Recovery().to(device)
    S = Supervisor().to(device)
    G = Generator().to(device)
    D = Discriminator().to(device)
    return E, R, S, G, D
