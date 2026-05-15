
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config
from data_loader import load_cells, get_static_matrix, get_curve_matrix
from test_function import evaluate
from plotting import plot_algorithm_results
from excel_writer import save_results

ALGO_NAME = "PCA + SOM"


class SOM:

    def __init__(self, K, n_features, n_iter=200, lr=0.5, sigma0=None, seed=42):
        rng = np.random.default_rng(seed)
        self.K        = K
        self.n_iter   = n_iter
        self.lr       = lr
        self.sigma0   = sigma0 or K / 2.0
        self.weights  = rng.standard_normal((K, n_features))
        self.loss_hist = []

    def _neighbourhood(self, winner, sigma):
        # Gaussian neighbourhood around winning neuron
        d = (np.arange(self.K) - winner) ** 2
        return np.exp(-d / (2 * sigma**2))

    def fit(self, X):
        N = len(X)
        for it in range(self.n_iter):
            # decay learning rate and neighbourhood width
            lr    = self.lr    * np.exp(-it / self.n_iter)
            sigma = self.sigma0 * np.exp(-it / self.n_iter)
            order = np.random.permutation(N)
            for i in order:
                x      = X[i]
                dists  = ((self.weights - x) ** 2).sum(1)
                winner = int(dists.argmin())
                h      = self._neighbourhood(winner, sigma)  # (K,)
                self.weights += lr * h[:, None] * (x - self.weights)
            # track total quantisation error
            loss = float(((X - self.weights[self._predict(X)]) ** 2).sum())
            self.loss_hist.append(loss)
        return self

    def _predict(self, X):
        return np.array([
            int(((self.weights - x) ** 2).sum(1).argmin())
            for x in X
        ])

    def predict(self, X):
        return self._predict(X)


# sorting

def segregate_into_packs(cells_df, labels, M):
    K, packs, partition, pack_id = int(labels.max()) + 1, [], [], 0
    for k in range(K):
        idx = np.where(labels == k)[0]
        sorted_idx = idx[np.argsort(cells_df["Q_Ah"].values[idx])]
        n_full = len(sorted_idx) // M
        for p in range(n_full):
            mb = sorted_idx[p*M:(p+1)*M]
            packs.append({
                "pack_id": pack_id, "cluster_id": k,
                "cell_ids": "|".join(cells_df["cell_id"].values[mb]),
                "cell_indices": mb.tolist(), "size": M,
                "mean_Q_Ah":   round(float(cells_df["Q_Ah"].values[mb].mean()), 4),
                "mean_R0_mOhm":round(float(cells_df["R0_mOhm"].values[mb].mean()), 4),
                "mean_VOCV_V": round(float(cells_df["VOCV_V"].values[mb].mean()), 4),
                "remainder": False,
            })
            partition.append(mb); pack_id += 1
        if len(sorted_idx) % M:
            mb = sorted_idx[n_full*M:]
            packs.append({
                "pack_id": pack_id, "cluster_id": k,
                "cell_ids": "|".join(cells_df["cell_id"].values[mb]),
                "cell_indices": mb.tolist(), "size": len(mb),
                "mean_Q_Ah":   round(float(cells_df["Q_Ah"].values[mb].mean()), 4),
                "mean_R0_mOhm":round(float(cells_df["R0_mOhm"].values[mb].mean()), 4),
                "mean_VOCV_V": round(float(cells_df["VOCV_V"].values[mb].mean()), 4),
                "remainder": True,
            })
            pack_id += 1
    return packs, partition


# function call

def run_pca_som(data_dir=None, out_dir=None, n_cells=None, M=None, weights=None,
                n_pca_components=10, curve_length=100):
    data_dir = data_dir or config.DATA_DIR
    out_dir  = out_dir  or os.path.join(config.OUTPUT_DIR, "pca_som")
    n_cells  = n_cells  or config.N_CELLS
    M        = M        or config.M
    weights  = weights  or config.WEIGHTS

    # 1. load cells + voltage curves
    cells_df, curves = load_cells(data_dir, n_cells=n_cells, load_curves=True)
    N = len(cells_df)
    K = max(1, round(N / M))
    print(f"\n{ALGO_NAME}  N={N}  K={K}  M={M}")

    # 2. build voltage curve matrix (N, curve_length)
    X_curves = get_curve_matrix(cells_df, curves, length=curve_length)

    # 3. PCA 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_curves)
   
    X_scaled += np.random.default_rng(0).normal(0, 1e-6, X_scaled.shape)
    n_comp = min(n_pca_components, X_scaled.shape[1], N)
    pca    = PCA(n_components=n_comp, random_state=42)
    X_pca  = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_comp} components explain {var_explained:.1%} of curve variance")

    # 4. SOM — clustering
    t0  = time.perf_counter()
    som = SOM(K=K, n_features=n_comp, n_iter=200, lr=0.5, seed=42)
    som.fit(X_pca)
    labels  = som.predict(X_pca).astype(int)
    runtime = round(time.perf_counter() - t0, 4)
    cells_df["cluster"] = labels
    print(f"  Done in {runtime}s  cluster sizes: {np.bincount(labels).tolist()}")

    # 5. segregate into packs + evaluate on static params
    packs, partition = segregate_into_packs(cells_df, labels, M)
    full    = [p for p in packs if not p["remainder"]]
    data_raw = get_static_matrix(cells_df)
    all_idx  = np.concatenate([np.array(p["cell_indices"]) for p in full])
    data_eval = data_raw[all_idx]
    idx_map   = {old: new for new, old in enumerate(all_idx)}
    remapped  = [np.array([idx_map[i] for i in p["cell_indices"]]) for p in full]
    scores = evaluate(data_eval, remapped, weights, runtime_s=runtime, name=ALGO_NAME)

    cells_df["pack_id"] = -1
    for p in packs:
        for ci in p["cell_indices"]:
            cells_df.at[ci, "pack_id"] = p["pack_id"]

    # 6. plot + save  (pass curve_matrix so panel B shows voltage curves)
    plot_algorithm_results(
        cells_df, labels, packs,
        inertia_hist=som.loss_hist,
        scores=scores,
        out_dir=os.path.join(out_dir, "plots"),
        algo_name=ALGO_NAME,
        curve_matrix=X_curves,
    )
    save_results(cells_df, packs, scores, som.loss_hist,
                 out_dir=os.path.join(out_dir, "results"), algo_name=ALGO_NAME)

    return cells_df, packs, scores, som.loss_hist


if __name__ == "__main__":
    run_pca_som()
