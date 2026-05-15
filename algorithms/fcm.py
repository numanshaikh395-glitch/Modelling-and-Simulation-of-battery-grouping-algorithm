
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import config
from data_loader import load_cells, get_static_matrix
from test_function import evaluate
from plotting import plot_algorithm_results
from excel_writer import save_results

ALGO_NAME = "Fuzzy C-Means"


# ── pack segregation (shared logic) ──────────────────────────────────────────

def segregate_into_packs(cells_df, labels, M):
    """Sort by Q within each cluster, split into packs of M."""
    K, packs, partition, pack_id = labels.max() + 1, [], [], 0
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


# fcm

def _fcm(X, K, m=2.0, max_iter=300, tol=1e-5, seed=42):

    N, F = X.shape
    rng  = np.random.default_rng(seed)
    # initialise membership matrix U (K, N) rows sum to 1
    U    = rng.dirichlet(np.ones(K), size=N).T   # (K, N)
    jm   = []

    for _ in range(max_iter):
        Um = U ** m   # (K, N)  — fuzzified memberships
        # cluster centres  (K, F)
        centres = (Um @ X) / Um.sum(axis=1, keepdims=True)
        # distances  (N, K)
        dist2 = np.array([((X - centres[k]) ** 2).sum(1) for k in range(K)]).T
        dist2 = np.maximum(dist2, 1e-10)
        # objective function
        j = float((Um.T * dist2).sum())
        jm.append(j)
        # update U
        exp   = 2.0 / (m - 1)
        new_U = np.zeros_like(U)
        for k in range(K):
            ratio  = dist2[:, k:k+1] / dist2   # (N, K)
            new_U[k] = 1.0 / (ratio ** exp).sum(1)
        # check convergence
        if np.max(np.abs(new_U - U)) < tol:
            U = new_U
            break
        U = new_U

    labels = np.argmax(U, axis=0).astype(int)
    return labels, jm



def run_fcm(data_dir=None, out_dir=None, n_cells=None, M=None, weights=None):
    data_dir = data_dir or config.DATA_DIR
    out_dir  = out_dir  or os.path.join(config.OUTPUT_DIR, "fcm")
    n_cells  = n_cells  or config.N_CELLS
    M        = M        or config.M
    weights  = weights  or config.WEIGHTS

    # 1. load static params only
    cells_df, _ = load_cells(data_dir, n_cells=n_cells, load_curves=False)
    N = len(cells_df)
    K = max(1, round(N / M))
    print(f"\n{ALGO_NAME}  N={N}  K={K}  M={M}")

    # 2. standardise
    data_raw = get_static_matrix(cells_df)
    mu  = data_raw.mean(0); sig = data_raw.std(0); sig[sig == 0] = 1
    X   = (data_raw - mu) / sig       # (N, F)

    # 3. FCM — pure numpy implementation
    t0 = time.perf_counter()
    labels, jm = _fcm(X, K, m=2.0, max_iter=300, tol=1e-5, seed=42)
    runtime = round(time.perf_counter() - t0, 4)
    cells_df["cluster"] = labels
    print(f"  Done in {runtime}s  cluster sizes: {np.bincount(labels).tolist()}")

    # 4. segregate into packs
    packs, partition = segregate_into_packs(cells_df, labels, M)
    full = [p for p in packs if not p["remainder"]]

    # 5. evaluate
    all_idx  = np.concatenate([np.array(p["cell_indices"]) for p in full])
    data_eval = data_raw[all_idx]
    idx_map   = {old: new for new, old in enumerate(all_idx)}
    remapped  = [np.array([idx_map[i] for i in p["cell_indices"]]) for p in full]
    scores = evaluate(data_eval, remapped, weights, runtime_s=runtime, name=ALGO_NAME)

    # 6. assign pack_id to cells_df
    cells_df["pack_id"] = -1
    for p in packs:
        for ci in p["cell_indices"]:
            cells_df.at[ci, "pack_id"] = p["pack_id"]

    # 7. plot + save
    plot_algorithm_results(
        cells_df, labels, packs,
        inertia_hist=list(jm),          # jm = objective function per iteration
        scores=scores,
        out_dir=os.path.join(out_dir, "plots"),
        algo_name=ALGO_NAME,
    )
    save_results(cells_df, packs, scores, list(jm),
                 out_dir=os.path.join(out_dir, "results"), algo_name=ALGO_NAME)

    return cells_df, packs, scores, list(jm)


if __name__ == "__main__":
    run_fcm()
