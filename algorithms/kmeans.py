

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import config
from data_loader import load_cells, get_static_matrix
from test_function import evaluate
from plotting import plot_algorithm_results
from excel_writer import save_results

ALGO_NAME = "K-Means"


def _standardise_weighted(data_raw, weights):
    mu  = data_raw.mean(0); sig = data_raw.std(0); sig[sig == 0] = 1
    return (data_raw - mu) / sig * np.sqrt(weights)


def _kmeans_fit(X, K, max_iter=300, n_init=10, seed=42):
   
    N   = len(X)
    rng = np.random.default_rng(seed)
    best_labels, best_inertia, best_hist = None, np.inf, []

    for _ in range(n_init):
        # K-Means++ init
        c = [X[rng.integers(0, N)]]
        for _ in range(1, K):
            d2  = np.array([min(np.sum((x - ci)**2) for ci in c) for x in X])
            c.append(X[rng.choice(N, p=d2 / d2.sum())])
        centres = np.array(c)
        hist    = []

        for _ in range(max_iter):
            diff   = X[:, None] - centres[None]
            dist2  = (diff**2).sum(2)
            labels = dist2.argmin(1)
            inert  = float(dist2[np.arange(N), labels].sum())
            hist.append(inert)
            new_c  = np.array([X[labels == k].mean(0) if (labels == k).any()
                                else X[rng.integers(0, N)] for k in range(K)])
            if np.max(np.linalg.norm(new_c - centres, axis=1)) < 1e-6:
                break
            centres = new_c

        if inert < best_inertia:
            best_inertia, best_labels, best_hist = inert, labels.copy(), hist.copy()

    return best_labels, best_hist


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


def run_kmeans(data_dir=None, out_dir=None, n_cells=None, M=None, weights=None):
    data_dir = data_dir or config.DATA_DIR
    out_dir  = out_dir  or os.path.join(config.OUTPUT_DIR, "kmeans")
    n_cells  = n_cells  or config.N_CELLS
    M        = M        or config.M
    weights  = weights  or config.WEIGHTS

    cells_df, _ = load_cells(data_dir, n_cells=n_cells, load_curves=False)
    N = len(cells_df)
    K = max(1, round(N / M))
    print(f"\n{ALGO_NAME}  N={N}  K={K}  M={M}")

    data_raw = get_static_matrix(cells_df)
    X        = _standardise_weighted(data_raw, np.array(weights))

    t0 = time.perf_counter()
    labels, hist = _kmeans_fit(X, K)
    runtime      = round(time.perf_counter() - t0, 4)
    cells_df["cluster"] = labels
    print(f"  Done in {runtime}s  cluster sizes: {np.bincount(labels).tolist()}")

    packs, partition = segregate_into_packs(cells_df, labels, M)
    full      = [p for p in packs if not p["remainder"]]
    all_idx   = np.concatenate([np.array(p["cell_indices"]) for p in full])
    data_eval = data_raw[all_idx]
    idx_map   = {old: new for new, old in enumerate(all_idx)}
    remapped  = [np.array([idx_map[i] for i in p["cell_indices"]]) for p in full]
    scores = evaluate(data_eval, remapped, weights, runtime_s=runtime, name=ALGO_NAME)

    cells_df["pack_id"] = -1
    for p in packs:
        for ci in p["cell_indices"]:
            cells_df.at[ci, "pack_id"] = p["pack_id"]

    plot_algorithm_results(cells_df, labels, packs, hist, scores,
                           out_dir=os.path.join(out_dir, "plots"), algo_name=ALGO_NAME)
    save_results(cells_df, packs, scores, hist,
                 out_dir=os.path.join(out_dir, "results"), algo_name=ALGO_NAME)

    return cells_df, packs, scores, hist


if __name__ == "__main__":
    run_kmeans()
