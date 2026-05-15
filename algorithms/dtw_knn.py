
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

import config
from data_loader import load_cells, get_static_matrix, get_curve_matrix
from test_function import evaluate
from plotting import plot_algorithm_results
from excel_writer import save_results

ALGO_NAME = "DTW-CDM + KNN"


# ── simplified DTW distance ───────────────────────────────────────────────────

def dtw_distance(a, b):

    n, m  = len(a), len(b)
    cost  = np.full((n, m), np.inf)
    cost[0, 0] = abs(a[0] - b[0])
    for i in range(1, n):
        cost[i, 0] = abs(a[i] - b[0]) + cost[i-1, 0]
    for j in range(1, m):
        cost[0, j] = abs(a[0] - b[j]) + cost[0, j-1]
    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = abs(a[i] - b[j]) + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    return cost[-1, -1]


def build_cdm(X, max_cells=300):

    N = min(len(X), max_cells)
    if N < len(X):
        print(f"  CDM: capping at {N} cells for speed (full DTW too slow for N>{max_cells})")
    X = X[:N]
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            d = dtw_distance(X[i], X[j])
            D[i, j] = d
            D[j, i] = d
    return D, N


# ── seed clusters from CDM 

def seed_clusters_from_cdm(D, K, seed=42):

    rng   = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, len(D)))]
    while len(seeds) < K:
        # next seed = cell farthest from any existing seed
        min_dist = D[:, seeds].min(axis=1)
        seeds.append(int(np.argmax(min_dist)))
    return np.array(seeds)


# pack sorting

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


# ── main 

def run_dtw_knn(data_dir=None, out_dir=None, n_cells=None, M=None, weights=None,
                curve_length=50, cdm_max=300, knn_k=5):
    data_dir = data_dir or config.DATA_DIR
    out_dir  = out_dir  or os.path.join(config.OUTPUT_DIR, "dtw_knn")
    n_cells  = n_cells  or config.N_CELLS
    M        = M        or config.M
    weights  = weights  or config.WEIGHTS

    # 1. load cells + voltage curves
    cells_df, curves = load_cells(data_dir, n_cells=n_cells, load_curves=True)
    N = len(cells_df)
    K = max(1, round(N / M))
    print(f"\n{ALGO_NAME}  N={N}  K={K}  M={M}")

    # 2. voltage curve matrix (shorter for DTW speed)
    X_curves = get_curve_matrix(cells_df, curves, length=curve_length)
    # small jitter prevents degenerate DTW distances on near-identical curves
    X_curves = X_curves + np.random.default_rng(0).normal(0, 1e-5, X_curves.shape).astype(np.float32)

    t0 = time.perf_counter()

    # 3. build CDM on a subset (DTW all-pairs is expensive)
    print("  Building DTW distance matrix ...")
    D, N_cdm = build_cdm(X_curves, max_cells=cdm_max)

    # 4. seed K clusters from farthest points in the CDM subset
    seeds = seed_clusters_from_cdm(D, K)

    # 5. assign CDM-subset cells to their nearest seed (initial labels)
    seed_curves   = X_curves[seeds]
    cdm_labels    = np.zeros(N_cdm, dtype=int)
    for i in range(N_cdm):
        cdm_labels[i] = int(np.argmin([dtw_distance(X_curves[i], seed_curves[k])
                                        for k in range(K)]))

    # 6. KNN — train on CDM subset, predict labels for all remaining cells
    knn = KNeighborsClassifier(n_neighbors=min(knn_k, N_cdm))
    knn.fit(X_curves[:N_cdm], cdm_labels)
    all_labels = knn.predict(X_curves).astype(int)
    runtime    = round(time.perf_counter() - t0, 4)

    cells_df["cluster"] = all_labels
    print(f"  Done in {runtime}s  cluster sizes: {np.bincount(all_labels).tolist()}")

    # track a simple loss: mean intra-cluster DTW distance per iteration
    
    loss_hist = [float(((X_curves - X_curves[all_labels].mean(0))**2).sum())]

    # 7. segregate + evaluate
    packs, partition = segregate_into_packs(cells_df, all_labels, M)
    full      = [p for p in packs if not p["remainder"]]
    data_raw  = get_static_matrix(cells_df)
    all_idx   = np.concatenate([np.array(p["cell_indices"]) for p in full])
    data_eval = data_raw[all_idx]
    idx_map   = {old: new for new, old in enumerate(all_idx)}
    remapped  = [np.array([idx_map[i] for i in p["cell_indices"]]) for p in full]
    scores = evaluate(data_eval, remapped, weights, runtime_s=runtime, name=ALGO_NAME)

    cells_df["pack_id"] = -1
    for p in packs:
        for ci in p["cell_indices"]:
            cells_df.at[ci, "pack_id"] = p["pack_id"]

    # 8. plot + save
    plot_algorithm_results(
        cells_df, all_labels, packs,
        inertia_hist=loss_hist,
        scores=scores,
        out_dir=os.path.join(out_dir, "plots"),
        algo_name=ALGO_NAME,
        curve_matrix=X_curves,
    )
    save_results(cells_df, packs, scores, loss_hist,
                 out_dir=os.path.join(out_dir, "results"), algo_name=ALGO_NAME)

    return cells_df, packs, scores, loss_hist


if __name__ == "__main__":
    run_dtw_knn()
