

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import config
from data_loader import load_cells, get_static_matrix, get_curve_matrix
from test_function import evaluate
from plotting import plot_algorithm_results
from excel_writer import save_results

ALGO_NAME = "CDM + KNN"


# ── pairwise distance matrix 

def build_distance_matrix(X):

    N = len(X)
    # vectorised: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    sq = (X ** 2).sum(axis=1)                    # (N,)
    D2 = sq[:, None] + sq[None, :] - 2 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)                     # numerical clamp
    return np.sqrt(D2).astype(np.float32)


# ── characteristic distribution density 

def compute_densities(D, k):

    N = len(D)
    S = np.zeros(N, dtype=np.float64)
    for i in range(N):
        row         = D[i].copy()
        row[i]      = np.inf        # exclude self
        nn_dists    = np.sort(row)[:k]
        S[i]        = 1.0 / (nn_dists.sum() + 1e-12)
    return S


# ── CDM + KNN grouping (Algorithm Steps 1-5) ─────────────────────────────────

def cdm_knn_grouping(D, S_init, M, k, eta):

    N        = D.shape[0]
    remaining = list(range(N))           # indices still available
    S        = S_init.copy()
    packs    = []
    orphans  = []
    pack_variances = []

    while len(remaining) >= M:
        # sort remaining cells by density ascending (lowest first)
        s_vals  = np.array([S[i] for i in remaining])
        order   = np.argsort(s_vals)
        seed_pos = order[0]
        seed    = remaining[seed_pos]

        # find k nearest neighbours of seed within remaining cells
        dists_to_remaining = np.array([D[seed, j] for j in remaining])
        dists_to_remaining[seed_pos] = np.inf    # exclude self
        nn_order  = np.argsort(dists_to_remaining)
        k_actual  = min(k, len(remaining) - 1)
        neighbours = [remaining[nn_order[j]] for j in range(k_actual)]

        #  fill a temporary pack
        pack = [seed]

        for candidate in neighbours:
            if len(pack) == M:
                break
            # check distance from candidate to every cell already in the pack
            max_dist = max(D[candidate, q] for q in pack)
            if max_dist <= eta:                  # Eq. 4 condition
                pack.append(candidate)

        # successful pack
        if len(pack) == M:
            packs.append(pack)
            # intra-pack variance: mean pairwise distance inside the pack
            intra = np.mean([D[pack[a], pack[b]]
                             for a in range(M) for b in range(a+1, M)])
            pack_variances.append(float(intra))
            for cell in pack:
                remaining.remove(cell)

        else:
            # seed is an orphan, remove it from the pool
            orphans.append(seed)
            remaining.remove(seed)

    # any leftover cells that never filled a pack are also orphans
    orphans.extend(remaining)

    return packs, orphans, pack_variances


# ── pack dict builder 

def build_pack_dicts(packs, orphans, cells_df):
    """Convert raw index lists into the standard pack dict format."""
    pack_dicts = []
    for pack_id, members in enumerate(packs):
        mb = np.array(members)
        pack_dicts.append({
            "pack_id"    : pack_id,
            "cluster_id" : pack_id,          # each pack is its own "cluster"
            "cell_ids"   : "|".join(cells_df["cell_id"].values[mb]),
            "cell_indices": mb.tolist(),
            "size"       : len(mb),
            "mean_Q_Ah"  : round(float(cells_df["Q_Ah"].values[mb].mean()), 4),
            "mean_R0_mOhm":round(float(cells_df["R0_mOhm"].values[mb].mean()), 4),
            "mean_VOCV_V": round(float(cells_df["VOCV_V"].values[mb].mean()), 4),
            "remainder"  : False,
        })
    for orp_id, cell in enumerate(orphans):
        pack_dicts.append({
            "pack_id"    : len(packs) + orp_id,
            "cluster_id" : -1,
            "cell_ids"   : cells_df["cell_id"].values[cell],
            "cell_indices": [cell],
            "size"       : 1,
            "mean_Q_Ah"  : round(float(cells_df["Q_Ah"].values[cell]), 4),
            "mean_R0_mOhm":round(float(cells_df["R0_mOhm"].values[cell]), 4),
            "mean_VOCV_V": round(float(cells_df["VOCV_V"].values[cell]), 4),
            "remainder"  : True,             # orphans flagged as remainder
        })
    return pack_dicts


# ── main 

def run_cdm_knn(data_dir=None, out_dir=None, n_cells=None, M=None, weights=None,
                k=None, eta=None, curve_length=100):
  
    data_dir = data_dir or config.DATA_DIR
    out_dir  = out_dir  or os.path.join(config.OUTPUT_DIR, "cdm_knn")
    n_cells  = n_cells  or config.N_CELLS
    M        = M        or config.M
    weights  = weights  or config.WEIGHTS
    k        = k        or max(M + 1, 2 * M)   # k > M as required by the paper

    # 1. load cells + voltage curves
    cells_df, curves = load_cells(data_dir, n_cells=n_cells, load_curves=True)
    N = len(cells_df)
    print(f"\n{ALGO_NAME}  N={N}  M={M}  k={k}")

    # 2. voltage curve matrix (N, curve_length)
    X_curves = get_curve_matrix(cells_df, curves, length=curve_length)

    t0 = time.perf_counter()

    # 3. pairwise distance matrix (Eq. 2)
    D = build_distance_matrix(X_curves)

    # 4. characteristic distribution densities (Eq. 1)
    S = compute_densities(D, k=min(k, N - 1))

    # 5. auto-set eta if not provided: use median of all pairwise distances
    if eta is None:
        finite_dists = D[D > 0].flatten()
        eta = float(np.percentile(finite_dists, 30))   # 30th percentile
        print(f"  Auto eta = {eta:.4f}  (30th pct of pairwise distances)")

    # 6. CDM + KNN grouping
    packs_idx, orphans, pack_variances = cdm_knn_grouping(D, S, M=M, k=k, eta=eta)
    runtime = round(time.perf_counter() - t0, 4)

    print(f"  Done in {runtime}s")
    print(f"  Successful packs : {len(packs_idx)}")
    print(f"  Orphan cells     : {len(orphans)}")

    # 7. assign labels to cells_df  (orphans get label -1)
    labels = np.full(N, -1, dtype=int)
    for pack_id, members in enumerate(packs_idx):
        for cell in members:
            labels[cell] = pack_id
    cells_df["cluster"] = labels
    cells_df["pack_id"] = labels

    # 8. build pack dicts
    packs = build_pack_dicts(packs_idx, orphans, cells_df)
    full  = [p for p in packs if not p["remainder"]]

    # 9. evaluate on static params (full packs only)
    data_raw  = get_static_matrix(cells_df)
    all_idx   = np.concatenate([np.array(p["cell_indices"]) for p in full])
    data_eval = data_raw[all_idx]
    idx_map   = {old: new for new, old in enumerate(all_idx)}
    remapped  = [np.array([idx_map[i] for i in p["cell_indices"]]) for p in full]
    scores = evaluate(data_eval, remapped, weights, runtime_s=runtime, name=ALGO_NAME)
    scores["orphan_cells"]     = len(orphans)
    scores["successful_packs"] = len(packs_idx)
    scores["grouping_failure_rate"] = round(len(orphans) / N, 4) if N else 0

    # 10. plot     
    plot_labels = np.maximum(labels, 0)
    plot_algorithm_results(
        cells_df, plot_labels, packs,
        inertia_hist=pack_variances,
        scores=scores,
        out_dir=os.path.join(out_dir, "plots"),
        algo_name=ALGO_NAME,
        curve_matrix=X_curves,
    )

    # 11. save Excel
    save_results(cells_df, packs, scores, pack_variances,
                 out_dir=os.path.join(out_dir, "results"), algo_name=ALGO_NAME)

    return cells_df, packs, scores, pack_variances


if __name__ == "__main__":
    run_cdm_knn()
