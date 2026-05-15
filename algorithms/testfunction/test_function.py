"""
test_function.py
Scores any battery grouping result against the mathematical model.
Call evaluate() after every algorithm to get J score + feasibility checks.
"""

import numpy as np


def _standardise(data_raw):
    mu    = data_raw.mean(0)
    sigma = data_raw.std(0)
    sigma[sigma == 0] = 1.0
    return (data_raw - mu) / sigma


def _compute_J(data_std, partition, weights):
    """Within-group variance (Eq. 3.17) — lower is better, ~1.0 = random."""
    w   = np.array(weights)
    K   = len(partition)
    ppv = np.zeros(3)
    for g in partition:
        grp  = data_std[g]
        ppv += ((grp - grp.mean(0)) ** 2).sum(0)
    ppv /= K
    return float(np.dot(w, ppv)), ppv


def _check_feasibility(data_raw, partition):
    N, K = data_raw.shape[0], len(partition)
    M    = N // K
    rep  = {}

    # C1 — every cell assigned exactly once
    all_idx        = np.concatenate(partition)
    unique, counts = np.unique(all_idx, return_counts=True)
    c1 = int(np.sum(counts != 1)) + int(len(unique) != N)
    rep["C1"] = c1
    if c1 > 0:
        return False, {**rep, "reason": "C1: duplicate or missing cells"}

    # C2 — equal group sizes (10% tolerance)
    rebal  = sum(abs(len(g) - M) for g in partition) // 2
    rep["C2_rebalance"] = rebal
    if rebal / N > 0.10:
        return False, {**rep, "reason": "C2: groups too unbalanced"}

    # C3 — mean Q >= 95% batch mean
    Q_floor = 0.95 * data_raw[:, 0].mean()
    c3 = sum(1 for g in partition if data_raw[g, 0].mean() < Q_floor)
    rep["C3"] = c3

    # C4 — mean R0 <= 110% batch mean
    R_ceil  = 1.10 * data_raw[:, 1].mean()
    c4 = sum(1 for g in partition if data_raw[g, 1].mean() > R_ceil)
    rep["C4"] = c4

    # C5 — OCV spread <= 10 mV per group
    c5 = sum(1 for g in partition if data_raw[g, 2].max() - data_raw[g, 2].min() > 0.010)
    rep["C5"] = c5

    passed     = c3 == 0 and c4 == 0 and c5 == 0
    rep["reason"] = "OK" if passed else f"C3={c3} C4={c4} C5={c5}"
    return passed, rep


def evaluate(data_raw, partition, weights=None, runtime_s=None, name="Algorithm"):
    """
    Score a partition against the mathematical model.

    Parameters
    ----------
    data_raw  : (N, 3) array  [Q_Ah, R0_mOhm, VOCV_V]
    partition : list of np.ndarray  (cell index groups, full packs only)
    weights   : [w_Q, w_R0, w_VOCV]
    runtime_s : float
    name      : algorithm label

    Returns   : dict with all metrics
    """
    if weights is None:
        weights = [0.5, 0.3, 0.2]

    N, K = data_raw.shape[0], len(partition)
    feasible, feas = _check_feasibility(data_raw, partition)

    result = {
        "algorithm" : name,
        "N"         : N,
        "K"         : K,
        "M"         : N // K,
        "feasible"  : feasible,
        "C1"        : feas.get("C1", 0),
        "C2_rebalance": feas.get("C2_rebalance", 0),
        "C3"        : feas.get("C3", 0),
        "C4"        : feas.get("C4", 0),
        "C5"        : feas.get("C5", 0),
        "reason"    : feas.get("reason", ""),
        "runtime_s" : runtime_s,
    }

    if not feasible:
        result.update({"J_score": None, "J_random": 1.0,
                        "M1": None, "M1_Q": None, "M1_R0": None, "M1_VOCV": None,
                        "M1_pass": False, "overall_pass": False})
        _print(result)
        return result

    data_std   = _standardise(data_raw)
    J, ppv     = _compute_J(data_std, partition, weights)
    J_rand     = float(np.sum(weights))   # ~1.0
    M1         = J / J_rand
    w          = np.array(weights)

    result.update({
        "J_score"   : round(J,  6),
        "J_random"  : round(J_rand, 4),
        "M1"        : round(M1, 4),
        "M1_Q"      : round(w[0] * ppv[0] / J_rand, 4),
        "M1_R0"     : round(w[1] * ppv[1] / J_rand, 4),
        "M1_VOCV"   : round(w[2] * ppv[2] / J_rand, 4),
        "M1_pass"   : bool(M1 <= 0.05),
        "overall_pass": feasible and M1 <= 0.05,
    })

    _print(result)
    return result


def _print(r):
    sep = "-" * 52
    print(f"\n{sep}")
    print(f"  {r['algorithm']}   N={r['N']}  K={r['K']}  M={r['M']}")
    print(sep)
    print(f"  Feasible      : {r['feasible']}  ({r['reason']})")
    if r["feasible"]:
        print(f"  J score       : {r['J_score']}  (random baseline = {r['J_random']})")
        print(f"  M1            : {r['M1']}   pass (<=0.05): {r['M1_pass']}")
        print(f"  M1  Q/R0/VOCV : {r['M1_Q']} / {r['M1_R0']} / {r['M1_VOCV']}")
    print(f"  Runtime       : {r['runtime_s']}s")
    print(f"  Overall pass  : {r['overall_pass']}")
    print(sep)
