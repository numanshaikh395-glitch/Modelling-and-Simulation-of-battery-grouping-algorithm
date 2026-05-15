"""
run_all.py
Runs all four algorithms in sequence, collects scores, and saves a comparison plot.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import config
from algorithms.kmeans  import run_kmeans
from algorithms.fcm     import run_fcm
from algorithms.pca_som import run_pca_som
from algorithms.dtw_knn import run_dtw_knn
from algorithms.cdm_knn import run_cdm_knn
from plotting import plot_runtime_comparison

OUT_DIR = config.OUTPUT_DIR


def main():
    print("=" * 60)
    print("  Battery Sorting — running all algorithms")
    print("=" * 60)

    all_scores = []

    # run each algorithm and collect its scores dict
    _, _, scores, _ = run_kmeans(out_dir=os.path.join(OUT_DIR, "kmeans"))
    all_scores.append(scores)

    _, _, scores, _ = run_fcm(out_dir=os.path.join(OUT_DIR, "fcm"))
    all_scores.append(scores)

    _, _, scores, _ = run_pca_som(out_dir=os.path.join(OUT_DIR, "pca_som"))
    all_scores.append(scores)

    _, _, scores, _ = run_dtw_knn(out_dir=os.path.join(OUT_DIR, "dtw_knn"))
    all_scores.append(scores)

    _, _, scores, _ = run_cdm_knn(out_dir=os.path.join(OUT_DIR, "cdm_knn"))
    all_scores.append(scores)

    # comparison plot
    plot_runtime_comparison(all_scores, out_dir=os.path.join(OUT_DIR, "comparison"))

    # print summary table
    print("\n" + "=" * 70)
    print(f"  {'Algorithm':<20} {'J score':>10} {'M1':>8} {'Runtime(s)':>12} {'Pass':>6}")
    print("  " + "-" * 66)
    for s in all_scores:
        print(f"  {s['algorithm']:<20} "
              f"{str(s.get('J_score','—')):>10} "
              f"{str(s.get('M1','—')):>8} "
              f"{str(s.get('runtime_s','—')):>12} "
              f"{str(s.get('overall_pass','—')):>6}")
    print("=" * 70)


if __name__ == "__main__":
    main()
