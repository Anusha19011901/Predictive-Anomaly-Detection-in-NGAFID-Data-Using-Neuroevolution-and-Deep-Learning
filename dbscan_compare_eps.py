#!/usr/bin/env python3
"""
@file dbscan_compare_eps.py
@brief Compare multiple DBSCAN prototype runs across different eps/min settings using ERROR windows.

This script evaluates the robustness and quality of DBSCAN clustering runs (e.g., eps=2.0, eps=2.1, eps=2.2)
by applying each prototype set to the SAME collection of pre-extracted anomaly windows (“ERROR windows”).
It computes nearest centroid distances, noise rates, and cluster counts to determine which DBSCAN
hyperparameters yield the most stable anomaly boxes.

**Workflow Summary**
1. Load all ERROR windows in `error_dir` (each is a W×F matrix extracted from anomalous subsequences).
2. Scale each window using the provided scaler (per-timestep or flattened).
3. Flatten each window to W×F = 210 dimensions.
4. For every prototype *.npz file matched by `prototypes_glob`:
   - Load centroids, half-widths, and optional PCA components.
   - Project scaled windows if PCA exists.
   - Compute nearest-centroid distances.
   - Extract cluster counts and noise rates (if saved).
5. Produce diagnostic plots:
   - Boxplot of nearest centroid distance per run.
   - Median-distance curve across runs.
   - Cluster-count bar chart.
   - Noise-rate bar chart.
6. Save a CSV (`compare_summary.csv`) containing statistics per run:
   - number of error windows
   - mean/median nearest distance
   - interquartile range (Q25–Q75)
   - number of clusters
   - noise rate

This tool is used to systematically compare DBSCAN “box” configurations and select the most
stable epsilon/minPts combination for anomaly detection in NGAFID flight data.
"""

import os, glob, argparse, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# re-use a couple helpers
def _canon(s: str):
    import re as _re
    return _re.sub(r'[^a-z0-9]', '', str(s).lower())

def _align_by_name(df: pd.DataFrame, want):
    m = { _canon(c): c for c in df.columns }
    res, miss = [], []
    for w in want:
        k = _canon(w)
        if k in m: res.append(m[k])
        else: miss.append(w)
    return res if not miss else None

def _all_numeric(names) -> bool:
    for c in names:
        try: float(str(c))
        except Exception: return False
    return True

def read_error_window_matrix(path: str, columns):
    df = pd.read_csv(path)
    if _all_numeric(list(df.columns)):  # headerless 7 columns
        if df.shape[1] != len(columns):
            raise ValueError(f"{os.path.basename(path)} has {df.shape[1]} columns, expected {len(columns)}.")
        df.columns = columns
        return df.apply(pd.to_numeric, errors="coerce").values.astype(float)
    cols = _align_by_name(df, columns)
    if cols is None:
        raise ValueError(f"Missing expected columns in {os.path.basename(path)}")
    return df[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)

def load_proto(npz_path):
    return dict(np.load(npz_path, allow_pickle=True))

def maybe_project(X_flat, proto):
    keys = proto.keys()
    mean_key, comp_key = None, None
    for mk, ck in [("pca_mean","pca_components"), ("proj_mean","proj_components"), ("pca_mu","pca_W")]:
        if mk in keys and ck in keys:
            mean_key, comp_key = mk, ck
            break
    if mean_key is None: return X_flat
    mu = np.asarray(proto[mean_key], dtype=float).reshape(1,-1)
    W  = np.asarray(proto[comp_key], dtype=float)
    return (X_flat - mu) @ W.T

def nearest_dists(X, C):
    x2 = (X*X).sum(axis=1, keepdims=True)
    c2 = (C*C).sum(axis=1)
    xc = X @ C.T
    d2 = x2 + c2[None,:] - 2.0*xc
    idx = d2.argmin(axis=1)
    return np.sqrt(d2[np.arange(len(X)), idx]), idx

def parse_stub(npz_path: str) -> str:
    # pull eps/min from filename if present
    m = re.search(r"eps([0-9.]+)_min(\d+)", os.path.basename(npz_path))
    if m: return f"eps{m.group(1)}_min{m.group(2)}"
    return os.path.splitext(os.path.basename(npz_path))[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error_dir", required=True)
    ap.add_argument("--scaler_path", required=True)
    ap.add_argument("--prototypes_glob", required=True,
                    help='Glob for multiple npz, e.g. "outputs/dbscan/prototypes_dbscan_eps*.npz"')
    ap.add_argument("--columns", nargs="+", required=True)
    ap.add_argument("--window_size", type=int, default=30)
    ap.add_argument("--out_dir", default="outputs/dbscan_compare")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from joblib import load
    scaler = load(args.scaler_path)

    # cache all error windows once
    files = sorted(glob.glob(os.path.join(args.error_dir, "window_*.csv")))
    M_list = []
    for f in files:
        try:
            M = read_error_window_matrix(f, args.columns)  # [W,F]
            if M.shape[0] != args.window_size: continue
            M_list.append(M)
        except Exception as e:
            print(f"[WARN] {os.path.basename(f)}: {e}")
    if not M_list:
        raise SystemExit("No error windows loaded.")
    # stack & scale to [N, 210]
    X210 = np.vstack([scaler.transform(M).reshape(1,-1) for M in M_list])

    # iterate over all prototypes npz
    records = []
    dist_map = {}  # stub -> nearest_dist array
    cluster_count = {}
    noise_rate = {}
    for npz_path in sorted(glob.glob(args.prototypes_glob)):
        proto = load_proto(npz_path)
        C = np.asarray(proto["centroids"], dtype=float)
        if C.size == 0:
            print(f"[INFO] {os.path.basename(npz_path)} has 0 clusters; skipping.")
            continue
        Z = maybe_project(X210, proto)
        if Z.shape[1] != C.shape[1]:
            print(f"[WARN] dim mismatch for {os.path.basename(npz_path)}: Z={Z.shape[1]} vs C={C.shape[1]} (skip)")
            continue
        d, _ = nearest_dists(Z, C)
        stub = parse_stub(npz_path)
        dist_map[stub] = d

        # training noise from saved labels (on AFTER)
        lbls = np.asarray(proto.get("cluster_labels", []))
        if lbls.size:
            noise_rate[stub] = float((lbls < 0).mean())
            cluster_count[stub] = int(len(set(lbls[lbls>=0])))
        else:
            noise_rate[stub] = np.nan
            cluster_count[stub] = int(C.shape[0])

        records.append({"run": stub, "median_dist": float(np.median(d)), "mean_dist": float(d.mean())})

    if not records:
        raise SystemExit("No comparable runs found.")

    # 1) Boxplot of nearest_dist across runs
    labels = list(dist_map.keys())
    data = [dist_map[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Nearest distance to centroid (ERROR windows) across DBSCAN runs")
    ax.set_xlabel("run (eps_min)"); ax.set_ylabel("nearest_dist")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "nearest_dist_boxplots.png"), dpi=200); plt.close(fig)

    # 2) Median distance curve
    dfm = pd.DataFrame(records).sort_values("run")
    fig, ax = plt.subplots(figsize=(10,4.2))
    ax.plot(dfm["run"], dfm["median_dist"], marker="o")
    ax.set_title("Median nearest_dist vs run")
    ax.set_xlabel("run"); ax.set_ylabel("median nearest_dist")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "median_dist_vs_run.png"), dpi=200); plt.close(fig)

    # 3) Cluster counts & noise rate bars
    cc = pd.Series(cluster_count).sort_index()
    nr = pd.Series(noise_rate).sort_index()

    fig, ax = plt.subplots(figsize=(10,4.2))
    ax.bar(cc.index, cc.values)
    ax.set_title("Number of clusters per run (from fit)")
    ax.set_xlabel("run"); ax.set_ylabel("# clusters")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "clusters_per_run.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,4.2))
    ax.bar(nr.index, nr.values)
    ax.set_title("Noise rate per run (from fit)")
    ax.set_xlabel("run"); ax.set_ylabel("noise rate")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "noise_rate_per_run.png"), dpi=200); plt.close(fig)

    # Save a csv summary
    # include also a robustness summary (IQR / spread)
    rows = []
    for k, d in dist_map.items():
        rows.append({
            "run": k,
            "n_error_windows": int(len(d)),
            "mean_nearest_dist": float(d.mean()),
            "median_nearest_dist": float(np.median(d)),
            "q25": float(np.percentile(d, 25)),
            "q75": float(np.percentile(d, 75)),
            "clusters": int(cc.get(k, np.nan)),
            "noise_rate": float(nr.get(k, np.nan))
        })
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "compare_summary.csv"), index=False)
    print(f"✓ Wrote comparison plots & summary to {args.out_dir}")

if __name__ == "__main__":
    main()
