#!/usr/bin/env python3
import os, argparse, glob, json
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- small helpers (match dbscan_boxes.py behavior) -------------------------
def _canon(s: str):
    import re
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _align_by_name(df: pd.DataFrame, want: List[str]):
    m = { _canon(c): c for c in df.columns }
    res, miss = [], []
    for w in want:
        k = _canon(w)
        if k in m: res.append(m[k])
        else: miss.append(w)
    return res if not miss else None

def _all_numeric(names: List[str]) -> bool:
    for c in names:
        try: float(str(c))
        except Exception: return False
    return True

def read_error_window_matrix(path: str, columns: List[str]) -> np.ndarray:
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

def load_proto_npz(npz_path: str) -> Dict[str, np.ndarray]:
    return dict(np.load(npz_path, allow_pickle=True))

# --- plotting ---------------------------------------------------------------
def plot_mean_shape(cluster_id: int, S: np.ndarray, columns: List[str], out_png: str):
    # S: [N, W, F]
    meanS = S.mean(axis=0)  # [W,F]
    W, F = meanS.shape[0], meanS.shape[1]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for f in range(F):
        ax.plot(meanS[:, f], label=columns[f], linewidth=2)
    ax.set_title(f"DBSCAN narratives — cluster {cluster_id}: mean sensor shapes (N={S.shape[0]})")
    ax.set_xlabel(f"t within window (0..{W-1})"); ax.set_ylabel("scaled value (relative)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def plot_exemplar(cluster_id: int, arr: np.ndarray, columns: List[str], dist: float, rank: int, out_png: str):
    # arr: [W,F]
    W = arr.shape[0]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for f, col in enumerate(columns):
        ax.plot(arr[:, f], label=col)
    ax.set_title(f"Cluster {cluster_id} exemplar #{rank} (nearest_dist={dist:.2f})")
    ax.set_xlabel(f"t within window (0..{W-1})"); ax.set_ylabel("scaled value (relative)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error_expl_csv", required=True, help="outputs/dbscan/explanations_error.csv from dbscan_boxes.py")
    ap.add_argument("--error_dir", required=True, help="Folder with error windows (window_*.csv)")
    ap.add_argument("--prototypes_path", required=True, help="NPZ used for explaining (for metadata)")
    ap.add_argument("--columns", nargs="+", required=True)
    ap.add_argument("--exemplars", type=int, default=3)
    ap.add_argument("--out_dir", default="outputs/dbscan_vis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load explanations
    df = pd.read_csv(args.error_expl_csv)
    if not {"file","cluster_id","nearest_dist"}.issubset(df.columns):
        raise SystemExit("explanations_error.csv must have columns: file, cluster_id, nearest_dist")

    # map file -> full matrix
    files = sorted(glob.glob(os.path.join(args.error_dir, "window_*.csv")))
    ts_map = {}
    for f in files:
        base = os.path.basename(f)
        try:
            ts_map[base] = read_error_window_matrix(f, args.columns)  # [W,F]
        except Exception as e:
            print(f"[WARN] {base}: {e}")

    # per cluster
    for cid, g in df.groupby("cluster_id"):
        series = []
        keep = []
        for _, row in g.iterrows():
            base = row["file"]
            if base in ts_map:
                series.append(ts_map[base])
                keep.append((base, float(row["nearest_dist"])))
        if not series:
            continue
        S = np.stack(series, axis=0)  # [N,W,F]
        plot_mean_shape(int(cid), S, args.columns, os.path.join(args.out_dir, f"dbscan_cluster_{cid}_mean.png"))

        # Exemplars (by nearest_dist ascending)
        keep_sorted = sorted(keep, key=lambda x: x[1])[:min(args.exemplars, len(keep))]
        for rank, (base, dist) in enumerate(keep_sorted, start=1):
            arr = ts_map[base]
            plot_exemplar(int(cid), arr, args.columns, dist, rank,
                          os.path.join(args.out_dir, f"dbscan_cluster_{cid}_exemplar_{rank}.png"))

    # simple distribution of nearest_dist overall
    fig, ax = plt.subplots(figsize=(6.5,4.5))
    ax.hist(df["nearest_dist"].values, bins=40, alpha=0.8)
    ax.set_title("Nearest distance to DBSCAN centroid (ERROR windows)")
    ax.set_xlabel("nearest_dist"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "nearest_dist_hist.png"), dpi=200); plt.close(fig)

    # per-cluster counts
    cnt = df["cluster_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5,4.5))
    ax.bar(cnt.index.astype(str), cnt.values)
    ax.set_title("Error windows per cluster"); ax.set_xlabel("cluster_id"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "cluster_counts.png"), dpi=200); plt.close(fig)

    # stash a tiny meta file
    meta = dict(prototypes_path=args.prototypes_path, columns=args.columns, exemplars=args.exemplars)
    Path = __import__("pathlib").Path
    Path(os.path.join(args.out_dir, "meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"✓ Wrote visuals to {args.out_dir}")

if __name__ == "__main__":
    main()
