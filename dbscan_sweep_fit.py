#!/usr/bin/env python3
# Improved DBSCAN sweep with k-distance ε candidates + optional PCA.
# Saves "prototype boxes" (centroid + percentile half-widths) for the best run.

import os, glob, argparse, json, math
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import joblib

# ---------- Robust NGAFID reading (template + positional fallback) ----------
NGAFID_HEADER_TEMPLATE = [
    "Lcl Date","Lcl Time","UTCOfst","AtvWpt","Latitude","Longitude","AltB","BaroA","AltMSL","OAT",
    "IAS","GndSpd","VSpd","Pitch","Roll","LatAc","NormAc","HDG","TRK","volt1","volt2","amp1","amp2",
    "FQtyL","FQtyR","E1 FFlow","E1 OilT","E1 OilP","E1 RPM","E1 CHT1","E1 CHT2","E1 CHT3","E1 CHT4",
    "E1 EGT1","E1 EGT2","E1 EGT3","E1 EGT4","AltGPS","TAS","HSIS","CRS","NAV1","NAV2","COM1","COM2",
    "HCDI","VCDI","WndSpd","WndDr","WptDst","WptBrg","MagVar","AfcsOn","RollM","PitchM","RollC","PichC",
    "VSpdG","GPSfix","HAL","VAL","HPLwas","HPLfd","VPLwas"
]
NGAFID_POS = {name: i for i, name in enumerate(NGAFID_HEADER_TEMPLATE)}

def _all_numeric(names: List[str]) -> bool:
    for c in names:
        try: float(str(c))
        except: return False
    return True

def _load_template_header(template_dir: str) -> List[str]:
    cand = sorted(glob.glob(os.path.join(template_dir, "*.csv")))
    for p in cand:
        try:
            df = pd.read_csv(p, skiprows=2, nrows=1)
            cols = [str(c).strip() for c in df.columns]
            if not _all_numeric(cols):  # real headers
                return cols
        except: pass
    return NGAFID_HEADER_TEMPLATE[:]  # fallback

def _align_by_name(df: pd.DataFrame, desired: List[str]) -> List[str]:
    cmap = {str(c).strip().lower().replace(" ",""): str(c) for c in df.columns}
    want, miss = [], []
    for s in desired:
        key = s.lower().replace(" ","")
        if key in cmap: want.append(cmap[key])
        else: miss.append(s)
    if miss:
        raise KeyError(f"Missing columns {miss}")
    return want

def read_ngafid(path: str, cols: List[str], template_dir: str) -> pd.DataFrame:
    # 1) Try NGAFID (skiprows=2) with real headers
    df = pd.read_csv(path, skiprows=2, low_memory=False)
    df.columns = df.columns.str.strip()
    try:
        use = _align_by_name(df, cols)
        out = df[use].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
        out.columns = cols
        return out
    except KeyError:
        # 2) Headerless → borrow template, then align
        raw = pd.read_csv(path, header=None, skiprows=2, low_memory=False)
        templ = _load_template_header(template_dir)
        m = raw.shape[1]
        header = templ[:m] if len(templ) >= m else templ + [f"Dummy_{i}" for i in range(m-len(templ))]
        raw.columns = header
        try:
            use = _align_by_name(raw, cols)
            out = raw[use].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
            out.columns = cols
            return out
        except KeyError:
            # 3) Final positional fallback
            missing = [c for c in cols if c not in NGAFID_POS]
            if missing:
                raise KeyError(f"{os.path.basename(path)}: cannot map {missing} to NGAFID template positions")
            idxs = [NGAFID_POS[c] for c in cols if NGAFID_POS[c] < m]
            if len(idxs) != len(cols):
                raise ValueError(f"{os.path.basename(path)}: file has {m} cols, template indices exceed width for {cols}")
            out = raw.iloc[:, idxs].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
            out.columns = cols
            return out

# ---------- windows / scaling ----------
def sliding_windows(A: np.ndarray, w: int, step: int) -> Tuple[np.ndarray, List[int]]:
    starts = list(range(0, len(A) - w + 1, step))
    if not starts: return np.empty((0,w,A.shape[1])), []
    W = np.stack([A[s:s+w, :] for s in starts], axis=0)
    return W, starts

def per_timestep_scale_then_flatten(win3d: np.ndarray, scaler) -> np.ndarray:
    # scaler should be fit on F=7 features
    n,w,f = win3d.shape
    X2 = win3d.reshape(-1, f)
    X2s = scaler.transform(X2)
    return X2s.reshape(n, w*f)

# ---------- prototypes ----------
def make_boxes_from_clusters(X: np.ndarray, labels: np.ndarray, perc: float):
    uniq = sorted([k for k in np.unique(labels) if k >= 0])
    if not uniq:
        return np.zeros((0, X.shape[1])), np.zeros((0, X.shape[1])), []
    cents, halfs, sizes = [], [], []
    for k in uniq:
        Xk = X[labels==k]
        if len(Xk)==0: continue
        c = Xk.mean(axis=0)
        d = np.abs(Xk - c[None,:])
        h = np.percentile(d, perc, axis=0)
        cents.append(c); halfs.append(h); sizes.append(len(Xk))
    return np.vstack(cents), np.vstack(halfs), sizes

# ---------- k-distance helpers ----------
def kdist_candidates(X: np.ndarray, k: int, sample: int = 2000,
                     percentiles: List[float] = [90,95,97,98,99,99.5,99.7,99.9]) -> List[float]:
    """Compute the distance to the k-th nearest neighbor on a sample, return ε percentiles."""
    if len(X) > sample:
        idx = np.random.RandomState(0).choice(len(X), size=sample, replace=False)
        Xs  = X[idx]
    else:
        Xs = X
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(Xs)
    # distances: each row sorted, col 0 is self (0.0), take col k-1
    dists = nn.kneighbors(Xs, return_distance=True)[0][:, k-1]
    dists = dists[np.isfinite(dists)]
    eps_list = [float(np.percentile(dists, p)) for p in percentiles]
    # dedupe & sort
    eps_list = sorted(set([e for e in eps_list if e > 0]))
    return eps_list

# ---------- sweep & fit ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after_dir", default="dataset/after")
    ap.add_argument("--template_dir", default="dataset/after_examm2")
    ap.add_argument("--scaler_path", default="outputs/scaler.pkl")
    ap.add_argument("--out_dir", default="outputs/dbscan")
    ap.add_argument("--columns", nargs="+", default=["AltMSL","E1 RPM","E1 FFlow","E1 CHT1","E1 EGT1","NormAc","IAS"])
    ap.add_argument("--window_size", type=int, default=30)
    ap.add_argument("--step_size", type=int, default=25)

    ap.add_argument("--use_pca", action="store_true", help="Apply PCA (retain 95%% var) before DBSCAN.")
    ap.add_argument("--pca_var", type=float, default=0.95, help="Variance to retain if --use_pca.")
    ap.add_argument("--min_samples_grid", nargs="+", type=int, default=[3,4,5,6,8,10])

    # Optional manual eps additions; otherwise chosen from k-distance percentiles
    ap.add_argument("--extra_eps", nargs="*", type=float, default=[])

    ap.add_argument("--perc", type=float, default=95.0, help="Percentile for half-widths.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scaler = joblib.load(args.scaler_path)
    assert getattr(scaler, "n_features_in_", None) == 7, "Scaler must be 7-feature per-timestep."

    # Gather AFTER windows
    files = sorted(glob.glob(os.path.join(args.after_dir, "*.csv")))
    all_win = []
    for p in files:
        try:
            df = read_ngafid(p, args.columns, args.template_dir)
        except Exception as e:
            print(f"[WARN] {os.path.basename(p)}: {e}")
            continue
        W, _ = sliding_windows(df.values.astype(float), args.window_size, args.step_size)
        if W.size: all_win.append(W)
    if not all_win:
        raise RuntimeError("No windows extracted from AFTER. Check headers/columns/window/step.")
    W3 = np.concatenate(all_win, axis=0)                 # [N, W, F]
    X  = per_timestep_scale_then_flatten(W3, scaler)     # [N, W*F]
    N, D = X.shape
    print(f"[info] Windows: N={N}, D={D}")

    # Optional PCA (strongly recommended for DBSCAN in high-D)
    pca = None
    if args.use_pca:
        pca = PCA(n_components=args.pca_var, svd_solver="full", random_state=0)
        Xp  = pca.fit_transform(X)
        print(f"[info] PCA: D={D} -> d={Xp.shape[1]} (var>={args.pca_var})")
    else:
        Xp = X

    # Build ε candidates from k-distance percentiles for each min_samples candidate
    sweep = []
    best  = None
    for ms in args.min_samples_grid:
        eps_list = kdist_candidates(Xp, k=ms, sample=min(4000, len(Xp)))
        eps_list += args.extra_eps
        eps_list = sorted(set(eps_list))
        print(f"[grid] min_samples={ms}, eps candidates ~ {eps_list[:6]}{'...' if len(eps_list)>6 else ''}")

        for eps in eps_list:
            db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
            labels = db.fit_predict(Xp)
            n_noise = int((labels==-1).sum())
            n_clu   = int(len([k for k in np.unique(labels) if k>=0]))
            noise_rate = n_noise / len(labels)

            # silhouette on core points if we have >=2 clusters
            sil = -1.0
            if n_clu >= 2:
                try:
                    core_idx = db.core_sample_indices_ if hasattr(db, "core_sample_indices_") else np.where(labels!=-1)[0]
                    if len(core_idx) > n_clu:
                        s = silhouette_samples(Xp[core_idx], labels[core_idx])
                        sil = float(np.nanmean(s))
                except Exception:
                    pass

            # fitness: prefer 2..12 clusters, low noise, higher silhouette
            fit = (0.0
                   + (1.0 if 2 <= n_clu <= 12 else 0.0)
                   + (1.0 if noise_rate < 0.5 else 0.0)
                   + max(0.0, sil))

            row = {"eps":eps, "min_samples":ms, "clusters":n_clu,
                   "noise_rate":noise_rate, "silhouette":sil, "fitness":fit}
            sweep.append(row)

            if (best is None) or (fit > best["fitness"]):
                best = {"eps":eps, "min_samples":ms, "labels":labels, "fitness":fit,
                        "clusters":n_clu, "noise_rate":noise_rate, "silhouette":sil}

    with open(os.path.join(args.out_dir, "dbscan_sweep.json"), "w") as f:
        json.dump(sweep, f, indent=2)

    if best is None or best["clusters"] == 0:
        raise SystemExit("DBSCAN still found no usable clusters. Try --use_pca and/or add --extra_eps 2.0 3.0")

    # Build prototype boxes in the same space used for clustering (Xp)
    cents, halfs, sizes = make_boxes_from_clusters(Xp, best["labels"], args.perc)

    # Save prototypes with enough info to reproduce transforms
    proto_path = os.path.join(
        args.out_dir,
        f"prototypes_dbscan_eps{best['eps']}_min{best['min_samples']}_p{int(args.perc)}.npz"
    )
    np.savez_compressed(
        proto_path,
        centroids=cents,
        halfwidths=halfs,
        columns=np.array(args.columns, dtype=object),
        window_size=np.array([args.window_size]),
        step_size=np.array([args.step_size]),
        perc=np.array([args.perc]),
        eps=np.array([best["eps"]]),
        min_samples=np.array([best["min_samples"]]),
        cluster_sizes=np.array(sizes),
        pca_components=(None if pca is None else pca.components_),
        pca_mean=(None if pca is None else pca.mean_),
        pca_explained_variance_ratio=(None if pca is None else pca.explained_variance_ratio_),
    )

    with open(os.path.join(args.out_dir, "dbscan_best.json"), "w") as f:
        json.dump({
            "best_eps": best["eps"],
            "best_min_samples": best["min_samples"],
            "clusters": best["clusters"],
            "noise_rate": best["noise_rate"],
            "silhouette": best["silhouette"],
            "proto_path": proto_path
        }, f, indent=2)

    print(f"✅ Saved prototypes → {proto_path}")
    print(f"Best: eps={best['eps']}  min_samples={best['min_samples']}  "
          f"clusters={best['clusters']}  noise={best['noise_rate']:.2f}  silhouette={best['silhouette']:.3f}")

if __name__ == "__main__":
    main()
