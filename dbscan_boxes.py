#!/usr/bin/env python3
# dbscan_boxes.py — DBSCAN “prototype boxes” with optional PCA projection
# - Fit on AFTER windows (healthy reference)
# - Explain BEFORE windows (assign cluster, nearest distance)
# - Explain ERROR windows (EXAMM anomaly windows)
#
# This version is PCA-aware: if the prototypes npz includes PCA params, we
# project flattened windows to that space before distance computations.

import os, glob, argparse, json
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

# -------- NGAFID helpers (robust header handling) ----------------------------

NGAFID_HEADER_TEMPLATE = [
    "Lcl Date","Lcl Time","UTCOfst","AtvWpt","Latitude","Longitude","AltB","BaroA","AltMSL","OAT",
    "IAS","GndSpd","VSpd","Pitch","Roll","LatAc","NormAc","HDG","TRK","volt1","volt2","amp1","amp2",
    "FQtyL","FQtyR","E1 FFlow","E1 OilT","E1 OilP","E1 RPM","E1 CHT1","E1 CHT2","E1 CHT3","E1 CHT4",
    "E1 EGT1","E1 EGT2","E1 EGT3","E1 EGT4","AltGPS","TAS","HSIS","CRS","NAV1","NAV2","COM1","COM2",
    "HCDI","VCDI","WndSpd","WndDr","WptDst","WptBrg","MagVar","AfcsOn","RollM","PitchM","RollC","PichC",
    "VSpdG","GPSfix","HAL","VAL","HPLwas","HPLfd","VPLwas"
]
NGAFID_POS = {name: i for i, name in enumerate(NGAFID_HEADER_TEMPLATE)}

def _canon(s: str) -> str:
    import re
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _all_numeric(names: List[str]) -> bool:
    for c in names:
        try:
            float(str(c))
        except Exception:
            return False
    return True

def _load_template_header(template_dir: str) -> List[str]:
    cand = sorted(glob.glob(os.path.join(template_dir, "*.csv")))
    for p in cand:
        try:
            df = pd.read_csv(p, skiprows=2, nrows=1)
            cols = [str(c).strip() for c in df.columns]
            if not _all_numeric(cols):
                return cols
        except Exception:
            pass
    return NGAFID_HEADER_TEMPLATE[:]

def _align_by_name(df: pd.DataFrame, want: List[str]) -> Optional[List[str]]:
    m = { _canon(c): c for c in df.columns }
    res, miss = [], []
    for w in want:
        k = _canon(w)
        if k in m: res.append(m[k])
        else: miss.append(w)
    return res if not miss else None

def read_ngafid(path: str, columns: List[str], template_dir: Optional[str]=None,
                force_template: bool=False) -> pd.DataFrame:
    """
    Robust NGAFID reader:
      - Try normal NGAFID (skiprows=2) + name alignment
      - If fails or force_template, read headerless (skiprows=2) and assign a template header
      - If still no names, select by fixed NGAFID positions
    """
    fname = os.path.basename(path)
    try:
        if not force_template:
            df = pd.read_csv(path, skiprows=2)
            df.columns = df.columns.str.strip()
            cols = _align_by_name(df, columns)
            if cols is not None:
                sub = df[cols].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
                sub.columns = columns
                return sub
    except Exception:
        pass

    # headerless route
    df_raw = pd.read_csv(path, header=None, skiprows=2)
    templ = _load_template_header(template_dir or ".")
    m = df_raw.shape[1]
    if len(templ) >= m:
        header = [str(x).strip() for x in templ[:m]]
    else:
        header = [str(x).strip() for x in templ] + [f"Dummy_{i}" for i in range(m - len(templ))]
    df_raw.columns = header

    cols = _align_by_name(df_raw, columns)
    if cols is not None:
        sub = df_raw[cols].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
        sub.columns = columns
        return sub

    # final fallback: select by template positions
    missing = [c for c in columns if c not in NGAFID_POS]
    if missing:
        raise ValueError(f'Missing columns {columns} in {fname}')
    idxs = [NGAFID_POS[c] for c in columns]
    if any(i >= m for i in idxs):
        raise ValueError(f'File {fname} has only {m} columns; some requested columns exceed width.')
    sub = df_raw.iloc[:, idxs].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
    sub.columns = columns
    return sub

# -------- windowing & scaling -------------------------------------------------

def sliding_windows(arr: np.ndarray, w: int, step: int) -> Tuple[np.ndarray, List[int]]:
    starts = list(range(0, len(arr) - w + 1, step))
    if not starts: return np.empty((0, w, arr.shape[1])), []
    win = np.stack([arr[s:s+w, :] for s in starts], axis=0)
    return win, starts

def scale_and_flatten(win3d: np.ndarray, scaler) -> np.ndarray:
    """
    Per-timestep scaling: scaler expects F features; we scale each row,
    then flatten to [N, W*F].
    """
    if win3d.size == 0: return np.empty((0, 0))
    n, w, f = win3d.shape
    need = getattr(scaler, "n_features_in_", None)
    if need not in (None, f):
        raise ValueError(f"Scaler expects {need} features, but F={f}.")
    Z = scaler.transform(win3d.reshape(-1, f)).reshape(n, w, f)
    return Z.reshape(n, w*f)

# -------- PCA projection helpers ---------------------------------------------

def load_proto_npz(npz_path: str) -> Dict[str, np.ndarray]:
    d = dict(np.load(npz_path, allow_pickle=True))
    return d

def maybe_project(X_flat: np.ndarray, proto_npz: Dict[str, np.ndarray]) -> np.ndarray:
    """
    If prototypes were trained with PCA, the npz should include PCA params.
    Supported keys (any one pair):
      - pca_mean / pca_components
      - proj_mean / proj_components
      - pca_mu   / pca_W
    If found, project X_flat -> Z (same dim as centroids).
    """
    keys = proto_npz.keys()
    mean_key, comp_key = None, None
    for mk, ck in [
        ("pca_mean", "pca_components"),
        ("proj_mean", "proj_components"),
        ("pca_mu",   "pca_W"),
    ]:
        if mk in keys and ck in keys:
            mean_key, comp_key = mk, ck
            break

    if mean_key is None:
        # No PCA in prototypes; return original
        return X_flat

    mu = np.asarray(proto_npz[mean_key], dtype=float).reshape(1, -1)  # [1, D]
    W  = np.asarray(proto_npz[comp_key], dtype=float)                  # [d, D]
    # Project: (X - mu) @ W.T   where W rows are principal axes
    return (X_flat - mu) @ W.T

# -------- distances -----------------------------------------------------------

def nearest_dists(X: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # X: [N, d], centroids: [K, d]
    # returns (min_dists[N], argmin_ids[N])
    if X.size == 0: return np.empty((0,)), np.empty((0,), dtype=int)
    # (x - c)^2 = x^2 + c^2 - 2 x.c
    x2 = (X * X).sum(axis=1, keepdims=True)          # [N,1]
    c2 = (centroids * centroids).sum(axis=1)         # [K]
    xc = X @ centroids.T                              # [N,K]
    d2 = x2 + c2[None, :] - 2.0 * xc                  # [N,K]
    idx = d2.argmin(axis=1)
    return np.sqrt(d2[np.arange(len(X)), idx]), idx

# -------- FIT ----------------------------------------------------------------

def fit_dbscan(after_dir: str,
               scaler_path: str,
               out_dir: str,
               columns: List[str],
               window_size: int,
               step_size: int,
               eps: float,
               min_samples: int,
               perc: float = 95.0,
               template_dir: Optional[str] = None,
               force_template: bool = False,
               use_pca: bool = False,
               pca_var: float = 0.95) -> str:
    """
    Fit DBSCAN on AFTER windows; optionally apply PCA before clustering.
    Saves:
      - prototypes_<...>.npz with keys:
          centroids[K,d], cluster_labels[N], assign_file, assign_start, assign_idx
          scaler_path, window_size, step_size, columns
          (optional) pca_mean, pca_components
    """
    os.makedirs(out_dir, exist_ok=True)
    scaler = joblib.load(scaler_path)

    files = sorted(glob.glob(os.path.join(after_dir, "*.csv")))
    X_all = []
    file_tags = []
    start_idx = []
    for f in files:
        try:
            df = read_ngafid(f, columns, template_dir, force_template)
        except Exception as e:
            print(f'[WARN] {os.path.basename(f)}: "{e}"')
            continue
        arr = df.values.astype(float)
        win, starts = sliding_windows(arr, window_size, step_size)
        if win.size == 0: continue
        X_flat = scale_and_flatten(win, scaler)  # [N, 210]
        X_all.append(X_flat)
        file_tags.extend([os.path.basename(f)] * len(starts))
        start_idx.extend(starts)
    if not X_all:
        raise RuntimeError("No windows extracted. Check columns/window/step.")
    X = np.vstack(X_all)  # [N, 210]

    # percentile clip (optional robustness)
    if perc is not None:
        lo, hi = np.percentile(X, (100 - perc, perc))
        X = np.clip(X, lo, hi)

    # PCA if requested
    pca = None
    Z = X
    if use_pca:
        pca = PCA(n_components=pca_var, svd_solver="full")
        Z = pca.fit_transform(X)

    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z)
    labels = db.labels_
    core = labels >= 0
    n_clusters = len(set(labels[labels >= 0]))

    # centroids per cluster in Z-space
    cents = []
    for k in sorted(set(labels)):
        if k < 0: continue
        cents.append(Z[labels == k].mean(axis=0))
    centroids = np.array(cents) if cents else np.empty((0, Z.shape[1]))

    # Save
    stub = f"prototypes_dbscan_eps{eps}_min{min_samples}_p{int(perc)}"
    npz_path = os.path.join(out_dir, f"{stub}.npz")
    np.savez_compressed(
        npz_path,
        centroids=centroids,
        cluster_labels=labels,
        assign_file=np.array(file_tags, dtype=object),
        assign_start=np.array(start_idx, dtype=int),
        window_size=window_size,
        step_size=step_size,
        columns=np.array(columns, dtype=object),
        scaler_path=scaler_path,
        eps=eps,
        min_samples=min_samples,
        percentile=perc,
        pca_mean=(pca.mean_ if pca is not None else np.array([])),
        pca_components=(pca.components_ if pca is not None else np.array([])),
    )

    # metadata/sweep info
    meta = {
        "n_windows": int(Z.shape[0]),
        "dim": int(Z.shape[1]),
        "n_clusters": int(n_clusters),
        "noise_windows": int((labels < 0).sum()),
        "eps": eps,
        "min_samples": min_samples,
        "percentile": perc,
        "columns": columns,
        "window_size": window_size,
        "step_size": step_size,
        "scaler_path": scaler_path,
        "note": "DBSCAN clusters in Z-space (PCA if provided)."
    }
    Path(os.path.join(out_dir, f"{stub}_meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"✓ Saved {npz_path}  (clusters={n_clusters}, dim={meta['dim']}, noise={(labels<0).mean():.2f})")
    return npz_path

# -------- EXPLAIN (BEFORE) ---------------------------------------------------

def explain_folder(input_dir: str,
                   scaler_path: str,
                   prototypes_path: str,
                   columns: List[str],
                   window_size: int,
                   step_size: int,
                   out_csv: str,
                   template_dir: Optional[str]=None,
                   force_template: bool=False):
    """
    Assign each BEFORE window to nearest DBSCAN centroid and compute distance.
    If prototypes include PCA, project windows the same way.
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    scaler = joblib.load(scaler_path)
    proto = load_proto_npz(prototypes_path)
    C = np.asarray(proto["centroids"], dtype=float)  # [K, d]
    if C.size == 0:
        print("No clusters in prototypes. Nothing to explain.")
        return

    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    rows = []
    for f in files:
        base = os.path.basename(f)
        try:
            df = read_ngafid(f, columns, template_dir, force_template)
        except Exception as e:
            print(f'[WARN] {base}: "{e}"')
            continue
        arr = df.values.astype(float)
        win, starts = sliding_windows(arr, window_size, step_size)
        if win.size == 0: continue

        X_flat = scale_and_flatten(win, scaler)              # [N, 210]
        Z = maybe_project(X_flat, proto)                     # [N, d] (maybe 210 if no PCA)
        if Z.shape[1] != C.shape[1]:
            print(f'[WARN] {base}: "feature dim mismatch: Z={Z.shape[1]} vs C={C.shape[1]}"')
            continue
        dists, idx = nearest_dists(Z, C)
        for s, di, ki in zip(starts, dists, idx):
            rows.append({"file": base, "window_idx": len(rows), "start_idx": int(s),
                        "cluster_id": int(ki), "nearest_dist": float(di)})

    if not rows:
        print("No windows explained (empty).")
        return
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✓ Wrote {out_csv}  ({len(rows)} rows)")

# -------- EXPLAIN ERROR (EXAMM windows) --------------------------------------

def read_error_window_matrix(path: str, columns: List[str]) -> np.ndarray:
    df = pd.read_csv(path)
    if _all_numeric(list(df.columns)):  # headerless 7 columns
        if df.shape[1] != len(columns):
            raise ValueError(f"{os.path.basename(path)} has {df.shape[1]} columns, expected {len(columns)}.")
        df.columns = columns
        return df.apply(pd.to_numeric, errors="coerce").values.astype(float)
    # name align
    cols = _align_by_name(df, columns)
    if cols is None:
        raise ValueError(f"Missing expected columns in {os.path.basename(path)}")
    return df[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)

def explain_error(error_dir: str,
                  scaler_path: str,
                  prototypes_path: str,
                  columns: List[str],
                  window_size: int,
                  out_csv: str):
    """
    Assign each EXAMM error window (30x7 CSV) to nearest centroid.
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    scaler = joblib.load(scaler_path)
    proto = load_proto_npz(prototypes_path)
    C = np.asarray(proto["centroids"], dtype=float)
    if C.size == 0:
        print("No clusters in prototypes. Nothing to explain.")
        return

    files = sorted(glob.glob(os.path.join(error_dir, "window_*.csv")))
    rows = []
    for f in files:
        base = os.path.basename(f)
        try:
            M = read_error_window_matrix(f, columns)  # [30,7]
        except Exception as e:
            print(f'[WARN] {base}: "{e}"')
            continue
        if M.shape[0] != window_size:
            print(f'[WARN] {base}: "unexpected window size {M.shape[0]}"')
            continue
        X_flat = scaler.transform(M).reshape(1, -1)   # [1,210]
        Z = maybe_project(X_flat, proto)              # [1,d]
        if Z.shape[1] != C.shape[1]:
            print(f'[WARN] {base}: "feature dim mismatch: Z={Z.shape[1]} vs C={C.shape[1]}"')
            continue
        dists, idx = nearest_dists(Z, C)
        rows.append({"file": base, "cluster_id": int(idx[0]), "nearest_dist": float(dists[0])})

    if not rows:
        print("No error windows explained (empty).")
        return
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✓ Wrote {out_csv}  ({len(rows)} rows)")

# -------- CLI ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_fit = sub.add_parser("fit", help="Fit DBSCAN on AFTER windows; save prototypes npz.")
    ap_fit.add_argument("--after_dir", required=True)
    ap_fit.add_argument("--scaler_path", required=True)
    ap_fit.add_argument("--out_dir", required=True)
    ap_fit.add_argument("--columns", nargs="+", required=True)
    ap_fit.add_argument("--window_size", type=int, default=30)
    ap_fit.add_argument("--step_size", type=int, default=25)
    ap_fit.add_argument("--eps", type=float, required=True)
    ap_fit.add_argument("--min_samples", type=int, required=True)
    ap_fit.add_argument("--perc", type=float, default=95.0)
    ap_fit.add_argument("--template_dir", default=None)
    ap_fit.add_argument("--force_template", action="store_true")
    ap_fit.add_argument("--use_pca", action="store_true")
    ap_fit.add_argument("--pca_var", type=float, default=0.95)

    ap_exp = sub.add_parser("explain", help="Explain BEFORE windows with saved DBSCAN prototypes.")
    ap_exp.add_argument("--input_dir", required=True)
    ap_exp.add_argument("--scaler_path", required=True)
    ap_exp.add_argument("--prototypes_path", required=True)
    ap_exp.add_argument("--columns", nargs="+", required=True)
    ap_exp.add_argument("--window_size", type=int, default=30)
    ap_exp.add_argument("--step_size", type=int, default=25)
    ap_exp.add_argument("--out_csv", required=True)
    ap_exp.add_argument("--template_dir", default=None)
    ap_exp.add_argument("--force_template", action="store_true")

    ap_expe = sub.add_parser("explain_error", help="Explain EXAMM error windows with DBSCAN prototypes.")
    ap_expe.add_argument("--error_dir", required=True)
    ap_expe.add_argument("--scaler_path", required=True)
    ap_expe.add_argument("--prototypes_path", required=True)
    ap_expe.add_argument("--columns", nargs="+", required=True)
    ap_expe.add_argument("--window_size", type=int, default=30)
    ap_expe.add_argument("--out_csv", required=True)

    args = ap.parse_args()
    if args.cmd == "fit":
        fit_dbscan(args.after_dir, args.scaler_path, args.out_dir, args.columns,
                   args.window_size, args.step_size, args.eps, args.min_samples,
                   args.perc, args.template_dir, args.force_template,
                   args.use_pca, args.pca_var)
    elif args.cmd == "explain":
        explain_folder(args.input_dir, args.scaler_path, args.prototypes_path,
                       args.columns, args.window_size, args.step_size,
                       args.out_csv, args.template_dir, args.force_template)
    elif args.cmd == "explain_error":
        explain_error(args.error_dir, args.scaler_path, args.prototypes_path,
                      args.columns, args.window_size, args.out_csv)

if __name__ == "__main__":
    main()
