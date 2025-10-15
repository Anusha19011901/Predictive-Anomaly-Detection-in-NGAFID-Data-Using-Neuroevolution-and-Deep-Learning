# prototype_boxes_errors.py
# Prototype-box XAI for EXAMM error windows (one window per CSV).

import os, re, glob, json, argparse
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib

# -----------------------------
# Defaults
# -----------------------------
OUT_DIR_DEFAULT = "outputs"
NORMAL_DIR_DEFAULT = "exact_data/normal"   # AFTER error windows
ANOMALY_DIR_DEFAULT = "exact_data/anomaly" # BEFORE error windows

COLUMNS_TO_USE_DEFAULT: List[str] = [
    "AltMSL", "E1 RPM", "E1 FFlow", "E1 CHT1", "E1 EGT1", "NormAc", "IAS"
]
WINDOW_SIZE_DEFAULT: int = 30
K_DEFAULT = 8
PERC_DEFAULT = 95.0

# -----------------------------
# Helpers
# -----------------------------
def list_window_files(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "window_*.csv")))

def read_error_window_vector(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    arr = df.values
    return arr.reshape(-1)

# -----------------------------
# Fit on AFTER error windows
# -----------------------------
def fit_prototypes_on_errors(
    normal_dir: str,
    scaler_path: str,
    out_dir: str,
    columns: List[str],
    window_size: int,
    k: int,
    perc: float,
    random_state: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    scaler = joblib.load(scaler_path) if (scaler_path and os.path.exists(scaler_path)) else None

    files = list_window_files(normal_dir)
    if not files:
        raise RuntimeError(f"No window_*.csv files in {normal_dir}")

    X_list = []
    for f in files:
        v = read_error_window_vector(f)
        X_list.append(v)
    X = np.vstack(X_list)

    if scaler is not None:
        X = scaler.transform(X)

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
    centroids = km.cluster_centers_
    labels = km.labels_

    halfwidths = np.zeros_like(centroids)
    for c_id in range(k):
        Xc = X[labels == c_id]
        if Xc.size == 0:
            continue
        deltas = np.abs(Xc - centroids[c_id])
        halfwidths[c_id] = np.percentile(deltas, perc, axis=0)

    np.savez_compressed(
        os.path.join(out_dir, "prototypes_errors.npz"),
        centroids=centroids,
        halfwidths=halfwidths,
        columns=np.array(columns, dtype=object),
        window_size=np.array([window_size]),
        k=np.array([k]),
        perc=np.array([perc]),
        note=np.array(["Prototypes fit on EXAMM error windows (AFTER)"], dtype=object),
    )
    meta = {
        "n_windows": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "k": int(k),
        "percentile": float(perc),
        "columns": columns,
        "window_size": window_size,
        "scaler_path": scaler_path,
        "train_dir": normal_dir
    }
    with open(os.path.join(out_dir, "prototypes_errors_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Prototypes (errors) fitted on {X.shape[0]} windows (D={X.shape[1]}), K={k}, perc={perc}.")
    print(f"Saved: {os.path.join(out_dir,'prototypes_errors.npz')}, {os.path.join(out_dir,'prototypes_errors_meta.json')}")

# -----------------------------
# Explain BEFORE error windows
# -----------------------------
def explain_error_folder(
    in_dir: str,
    scaler_path: str,
    prototypes_path: str,
    out_csv: str,
    columns: List[str],
    window_size: int,
):
    data = np.load(prototypes_path, allow_pickle=True)
    centroids = data["centroids"]
    halfwidths = data["halfwidths"]
    F = len(columns)
    D_expected = window_size * F

    scaler = joblib.load(scaler_path) if (scaler_path and os.path.exists(scaler_path)) else None

    files = list_window_files(in_dir)
    if not files:
        raise RuntimeError(f"No window_*.csv files in {in_dir}")

    rows = []
    for f in files:
        x = read_error_window_vector(f)
        if scaler is not None:
            x = scaler.transform(x.reshape(1, -1)).ravel()

        if x.shape[0] != D_expected:
            print(f"[WARN] {os.path.basename(f)}: vector dim {x.shape[0]} != expected {D_expected}; skipping")
            continue

        diffs = centroids - x[None, :]
        d2 = np.sum(diffs * diffs, axis=1)
        k_id = int(np.argmin(d2))
        c = centroids[k_id]
        h = halfwidths[k_id]
        lo, hi = c - h, c + h

        below = x < lo
        above = x > hi
        mask = (below | above).reshape(window_size, F)
        counts = mask.sum(axis=0)

        hw = np.maximum(h, 1e-8)
        sev = (np.abs(x - c) / hw).reshape(window_size, F).sum(axis=0)

        order = sorted(range(F), key=lambda i: (counts[i], sev[i]), reverse=True)
        top3 = [columns[i] for i in order[:3] if counts[i] > 0]
        top3_str = ", ".join(top3)

        row = {
            "file": os.path.basename(f),
            "prototype_id": k_id,
            "nearest_dist": float(np.sqrt(d2[k_id])),
            "top3_sensors": top3_str,
        }
        for i, col in enumerate(columns):
            row[f"viol_count_{col}"] = int(counts[i])
            row[f"viol_sev_{col}"] = float(sev[i])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Wrote ERROR explanations for {len(out_df)} windows → {out_csv}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_fit = sub.add_parser("fit", help="Fit K-Means prototypes on EXAMM error windows (AFTER) in exact_data/normal.")
    ap_fit.add_argument("--normal_dir", default=NORMAL_DIR_DEFAULT)
    ap_fit.add_argument("--scaler_path", default=os.path.join(OUT_DIR_DEFAULT, "error_scaler.pkl"))
    ap_fit.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap_fit.add_argument("--columns", nargs="+", default=COLUMNS_TO_USE_DEFAULT)
    ap_fit.add_argument("--window_size", type=int, default=WINDOW_SIZE_DEFAULT)
    ap_fit.add_argument("--k", type=int, default=K_DEFAULT)
    ap_fit.add_argument("--perc", type=float, default=PERC_DEFAULT)
    ap_fit.add_argument("--random_state", type=int, default=0)

    ap_ex = sub.add_parser("explain", help="Explain EXAMM error windows (BEFORE) in exact_data/anomaly using fitted prototypes.")
    ap_ex.add_argument("--input_dir", default=ANOMALY_DIR_DEFAULT)
    ap_ex.add_argument("--scaler_path", default=os.path.join(OUT_DIR_DEFAULT, "error_scaler.pkl"))
    ap_ex.add_argument("--prototypes_path", default=os.path.join(OUT_DIR_DEFAULT, "prototypes_errors.npz"))
    ap_ex.add_argument("--out_csv", default=os.path.join(OUT_DIR_DEFAULT, "prototype_explanations_errors.csv"))
    ap_ex.add_argument("--columns", nargs="+", default=COLUMNS_TO_USE_DEFAULT)
    ap_ex.add_argument("--window_size", type=int, default=WINDOW_SIZE_DEFAULT)

    args = ap.parse_args()

    if args.cmd == "fit":
        fit_prototypes_on_errors(
            normal_dir=args.normal_dir,
            scaler_path=args.scaler_path,
            out_dir=args.out_dir,
            columns=args.columns,
            window_size=args.window_size,
            k=args.k,
            perc=args.perc,
            random_state=args.random_state,
        )
    elif args.cmd == "explain":
        explain_error_folder(
            in_dir=args.input_dir,
            scaler_path=args.scaler_path,
            prototypes_path=args.prototypes_path,
            out_csv=args.out_csv,
            columns=args.columns,
            window_size=args.window_size,
        )

if __name__ == "__main__":
    main()
