# prototype_boxes.py
# Prototype "hypercube" explainability for windowed NGAFID data.
# Fits K-Means on healthy windows (AFTER), builds axis-aligned ranges per dimension,
# and explains new windows by nearest prototype + per-sensor range violations.

import os
import re
import glob
import json
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib


# -----------------------------
# Defaults (mirror your OCSVM script)
# -----------------------------
AFTER_DIR_DEFAULT = "dataset/after"
OUT_DIR_DEFAULT = "outputs"

COLUMNS_TO_USE_DEFAULT: List[str] = [
    "AltMSL", "E1 RPM", "E1 FFlow", "E1 CHT1", "E1 EGT1", "NormAc", "IAS"
]
WINDOW_SIZE_DEFAULT: int = 30
STEP_SIZE_DEFAULT: int = 25

# Prototypes
K_DEFAULT = 8          # number of clusters (prototypes)
PERC_DEFAULT = 95.0    # percentile for half-widths


# -----------------------------
# IO & PREP (reuse your conventions)
# -----------------------------
def list_after_files(folder: str, exclude_after01: bool = True) -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if exclude_after01:
        files = [f for f in files if not re.search(r"_after_(0|1)_", os.path.basename(f))]
    return files

def list_any_files(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.csv")))

def read_and_clean(path: str, cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2)
    df.columns = df.columns.str.strip()
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {os.path.basename(path)}: {sorted(missing)}")
    sub = df[cols].replace("", np.nan)
    sub = sub.apply(pd.to_numeric, errors="coerce").dropna()
    return sub

def sliding_windows_from_scaled(
    scaled_array: np.ndarray, window_size: int, step: int
) -> Tuple[np.ndarray, List[int]]:
    starts = list(range(0, len(scaled_array) - window_size + 1, step))
    if not starts:
        return np.empty((0, window_size * scaled_array.shape[1])), []
    windows = np.stack([scaled_array[s:s + window_size, :] for s in starts], axis=0)  # [N, W, F]
    flat = windows.reshape(windows.shape[0], -1)  # [N, W*F]
    return flat, starts


# -----------------------------
# FIT: build prototypes on healthy windows
# -----------------------------
def fit_prototypes(
    after_dir: str,
    scaler_path: str,
    out_dir: str,
    columns: List[str],
    window_size: int,
    step_size: int,
    k: int,
    perc: float,
    random_state: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    scaler = joblib.load(scaler_path)

    # 1) Gather healthy AFTER data (same as OCSVM training)
    after_files = list_after_files(after_dir, exclude_after01=True)
    if not after_files:
        raise RuntimeError(f"No AFTER files found in {after_dir} (after_0/1 excluded).")

    # First pass: concat for scaling indices (we will use the loaded scaler to transform)
    concat_df = []
    lengths = []
    for f in after_files:
        df = read_and_clean(f, columns)
        concat_df.append(df)
        lengths.append(len(df))
    concat = pd.concat(concat_df, axis=0, ignore_index=True)

    # Use the SAME scaler that OCSVM used
    concat_scaled = scaler.transform(concat.values.astype(float))

    # 2) Build flattened windows per file
    X_list = []
    starts_all: Dict[str, List[int]] = {}
    cursor = 0
    for f, n in zip(after_files, lengths):
        scaled_block = concat_scaled[cursor: cursor + n, :]
        cursor += n
        flat_windows, starts = sliding_windows_from_scaled(scaled_block, window_size, step_size)
        if flat_windows.size:
            X_list.append(flat_windows)
            starts_all[os.path.basename(f)] = starts
    if not X_list:
        raise RuntimeError("No windows generated. Check window/step sizes and input data.")

    X_train = np.vstack(X_list)  # [N, D], D = window_size * F
    N, D = X_train.shape

    # 3) K-Means clustering
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(X_train)
    centroids = km.cluster_centers_             # (K, D)
    labels = km.labels_

    # 4) Axis-aligned half-widths per cluster using percentile
    halfwidths = np.zeros_like(centroids)
    for c_id in range(k):
        Xc = X_train[labels == c_id]
        if len(Xc) == 0:
            continue
        deltas = np.abs(Xc - centroids[c_id][None, :])  # (Nc, D)
        halfwidths[c_id] = np.percentile(deltas, perc, axis=0)

    # 5) Persist artifacts
    np.savez_compressed(
        os.path.join(out_dir, "prototypes.npz"),
        centroids=centroids,
        halfwidths=halfwidths,
        columns=np.array(columns, dtype=object),
        window_size=np.array([window_size]),
        step_size=np.array([step_size]),
        k=np.array([k]),
        perc=np.array([perc]),
    )
    meta = {
        "n_windows": int(N),
        "dim": int(D),
        "k": int(k),
        "percentile": float(perc),
        "columns": columns,
        "window_size": window_size,
        "step_size": step_size,
        "scaler_path": scaler_path,
        "note": "K-Means prototypes with axis-aligned percentile ranges on healthy AFTER windows."
    }
    with open(os.path.join(out_dir, "prototypes_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Prototypes fitted on {N} windows (D={D}), K={k}, perc={perc}.")
    print(f"Saved: {os.path.join(out_dir,'prototypes.npz')}, {os.path.join(out_dir,'prototypes_meta.json')}")


# -----------------------------
# EXPLAIN: nearest prototype + per-sensor violations
# -----------------------------
def load_prototypes(prototypes_path: str):
    data = np.load(prototypes_path, allow_pickle=True)
    centroids = data["centroids"]
    halfwidths = data["halfwidths"]
    columns = list(data["columns"])
    window_size = int(data["window_size"][0])
    step_size = int(data["step_size"][0])
    return centroids, halfwidths, columns, window_size, step_size

def explain_file_windows(
    csv_path: str,
    scaler,
    centroids: np.ndarray,
    halfwidths: np.ndarray,
    columns: List[str],
    window_size: int,
    step_size: int,
) -> pd.DataFrame:
    df = read_and_clean(csv_path, columns)
    X = scaler.transform(df.values.astype(float))
    flat, starts = sliding_windows_from_scaled(X, window_size, step_size)
    if flat.size == 0:
        return pd.DataFrame(columns=[
            "file","window_idx","start_idx","end_idx","prototype_id","nearest_dist","top3_sensors"
        ] + [f"viol_count_{c}" for c in columns] + [f"viol_sev_{c}" for c in columns])

    F = len(columns)
    K = centroids.shape[0]
    D = window_size * F

    rows = []
    for w_idx, x in enumerate(flat):
        # nearest prototype
        diffs = centroids - x[None, :]
        d2 = np.sum(diffs * diffs, axis=1)
        k = int(np.argmin(d2))
        c = centroids[k]
        h = halfwidths[k]
        lo, hi = c - h, c + h

        # per-dimension out-of-box flags and severity
        below = x < lo
        above = x > hi
        # counts per sensor across the 30 positions
        mask = (below | above).reshape(window_size, F)     # [W,F]
        counts = mask.sum(axis=0)                          # [F]
        # severity: how far outside, normalized by half-width (sum across time)
        hw = np.maximum(h, 1e-8)
        sev = (np.abs(x - c) / hw).reshape(window_size, F).sum(axis=0)  # [F]

        # top-3 sensors by (count, then severity)
        order = sorted(range(F), key=lambda i: (counts[i], sev[i]), reverse=True)
        top3 = [columns[i] for i in order[:3] if counts[i] > 0]
        top3_str = ", ".join(top3) if top3 else ""

        row = {
            "file": os.path.basename(csv_path),
            "window_idx": w_idx,
            "start_idx": starts[w_idx],
            "end_idx": starts[w_idx] + window_size - 1,
            "prototype_id": k,
            "nearest_dist": float(np.sqrt(d2[k])),
            "top3_sensors": top3_str,
        }
        for i, col in enumerate(columns):
            row[f"viol_count_{col}"] = int(counts[i])
        for i, col in enumerate(columns):
            row[f"viol_sev_{col}"] = float(sev[i])
        rows.append(row)

    return pd.DataFrame(rows)

def explain_folder(
    in_dir: str,
    scaler_path: str,
    prototypes_path: str,
    out_csv: str,
):
    centroids, halfwidths, columns, window_size, step_size = load_prototypes(prototypes_path)
    scaler = joblib.load(scaler_path)

    files = list_any_files(in_dir)
    all_rows = []
    for f in files:
        try:
            df_expl = explain_file_windows(
                f, scaler, centroids, halfwidths, columns, window_size, step_size
            )
            if not df_expl.empty:
                all_rows.append(df_expl)
        except Exception as e:
            print(f"[WARN] {os.path.basename(f)}: {e}")

    if not all_rows:
        print("No windows explained (empty).")
        return

    out_df = pd.concat(all_rows, axis=0, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Wrote explanations for {len(out_df)} windows → {out_csv}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_fit = sub.add_parser("fit", help="Fit K-Means prototypes on AFTER windows.")
    ap_fit.add_argument("--after_dir", default=AFTER_DIR_DEFAULT)
    ap_fit.add_argument("--scaler_path", required=True)
    ap_fit.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap_fit.add_argument("--columns", nargs="+", default=COLUMNS_TO_USE_DEFAULT)
    ap_fit.add_argument("--window_size", type=int, default=WINDOW_SIZE_DEFAULT)
    ap_fit.add_argument("--step_size", type=int, default=STEP_SIZE_DEFAULT)
    ap_fit.add_argument("--k", type=int, default=K_DEFAULT)
    ap_fit.add_argument("--perc", type=float, default=PERC_DEFAULT)
    ap_fit.add_argument("--random_state", type=int, default=0)

    ap_ex = sub.add_parser("explain", help="Explain windows in a folder by nearest prototype + per-sensor violations.")
    ap_ex.add_argument("--input_dir", required=True)
    ap_ex.add_argument("--scaler_path", required=True)
    ap_ex.add_argument("--prototypes_path", default=os.path.join(OUT_DIR_DEFAULT, "prototypes.npz"))
    ap_ex.add_argument("--out_csv", default=os.path.join(OUT_DIR_DEFAULT, "prototype_explanations.csv"))

    args = ap.parse_args()

    if args.cmd == "fit":
        fit_prototypes(
            after_dir=args.after_dir,
            scaler_path=args.scaler_path,
            out_dir=args.out_dir,
            columns=args.columns,
            window_size=args.window_size,
            step_size=args.step_size,
            k=args.k,
            perc=args.perc,
            random_state=args.random_state,
        )
    elif args.cmd == "explain":
        explain_folder(
            in_dir=args.input_dir,
            scaler_path=args.scaler_path,
            prototypes_path=args.prototypes_path,
            out_csv=args.out_csv,
        )

if __name__ == "__main__":
    main()
