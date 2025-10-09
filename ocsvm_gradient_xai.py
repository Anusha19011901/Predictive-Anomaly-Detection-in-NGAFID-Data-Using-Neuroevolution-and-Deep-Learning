# ocsvm_gradient_xai.py
# Model-specific explainability for RBF One-Class SVM:
# Per-window gradient attribution → per-sensor contributions.

import os
import re
import glob
import json
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Defaults (match your pipeline)
# -----------------------------
AFTER_DIR_DEFAULT = "dataset/after"
BEFORE_DIR_DEFAULT = "dataset/before"
OUT_DIR_DEFAULT = "outputs"

COLUMNS_TO_USE_DEFAULT: List[str] = [
    "AltMSL", "E1 RPM", "E1 FFlow", "E1 CHT1", "E1 EGT1", "NormAc", "IAS"
]
WINDOW_SIZE_DEFAULT: int = 30
STEP_SIZE_DEFAULT: int = 25

# -----------------------------
# IO helpers
# -----------------------------
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
# OCSVM gradient (RBF)
# -----------------------------
def ocsvm_rbf_gradient(ocsvm, x_flat: np.ndarray) -> np.ndarray:
    sv = ocsvm.support_vectors_          # (n_sv, D)
    alpha = ocsvm._dual_coef_.ravel()    # (n_sv,)
    gamma = ocsvm._gamma                 # float
    diffs = (x_flat[None, :] - sv)       # (n_sv, D)
    sqn = np.sum(diffs * diffs, axis=1)  # (n_sv,)
    K = np.exp(-gamma * sqn)             # (n_sv,)
    grad = (-2.0 * gamma) * np.sum((alpha * K)[:, None] * diffs, axis=0)  # (D,)
    return grad

def sensor_contributions_from_grad(
    grad_flat: np.ndarray, window_size: int, sensor_names: List[str]
) -> Dict[str, float]:
    F = len(sensor_names)
    G = np.abs(grad_flat).reshape(window_size, F).sum(axis=0)  # (F,)
    w = G / (G.sum() + 1e-12)
    return dict(zip(sensor_names, map(float, w)))

# -----------------------------
# Explain a single CSV file
# -----------------------------
def explain_file_with_gradients(
    csv_path: str,
    scaler,
    ocsvm,
    columns: List[str],
    window_size: int,
    step_size: int,
) -> pd.DataFrame:
    df = read_and_clean(csv_path, columns)
    X = scaler.transform(df.values.astype(float))
    flat, starts = sliding_windows_from_scaled(X, window_size, step_size)
    if flat.size == 0:
        return pd.DataFrame(columns=["file","window_idx","start_idx","end_idx","decision_score","top3_sensors"]
                            + [f"contrib_{c}" for c in columns])

    F = len(columns)
    rows = []
    # decision_function: positive ~ normal, negative ~ outlier
    scores = ocsvm.decision_function(flat)  # (N,)

    for w_idx, (x, score) in enumerate(zip(flat, scores)):
        grad = ocsvm_rbf_gradient(ocsvm, x)
        contrib = sensor_contributions_from_grad(grad, window_size, columns)
        # top-3 sensors by contribution
        top3 = ", ".join([k for k, _ in sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:3]])

        row = {
            "file": os.path.basename(csv_path),
            "window_idx": int(w_idx),
            "start_idx": int(starts[w_idx]),
            "end_idx": int(starts[w_idx] + window_size - 1),
            "decision_score": float(score),
            "top3_sensors": top3
        }
        for c in columns:
            row[f"contrib_{c}"] = float(contrib[c])
        rows.append(row)

    return pd.DataFrame(rows)

# -----------------------------
# Folder runner
# -----------------------------
def explain_folder_with_gradients(
    in_dir: str,
    scaler_path: str,
    ocsvm_model_path: str,
    columns: List[str],
    window_size: int,
    step_size: int,
    out_csv: str
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    scaler = joblib.load(scaler_path)
    ocsvm = joblib.load(ocsvm_model_path)

    files = list_any_files(in_dir)
    if not files:
        raise RuntimeError(f"No CSV files found in {in_dir}")

    all_rows = []
    for f in files:
        try:
            df_rows = explain_file_with_gradients(f, scaler, ocsvm, columns, window_size, step_size)
            if not df_rows.empty:
                all_rows.append(df_rows)
        except Exception as e:
            print(f"[WARN] {os.path.basename(f)}: {e}")

    if not all_rows:
        print("No windows explained (empty).")
        return

    out_df = pd.concat(all_rows, axis=0, ignore_index=True)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Wrote gradient attributions for {len(out_df)} windows → {out_csv}")

# -----------------------------
# Optional: bar plot for a selected window
# -----------------------------
def plot_bar_for_window(
    csv_path: str,
    out_path: str,
    file_name: str = None,
    window_idx: int = None
):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    contrib_cols = [c for c in df.columns if c.startswith("contrib_")]
    sensors = [c.replace("contrib_", "") for c in contrib_cols]

    if file_name is not None and window_idx is not None:
        row = df[(df["file"] == file_name) & (df["window_idx"] == window_idx)]
        if row.empty:
            raise ValueError("No row matches the given file and window_idx.")
        row = row.iloc[0]
    else:
        # pick the most anomalous (lowest decision score)
        row = df.iloc[int(df["decision_score"].idxmin())]

    vals = row[contrib_cols].astype(float).values
    order = np.argsort(-vals)
    vals_sorted = vals[order]
    sensors_sorted = [sensors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(sensors_sorted)), vals_sorted)
    ax.set_xticks(range(len(sensors_sorted)))
    ax.set_xticklabels(sensors_sorted, rotation=45, ha="right")
    ax.set_ylabel("Contribution (normalized)")
    ax.set_title(f"OC-SVM gradient attribution — file={row['file']}, window_idx={int(row['window_idx'])}, score={float(row['decision_score']):.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_exp = sub.add_parser("explain", help="Compute OC-SVM gradient attributions per window for all CSVs in a folder.")
    ap_exp.add_argument("--input_dir", required=True)
    ap_exp.add_argument("--scaler_path", default=os.path.join(OUT_DIR_DEFAULT, "scaler.pkl"))
    ap_exp.add_argument("--ocsvm_model_path", default=os.path.join(OUT_DIR_DEFAULT, "ocsvm_model.pkl"))
    ap_exp.add_argument("--columns", nargs="+", default=COLUMNS_TO_USE_DEFAULT)
    ap_exp.add_argument("--window_size", type=int, default=WINDOW_SIZE_DEFAULT)
    ap_exp.add_argument("--step_size", type=int, default=STEP_SIZE_DEFAULT)
    ap_exp.add_argument("--out_csv", default=os.path.join(OUT_DIR_DEFAULT, "ocsvm_gradient_attributions.csv"))

    ap_bar = sub.add_parser("bar", help="Plot a bar chart of contributions for a selected (or most anomalous) window.")
    ap_bar.add_argument("--csv", required=True, help="CSV produced by the 'explain' command")
    ap_bar.add_argument("--out", default=os.path.join(OUT_DIR_DEFAULT, "ocsvm_gradient_bar.png"))
    ap_bar.add_argument("--select_file", default=None)
    ap_bar.add_argument("--select_window", type=int, default=None)

    args = ap.parse_args()

    if args.cmd == "explain":
        explain_folder_with_gradients(
            in_dir=args.input_dir,
            scaler_path=args.scaler_path,
            ocsvm_model_path=args.ocsvm_model_path,
            columns=args.columns,
            window_size=args.window_size,
            step_size=args.step_size,
            out_csv=args.out_csv
        )
    elif args.cmd == "bar":
        plot_bar_for_window(
            csv_path=args.csv,
            out_path=args.out,
            file_name=args.select_file,
            window_idx=args.select_window
        )

if __name__ == "__main__":
    main()
