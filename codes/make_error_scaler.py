# make_error_scaler.py
# Fit a StandardScaler on EXAMM error windows in exact_data/normal and save it.

import os, glob, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

DEF_NORMAL_DIR = "exact_data/normal"
DEF_OUT = "outputs/error_scaler.pkl"
DEF_COLS = ["AltMSL","E1 RPM","E1 FFlow","E1 CHT1","E1 EGT1","NormAc","IAS"]
DEF_W = 30

def list_window_files(folder):
    return sorted(glob.glob(os.path.join(folder, "window_*.csv")))

def read_vec(path, columns):
    df = pd.read_csv(path)
    df = df[columns]  # drop anomaly_flag etc.
    return df.values.astype(float).reshape(-1)  # flatten W x F -> (W*F,)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_dir", default=DEF_NORMAL_DIR)
    ap.add_argument("--columns", nargs="+", default=DEF_COLS)
    ap.add_argument("--window_size", type=int, default=DEF_W)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    files = list_window_files(args.normal_dir)
    if not files:
        raise SystemExit(f"No window_*.csv in {args.normal_dir}")

    X = [read_vec(f, args.columns) for f in files]
    X = np.vstack(X)  # [N, W*F]

    scaler = StandardScaler().fit(X)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(scaler, args.out)
    print(f"✅ Saved scaler to {args.out} (fit on {len(files)} windows, dim={X.shape[1]})")

if __name__ == "__main__":
    main()
