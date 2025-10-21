# fit_error_scaler.py
# Fits a StandardScaler on AFTER error data and saves it as outputs/error_scaler.pkl
import os, glob, argparse, numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler

def list_csvs(folder): return sorted(glob.glob(os.path.join(folder, "*.csv")))
def only_numeric(df):  return df.select_dtypes(include=[np.number])

def load_error_frame(path, auto_drop_non_numeric=True):
    df = pd.read_csv(path)
    if auto_drop_non_numeric:
        df = only_numeric(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise ValueError(f"No usable numeric data in {os.path.basename(path)}")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after_error_dir", required=True, help="Folder with AFTER error CSVs (EXAMM outputs)")
    ap.add_argument("--out_scaler", default="outputs/error_scaler.pkl")
    args = ap.parse_args()

    files = list_csvs(args.after_error_dir)
    if not files: raise SystemExit(f"No CSVs in {args.after_error_dir}")
    parts = []
    for f in files:
        try:
            parts.append(load_error_frame(f))
        except Exception as e:
            print(f"[WARN] {os.path.basename(f)}: {e}")
    if not parts: raise SystemExit("No numeric data to fit scaler.")
    big = pd.concat(parts, axis=0, ignore_index=True)
    scaler = StandardScaler().fit(big.values.astype(float))
    os.makedirs(os.path.dirname(args.out_scaler), exist_ok=True)
    joblib.dump(scaler, args.out_scaler)
    print(f"✅ Saved error scaler → {args.out_scaler} (fit on {len(big)} rows from {len(files)} files)")

if __name__ == "__main__":
    main()
