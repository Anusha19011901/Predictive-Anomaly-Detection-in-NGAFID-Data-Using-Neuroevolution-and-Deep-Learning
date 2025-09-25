#!/usr/bin/env python3
"""
Early-fusion OC-SVM on EXAMM errors + raw-window stats (aligned by EXAMM window_idx).

Inputs (existing from your EXAMM pipeline)
  artifacts/errors/per_window/ocsvm_input_after.csv
  artifacts/errors/per_window/ocsvm_input_before.csv
  dataset/after_examm2/*.csv
  dataset/before_examm2/*.csv

Outputs
  outputs/fusion/before_scores.csv
  outputs/fusion/before_topk_contributors.csv
  outputs/fusion/ocsvm_fusion_after_train_summary.json
  outputs/fusion/feature_columns.json
"""

import os, glob, json, argparse
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

# --------------------
# Defaults
# --------------------
AFTER_ERRORS_CSV = "artifacts/errors/per_window/ocsvm_input_after.csv"
BEFORE_ERRORS_CSV = "artifacts/errors/per_window/ocsvm_input_before.csv"
RAW_AFTER_DIR  = "dataset/after_examm2"
RAW_BEFORE_DIR = "dataset/before_examm2"
OUT_DIR = "outputs/fusion"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 30
STEP_SIZE   = 5
MERGE_KEYS  = ["flight_id","subseq_id","window_idx"]

NU    = 0.05
GAMMA = "scale"

TOPK_CONTRIB = 5

# --------------------
# Helpers
# --------------------
def parse_ids_from_filename(path: str) -> Tuple[str, str]:
    base = os.path.basename(path)
    flight_id = os.path.splitext(base)[0]
    subseq_id = flight_id
    return flight_id, subseq_id

def read_clean_csv(path: str, keep_cols: List[str] | None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # If keep_cols given, intersect to avoid reading junk
    if keep_cols is not None:
        keep = [c for c in keep_cols if c in df.columns]
        df = df[keep]
    # coerce to numeric and drop NaN rows (EXAMM already cleaned these)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df

def sliding_windows(n_rows: int, window: int, step: int) -> List[Tuple[int,int]]:
    starts = list(range(0, max(n_rows - window + 1, 0), step))
    return [(s, s+window-1) for s in starts]

def window_stats_vectorized(block: pd.DataFrame) -> pd.DataFrame:
    """Vectorized stats for speed: mean, std, slope, q25, q75, rng (per column)."""
    X = block.values.astype(float)     # shape (T, F)
    T, F = X.shape
    t = np.arange(T, dtype=float)
    t_c = t - t.mean()
    t_var = (t_c ** 2).sum() if T > 1 else 0.0

    mean = X.mean(axis=0)
    std  = X.std(axis=0, ddof=1) if T > 1 else np.zeros(F)
    q25  = np.percentile(X, 25, axis=0)
    q75  = np.percentile(X, 75, axis=0)
    rng  = X.max(axis=0) - X.min(axis=0)

    if t_var > 0:
        # slope = cov(t, x) / var(t); cov(t,x) = sum((t-mean_t)*(x-mean_x)) / (T-1)
        X_c = X - mean
        cov_num = (t_c[:, None] * X_c).sum(axis=0)
        slope = cov_num / t_var
    else:
        slope = np.zeros(F)

    cols = []
    data = []
    for c in block.columns:
        cols.extend([f"{c}__mean", f"{c}__std", f"{c}__slope", f"{c}__q25", f"{c}__q75", f"{c}__rng"])
    # Interleave stats in the same order as columns
    for j in range(F):
        data.extend([mean[j], std[j], slope[j], q25[j], q75[j], rng[j]])
    return pd.DataFrame([data], columns=cols)

def make_raw_window_table(raw_dir: str,
                          feature_cols: List[str],
                          max_flights: int | None,
                          max_windows_per_flight: int | None) -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if max_flights is not None:
        files = files[:max_flights]

    for fi, f in enumerate(files, 1):
        flight_id, subseq_id = parse_ids_from_filename(f)
        df = read_clean_csv(f, keep_cols=feature_cols)

        win_ranges = sliding_windows(len(df), WINDOW_SIZE, STEP_SIZE)
        if max_windows_per_flight is not None:
            win_ranges = win_ranges[:max_windows_per_flight]

        for widx, (s, e) in enumerate(win_ranges):
            block = df.iloc[s:e+1]
            stats = window_stats_vectorized(block)
            stats.insert(0, "window_idx", widx)
            stats.insert(0, "subseq_id", subseq_id)
            stats.insert(0, "flight_id", flight_id)
            rows.append(stats)

        print(f"  built {len(win_ranges):4d} windows  [{fi:02d}/{len(files)}]  {os.path.basename(f)}")

    if not rows:
        return pd.DataFrame(columns=MERGE_KEYS)
    return pd.concat(rows, axis=0, ignore_index=True)

def standardize_train_apply(train_df: pd.DataFrame, test_df: pd.DataFrame):
    stats = {}
    train_z = train_df.copy()
    test_z  = test_df.copy()
    for c in train_df.columns:
        mu = float(train_df[c].mean())
        sd = float(train_df[c].std(ddof=1)) or 1.0
        stats[c] = (mu, sd)
        train_z[c] = (train_df[c] - mu) / sd
        if c in test_z:
            test_z[c] = (test_df[c] - mu) / sd
    return train_z, test_z, stats

def rank_contributors(zrow: pd.Series, topk: int):
    zabs = zrow.abs().sort_values(ascending=False)
    zabs = zabs.head(topk)
    return list(zabs.items())

# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser(description="Early-fusion OC-SVM on EXAMM errors + raw-window stats")
    ap.add_argument("--after_errors",  default=AFTER_ERRORS_CSV)
    ap.add_argument("--before_errors", default=BEFORE_ERRORS_CSV)
    ap.add_argument("--raw_after_dir",  default=RAW_AFTER_DIR)
    ap.add_argument("--raw_before_dir", default=RAW_BEFORE_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--cols_mode", choices=["errors_only","intersection","all_raw"], default="errors_only",
                    help="Which raw columns to compute stats for: "
                         "'errors_only' = sensors present in mae_*; "
                         "'intersection' = intersection with raw csv columns (safer); "
                         "'all_raw' = all numeric columns in raw files (slow).")
    ap.add_argument("--max_flights", type=int, default=None, help="Limit number of flights per split for quick runs.")
    ap.add_argument("--max_windows_per_flight", type=int, default=30, help="Limit windows per flight (set None for all).")
    ap.add_argument("--nu", type=float, default=NU)
    ap.add_argument("--gamma", default=GAMMA)
    ap.add_argument("--topk", type=int, default=TOPK_CONTRIB)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load EXAMM per-window errors
    err_after  = pd.read_csv(args.after_errors)
    err_before = pd.read_csv(args.before_errors)
    id_cols = MERGE_KEYS
    mae_cols_after  = [c for c in err_after.columns  if c.startswith("mae_")]
    mae_cols_before = [c for c in err_before.columns if c.startswith("mae_")]

    # → Feature list driven by errors (fastest path)
    mae_feature_names = [c.replace("mae_", "") for c in sorted(set(mae_cols_after) & set(mae_cols_before))]

    # 2) Decide which raw columns to compute stats for
    if args.cols_mode == "errors_only":
        feature_cols = mae_feature_names
    elif args.cols_mode == "intersection":
        # peek one raw file to intersect
        sample_files = sorted(glob.glob(os.path.join(args.raw_after_dir, "*.csv")))
        if not sample_files:
            raise RuntimeError(f"No files in {args.raw_after_dir}")
        sample_cols = pd.read_csv(sample_files[0], nrows=1).columns.tolist()
        feature_cols = [c for c in mae_feature_names if c in sample_cols]
    else:  # all_raw (slow)
        # gather union of numeric columns from a sample file
        sample_files = sorted(glob.glob(os.path.join(args.raw_after_dir, "*.csv")))
        if not sample_files:
            raise RuntimeError(f"No files in {args.raw_after_dir}")
        tmp = pd.read_csv(sample_files[0], low_memory=False)
        feature_cols = [c for c in tmp.columns if pd.api.types.is_numeric_dtype(pd.to_numeric(tmp[c], errors="coerce"))]

    # Keep only the common mae_* across both splits
    common_mae = ["mae_" + c for c in feature_cols if ("mae_" + c) in mae_cols_after and ("mae_" + c) in mae_cols_before]
    err_after  = err_after[id_cols + common_mae]
    err_before = err_before[id_cols + common_mae]

    print("==== Fusion run config ====")
    print(f"after_errors : {args.after_errors}")
    print(f"before_errors: {args.before_errors}")
    print(f"raw_after    : {args.raw_after_dir}")
    print(f"raw_before   : {args.raw_before_dir}")
    print(f"out_dir      : {args.out_dir}")
    print(f"WINDOW/STEP  : {WINDOW_SIZE}/{STEP_SIZE}")
    print(f"cols_mode    : {args.cols_mode}")
    print(f"features     : {len(feature_cols)} → {feature_cols[:10]}{'...' if len(feature_cols)>10 else ''}")
    print(f"max_flights  : {args.max_flights}   max_windows_per_flight: {args.max_windows_per_flight}")
    print("===========================")

    # 3) Build RAW window-stat tables from *_examm2 (vectorized & limited)
    print("\nBuilding AFTER raw window table...")
    raw_after_tbl  = make_raw_window_table(args.raw_after_dir,  feature_cols,
                                           max_flights=args.max_flights,
                                           max_windows_per_flight=args.max_windows_per_flight)
    print("Building BEFORE raw window table...")
    raw_before_tbl = make_raw_window_table(args.raw_before_dir, feature_cols,
                                           max_flights=args.max_flights,
                                           max_windows_per_flight=args.max_windows_per_flight)

    # 4) Merge (errors + raw stats) by KEYS ONLY
    fused_after  = err_after.merge(raw_after_tbl,  on=id_cols, how="inner")
    fused_before = err_before.merge(raw_before_tbl, on=id_cols, how="inner")

    print("\n---- Merge debug ----")
    print(f"err_after rows:  {len(err_after):6d}  raw_after:  {len(raw_after_tbl):6d}  fused_after:  {len(fused_after):6d}")
    print(f"err_before rows: {len(err_before):6d}  raw_before: {len(raw_before_tbl):6d}  fused_before: {len(fused_before):6d}")

    if fused_after.empty or fused_before.empty:
        print("\nSample AFTER error keys:", err_after[id_cols].head().to_dict(orient="records"))
        print("Sample AFTER raw keys:",   raw_after_tbl[id_cols].head().to_dict(orient="records"))
        print("Sample BEFORE error keys:", err_before[id_cols].head().to_dict(orient="records"))
        print("Sample BEFORE raw keys:",   raw_before_tbl[id_cols].head().to_dict(orient="records"))
        raise RuntimeError("Fusion merge produced empty tables. Check ID alignment / filenames.")

    # 5) Build feature matrices (drop ids)
    X_after  = fused_after.drop(columns=id_cols)
    X_before = fused_before.drop(columns=id_cols)

    # 6) Standardize by AFTER statistics
    X_after_z, X_before_z, zstats = standardize_train_apply(X_after, X_before)

    # 7) Train OC-SVM on AFTER fused features
    ocsvm = OneClassSVM(kernel="rbf", nu=args.nu, gamma=args.gamma)
    ocsvm.fit(X_after_z.values)

    # 8) Score BEFORE
    scores = ocsvm.decision_function(X_before_z.values)
    preds  = ocsvm.predict(X_before_z.values)
    flags  = (preds == -1).astype(int)

    out = fused_before[id_cols].copy()
    out["ocsvm_score"]  = scores
    out["anomaly_flag"] = flags

    # 9) Contributor ranking (absolute z across fused features)
    contrib_rows = []
    for i in range(len(X_before_z)):
        top = rank_contributors(X_before_z.iloc[i], args.topk)
        row = {**fused_before.iloc[i][id_cols].to_dict()}
        for j, (feat, zval) in enumerate(top, 1):
            row[f"top{j}_feature"] = feat
            row[f"top{j}_z"] = float(zval)
        contrib_rows.append(row)
    contrib_df = pd.DataFrame(contrib_rows)

    # 10) Save
    out_csv = os.path.join(args.out_dir, "before_scores.csv")
    out.to_csv(out_csv, index=False)
    contrib_df.to_csv(os.path.join(args.out_dir, "before_topk_contributors.csv"), index=False)

    with open(os.path.join(args.out_dir, "feature_columns.json"), "w") as f:
        json.dump({"columns": list(X_after_z.columns)}, f, indent=2)

    summary = {
        "nu": args.nu,
        "gamma": args.gamma,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "n_train_windows": int(len(X_after_z)),
        "n_test_windows": int(len(X_before_z)),
        "merge_keys": MERGE_KEYS,
        "cols_mode": args.cols_mode,
        "max_flights": args.max_flights,
        "max_windows_per_flight": args.max_windows_per_flight,
        "note": "Early fusion on EXAMM MAE features + vectorized raw-window stats (from *_examm2).",
    }
    with open(os.path.join(args.out_dir, "ocsvm_fusion_after_train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Saved: {out_csv}")
    print(f"✅ Saved: {os.path.join(args.out_dir, 'before_topk_contributors.csv')}")
    print(f"✅ Train summary: {os.path.join(args.out_dir, 'ocsvm_fusion_after_train_summary.json')}")
    print(f"✅ Features: {os.path.join(args.out_dir, 'feature_columns.json')}")

if __name__ == "__main__":
    main()
