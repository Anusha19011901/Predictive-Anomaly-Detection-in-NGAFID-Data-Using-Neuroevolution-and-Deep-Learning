#!/usr/bin/env python3
"""
@file ocsvm_examm_only.py
@brief One-Class SVM anomaly detection using EXAMM-derived MAE features (no raw-window fusion).

This module performs anomaly detection solely on EXAMM forecast-error features (`mae_*`),
ignoring raw sensor windows. It trains an OC-SVM model on AFTER-maintenance windows to model
"normal" aircraft behavior, then scores BEFORE-maintenance windows to detect anomalies that
anticipate future maintenance events.

-------------------------------------------------------------------------------
Inputs (from EXAMM Forecasting Pipeline)
-------------------------------------------------------------------------------
artifacts/errors/per_window/ocsvm_input_after.csv  
artifacts/errors/per_window/ocsvm_input_before.csv  

These must contain:
- `flight_id`, `subseq_id`, `window_idx`  
- `mae_<feature>` columns for EXAMM prediction errors

-------------------------------------------------------------------------------
Core Workflow
-------------------------------------------------------------------------------

1. **Load EXAMM Per-Window Errors**
   - Extract MAE features (`mae_*`) from AFTER and BEFORE datasets.
   - Ensure consistent feature sets across both.

2. **Z-Score Standardization**
   - Fit mean and std on AFTER (normal baseline).
   - Apply same parameters to BEFORE.
   - Export z-score statistics to `zstats.json`.

3. **Train One-Class SVM**
   - Kernel: RBF  
   - nu: anomaly fraction (default 0.05)  
   - gamma: kernel width (`scale` by default)

4. **Score BEFORE Windows**
   - `decision_function`: normality score  
   - `predict`: assigns +1 (normal) or -1 (anomaly)  
   - Produces:
       `before_scores.csv`

5. **Top-K Contributing Features (Explainability)**
   - For each window, rank features by absolute z-score.
   - Saves:
       `before_topk_contributors.csv`

6. **Persist Core Artifacts**
   - OC-SVM model (`ocsvm_model.pkl`)
   - Feature list (`feature_columns.json`)
   - Training summary (`ocsvm_examm_only_summary.json`)
   - Z-score stats (`zstats.json`)

-------------------------------------------------------------------------------
Optional Outputs (Controlled by Flags)
-------------------------------------------------------------------------------

--export_group_means  
    Compute mean z-score per (flight_id, subseq_id) for heatmaps.
    Exports:
      - `before_groupmeans_wide.csv`
      - `before_groupmeans_long.csv`

--save_before_zscores  
    Save full per-window BEFORE z-score matrix:
      - `before_zscores.parquet`

-------------------------------------------------------------------------------
Intended Use
-------------------------------------------------------------------------------
This script is used in the NGAFID anomaly pipeline to evaluate whether EXAMM forecast
errors alone — without raw-window fusion or multivariate modeling — can detect early,
subtle deviations in aircraft behavior. It provides clear anomaly scores, interpretable
feature contributors, and optional aggregated visual layers for deeper analysis.
"""


import os, json, argparse
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
import joblib

AFTER_ERRORS_CSV = "artifacts/errors/per_window/ocsvm_input_after.csv"
BEFORE_ERRORS_CSV = "artifacts/errors/per_window/ocsvm_input_before.csv"
OUT_DIR = "outputs/ocsvm_examm_only"
os.makedirs(OUT_DIR, exist_ok=True)

ID_COLS = ["flight_id","subseq_id","window_idx"]
NU = 0.05
GAMMA = "scale"
TOPK_CONTRIB = 5

def standardize_train_apply(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Return (train_z, test_z, stats) where stats[c]=(mu, sd). ddof=1; fallback sd=1.0."""
    stats: Dict[str, Tuple[float, float]] = {}
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
    zabs = zrow.abs().sort_values(ascending=False).head(topk)
    return list(zabs.items())

def main():
    ap = argparse.ArgumentParser(description="OC-SVM using EXAMM mae_* only (no fusion)")
    ap.add_argument("--after_errors",  default=AFTER_ERRORS_CSV)
    ap.add_argument("--before_errors", default=BEFORE_ERRORS_CSV)
    ap.add_argument("--out_dir",       default=OUT_DIR)
    ap.add_argument("--nu", type=float, default=NU)
    ap.add_argument("--gamma", default=GAMMA)
    ap.add_argument("--topk", type=int, default=TOPK_CONTRIB)
    ap.add_argument("--export_group_means", action="store_true",
                    help="Export groupmeans of BEFORE z by (flight_id, subseq_id) for heatmaps.")
    ap.add_argument("--save_before_zscores", action="store_true",
                    help="Save per-window BEFORE z-scores (parquet).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load EXAMM per-window errors
    A = pd.read_csv(args.after_errors)
    B = pd.read_csv(args.before_errors)

    # Basic checks
    for k in ID_COLS:
        if k not in A.columns or k not in B.columns:
            raise RuntimeError(f"Missing key '{k}' in EXAMM error csvs.")
    mae_cols_A = [c for c in A.columns if c.startswith("mae_")]
    mae_cols_B = [c for c in B.columns if c.startswith("mae_")]
    if not mae_cols_A or not mae_cols_B:
        raise RuntimeError("No mae_* columns found in EXAMM error csvs.")
    common_mae = sorted(set(mae_cols_A) & set(mae_cols_B))
    if not common_mae:
        raise RuntimeError("No overlapping mae_* columns between AFTER and BEFORE.")

    # 2) Slice feature matrices
    A_X = A[common_mae].copy()
    B_X = B[common_mae].copy()

    # 3) Standardize (fit on AFTER, apply to BEFORE)
    A_Z, B_Z, zstats = standardize_train_apply(A_X, B_X)

    # 4) Train OC-SVM on AFTER z-features
    ocsvm = OneClassSVM(kernel="rbf", nu=args.nu, gamma=args.gamma)
    ocsvm.fit(A_Z.values)

    # 5) Score BEFORE
    scores = ocsvm.decision_function(B_Z.values)  # higher is more normal
    preds  = ocsvm.predict(B_Z.values)            # -1 anomaly, +1 normal
    flags  = (preds == -1).astype(int)

    out = B[ID_COLS].copy()
    out["ocsvm_score"]  = scores
    out["anomaly_flag"] = flags

    # 6) Top-k contributors by |z| across mae_* features (diagnostics)
    contrib_rows = []
    for i in range(len(B_Z)):
        top = rank_contributors(B_Z.iloc[i], args.topk)
        row = {**B.iloc[i][ID_COLS].to_dict()}
        for j, (feat, zval) in enumerate(top, 1):
            row[f"top{j}_feature"] = feat
            row[f"top{j}_z"] = float(zval)
        contrib_rows.append(row)
    contrib_df = pd.DataFrame(contrib_rows)

    # 7) Persist core artifacts
    scores_csv   = os.path.join(args.out_dir, "before_scores.csv")
    contrib_csv  = os.path.join(args.out_dir, "before_topk_contributors.csv")
    feats_json   = os.path.join(args.out_dir, "feature_columns.json")
    summary_json = os.path.join(args.out_dir, "ocsvm_examm_only_summary.json")
    model_pkl    = os.path.join(args.out_dir, "ocsvm_model.pkl")
    zstats_json  = os.path.join(args.out_dir, "zstats.json")

    out.to_csv(scores_csv, index=False)
    contrib_df.to_csv(contrib_csv, index=False)

    with open(feats_json, "w") as f:
        json.dump({"columns": common_mae}, f, indent=2)

    with open(zstats_json, "w") as f:
        json.dump({c: {"mu": mu, "sd": sd} for c, (mu, sd) in zstats.items()}, f, indent=2)

    joblib.dump(ocsvm, model_pkl)

    summary = {
        "mode": "OC-SVM on EXAMM mae_* only",
        "nu": args.nu,
        "gamma": args.gamma,
        "n_after_windows": int(len(A_Z)),
        "n_before_windows": int(len(B_Z)),
        "id_keys": ID_COLS,
        "topk_contributors": args.topk,
        "note": "Trained solely on z-scored EXAMM per-window errors; no raw-window fusion."
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Saved: {scores_csv}")
    print(f"✅ Saved: {contrib_csv}")
    print(f"✅ Saved model: {model_pkl}")
    print(f"✅ Saved: {summary_json}")
    print(f"✅ Saved: {feats_json}")
    print(f"✅ Saved: {zstats_json}")

    # 8) Optional: export BEFORE z-scores (for fine-grained viz & QC)
    if args.save_before_zscores:
        z_df = pd.concat([B[ID_COLS].reset_index(drop=True), B_Z.reset_index(drop=True)], axis=1)
        z_path = os.path.join(args.out_dir, "before_zscores.parquet")
        z_df.to_parquet(z_path, index=False)
        print(f"✅ Saved: {z_path}")

    # 9) Optional: groupmeans for heatmaps (mean z by flight_id, subseq_id)
    if args.export_group_means:
        z_df = pd.concat([B[ID_COLS].reset_index(drop=True), B_Z.reset_index(drop=True)], axis=1)
        key_cols = ["flight_id", "subseq_id"]
        mae_cols = [c for c in z_df.columns if c.startswith("mae_")]
        gm = z_df.groupby(key_cols)[mae_cols].mean().reset_index()

        # wide
        gm_wide_path = os.path.join(args.out_dir, "before_groupmeans_wide.csv")
        gm.to_csv(gm_wide_path, index=False)

        # long (tidy) -> easier heatmaps
        gm_long = gm.melt(id_vars=key_cols, var_name="feature", value_name="z_mean")
        gm_long_path = os.path.join(args.out_dir, "before_groupmeans_long.csv")
        gm_long.to_csv(gm_long_path, index=False)

        print(f"✅ Saved: {gm_wide_path}")
        print(f"✅ Saved: {gm_long_path}")

if __name__ == "__main__":
    main()
