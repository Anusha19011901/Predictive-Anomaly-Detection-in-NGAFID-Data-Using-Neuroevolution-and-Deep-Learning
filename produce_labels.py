#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce binary anomaly labels (window + subsequence) from DBSCAN analysis outputs.

Inputs (existing from your run):
- exemplars_per_prototype.csv   (file, prototype_id, nearest_dist, viol_count_total, viol_sev_total)
- distribution_by_prototype.csv (prototype_id, n|count, mean_dist, mean_viol_cnt, mean_viol_sev)
- sensor_contrib_per_proto_zscore.csv          (prototype_id, feature, contribution)
- sensor_contrib_linear_probe_per_proto.csv    (prototype_id, feature, abs_coef)

Also needs access to window CSV filenames to derive subsequence IDs from file names.

Scoring (window-level):
  score = w_proto * proto_risk
        + w_engZ  * engine_zscore
        + w_accZ  * normac_zscore
        + w_lp    * engine_linear_probe

Where:
  proto_risk = norm(mean_viol_sev) + α * norm(mean_dist)
  engine_zscore = mean of z | for {E1 RPM, E1 FFlow, E1 EGT1, E1 CHT1} for that window's prototype
  normac_zscore = z | of NormAc for that prototype (or 0 if missing)
  engine_linear_probe = mean |coef| for the same engine features for that prototype

Labeling:
  - By default, label windows as anomaly if score >= P-th percentile (default P=85).
    (You can switch to an absolute threshold via --score_thresh.)
  - Subsequence label = 1 if >= K anomalous windows (default K=3) OR a run of >= R consecutive anomalies (default R=2).

Outputs:
  out_dir/labels_window.csv
  out_dir/labels_subsequence.csv

Usage example:
  python3 produce_labels.py \
    --exemplars_csv outputs/dbscan_eps2.1_run/analysis/exemplars_per_prototype.csv \
    --distribution_csv outputs/dbscan_eps2.1_run/analysis/distribution_by_prototype.csv \
    --zscore_csv outputs/proto_diagnostics/sensor_contrib_per_proto_zscore.csv \
    --linear_csv outputs/proto_diagnostics/sensor_contrib_linear_probe_per_proto.csv \
    --windows_dir exact_data/anomaly \
    --out_dir outputs/labels \
    --percentile 85 \
    --min_anom_windows 3 \
    --min_consec 2

"""
from __future__ import annotations
import argparse, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd

ENGINE_FEATS = ["E1 RPM","E1 FFlow","E1 EGT1","E1 CHT1"]
ENERGY_FEATS  = ["NormAc"]
AIRSPEED_FEAT = "IAS"  # optional bonus if desired

def read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return pd.read_csv(path)

def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    v = s.values
    mn, mx = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn == 0:
        return pd.Series(np.zeros_like(v), index=s.index)
    return (s - mn) / (mx - mn)

def guess_subsequence_id(filename: str) -> str:
    """
    Heuristic to extract a subsequence/group id from window file names.
    Adjust this to your actual naming. Examples we’ve seen:
      window_000123.csv                -> takes prefix 'window_000123'
      N513ND_before_4_338910_012.csv   -> could group by 'N513ND_before_4_338910'
    Here we strip trailing numeric chunk or extension to form a group key.
    """
    base = Path(filename).stem  # no .csv
    # Remove trailing numeric-like window index (e.g., _000123)
    m = re.match(r"(.+?)(?:_[0-9]+)$", base)
    if m:
        return m.group(1)
    return base

def compute_component_tables(df_dist, df_z, df_lin) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return:
       - proto_risk table: prototype_id -> proto_risk
       - z_engine table: prototype_id -> mean |z| across ENGINE_FEATS + NormAc separately
       - lin_engine table: prototype_id -> mean |coef| across ENGINE_FEATS
    """
    # Harmonize count column
    if "count" not in df_dist.columns and "n" in df_dist.columns:
        df_dist = df_dist.rename(columns={"n":"count"})

    # Prototype risk from mean_viol_sev + α * mean_dist (normalized)
    α = 0.6
    r1 = normalize_series(df_dist["mean_viol_sev"]) if "mean_viol_sev" in df_dist else pd.Series(0, index=df_dist.index)
    r2 = normalize_series(df_dist["mean_dist"]) if "mean_dist" in df_dist else pd.Series(0, index=df_dist.index)
    proto_risk = r1 + α * r2
    df_proto_risk = pd.DataFrame({"prototype_id": df_dist["prototype_id"], "proto_risk": proto_risk})

    # Z-score contributions per prototype
    def pivot_contrib(df_in, value_col, feats):
        # df_in: [prototype_id, feature, contribution/abs_coef]
        sub = df_in[df_in["feature"].isin(feats)].copy()
        if sub.empty:
            # all zeros fallback
            return pd.DataFrame({"prototype_id": df_in["prototype_id"].unique(), "score": 0.0})
        agg = sub.groupby("prototype_id")[value_col].mean().reset_index()
        agg = agg.rename(columns={value_col:"score"})
        return agg

    z_eng   = pivot_contrib(df_z,  "contribution", ENGINE_FEATS)
    z_acc   = pivot_contrib(df_z,  "contribution", ENERGY_FEATS)

    # Linear probe contributions
    lin_eng = pivot_contrib(df_lin, "abs_coef", ENGINE_FEATS)

    return df_proto_risk, z_eng, z_acc, lin_eng

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exemplars_csv", required=True, type=str)
    ap.add_argument("--distribution_csv", required=True, type=str)
    ap.add_argument("--zscore_csv", required=True, type=str)
    ap.add_argument("--linear_csv", required=True, type=str)
    ap.add_argument("--windows_dir", required=True, type=str, help="Directory that contains the window CSVs (for grouping).")
    ap.add_argument("--out_dir", required=True, type=str)

    # Labeling knobs
    ap.add_argument("--percentile", type=float, default=85.0, help="Percentile cutoff for anomaly label. Ignored if --score_thresh is set.")
    ap.add_argument("--score_thresh", type=float, default=None, help="Absolute score threshold; overrides percentile if given.")
    ap.add_argument("--min_anom_windows", type=int, default=3, help="Min # anomalous windows to label a subsequence as anomalous.")
    ap.add_argument("--min_consec", type=int, default=2, help="Min consecutive anomalous windows to flip subsequence label.")
    args = ap.parse_args()

    exemplars_csv = Path(args.exemplars_csv).expanduser().resolve()
    distribution_csv = Path(args.distribution_csv).expanduser().resolve()
    zscore_csv = Path(args.zscore_csv).expanduser().resolve()
    linear_csv = Path(args.linear_csv).expanduser().resolve()
    windows_dir = Path(args.windows_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_ex = read_csv(exemplars_csv, "exemplars_per_prototype.csv")
    df_dist = read_csv(distribution_csv, "distribution_by_prototype.csv")
    df_z = read_csv(zscore_csv, "sensor_contrib_per_proto_zscore.csv")
    df_lin = read_csv(linear_csv, "sensor_contrib_linear_probe_per_proto.csv")

    # Component tables
    df_proto_risk, z_eng, z_acc, lin_eng = compute_component_tables(df_dist, df_z, df_lin)

    # Merge prototype-level components into exemplars (window table)
    # Keep useful columns
    keep_cols = ["file","prototype_id","nearest_dist","viol_count_total","viol_sev_total"]
    dfw = df_ex[keep_cols].copy()
    dfw["prototype_id"] = dfw["prototype_id"].astype(int)

    dfw = dfw.merge(df_proto_risk, on="prototype_id", how="left")
    dfw = dfw.merge(z_eng.rename(columns={"score":"z_engine"}), on="prototype_id", how="left")
    dfw = dfw.merge(z_acc.rename(columns={"score":"z_normac"}), on="prototype_id", how="left")
    dfw = dfw.merge(lin_eng.rename(columns={"score":"lp_engine"}), on="prototype_id", how="left")

    for c in ["proto_risk","z_engine","z_normac","lp_engine"]:
        if c not in dfw.columns:
            dfw[c] = 0.0
        dfw[c] = dfw[c].fillna(0.0).astype(float)

    # Combine into a single score (weights chosen to emphasize engine cues + prototype risk)
    w_proto = 0.45
    w_engZ  = 0.30
    w_accZ  = 0.10
    w_lp    = 0.15

    # Normalize components to [0,1] before combining
    for comp in ["proto_risk","z_engine","z_normac","lp_engine"]:
        dfw[f"{comp}_norm"] = normalize_series(dfw[comp])

    dfw["anom_score"] = (
        w_proto * dfw["proto_risk_norm"] +
        w_engZ  * dfw["z_engine_norm"]   +
        w_accZ  * dfw["z_normac_norm"]   +
        w_lp    * dfw["lp_engine_norm"]
    )

    # Decide window labels
    if args.score_thresh is not None:
        thresh = float(args.score_thresh)
    else:
        p = float(args.percentile)
        thresh = np.nanpercentile(dfw["anom_score"].values, p)
    dfw["label_window"] = (dfw["anom_score"] >= thresh).astype(int)
    dfw["label_rule"] = np.where(dfw["label_window"]==1,
                                 f"score>= {thresh:.3f}",
                                 f"score< {thresh:.3f}")

    # Derive subsequence IDs (grouping)
    dfw["subsequence_id"] = dfw["file"].astype(str).apply(guess_subsequence_id)

    # Save window labels
    window_cols = [
        "file","subsequence_id","prototype_id",
        "nearest_dist","viol_count_total","viol_sev_total",
        "proto_risk","z_engine","z_normac","lp_engine",
        "anom_score","label_window","label_rule"
    ]
    dfw[window_cols].to_csv(out_dir / "labels_window.csv", index=False)

    # Roll up to subsequence labels
    # Anomalous if >= min_anom_windows or a run of >= min_consec anomalies
    rows = []
    for sid, sub in dfw.groupby("subsequence_id"):
        lab = sub["label_window"].values.tolist()
        count_anom = int(np.sum(lab))
        max_run = 0
        run = 0
        for v in lab:
            if v==1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        label_sub = int((count_anom >= args.min_anom_windows) or (max_run >= args.min_consec))

        rows.append({
            "subsequence_id": sid,
            "n_windows": int(len(sub)),
            "n_anom_windows": count_anom,
            "max_consec_anom": int(max_run),
            "mean_anom_score": float(np.mean(sub["anom_score"])),
            "proto_ids": ",".join(map(str, sub["prototype_id"].astype(int).unique().tolist())),
            "label_subsequence": label_sub,
            "rule": f"n_anom>={args.min_anom_windows} OR consec>={args.min_consec}"
        })
    dfs = pd.DataFrame(rows).sort_values(["label_subsequence","mean_anom_score","n_anom_windows"], ascending=[False,False,False])
    dfs.to_csv(out_dir / "labels_subsequence.csv", index=False)

    print(f"[OK] Wrote:\n  - {out_dir/'labels_window.csv'}\n  - {out_dir/'labels_subsequence.csv'}\n  Threshold used: {thresh:.4f}")

if __name__ == "__main__":
    main()
