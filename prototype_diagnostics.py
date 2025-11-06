#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototype Diagnostics for NGAFID DBSCAN runs
--------------------------------------------
Generates robust plots from /analysis CSVs and computes sensor contributions
per prototype. Handles common pitfalls:
 - 'n' vs 'count' column mismatch in distribution_by_prototype.csv
 - empty overlays due to wrong windows_dir
 - heatmaps turning "all red" due to NaNs / zero-variance columns

Outputs (examples):
  out_dir/
    cluster_sizes.png
    mean_severity_by_proto.png
    mean_count_by_proto.png
    mean_nearest_distance_by_proto.png
    cluster_summary_for_slides.csv
    sensor_contrib_per_proto_zscore.csv
    sensor_contrib_global_zscore.csv
    sensor_contrib_linear_probe_per_proto.csv
    sensor_contrib_linear_probe_global.csv
    corr_proto_<k>.png         (optional, if --correlations is set)
    overlay_proto_<k>.png      (optional, if --overlays is set)

USAGE (example):
  python3 prototype_diagnostics.py \
    --windows_dir dataset/after_examm2/windows_30_25 \
    --exemplars_csv outputs/dbscan_eps2.1_run/analysis/exemplars_per_prototype.csv \
    --distribution_csv outputs/dbscan_eps2.1_run/analysis/distribution_by_prototype.csv \
    --out_dir outputs/proto_diagnostics \
    --small_cluster_thresh 20 \
    --correlations \
    --linear_probe \
    --baseline_windows_dir dataset/after_examm2/windows_30_25

Notes:
 - If you don't have a separate "healthy" baseline, omit --baseline_windows_dir.
   The script falls back to a global baseline (all windows).
 - Linear probe needs scikit-learn installed.
"""

import argparse
import os
from pathlib import Path
import sys
import math
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: for linear probe
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_csv_required(p: Path, name: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing required CSV: {name} at {p}")
    try:
        return pd.read_csv(p)
    except Exception as e:
        raise RuntimeError(f"Could not read {name} at {p}: {e}")

def safe_corr(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Remove zero-variance columns, replace NaN/Inf, clip to [-1,1].
    Returns None if nothing to correlate.
    """
    if df is None or df.empty:
        return None
    # numeric only
    df_num = df.select_dtypes(include=[np.number])
    # drop constant columns
    std = df_num.std(axis=0, ddof=0)
    keep_cols = std[std > 0].index.tolist()
    df2 = df_num[keep_cols]
    if df2.empty:
        return None
    c = df2.corr()
    c = c.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return c.clip(lower=-1.0, upper=1.0)

def plot_bar(xlabels, values, title, xlabel, ylabel, out_png: Path):
    plt.figure(figsize=(9, 4.8))
    plt.bar([str(x) for x in xlabels], values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def auto_feature_columns(df: pd.DataFrame, include_regex: str | None = None, exclude_regex: str | None = None) -> list[str]:
    """
    Heuristic to pick sensor columns from a window CSV.
    - includes numeric columns only
    - excludes common non-signal columns (index, label, time)
    - optional regex filters
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_like = {"index", "idx", "label", "y", "time", "timestamp", "t", "window_id"}
    # remove obvious admin cols
    filtered = [c for c in num_cols if c.lower() not in drop_like]
    if include_regex:
        rgx = re.compile(include_regex)
        filtered = [c for c in filtered if rgx.search(c)]
    if exclude_regex:
        rgx = re.compile(exclude_regex)
        filtered = [c for c in filtered if not rgx.search(c)]
    return filtered

def load_window_csvs(windows_dir: Path, filenames: list[str], feature_cols: list[str]) -> pd.DataFrame:
    """
    Concatenate the selected feature columns for multiple window files.
    Returns a DataFrame with shape [sum_rows, len(feature_cols)].
    Skips files that do not exist; warns if many are missing.
    """
    frames = []
    missing = 0
    for fn in filenames:
        fp = (windows_dir / fn)
        if not fp.exists():
            missing += 1
            continue
        try:
            df = pd.read_csv(fp)
            if not feature_cols:
                # discover on first valid file
                feature_cols = auto_feature_columns(df)
            frames.append(df[feature_cols])
        except Exception as e:
            warnings.warn(f"Failed reading {fp}: {e}")
    if missing > 0:
        warnings.warn(f"{missing} window files listed but not found under {windows_dir}")
    if not frames:
        return pd.DataFrame(columns=feature_cols)
    return pd.concat(frames, ignore_index=True)

def summarize_window_means(windows_dir: Path, filenames: list[str], feature_cols: list[str]) -> pd.DataFrame:
    """
    Returns per-window feature means table shape [n_windows, n_features].
    """
    rows = []
    missing = 0
    for fn in filenames:
        fp = (windows_dir / fn)
        if not fp.exists():
            missing += 1
            continue
        df = pd.read_csv(fp)
        if not feature_cols:
            feature_cols = auto_feature_columns(df)
        if not feature_cols:
            continue
        row = df[feature_cols].mean(axis=0, numeric_only=True).to_dict()
        row["__file"] = fn
        rows.append(row)
    if missing > 0:
        warnings.warn(f"{missing} window files listed but not found under {windows_dir}")
    if not rows:
        return pd.DataFrame(columns=(feature_cols + ["__file"]))
    return pd.DataFrame(rows)

def compute_baseline_stats(windows_dir: Path | None,
                           all_files: list[str],
                           feature_cols: list[str]) -> dict[str, pd.Series]:
    """
    If a baseline windows_dir is given, load all_files there to build mean/std.
    Otherwise, use global baseline from the union in windows_dir of exemplars.
    """
    if windows_dir is None:
        # no baseline provided — we'll defer to caller
        return {}

    # Try to sample a reasonable number if huge (cap to 500 windows)
    sample_files = all_files
    if len(sample_files) > 500:
        rng = np.random.default_rng(42)
        sample_files = list(rng.choice(sample_files, size=500, replace=False))

    X = load_window_csvs(windows_dir, sample_files, feature_cols)
    if X.empty:
        warnings.warn("Baseline windows produced empty frame; using zeros.")
        mean = pd.Series(0.0, index=feature_cols)
        std = pd.Series(1.0, index=feature_cols)
    else:
        mean = X.mean(axis=0, numeric_only=True)
        std = X.std(axis=0, ddof=0, numeric_only=True).replace(0, 1e-9)
        # Align to feature_cols to be safe
        mean = mean.reindex(feature_cols, fill_value=0.0)
        std = std.reindex(feature_cols, fill_value=1.0)
    return {"mean": mean, "std": std}

def zscore_contributions_for_proto(windows_dir: Path,
                                   filenames: list[str],
                                   feature_cols: list[str],
                                   baseline_stats: dict | None) -> pd.Series:
    """
    Mean absolute z-score across all rows in all windows for a prototype.
    """
    X = load_window_csvs(windows_dir, filenames, feature_cols)
    if X.empty or not feature_cols:
        return pd.Series(dtype=float)
    if not baseline_stats:
        # global baseline
        mu = X.mean(axis=0, numeric_only=True)
        sigma = X.std(axis=0, ddof=0, numeric_only=True).replace(0, 1e-9)
    else:
        mu = baseline_stats["mean"].reindex(feature_cols, fill_value=0.0)
        sigma = baseline_stats["std"].reindex(feature_cols, fill_value=1.0).replace(0, 1e-9)

    Z = (X[feature_cols] - mu) / sigma
    mabs = Z.abs().mean(axis=0, numeric_only=True)
    return mabs

def linear_probe_contributions(windows_dir: Path,
                               df_exemplars: pd.DataFrame,
                               feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """
    Build a per-window mean feature matrix and train a multinomial (OVR) LR.
    Returns:
      per_proto_df: rows = (prototype_id, feature, |coef|)
      global_df: mean |coef| per feature across classes (Series)
    """
    if not SKLEARN_AVAILABLE:
        warnings.warn("scikit-learn not available; skipping linear probe.")
        return None, None

    groups = []
    for pid, sub in df_exemplars.groupby("prototype_id"):
        filenames = sub["file"].tolist()
        wi = summarize_window_means(windows_dir, filenames, feature_cols)
        if wi.empty:
            continue
        wi["prototype_id"] = pid
        groups.append(wi)
    if not groups:
        warnings.warn("No per-window summaries available; skipping linear probe.")
        return None, None

    W = pd.concat(groups, ignore_index=True)
    if "__file" in W.columns:
        W = W.drop(columns=["__file"])
    y = W["prototype_id"].astype(int).values
    X = W[feature_cols].values
    if X.size == 0:
        warnings.warn("No features found for linear probe.")
        return None, None

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=4000, multi_class="ovr", n_jobs=None)
    )
    clf.fit(X, y)
    lr = clf.named_steps["logisticregression"]
    # Coeff shape: [n_classes, n_features], class order = lr.classes_
    abs_coef = np.abs(lr.coef_)
    class_ids = lr.classes_

    per_rows = []
    for i, cls in enumerate(class_ids):
        for j, feat in enumerate(feature_cols):
            per_rows.append({"prototype_id": int(cls),
                             "feature": feat,
                             "abs_coef": float(abs_coef[i, j])})
    per_df = pd.DataFrame(per_rows)

    global_mean = pd.Series(abs_coef.mean(axis=0), index=feature_cols, name="abs_coef_mean")
    return per_df, global_mean


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Prototype diagnostics + sensor contributions")
    ap.add_argument("--windows_dir", type=str, required=True,
                    help="Directory containing per-window CSV files used to create the analysis artifacts.")
    ap.add_argument("--exemplars_csv", type=str, required=True,
                    help="Path to outputs/.../analysis/exemplars_per_prototype.csv")
    ap.add_argument("--distribution_csv", type=str, required=True,
                    help="Path to outputs/.../analysis/distribution_by_prototype.csv")
    ap.add_argument("--scores_csv", type=str, default=None,
                    help="(Optional) Path to a scores csv if needed; not required for core plots.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Directory to write plots and csvs.")
    ap.add_argument("--small_cluster_thresh", type=int, default=20,
                    help="Clusters smaller than this may be flagged as 'small'.")
    ap.add_argument("--include_features", type=str, default=None,
                    help="Regex to include only matching feature columns from window CSVs.")
    ap.add_argument("--exclude_features", type=str, default=None,
                    help="Regex to exclude matching feature columns from window CSVs.")
    ap.add_argument("--correlations", action="store_true",
                    help="If set, compute and save correlation heatmaps per prototype.")
    ap.add_argument("--overlays", action="store_true",
                    help="If set, attempt simple overlay plots per prototype (means with CI).")
    ap.add_argument("--baseline_windows_dir", type=str, default=None,
                    help="Optional directory with 'healthy' windows to build baseline for z-scores.")
    ap.add_argument("--no_linear_probe", dest="linear_probe", action="store_false",
                    help="Disable linear-probe contributions even if sklearn is available.")
    ap.add_argument("--linear_probe", dest="linear_probe", action="store_true",
                    help=argparse.SUPPRESS)
    ap.set_defaults(linear_probe=True)
    args = ap.parse_args()

    windows_dir = Path(args.windows_dir).expanduser().resolve()
    assert windows_dir.exists(), f"windows_dir not found: {windows_dir}"

    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())

    exemplars_csv = Path(args.exemplars_csv).expanduser().resolve()
    distribution_csv = Path(args.distribution_csv).expanduser().resolve()

    df_ex = read_csv_required(exemplars_csv, "exemplars_per_prototype.csv")
    df_dist = read_csv_required(distribution_csv, "distribution_by_prototype.csv")

    # ---- Harmonize column names in distribution (n -> count)
    if "count" not in df_dist.columns and "n" in df_dist.columns:
        df_dist = df_dist.rename(columns={"n": "count"})
    # Expected remaining columns:
    # prototype_id, count, mean_dist, mean_viol_cnt, mean_viol_sev

    # Basic sanity plots
    plot_bar(
        xlabels=df_dist["prototype_id"],
        values=df_dist["count"],
        title="Cluster sizes (windows per prototype)",
        xlabel="Prototype ID", ylabel="Windows",
        out_png=out_dir / "cluster_sizes.png"
    )
    plot_bar(
        xlabels=df_dist["prototype_id"],
        values=df_dist.get("mean_viol_sev", pd.Series([0]*len(df_dist))),
        title="Mean violation severity by prototype",
        xlabel="Prototype ID", ylabel="Mean severity",
        out_png=out_dir / "mean_severity_by_proto.png"
    )
    plot_bar(
        xlabels=df_dist["prototype_id"],
        values=df_dist.get("mean_viol_cnt", pd.Series([0]*len(df_dist))),
        title="Mean violation count by prototype",
        xlabel="Prototype ID", ylabel="Mean count",
        out_png=out_dir / "mean_count_by_proto.png"
    )
    plot_bar(
        xlabels=df_dist["prototype_id"],
        values=df_dist.get("mean_dist", pd.Series([0]*len(df_dist))),
        title="Mean exemplar nearest distance by prototype",
        xlabel="Prototype ID", ylabel="Mean distance",
        out_png=out_dir / "mean_nearest_distance_by_proto.png"
    )

    # Slide-ready summary
    slide_cols_map = {
        "prototype_id": "prototype_id",
        "count": "windows",
        "mean_viol_cnt": "mean_violation_count",
        "mean_viol_sev": "mean_violation_severity",
        "mean_dist": "mean_exemplar_distance",
    }
    df_slide = df_dist.rename(columns=slide_cols_map)[list(slide_cols_map.values())]
    df_slide.to_csv(out_dir / "cluster_summary_for_slides.csv", index=False)

    # Determine feature columns from any one window file (fallback).
    # Try the first exemplar file that exists.
    sample_file = None
    for f in df_ex["file"]:
        fp = windows_dir / f
        if fp.exists():
            sample_file = fp
            break
    feature_cols = []
    if sample_file is not None:
        tmp = pd.read_csv(sample_file)
        feature_cols = auto_feature_columns(tmp, args.include_features, args.exclude_features)
    if not feature_cols:
        # final fallback: introspect all numeric columns later during loads
        warnings.warn("Could not auto-detect feature columns from sample file; will infer per-load.")

    # Baseline stats for z-scores
    baseline_dir = Path(args.baseline_windows_dir).expanduser().resolve() if args.baseline_windows_dir else None
    if baseline_dir and not baseline_dir.exists():
        warnings.warn(f"baseline_windows_dir not found: {baseline_dir}, ignoring.")
        baseline_dir = None

    # Build a list of all referenced window filenames for baseline computation
    all_files = df_ex["file"].dropna().astype(str).tolist()
    baseline_stats = compute_baseline_stats(baseline_dir if baseline_dir else windows_dir,
                                           all_files, feature_cols if feature_cols else [])

    # ----------------------------
    # Z-score contributions
    # ----------------------------
    z_rows = []
    for pid, sub in df_ex.groupby("prototype_id"):
        file_list = sub["file"].dropna().astype(str).tolist()
        mabs = zscore_contributions_for_proto(windows_dir, file_list,
                                              feature_cols if feature_cols else [],
                                              baseline_stats if baseline_stats else None)
        if mabs.empty:
            continue
        for feat, val in mabs.items():
            z_rows.append({"prototype_id": int(pid), "feature": feat, "contribution": float(val)})

    if z_rows:
        df_z = pd.DataFrame(z_rows)
        df_z.to_csv(out_dir / "sensor_contrib_per_proto_zscore.csv", index=False)

        # Global (weighted by cluster size)
        weights = df_dist.set_index("prototype_id")["count"]
        # Align weights
        def wgt(pid): return float(weights.get(pid, 1.0))
        g = []
        for feat, grp in df_z.groupby("feature"):
            num = (grp["contribution"] * grp["prototype_id"].map(wgt)).sum()
            den = grp["prototype_id"].map(wgt).sum()
            g.append({"feature": feat, "contribution_weighted": float(num / max(den, 1e-9))})
        df_g = pd.DataFrame(g).sort_values("contribution_weighted", ascending=False)
        df_g.to_csv(out_dir / "sensor_contrib_global_zscore.csv", index=False)

    # ----------------------------
    # Linear probe contributions (optional)
    # ----------------------------
    if args.linear_probe and SKLEARN_AVAILABLE:
        per_df, global_mean = linear_probe_contributions(windows_dir, df_ex, feature_cols if feature_cols else [])
        if per_df is not None:
            per_df.to_csv(out_dir / "sensor_contrib_linear_probe_per_proto.csv", index=False)
        if global_mean is not None:
            global_mean.sort_values(ascending=False).to_csv(out_dir / "sensor_contrib_linear_probe_global.csv", header=True)
    elif args.linear_probe and not SKLEARN_AVAILABLE:
        warnings.warn("Linear probe requested but scikit-learn is not installed; skipping.")

    # ----------------------------
    # Correlations per prototype (optional)
    # ----------------------------
    if args.correlations:
        for pid, sub in df_ex.groupby("prototype_id"):
            file_list = sub["file"].dropna().astype(str).tolist()
            X = load_window_csvs(windows_dir, file_list, feature_cols if feature_cols else [])
            c = safe_corr(X)
            if c is None or c.empty:
                warnings.warn(f"Skipping corr for prototype {pid}: no usable data.")
                continue
            plt.figure(figsize=(7.2, 6.4))
            plt.imshow(c.values, vmin=-1, vmax=1)
            plt.xticks(ticks=range(len(c.columns)), labels=c.columns, rotation=90)
            plt.yticks(ticks=range(len(c.index)), labels=c.index)
            plt.title(f"Correlation heatmap (proto {pid})")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(out_dir / f"corr_proto_{int(pid)}.png", dpi=180)
            plt.close()

    # ----------------------------
    # Simple overlays per prototype (optional)
    # ----------------------------
    if args.overlays:
        # For overlays we’ll plot per-feature mean trajectory with 95% CI if time-like axis exists.
        # Many NGAFID windows are fixed-length time series rows; we assume each window CSV has
        # shape [T, features]. We'll plot only a small set (top-k by variance) to avoid clutter.
        MAX_FEATURES = 8

        for pid, sub in df_ex.groupby("prototype_id"):
            file_list = sub["file"].dropna().astype(str).tolist()
            # Load a small sample of windows (cap)
            if len(file_list) > 50:
                rng = np.random.default_rng(0)
                file_list = list(rng.choice(file_list, size=50, replace=False))

            # Stack windows; ensure consistent T across windows by truncation to min length
            stacks = []
            valid_cols = None
            minT = math.inf
            for fn in file_list:
                fp = windows_dir / fn
                if not fp.exists():
                    continue
                dfw = pd.read_csv(fp)
                fcols = auto_feature_columns(dfw, args.include_features, args.exclude_features)
                if not fcols:
                    continue
                if valid_cols is None:
                    valid_cols = fcols
                else:
                    # intersect
                    valid_cols = [c for c in valid_cols if c in fcols]
                minT = min(minT, len(dfw))
                stacks.append(dfw)

            if not stacks or valid_cols is None or minT == math.inf or minT == 0:
                warnings.warn(f"No data to overlay for proto {pid}")
                continue

            # pick features with highest variance across concatenated slices
            cat = pd.concat([s[valid_cols].iloc[:minT] for s in stacks], axis=0, ignore_index=True)
            var_rank = cat.var(axis=0, ddof=0).sort_values(ascending=False)
            use_feats = var_rank.index[:min(MAX_FEATURES, len(var_rank))].tolist()

            # build 3D array [N_windows, T, F]
            arr = np.stack([s[use_feats].iloc[:minT].to_numpy() for s in stacks], axis=0)  # [N, T, F]
            mean = arr.mean(axis=0)   # [T, F]
            std = arr.std(axis=0, ddof=0)
            n = arr.shape[0]
            se = std / np.sqrt(max(n, 1))
            # 95% CI
            ci = 1.96 * se

            # plot one figure with subplots per feature
            nF = len(use_feats)
            ncols = min(3, nF)
            nrows = math.ceil(nF / ncols)
            plt.figure(figsize=(4*ncols, 2.6*nrows))
            t = np.arange(minT)
            for i, feat in enumerate(use_feats, start=1):
                ax = plt.subplot(nrows, ncols, i)
                ax.plot(t, mean[:, i-1], label=f"{feat} mean")
                ax.fill_between(t, mean[:, i-1]-ci[:, i-1], mean[:, i-1]+ci[:, i-1], alpha=0.3)
                ax.set_title(feat)
                ax.set_xlabel("t (samples)")
                ax.set_ylabel(feat)
            plt.suptitle(f"Overlay (mean ±95% CI) — prototype {int(pid)}")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(out_dir / f"overlay_proto_{int(pid)}.png", dpi=180)
            plt.close()

    print(f"Done. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
