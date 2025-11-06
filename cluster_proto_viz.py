#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_proto_viz.py
Concrete visualizations & stats for DBSCAN prototypes.

Inputs expected (give paths via CLI; sensible defaults shown):
- exemplars_per_prototype.csv  (per-window metrics; REQUIRED)
    required cols: file, prototype_id, nearest_dist, viol_count_total, viol_sev_total
    optional cols: any feature-level severities like viol_sev_AltMSL, viol_sev_E1_RPM, ...
- distribution_by_prototype.csv  (prototype_id, n_windows) [optional; computed if missing]
- (optional) windows_dir/  raw time-series e.g., window_12345.csv for overlay plots

Outputs:
- ./proto_viz/  (PNG figures and CSV summaries)

Usage:
python3 cluster_proto_viz.py \
  --exemplars_csv /path/to/exemplars_per_prototype.csv \
  --dist_csv /path/to/distribution_by_prototype.csv \
  --windows_dir /path/to/dataset/after_examm2 \
  --out_dir outputs/proto_viz \
  --top_k 5
"""

import argparse
import os
import sys
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- util

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def has_cols(df, cols):
    return all(c in df.columns for c in cols)

def find_feature_sev_cols(df):
    # anything like viol_sev_*
    return [c for c in df.columns if c.startswith("viol_sev_")]

def clean_featname(c):
    # prettify "viol_sev_E1_EGT1" -> "E1 EGT1"
    return c.replace("viol_sev_", "").replace("_", " ")

def read_csv_loose(path):
    if path and Path(path).exists():
        return pd.read_csv(path)
    return None

# -------- core plots

def plot_cluster_size_vs_severity(df_ex, df_dist, out_dir: Path):
    # cluster size
    if df_dist is None or not has_cols(df_dist, ["prototype_id", "n_windows"]):
        df_dist = df_ex.groupby("prototype_id").size().reset_index(name="n_windows")
    # severity per cluster
    sev = df_ex.groupby("prototype_id")["viol_sev_total"].mean().reset_index(name="mean_severity")
    merged = pd.merge(df_dist, sev, on="prototype_id", how="inner")
    # risk index: small * severe
    total = merged["n_windows"].sum()
    merged["size_frac"] = merged["n_windows"] / max(total, 1)
    merged["risk_index"] = merged["mean_severity"] * (1.0 - merged["size_frac"])

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(merged["n_windows"], merged["mean_severity"], s=60, alpha=0.8)
    for _, r in merged.iterrows():
        ax.text(r["n_windows"], r["mean_severity"], str(int(r["prototype_id"])), fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("Cluster size (# windows)")
    ax.set_ylabel("Mean violation severity")
    ax.set_title("Cluster size vs mean severity (numbers = prototype_id)")
    fig.tight_layout()
    fig.savefig(out_dir / "size_vs_mean_severity.png", dpi=160)
    plt.close(fig)

    merged.sort_values("risk_index", ascending=False).to_csv(out_dir / "cluster_risk_ranking.csv", index=False)
    return merged.sort_values("risk_index", ascending=False)

def bar_top_prototypes(merged_risk, out_dir: Path, top_k=10):
    top = merged_risk.head(top_k)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(top["prototype_id"].astype(str), top["risk_index"])
    ax.set_xlabel("prototype_id")
    ax.set_ylabel("Risk Index = MeanSeverity * (1 - SizeFrac)")
    ax.set_title(f"Top {top_k} high-risk prototypes")
    fig.tight_layout()
    fig.savefig(out_dir / "top_risk_prototypes.png", dpi=160)
    plt.close(fig)

def parallel_coords_features(df_ex, out_dir: Path):
    feat_cols = find_feature_sev_cols(df_ex)
    if not feat_cols:
        return False  # nothing to do

    # aggregate per prototype (median more robust)
    agg = df_ex.groupby("prototype_id")[feat_cols].median().reset_index()
    # min-max scale for parallel coordinates
    vals = agg[feat_cols].values.astype(float)
    vmin = vals.min(axis=0)
    vmax = vals.max(axis=0)
    denom = np.where((vmax - vmin) == 0, 1.0, (vmax - vmin))
    scaled = (vals - vmin) / denom
    ax_labels = [clean_featname(c) for c in feat_cols]

    # draw
    fig, ax = plt.subplots(figsize=(12,6))
    colors = plt.cm.tab20(np.linspace(0,1,len(agg)))
    for i, row in enumerate(scaled):
        ax.plot(range(len(feat_cols)), row, color=colors[i], alpha=0.85, label=f"proto {int(agg['prototype_id'].iloc[i])}")
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(ax_labels, rotation=35, ha="right")
    ax.set_ylim(0,1)
    ax.set_title("Feature severity fingerprint per prototype (median, min-max scaled)")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "parallel_coords_feature_fingerprints.png", dpi=160)
    plt.close(fig)
    return True

def per_proto_feature_bars(df_ex, out_dir: Path, top_k=10):
    feat_cols = find_feature_sev_cols(df_ex)
    if not feat_cols:
        return False

    # long-form bars: mean severity per feature per prototype
    melted = (
        df_ex
        .groupby("prototype_id")[feat_cols].mean()
        .reset_index()
        .melt(id_vars="prototype_id", var_name="feature", value_name="avg_severity")
    )
    melted["feature_clean"] = melted["feature"].apply(clean_featname)

    # show only top_k prototypes by total avg severity for readability
    proto_order = (
        melted.groupby("prototype_id")["avg_severity"]
        .mean().sort_values(ascending=False).head(top_k).index.tolist()
    )
    plot_df = melted[melted["prototype_id"].isin(proto_order)]

    # one big barplot
    fig, ax = plt.subplots(figsize=(12,6))
    # simple manual "grouped bars": pivot and plot
    P = (plot_df
         .pivot(index="feature_clean", columns="prototype_id", values="avg_severity")
         .fillna(0.0)
         .sort_index())
    P.plot(kind="bar", ax=ax)
    ax.set_ylabel("Avg feature severity")
    ax.set_title(f"Avg per-feature severity — top {top_k} prototypes by overall severity")
    ax.legend(title="prototype_id", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "bars_feature_by_prototype.png", dpi=160)
    plt.close(fig)
    return True

def correlation_heatmaps(df_ex, out_dir: Path, max_protos=6):
    feat_cols = find_feature_sev_cols(df_ex)
    if not feat_cols:
        return False

    # choose prototypes with highest mean severity to focus
    order = (df_ex.groupby("prototype_id")["viol_sev_total"]
             .mean().sort_values(ascending=False).head(max_protos).index.tolist())

    for pid in order:
        sub = df_ex[df_ex["prototype_id"] == pid]
        if len(sub) < 3:
            continue
        corr = sub[feat_cols].corr()
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(feat_cols)))
        ax.set_yticks(range(len(feat_cols)))
        labels = [clean_featname(c) for c in feat_cols]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(f"Prototype {pid} — feature severity correlation")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"corr_proto_{pid}.png", dpi=160)
        plt.close(fig)
    return True

def overlay_timeseries_for_prototypes(df_ex, windows_dir: Path, out_dir: Path, top_proto_ids, 
                                     signals=("AltMSL","E1 RPM","E1 EGT1","IAS","NormAc"),
                                     max_windows_per_proto=60):
    """
    For each prototype in top_proto_ids, overlay raw time-series for a few signals to show consistency/spread.
    Assumes window CSVs exist in windows_dir with corresponding 'file' names from df_ex.
    """
    if windows_dir is None or not windows_dir.exists():
        return False

    for pid in top_proto_ids:
        sub = df_ex[df_ex["prototype_id"] == pid].copy()
        if sub.empty:
            continue
        # sample to avoid overplotting
        sub = sub.sample(min(len(sub), max_windows_per_proto), random_state=42)

        # try to read and stack
        for sig in signals:
            fig, ax = plt.subplots(figsize=(9,4))
            plotted = 0
            for _, r in sub.iterrows():
                fpath = windows_dir / r["file"]
                if not fpath.exists():
                    continue
                try:
                    W = pd.read_csv(fpath)
                    if sig not in W.columns:
                        continue
                    # If a column 'anomaly_flag' exists, we ignore it for overlay; we want raw sig
                    y = W[sig].values
                    if len(y) == 0:
                        continue
                    ax.plot(range(len(y)), y, alpha=0.1, linewidth=1)
                    plotted += 1
                except Exception:
                    continue
            ax.set_title(f"Prototype {pid} — overlay of '{sig}' across {plotted} windows")
            ax.set_xlabel("timestep")
            ax.set_ylabel(sig)
            fig.tight_layout()
            fig.savefig(out_dir / f"overlay_proto{pid}_{sig.replace(' ','_').replace('/','-')}.png", dpi=160)
            plt.close(fig)
    return True

# -------- main

def main():
    ap = argparse.ArgumentParser(
        description="Concrete visualizations for DBSCAN prototypes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python3 cluster_proto_viz.py \\
            --exemplars_csv /mnt/data/exemplars_per_prototype.csv \\
            --dist_csv /mnt/data/distribution_by_prototype.csv \\
            --windows_dir dataset/after_examm2 \\
            --out_dir outputs/proto_viz \\
            --top_k 6
        """)
    )
    ap.add_argument("--exemplars_csv", required=True, help="Path to exemplars_per_prototype.csv")
    ap.add_argument("--dist_csv", default="", help="Path to distribution_by_prototype.csv (optional)")
    ap.add_argument("--windows_dir", default="", help="Directory containing window_*.csv (optional for overlays)")
    ap.add_argument("--out_dir", default="outputs/proto_viz", help="Where to write figures & tables")
    ap.add_argument("--top_k", type=int, default=6, help="Top K prototypes to emphasize (risk & feature bars)")
    ap.add_argument("--overlay", action="store_true", help="Enable time-series overlay plots if windows_dir is set")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    df_ex = read_csv_loose(args.exemplars_csv)
    if df_ex is None:
        print(f"[error] Could not read {args.exemplars_csv}", file=sys.stderr)
        sys.exit(2)

    # Minimal required cols
    req_cols = ["file","prototype_id","nearest_dist","viol_count_total","viol_sev_total"]
    for c in req_cols:
        if c not in df_ex.columns:
            print(f"[error] Missing required column '{c}' in exemplars CSV.", file=sys.stderr)
            sys.exit(2)

    # normalize prototype_id to int
    try:
        df_ex["prototype_id"] = df_ex["prototype_id"].astype(int)
    except Exception:
        pass

    df_dist = read_csv_loose(args.dist_csv)

    # 1) size vs severity + risk ranking
    print("[info] Plotting size vs severity & computing risk index…")
    risk = plot_cluster_size_vs_severity(df_ex, df_dist, out_dir)
    bar_top_prototypes(risk, out_dir, top_k=args.top_k)
    top_proto_ids = risk["prototype_id"].head(args.top_k).tolist()

    # 2) parallel coords & bar summaries if feature severity columns exist
    feat_ok = parallel_coords_features(df_ex, out_dir)
    bars_ok = per_proto_feature_bars(df_ex, out_dir, top_k=args.top_k)
    if not feat_ok:
        print("[warn] No per-feature severity columns (viol_sev_*) found; skipping feature fingerprint plots.")
    if not bars_ok:
        print("[warn] No per-feature severity columns found; skipping grouped bar plot.")

    # 3) correlation heatmaps (for top-severity prototypes)
    corr_ok = correlation_heatmaps(df_ex, out_dir)
    if not corr_ok:
        print("[warn] No per-feature severity columns found; skipping correlation heatmaps.")

    # 4) time-series overlays for selected prototypes (requires windows_dir)
    windows_dir = Path(args.windows_dir) if args.windows_dir else None
    if args.overlay and windows_dir and windows_dir.exists():
        print("[info] Building time-series overlays for top prototypes…")
        overlay_timeseries_for_prototypes(
            df_ex, windows_dir, out_dir, top_proto_ids,
            signals=("AltMSL","E1 RPM","E1 EGT1","IAS","NormAc"),
            max_windows_per_proto=80
        )
    elif args.overlay:
        print("[warn] --overlay set but windows_dir missing or not found; skipping overlays.")

    # 5) summary CSVs
    # per-prototype summary
    proto_summary = (
        df_ex.groupby("prototype_id")
        .agg(
            n_windows=("file","count"),
            mean_dist=("nearest_dist","mean"),
            mean_viol_count=("viol_count_total","mean"),
            mean_viol_sev=("viol_sev_total","mean"),
        )
        .reset_index()
        .sort_values(["mean_viol_sev","n_windows"], ascending=[False, True])
    )
    proto_summary.to_csv(out_dir / "prototype_summary.csv", index=False)

    print(f"[done] Wrote figures & tables to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
