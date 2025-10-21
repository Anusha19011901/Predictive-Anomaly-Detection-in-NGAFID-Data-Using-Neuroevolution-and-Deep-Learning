#!/usr/bin/env python3
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sweep_plots(summary_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(summary_csv)
    if df.empty: return
    df = df.sort_values("eps")

    # clusters vs eps
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df["eps"], df["n_clusters"], marker="o")
    ax.set_xlabel("eps"); ax.set_ylabel("#clusters")
    ax.set_title("DBSCAN sweep: clusters vs eps")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"clusters_vs_eps.png"), dpi=180); plt.close(fig)

    # noise fraction vs eps
    if "n_noise" in df.columns:
        total = (df["n_clusters"] + df["n_noise"]).replace(0, np.nan)
        nf = df["n_noise"] / total
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(df["eps"], nf, marker="o")
        ax.set_xlabel("eps"); ax.set_ylabel("noise fraction")
        ax.set_title("DBSCAN sweep: noise fraction vs eps")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir,"noise_vs_eps.png"), dpi=180); plt.close(fig)

def detail_plots(expl_csv: str, out_dir: str, annotate_pct: float, max_annos: int):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(expl_csv)
    if df.empty: return

    # heatmap of nearest_dist per window (by file)
    df["file_ord"] = df["file"].astype("category").cat.codes
    pivot = df.pivot_table(index="file_ord", columns="window_idx", values="nearest_dist", aggfunc="first")
    fig, ax = plt.subplots(figsize=(8,4.5))
    im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
    ax.set_xlabel("window_idx"); ax.set_ylabel("file (ordinal)")
    ax.set_title("Nearest distance to DBSCAN prototype box")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"nearest_dist_heatmap.png"), dpi=180); plt.close(fig)

    # annotate top sensors for extreme windows
    thr = np.percentile(df["nearest_dist"], annotate_pct)
    top = df[df["nearest_dist"] >= thr].copy()
    if "top3_sensors" in top.columns:
        counts = top["top3_sensors"].str.get_dummies(sep=", ").sum().sort_values(ascending=False)
        counts = counts.head(max_annos)
        fig, ax = plt.subplots(figsize=(6,4))
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Top sensors on high-distance windows (≥ P{annotate_pct})")
        ax.set_ylabel("count")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir,"top_sensors_extremes.png"), dpi=180); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap1 = sub.add_parser("sweep")
    ap1.add_argument("--summary_csv", required=True)
    ap1.add_argument("--out_dir", default="outputs/dbscan/plots")

    ap2 = sub.add_parser("detail")
    ap2.add_argument("--explanations_csv", required=True)
    ap2.add_argument("--out_dir", default="outputs/dbscan/plots")
    ap2.add_argument("--annotate_pct", type=float, default=95.0)
    ap2.add_argument("--max_annos", type=int, default=20)

    args = ap.parse_args()
    if args.cmd == "sweep":
        sweep_plots(args.summary_csv, args.out_dir)
    else:
        detail_plots(args.explanations_csv, args.out_dir, args.annotate_pct, args.max_annos)

if __name__ == "__main__":
    main()
