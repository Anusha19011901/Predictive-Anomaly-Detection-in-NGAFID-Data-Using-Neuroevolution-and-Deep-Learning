# visualize_prototype_boxes.py
# Visuals for prototype-box explanations:
# 1) Heatmap of nearest_dist (rows=files, cols=window_idx) + annotations for top percentile
# 2) Bar chart of violation counts for a selected window
# 3) Sensor-level summary across all windows
# 4) Per-file nearest_dist timelines

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import shorten

def load_explanations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    req = {"file","window_idx","prototype_id","nearest_dist","top3_sensors"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    if "start_idx" not in df.columns:
        df["start_idx"] = df["window_idx"]
    return df

def get_sensor_columns(df: pd.DataFrame):
    viol_count_cols = [c for c in df.columns if c.startswith("viol_count_")]
    viol_sev_cols   = [c for c in df.columns if c.startswith("viol_sev_")]
    sensors = [c.replace("viol_count_", "") for c in viol_count_cols]
    return sensors, viol_count_cols, viol_sev_cols

# ---------------- Heatmap ----------------
def plot_heatmap(df: pd.DataFrame, out_path: str, annotate_pct: float = 95.0, max_annos: int = 40):
    sensors, _, _ = get_sensor_columns(df)
    files = sorted(df["file"].unique())
    max_w = int(df["window_idx"].max())
    H = np.full((len(files), max_w+1), np.nan, dtype=float)

    for r, fname in enumerate(files):
        sub = df[df["file"] == fname]
        H[r, sub["window_idx"].values] = sub["nearest_dist"].values

    flat = H[~np.isnan(H)]
    if flat.size == 0:
        raise ValueError("No nearest_dist values to plot.")
    pthr = float(np.percentile(flat, annotate_pct))

    fig, ax = plt.subplots(figsize=(14, max(3, 0.5*len(files))))
    im = ax.imshow(H, aspect="auto", interpolation="nearest")
    ax.set_title(f"All Windows — nearest_dist (rows: files, cols: window_idx)")
    ax.set_xlabel("window_idx")
    ax.set_ylabel("file")

    yticks = np.arange(len(files))
    ylabels = [shorten(f, width=60, placeholder="…") for f in files]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    if max_w > 0:
        ax.set_xticks(np.linspace(0, max_w, num=min(max_w+1, 12), dtype=int))

    # annotate windows above percentile
    annos = []
    for r, fname in enumerate(files):
        sub = df[df["file"] == fname]
        hit = sub[sub["nearest_dist"] >= pthr]
        for _, row in hit.iterrows():
            w = int(row["window_idx"])
            label = ", ".join([s.strip() for s in str(row["top3_sensors"]).split(",")[:2]])
            if not np.isnan(H[r, w]) and label:
                annos.append((r, w, label, float(row["nearest_dist"])))
    annos.sort(key=lambda x: x[3], reverse=True)
    for (ri, cj, lab, _) in annos[:max_annos]:
        ax.text(cj, ri, lab, ha="left", va="bottom", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"nearest_dist (P{int(annotate_pct)} = {pthr:.2f})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")

# ---------------- Selected window bar ----------------
def plot_bar_for_window(df: pd.DataFrame, out_path: str, file_name: str = None, window_idx: int = None):
    sensors, count_cols, _ = get_sensor_columns(df)

    if file_name is not None and window_idx is not None:
        row = df[(df["file"] == file_name) & (df["window_idx"] == window_idx)]
        if row.empty:
            raise ValueError("No row matches the given file and window_idx.")
        row = row.iloc[0]
    else:
        row = df.iloc[int(df["nearest_dist"].idxmax())]

    counts = row[count_cols].astype(float).values
    order = np.argsort(-counts)
    counts_sorted = counts[order]
    sensors_sorted = [sensors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(sensors_sorted)), counts_sorted)
    ax.set_xticks(range(len(sensors_sorted)))
    ax.set_xticklabels(sensors_sorted, rotation=45, ha="right")
    ax.set_ylabel("Violation count (out of 30)")
    ax.set_title(f"Window violations — file={row['file']}, window_idx={int(row['window_idx'])}, dist={float(row['nearest_dist']):.2f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")

# ---------------- Sensor summary ----------------
def plot_sensor_summary(df: pd.DataFrame, out_path: str):
    sensors, count_cols, sev_cols = get_sensor_columns(df)
    counts_mat = df[count_cols].astype(float).values   # [N, F]
    sevs_mat   = df[sev_cols].astype(float).values     # [N, F]

    mean_counts = counts_mat.mean(axis=0)              # avg #positions out of 30
    median_sev  = np.median(sevs_mat, axis=0)          # robust severity

    # Two separate figures (one chart per figure for clarity)
    # A) mean violation count
    order = np.argsort(-mean_counts)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(range(len(sensors)), mean_counts[order])
    ax1.set_xticks(range(len(sensors)))
    ax1.set_xticklabels([sensors[i] for i in order], rotation=45, ha="right")
    ax1.set_ylabel("Mean violation count (0–30)")
    ax1.set_title("Sensor summary — mean violation count across all windows")
    fig1.tight_layout()
    out1 = out_path.replace(".png", "_mean_count.png")
    fig1.savefig(out1, dpi=200)
    plt.close(fig1)
    print(f"[saved] {out1}")

    # B) median severity
    order2 = np.argsort(-median_sev)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(range(len(sensors)), median_sev[order2])
    ax2.set_xticks(range(len(sensors)))
    ax2.set_xticklabels([sensors[i] for i in order2], rotation=45, ha="right")
    ax2.set_ylabel("Median severity (normalized sum)")
    ax2.set_title("Sensor summary — median violation severity across all windows")
    fig2.tight_layout()
    out2 = out_path.replace(".png", "_median_sev.png")
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)
    print(f"[saved] {out2}")

# ---------------- Per-file nearest_dist timelines ----------------
def plot_file_timelines(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for fname, sub in df.groupby("file"):
        sub = sub.sort_values("window_idx")
        fig, ax = plt.subplots(figsize=(10, 2.8))
        ax.plot(sub["window_idx"].values, sub["nearest_dist"].values, marker="o", linestyle="-")
        ax.set_title(f"nearest_dist timeline — {fname}")
        ax.set_xlabel("window_idx")
        ax.set_ylabel("nearest_dist")
        fig.tight_layout()
        path = os.path.join(out_dir, f"timeline_{os.path.splitext(os.path.basename(fname))[0]}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[saved] {path}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="prototype_explanations_before.csv (or after)")
    ap.add_argument("--outdir", default="outputs/plots", help="where to save plots")
    ap.add_argument("--annotate_pct", type=float, default=95.0, help="percentile for heatmap annotations")
    ap.add_argument("--max_annos", type=int, default=40, help="max number of labels to draw on heatmap")
    ap.add_argument("--select_file", default=None, help="file name for selected window bar plot")
    ap.add_argument("--select_window", type=int, default=None, help="window_idx for selected window bar plot")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_explanations(args.csv)

    # 1) Heatmap
    heatmap_path = os.path.join(args.outdir, "heatmap_nearest_dist.png")
    plot_heatmap(df, heatmap_path, annotate_pct=args.annotate_pct, max_annos=args.max_annos)

    # 2) Bar for a selected (or max) window
    bar_path = os.path.join(args.outdir, "selected_window_violation_counts.png")
    plot_bar_for_window(df, bar_path, file_name=args.select_file, window_idx=args.select_window)

    # 3) Sensor summary (mean count + median severity)
    summary_path = os.path.join(args.outdir, "sensor_summary.png")
    plot_sensor_summary(df, summary_path)

    # 4) Per-file timelines
    file_tl_dir = os.path.join(args.outdir, "timelines")
    plot_file_timelines(df, file_tl_dir)

if __name__ == "__main__":
    main()
