# visualize_prototype_boxes_errors.py
# Visuals for prototype-box XAI on EXAMM error windows.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_explanations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"file","window_idx","nearest_dist","top3_sensors"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    # sort by file then window_idx
    df = df.sort_values(["file","window_idx"]).reset_index(drop=True)
    return df

def get_sensor_cols(df: pd.DataFrame):
    viol_count_cols = [c for c in df.columns if c.startswith("viol_count_")]
    viol_sev_cols   = [c for c in df.columns if c.startswith("viol_sev_")]
    sensors = [c.replace("viol_count_","") for c in viol_count_cols]
    return sensors, viol_count_cols, viol_sev_cols

def plot_all_windows_heatmap(df: pd.DataFrame, outdir: str, annotate_pct: float = 95.0, max_annos: int = 30):
    files = df["file"].unique().tolist()
    mats = []
    col_counts = []

    # Build a ragged heatmap by padding each file row to the max width
    for f in files:
        sub = df[df["file"] == f].sort_values("window_idx")
        vals = sub["nearest_dist"].to_numpy().astype(float)
        mats.append(vals)
        col_counts.append(len(vals))
    max_len = max(col_counts) if col_counts else 0
    if max_len == 0:
        print("No windows to plot.")
        return

    H = len(files)
    W = max_len
    M = np.full((H, W), np.nan)
    for i, vals in enumerate(mats):
        M[i, :len(vals)] = vals

    # global percentile threshold on the observed nearest_dist values
    flat_vals = np.concatenate([v for v in mats if len(v) > 0])
    thr = np.percentile(flat_vals, annotate_pct)

    fig, ax = plt.subplots(figsize=(min(16, 2 + W*0.08), 1.5 + H*0.35))
    im = ax.imshow(M, aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("nearest_dist")

    ax.set_yticks(np.arange(H))
    ax.set_yticklabels([os.path.basename(f) for f in files], fontsize=8)
    ax.set_xlabel("window_idx")
    ax.set_title(f"All windows (error mode): nearest_dist heatmap (annotate ≥ P{int(annotate_pct)})")

    # annotations for high-distance windows
    annos = 0
    for i, f in enumerate(files):
        sub = df[df["file"] == f].sort_values("window_idx")
        for _, row in sub.iterrows():
            j = int(row["window_idx"])
            # find column index in ragged row (window_idx assumed dense starting at 0; if not, offset by min)
            # safer: map order by position within this file
            # Build an index from position in sub
        # second pass: annotate by position to avoid gaps
        sub_positions = sub.reset_index(drop=True)
        for pos in range(len(sub_positions)):
            row = sub_positions.iloc[pos]
            nd = float(row["nearest_dist"])
            if nd >= thr and annos < max_annos:
                ax.scatter([pos], [i], marker='o', s=10)
                txt = str(row.get("top3_sensors",""))
                if txt:
                    ax.text(pos+0.2, i+0.15, txt[:30], fontsize=6, rotation=15)
                annos += 1
            if annos >= max_annos:
                break

    ensure_dir(outdir)
    out_path = os.path.join(outdir, "errors_all_windows_heatmap.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved heatmap → {out_path}")

def plot_window_bars(df: pd.DataFrame, outdir: str, sel_file: str, sel_window: int):
    row = df[(df["file"] == sel_file) & (df["window_idx"] == sel_window)]
    if row.empty:
        print(f"No row found for file={sel_file}, window_idx={sel_window}")
        return
    row = row.iloc[0]
    sensors, viol_count_cols, viol_sev_cols = get_sensor_cols(df)
    if not sensors:
        print("No per-sensor violation columns present.")
        return

    # counts
    counts = np.array([row[c] for c in viol_count_cols], dtype=float)
    order_c = np.argsort(-counts)
    fig1, ax1 = plt.subplots(figsize=(max(7, len(sensors)*0.5), 4))
    ax1.bar(range(len(sensors)), counts[order_c])
    ax1.set_xticks(range(len(sensors)))
    ax1.set_xticklabels([sensors[i] for i in order_c], rotation=30, ha="right")
    ax1.set_ylabel("Violation count (0–30)")
    ax1.set_title(f"Violations (count) — {os.path.basename(sel_file)} / window {sel_window}")
    ensure_dir(outdir)
    out1 = os.path.join(outdir, f"errors_window_{sel_window}_viol_counts.png")
    fig1.tight_layout()
    fig1.savefig(out1, dpi=200)
    plt.close(fig1)
    print(f"✓ Saved counts bar → {out1}")

    # severity
    sevs = np.array([row[c] for c in viol_sev_cols], dtype=float)
    order_s = np.argsort(-sevs)
    fig2, ax2 = plt.subplots(figsize=(max(7, len(sensors)*0.5), 4))
    ax2.bar(range(len(sensors)), sevs[order_s])
    ax2.set_xticks(range(len(sensors)))
    ax2.set_xticklabels([sensors[i] for i in order_s], rotation=30, ha="right")
    ax2.set_ylabel("Violation severity (sum of normalized excess)")
    ax2.set_title(f"Violations (severity) — {os.path.basename(sel_file)} / window {sel_window}")
    out2 = os.path.join(outdir, f"errors_window_{sel_window}_viol_severity.png")
    fig2.tight_layout()
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)
    print(f"✓ Saved severity bar → {out2}")

def plot_collapsed_heatmap(df: pd.DataFrame, outdir: str, annotate_pct: float = 95.0, max_annos: int = 30):
    # single-row timeline: cols sorted by window_idx
    sub = df.sort_values("window_idx").reset_index(drop=True)
    vals = sub["nearest_dist"].to_numpy().astype(float)
    M = vals.reshape(1, -1)  # 1 x W

    thr = np.percentile(vals, annotate_pct)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(min(18, 2 + M.shape[1]*0.05), 3.0))
    im = ax.imshow(M, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("nearest_dist")
    ax.set_yticks([0])
    ax.set_yticklabels(["anomaly_set"])
    ax.set_xlabel("window_idx")
    ax.set_title(f"All windows (error mode, collapsed): nearest_dist (annotate ≥ P{int(annotate_pct)})")

    # annotate top sensors for high-distance windows
    annos = 0
    for pos, (_, row) in enumerate(sub.iterrows()):
        nd = float(row["nearest_dist"])
        if nd >= thr and annos < max_annos:
            ax.scatter([pos], [0], marker="o", s=12)
            txt = str(row.get("top3_sensors",""))
            if txt:
                ax.text(pos+0.2, 0.0+0.25, txt[:30], fontsize=7, rotation=15)
            annos += 1
        if annos >= max_annos:
            break

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "heatmap_nearest_dist.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved collapsed heatmap → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="outputs/prototype_explanations_errors.csv")
    ap.add_argument("--outdir", default="outputs/plots")
    ap.add_argument("--annotate_pct", type=float, default=95.0, help="Percentile for heatmap annotations.")
    ap.add_argument("--max_annos", type=int, default=30, help="Max annotations on heatmap.")
    ap.add_argument("--select_file", default="", help="Exact basename of file to bar-plot.")
    ap.add_argument("--select_window", type=int, default=-1, help="Window index to bar-plot.")
    args = ap.parse_args()

    df = load_explanations(args.csv)
    plot_collapsed_heatmap(df, args.outdir, annotate_pct=args.annotate_pct, max_annos=args.max_annos)

    if args.select_file and args.select_window >= 0:
        plot_window_bars(df, args.outdir, args.select_file, args.select_window)

if __name__ == "__main__":
    main()
