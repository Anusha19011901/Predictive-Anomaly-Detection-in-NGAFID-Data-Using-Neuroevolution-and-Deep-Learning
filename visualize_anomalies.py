# visualize_anomalies.py
# Visual validation of OC-SVM anomaly windows against normal-window envelopes.
# Saves per-window figures to outputs/plots/.

from __future__ import annotations
import argparse
from pathlib import Path
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utility: filesystem + loading
# -----------------------------
def list_csvs(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted([p for p in folder.glob("*.csv")])

def load_windows(csv_paths: List[Path]) -> Tuple[List[pd.DataFrame], List[str]]:
    dfs, names = [], []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            raise RuntimeError(f"Failed reading {p}: {e}")
        dfs.append(df)
        names.append(p.stem)
    return dfs, names

def window_length_harmonize(dfs: List[pd.DataFrame]) -> int:
    """Ensure consistent length across windows by truncating to min length."""
    lengths = [len(d) for d in dfs if len(d) > 0]
    if not lengths:
        raise ValueError("No non-empty CSVs found.")
    min_len = int(np.min(lengths))
    for i in range(len(dfs)):
        dfs[i] = dfs[i].iloc[:min_len].reset_index(drop=True)
    return min_len

def select_numeric_columns(dfs: List[pd.DataFrame], drop_like: Tuple[str, ...] = ("time", "timestamp")) -> List[str]:
    # Columns present in all windows
    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    # Keep numeric only
    numeric_cols = []
    for c in sorted(common_cols):
        if pd.api.types.is_numeric_dtype(dfs[0][c]):
            # Optionally drop obvious time columns if present
            if any(k.lower() in c.lower() for k in drop_like):
                continue
            numeric_cols.append(c)
    if not numeric_cols:
        raise ValueError("No numeric (non-time) columns common across windows.")
    return numeric_cols

# -------------------------------------
# Statistics from normal windows (envelope)
# -------------------------------------
def build_normal_stats(normal_dfs: List[pd.DataFrame], cols: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns per-feature dict with:
      'mean', 'std', 'median', 'q25', 'q75' arrays over time (length = window_len).
    """
    stats = {}
    # Stack each feature across windows -> shape (n_windows, T)
    for c in cols:
        stacked = np.stack([df[c].to_numpy(dtype=float) for df in normal_dfs], axis=0)  # (W, T)
        stats[c] = {
            "mean":   np.nanmean(stacked, axis=0),
            "std":    np.nanstd(stacked, axis=0, ddof=1),
            "median": np.nanmedian(stacked, axis=0),
            "q25":    np.nanpercentile(stacked, 25, axis=0),
            "q75":    np.nanpercentile(stacked, 75, axis=0),
        }
        # Avoid zero std for z-score
        stats[c]["std"][stats[c]["std"] == 0] = 1.0
    return stats

# -------------------------------------
# Feature selection: top-K by variance
# -------------------------------------
def pick_topk_features(normal_dfs: List[pd.DataFrame], cols: List[str], k: int) -> List[str]:
    # Compute variance over (windows x time) for each feature using normal data
    variances = {}
    for c in cols:
        concat = np.concatenate([df[c].to_numpy(dtype=float) for df in normal_dfs], axis=0)
        variances[c] = float(np.nanvar(concat))
    ranked = sorted(variances.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:max(1, k)]]

# -----------------------------
# Plotting
# -----------------------------
def plot_one_window(
    anomaly_df: pd.DataFrame,
    window_name: str,
    feature_list: List[str],
    normal_stats: Dict[str, Dict[str, np.ndarray]],
    outdir: Path,
    normalize: bool = True,
):
    T = len(anomaly_df)
    x = np.arange(T)

    n_features = len(feature_list)
    ncols = 2 if n_features > 1 else 1
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, max(3.5 * nrows, 4)))
    axes = np.array(axes).reshape(-1)  # flatten in case of single subplot

    for i, c in enumerate(feature_list):
        ax = axes[i]
        stats_c = normal_stats[c]

        # Normalize by normal mean/std for visibility (z-score)
        if normalize:
            z_med = (stats_c["median"] - stats_c["mean"]) / stats_c["std"]
            z_q25 = (stats_c["q25"]    - stats_c["mean"]) / stats_c["std"]
            z_q75 = (stats_c["q75"]    - stats_c["mean"]) / stats_c["std"]
            z_anom = (anomaly_df[c].to_numpy(dtype=float) - stats_c["mean"]) / stats_c["std"]
            y_label = f"{c} (z-score vs normal)"
        else:
            z_med = stats_c["median"]
            z_q25 = stats_c["q25"]
            z_q75 = stats_c["q75"]
            z_anom = anomaly_df[c].to_numpy(dtype=float)
            y_label = c

        # Normal envelope (IQR) + median
        ax.fill_between(x, z_q25, z_q75, alpha=0.25, label="Normal IQR")
        ax.plot(x, z_med, linewidth=1.5, label="Normal median")

        # Anomaly curve
        ax.plot(x, z_anom, linewidth=2.0, label="Anomaly", linestyle="-")

        ax.set_title(c)
        ax.set_xlabel("Window index (t)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Anomalous Window: {window_name}", fontsize=14, y=1.02)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{window_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# -----------------------------
# Orchestration
# -----------------------------
def parse_window_index(stem: str) -> int | None:
    m = re.search(r"(\d+)$", stem.replace("window_", ""))
    return int(m.group(1)) if m else None

def main(
    anomaly_dir: Path,
    normal_dir: Path,
    out_dir: Path,
    topk: int = 6,
    normalize: bool = True,
):
    # Read windows
    anom_paths = list_csvs(anomaly_dir)
    norm_paths = list_csvs(normal_dir)

    if not anom_paths:
        raise RuntimeError(f"No anomaly CSVs in: {anomaly_dir}")
    if not norm_paths:
        raise RuntimeError(f"No normal CSVs in: {normal_dir}")

    anomaly_dfs, anomaly_names = load_windows(anom_paths)
    normal_dfs, _ = load_windows(norm_paths)

    # Harmonize lengths
    T1 = window_length_harmonize(anomaly_dfs)
    T2 = window_length_harmonize(normal_dfs)
    T = min(T1, T2)
    # Re-trim both to the same min length (safety)
    anomaly_dfs = [df.iloc[:T].reset_index(drop=True) for df in anomaly_dfs]
    normal_dfs  = [df.iloc[:T].reset_index(drop=True) for df in normal_dfs]

    # Columns
    cols = select_numeric_columns(anomaly_dfs + normal_dfs)
    top_features = pick_topk_features(normal_dfs, cols, k=topk)

    # Stats from normal windows
    norm_stats = build_normal_stats(normal_dfs, cols)

    # Sort anomalies by numeric index if present
    name_idx_pairs = []
    for name in anomaly_names:
        idx = parse_window_index(name)
        name_idx_pairs.append((idx if idx is not None else 10**9, name))
    # Ensure order by window index
    name_idx_pairs.sort()

    # Plot each anomaly
    for _, name in name_idx_pairs:
        df = anomaly_dfs[anomaly_names.index(name)]
        plot_one_window(
            anomaly_df=df,
            window_name=name,
            feature_list=top_features,
            normal_stats=norm_stats,
            outdir=out_dir,
            normalize=normalize,
        )

def main_with_filters(args):
    """
    Wraps main() so we can filter which anomaly windows to plot and optionally
    also emit a single multi-page PDF.
    """
    # First, run the core pipeline up to feature selection and stats
    # We replicate just enough of main() to filter windows before plotting.
    anom_paths = list_csvs(args.anomaly_dir)
    norm_paths = list_csvs(args.normal_dir)
    if not anom_paths:
        raise RuntimeError(f"No anomaly CSVs in: {args.anomaly_dir}")
    if not norm_paths:
        raise RuntimeError(f"No normal CSVs in: {args.normal_dir}")

    anomaly_dfs, anomaly_names = load_windows(anom_paths)
    normal_dfs, _ = load_windows(norm_paths)

    # Harmonize lengths across both sets
    T1 = window_length_harmonize(anomaly_dfs)
    T2 = window_length_harmonize(normal_dfs)
    T = min(T1, T2)
    anomaly_dfs = [df.iloc[:T].reset_index(drop=True) for df in anomaly_dfs]
    normal_dfs  = [df.iloc[:T].reset_index(drop=True) for df in normal_dfs]

    cols = select_numeric_columns(anomaly_dfs + normal_dfs)
    top_features = pick_topk_features(normal_dfs, cols, k=args.topk)
    norm_stats = build_normal_stats(normal_dfs, cols)

    # Pair names with numeric index
    pairs = []
    for name in anomaly_names:
        idx = parse_window_index(name)
        if idx is None:
            continue
        pairs.append((idx, name))
    pairs.sort(key=lambda x: x[0])  # sort by index asc

    # === Apply filters ===
    if args.subset:
        want = {int(x.strip()) for x in args.subset.split(",") if x.strip().isdigit()}
        pairs = [p for p in pairs if p[0] in want]

    if args.start_idx is not None:
        pairs = [p for p in pairs if p[0] >= args.start_idx]
    if args.end_idx is not None:
        pairs = [p for p in pairs if p[0] <= args.end_idx]

    if args.stride and args.stride > 1:
        pairs = pairs[::args.stride]

    if args.max_plots is not None:
        pairs = pairs[:args.max_plots]

    # For optional single-PDF
    pdf_writer = None
    if args.pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pdf_writer = PdfPages(args.pdf)

    # Plot
    saved = []
    for idx, name in pairs:
        df = anomaly_dfs[anomaly_names.index(name)]
        plot_one_window(
            anomaly_df=df,
            window_name=name,
            feature_list=top_features,
            normal_stats=norm_stats,
            outdir=args.out_dir,
            normalize=not args.no_normalize,
        )
        saved.append(name)

        # If PDF requested, also add current figure to PDF (re-render on the fly)
        if pdf_writer is not None:
            # Re-render the same plot onto a fresh figure and save into PDF
            # (This avoids coupling to plot_one_window internals.)
            plt.figure()
            # Simple 1-panel overview: median + IQR (mean z-norm) of first feature for thumbnail
            # but better: reopen the saved PNG into the PDF
            import matplotlib.image as mpimg
            img = mpimg.imread(args.out_dir / f"{name}.png")
            plt.imshow(img)
            plt.axis('off')
            pdf_writer.savefig()
            plt.close()

    if pdf_writer is not None:
        pdf_writer.close()
        print(f"Multi-page PDF written to: {Path(args.pdf).resolve()}")

    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize anomaly windows vs normal envelopes.")
    parser.add_argument("--anomaly_dir", type=Path, default=Path("exact_data/anomaly"))
    parser.add_argument("--normal_dir",  type=Path, default=Path("exact_data/normal"))
    parser.add_argument("--out_dir",     type=Path, default=Path("outputs/plots"))
    parser.add_argument("--topk",        type=int,  default=6, help="How many top-variance features to plot per window")
    parser.add_argument("--no_normalize", action="store_true", help="Disable z-score normalization")

    # NEW controls
    parser.add_argument("--start_idx", type=int, default=None, help="Only plot windows with index >= this")
    parser.add_argument("--end_idx",   type=int, default=None, help="Only plot windows with index <= this")
    parser.add_argument("--stride",    type=int, default=1,    help="Plot every k-th window after filtering (default 1)")
    parser.add_argument("--max_plots", type=int, default=None, help="Stop after plotting at most this many windows")
    parser.add_argument("--subset",    type=str, default=None,
                        help="Comma-separated list of specific window indices to plot (e.g., '50,75,325'). Overrides other filters.")
    parser.add_argument("--pdf",       type=Path, default=None, help="If set, also write a single multi-page PDF to this path")

    args = parser.parse_args()

    print("==== Visualization run configuration ====")
    print(f"Anomaly dir : {args.anomaly_dir.resolve()}")
    print(f"Normal dir  : {args.normal_dir.resolve()}")
    print(f"Output dir  : {args.out_dir.resolve()}")
    print(f"Top-K feats : {args.topk}")
    print(f"Normalize   : {not args.no_normalize}")
    print(f"start_idx   : {args.start_idx}   end_idx: {args.end_idx}   stride: {args.stride}   max_plots: {args.max_plots}")
    print(f"subset      : {args.subset}")
    print(f"pdf         : {args.pdf}")
    print("=========================================")

    # Run
    filtered_pngs = main_with_filters(args)  # call helper below

    # Build a tiny HTML gallery
    try:
        outdir = Path(args.out_dir)
        pngs = sorted(p for p in outdir.glob("*.png"))
        print(f"Total figures saved: {len(pngs)}")
        if pngs:
            html = [
                "<!doctype html><meta charset='utf-8'>",
                "<title>NGAFID Anomaly Plots</title>",
                "<style>body{font-family:system-ui;margin:24px} img{max-width:100%;height:auto;border:1px solid #ddd;margin:12px 0}</style>",
                "<h1>NGAFID Anomaly Plots</h1>",
                "<p>Auto-generated gallery from visualize_anomalies.py</p>",
            ]
            for p in pngs:
                html.append(f"<h3>{p.name}</h3>")
                html.append(f"<img src='{p.name}' alt='{p.name}'/>")
            (outdir / "index.html").write_text("\n".join(html), encoding="utf-8")
            print(f"Gallery: {(outdir / 'index.html').resolve()}")
    except Exception as e:
        print(f"(Non-fatal) Failed to write gallery: {e}")


