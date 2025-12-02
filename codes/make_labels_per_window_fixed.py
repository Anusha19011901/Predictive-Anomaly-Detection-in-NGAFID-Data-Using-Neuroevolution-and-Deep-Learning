#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

"""
This script creates a CORRECT labels_per_window.csv by merging:
1. window list
2. window → dbscan_label assignments from DBSCAN sweep
"""

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_dir", required=True,
                    help="Directory containing window CSVs")
    ap.add_argument("--assignments_csv", required=True,
                    help="assignments_dbscan_epsX_minY.csv produced by patched dbscan_sweep_fit.py")
    ap.add_argument("--out_csv", required=True,
                    help="path to write labels_per_window.csv")
    return ap.parse_args()

def main():
    args = parse_args()

    windows_dir = Path(args.windows_dir)
    assign_csv = Path(args.assignments_csv)

    # ---------------------------------------------------------------------
    # 1. Build window list (filename + index)
    # ---------------------------------------------------------------------
    files = sorted(list(windows_dir.glob("window_*.csv")))
    if len(files) == 0:
        raise RuntimeError(f"No window files found in {windows_dir}")

    win_df = pd.DataFrame({
        "window_idx": range(len(files)),
        "filename": [f.name for f in files],
        "path": [str(f) for f in files],
    })

    # ---------------------------------------------------------------------
    # 2. Load DBSCAN assignments (window_idx → dbscan_label)
    # ---------------------------------------------------------------------
    assign_df = pd.read_csv(assign_csv)

    if "window_idx" not in assign_df.columns:
        raise RuntimeError("assignments CSV must contain a window_idx column")

    # ---------------------------------------------------------------------
    # 3. Merge
    # ---------------------------------------------------------------------
    merged = win_df.merge(assign_df, on="window_idx", how="left")

    # ---------------------------------------------------------------------
    # 4. Save final
    # ---------------------------------------------------------------------
    merged.to_csv(args.out_csv, index=False)
    print(f"[INFO] wrote corrected labels → {args.out_csv}")

if __name__ == "__main__":
    main()
