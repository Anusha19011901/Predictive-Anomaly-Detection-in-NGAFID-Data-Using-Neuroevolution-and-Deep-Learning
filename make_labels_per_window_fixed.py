#!/usr/bin/env python3
"""
@file make_labels_per_window_fixed.py
@brief Generate a corrected `labels_per_window.csv` by merging window files with DBSCAN label assignments.

This script fixes DBSCAN window-to-label mapping by rebuilding the true window index
ordering directly from the window directory. It then merges that ordering with the
DBSCAN assignments produced by the patched `dbscan_sweep_fit.py`.

**Inputs**
1. **windows_dir**  
   Directory containing window_*.csv files (e.g., window_0.csv … window_N.csv).  
   These files define the *true* ordering of window indices used everywhere else.

2. **assignments_csv**  
   A DBSCAN assignment table mapping:
       window_idx → dbscan_label  
   produced by the DBSCAN sweep stage.

**Process**
1. Scan `windows_dir` in sorted order and reconstruct a window table:
       window_idx, filename, path
2. Load DBSCAN assignments.
3. Merge using window_idx to ensure exact index alignment.
4. Write the corrected table to `labels_per_window.csv`.

**Output**
A fully aligned labels file containing:
- window_idx  
- filename  
- path  
- dbscan_label  

This corrected version resolves index drift issues and ensures all later steps
(EXAMM scoring, clustering diagnostics, hybrid labeling, prototype analysis)
use consistent, accurate DBSCAN label assignments.
"""

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
