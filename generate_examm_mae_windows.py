#!/usr/bin/env python3
"""
@file generate_examm_mae_windows.py
@brief Generate per-window EXAMM MAE features for downstream OC-SVM anomaly detection.

This script reads EXAMM prediction CSVs produced during sequence forecasting evaluation.
Each EXAMM output file contains paired columns of the form:

    expected_<feature>
    predicted_<feature>

For every window, the script:
1. Identifies the flight ID, subsequence ID, and window index from the filename.
2. Extracts all expected/predicted feature pairs.
3. Computes Mean Absolute Error (MAE) for each feature:
       mae_<feature> = mean(|expected - predicted|)
4. Produces a unified table where each row corresponds to one window of one subsequence.

**Output**
A consolidated CSV is created at:

    artifacts/errors/per_window/ocsvm_input_before.csv

containing columns:
- flight_id  
- subseq_id  
- window_idx  
- mae_<feature1>, mae_<feature2>, ...

**Usage**
This dataset forms the EXAMM-derived feature set used by the OC-SVM pipeline
to detect anomalous subsequences based on EXAMM's reconstruction/prediction errors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import os

"""
Generate EXAMM MAE features for OC-SVM:
Reads EXAMM prediction CSVs with columns:
  expected_<feat>
  predicted_<feat>

Outputs:
  artifacts/errors/per_window/ocsvm_input_before.csv
"""

INPUT_DIR = "artifacts/evaluation_output"
OUT_DIR = "artifacts/errors/per_window"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_ids(filename):
    """
    Extract flight_id, subseq_id, window_idx from NAME.
    Example filename:
      open_..._before_2_189656_predictions.csv
    """
    base = filename.name
    parts = base.split("_")
    # flight id = aircraft tail number, e.g. N550ND
    m = re.search(r"N[0-9A-Z]+", base)
    flight_id = m.group(0) if m else "UNK"

    # subseq = the number before the ID (the before_<subseq>_<id>.csv)
    m2 = re.search(r"before_(\d+)_", base)
    subseq = int(m2.group(1)) if m2 else -1

    # window idx = trailing digits before "_predictions"
    m3 = re.search(r"_(\d+)_predictions\.csv$", base)
    window_idx = int(m3.group(1)) if m3 else -1

    return flight_id, subseq, window_idx

def main():
    files = sorted(Path(INPUT_DIR).glob("*_predictions.csv"))
    if not files:
        print("No EXAMM prediction files found.")
        return

    rows = []

    for f in files:
        df = pd.read_csv(f)

        # all expected_* columns
        expected_cols = [c for c in df.columns if c.startswith("expected_")]
        predicted_cols = [c for c in df.columns if c.startswith("predicted_")]

        # match feature names
        feats = sorted([c.replace("expected_", "") for c in expected_cols
                        if "predicted_"+c.replace("expected_","") in predicted_cols])

        if not feats:
            print(f"[WARN] No matching expected/predicted pairs in {f}")
            continue

        flight_id, subseq, window_idx = extract_ids(f)

        # compute MAE for each feature
        mae_vals = {}
        for ft in feats:
            e = df["expected_"+ft]
            p = df["predicted_"+ft]
            mae_vals["mae_"+ft] = float(np.mean(np.abs(e - p)))

        row = {
            "flight_id": flight_id,
            "subseq_id": subseq,
            "window_idx": window_idx,
        }
        row.update(mae_vals)

        rows.append(row)

    if not rows:
        print("No data rows generated.")
        return

    out_df = pd.DataFrame(rows)
    out_path = Path(OUT_DIR) / "ocsvm_input_before.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✅ Wrote {len(out_df)} rows → {out_path}")

if __name__ == "__main__":
    main()
