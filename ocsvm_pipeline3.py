"""
@file ocsvm_pipeline3.py
@brief Train a global One-Class SVM anomaly detector on AFTER-maintenance NGAFID flights 
       using flattened, scaled sliding windows of selected sensor features.

This script builds the simplest standalone OC-SVM pipeline for NGAFID anomaly detection:
it extracts windows from AFTER flights (representing "healthy" post-maintenance behavior),
scales them, flattens them, trains one global OC-SVM model, and saves the resulting model
and metadata. No EXAMM features or DBSCAN prototypes are used here — this is a pure
window-based OC-SVM baseline.

-------------------------------------------------------------------------------
Core Workflow
-------------------------------------------------------------------------------

1. **File Discovery**
   - Loads all AFTER flight CSVs from `dataset/after/`.
   - Excludes `_after_0_` and `_after_1_` files by default (often low-quality segments).

2. **Feature Selection**
   Uses a fixed 7-feature subset aligned with EXAMM/DBSCAN pipelines:
       AltMSL, E1 RPM, E1 FFlow, E1 CHT1, E1 EGT1, NormAc, IAS

3. **Scaling**
   - Concatenates *all numeric rows* from all AFTER flights.
   - Fits a `StandardScaler` (mean/std) across the merged dataset.
   - Ensures consistent per-feature normalization for ALL windows.

4. **Sliding Window Extraction**
   For each AFTER file:
   - Apply the fitted scaler.
   - Build overlapping windows with:
         WINDOW_SIZE = 30 timesteps  
         STEP_SIZE   = 25  
   - Flatten each (30 × 7) window into a 210-dimensional vector.

5. **OC-SVM Training**
   - A single One-Class SVM is trained on *all* flattened healthy windows.
   - Recommended hyperparameters:
         nu    = 0.05  
         gamma = "scale" (RBF kernel)  

6. **Outputs**
   Saved to `outputs/`:
   - `ocsvm_model.pkl`  → trained OC-SVM model
   - `scaler.pkl`       → StandardScaler used for preprocessing
   - `ocsvm_meta.json`  → metadata describing training parameters, hyperparameters,
                          file-window counts, and preprocessing configuration

-------------------------------------------------------------------------------
Intended Use
-------------------------------------------------------------------------------
This OC-SVM model represents a baseline anomaly detector for airborne telemetry.
It is trained exclusively on AFTER-maintenance "healthy" behavior and can be used
to score BEFORE-maintenance windows, identifying those that deviate significantly
from the learned normal pattern.

It is also useful as:
- A standalone anomaly model  
- A comparison baseline against EXAMM-enhanced OC-SVM  
- A component in fusion/hybrid labeling pipelines  
- A sanity-check baseline for DBSCAN prototype behavior

-------------------------------------------------------------------------------
Notes
-------------------------------------------------------------------------------
- All windows are flattened, no sequence modeling is used.
- Scaling is per-feature, not per-window.
- Assumes NGAFID CSVs with 2-line metadata headers (skiprows=2).
"""

import os
import glob
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib
import re
import json

# -----------------------------
# CONFIG
# -----------------------------
AFTER_DIR = "dataset/after"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Train/feature settings
COLUMNS_TO_USE: List[str] = [
    "AltMSL", "E1 RPM", "E1 FFlow", "E1 CHT1", "E1 EGT1", "NormAc", "IAS"
]
WINDOW_SIZE: int = 30
STEP_SIZE: int = 25

# Model hyperparams (sane defaults for healthy-only boundary)
NU = 0.05           # smaller => tighter normal region
GAMMA = "scale"     # RBF gamma auto-scaling


# -----------------------------
# IO & PREP
# -----------------------------
def list_after_files(folder: str, exclude_after01: bool = True) -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if exclude_after01:
        files = [f for f in files if not re.search(r"_after_(0|1)_", os.path.basename(f))]
    return files

def read_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2)  # NGAFID header rows
    df.columns = df.columns.str.strip()
    missing = set(COLUMNS_TO_USE) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {os.path.basename(path)}: {sorted(missing)}")

    sub = df[COLUMNS_TO_USE].replace("", np.nan)
    sub = sub.apply(pd.to_numeric, errors="coerce").dropna()
    return sub


def sliding_windows_from_scaled(
    scaled_array: np.ndarray, window_size: int, step: int
) -> Tuple[np.ndarray, List[int]]:
    # scaled_array shape: [T, F]
    starts = list(range(0, len(scaled_array) - window_size + 1, step))
    if not starts:
        return np.empty((0, window_size * scaled_array.shape[1])), []
    windows = np.stack([scaled_array[s:s + window_size, :] for s in starts], axis=0)  # [N, W, F]
    flat = windows.reshape(windows.shape[0], -1)  # [N, W*F]
    return flat, starts


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    after_files = list_after_files(AFTER_DIR, exclude_after01=True)
    if not after_files:
        raise RuntimeError("No AFTER files found for training (after_0/1 excluded).")

    # 1) Read and concat all AFTER data for scaler fit
    all_after_rows = []
    per_file_scaled_windows = []
    total_windows = 0

    # First pass: read & concat for scaler fit
    concat_df = []
    for f in after_files:
        df = read_and_clean(f)
        concat_df.append(df)
    concat_df = pd.concat(concat_df, axis=0, ignore_index=True)

    # 2) Fit scaler on healthy samples (per-feature scaling)
    scaler = StandardScaler()
    concat_scaled = scaler.fit_transform(concat_df.values.astype(float))

    # 3) Second pass: create flattened windows from each file using the fitted scaler
    X_train_list = []
    file_window_counts: Dict[str, int] = {}

    cursor = 0
    for f in after_files:
        df = read_and_clean(f)
        n = len(df)
        # take the corresponding block from the already-scaled big array
        scaled_block = concat_scaled[cursor: cursor + n, :]
        cursor += n

        flat_windows, starts = sliding_windows_from_scaled(
            scaled_block, WINDOW_SIZE, STEP_SIZE
        )
        if flat_windows.size:
            X_train_list.append(flat_windows)
            file_window_counts[os.path.basename(f)] = flat_windows.shape[0]
            total_windows += flat_windows.shape[0]

    if not X_train_list:
        raise RuntimeError("Training windows set is empty; check data/columns/window params.")

    X_train = np.vstack(X_train_list)  # [TotalWindows, W*F]

    # 4) Train ONE OCSVM on ALL healthy windows
    ocsvm = OneClassSVM(kernel="rbf", nu=NU, gamma=GAMMA)
    ocsvm.fit(X_train)

    # 5) Persist model artifacts + metadata
    model_path = os.path.join(OUT_DIR, "ocsvm_model.pkl")
    scaler_path = os.path.join(OUT_DIR, "scaler.pkl")
    joblib.dump(ocsvm, model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "columns": COLUMNS_TO_USE,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "nu": NU,
        "gamma": GAMMA,
        "n_after_files": len(after_files),
        "total_windows": int(total_windows),
        "file_window_counts": file_window_counts,
        "note": "Trained on AFTER flights (excluding after_0/1) with per-feature scaling; windows flattened."
    }
    with open(os.path.join(OUT_DIR, "ocsvm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Trained ONE OCSVM on {total_windows} healthy windows from {len(after_files)} files.")
    print(f"Saved: {model_path}, {scaler_path}, {os.path.join(OUT_DIR, 'ocsvm_meta.json')}")
