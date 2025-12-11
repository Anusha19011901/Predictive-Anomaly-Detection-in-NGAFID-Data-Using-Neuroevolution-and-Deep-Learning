# make_windows.py
"""
@file make_windows.py
@brief Generate scaled sliding windows for EXAMM forecasting and OC-SVM anomaly detection.

This module constructs W×F input windows and next-step prediction targets from raw NGAFID
flight data, applying the same feature selection and scaling used during EXAMM training.
It enables consistent preprocessing across EXAMM, OC-SVM, DBSCAN, and hybrid labeling pipelines.

**Core Functionality**

1. **Load Features and Scaler**
   - Reads the selected feature list from:
         artifacts/features/selected_features.json
   - Loads the StandardScaler parameters (mean, scale) from:
         artifacts/scalers/standardizer.pkl

2. **Window Construction (Sliding Windows)**
   Using window size `WIN = 30` and step size `STEP = 5`, this script generates:
   - `X`:  Input windows of shape (N, WIN, F)
   - `Y`:  Next-timestep targets of shape (N, F) aligned with EXAMM’s 1-step-ahead forecasting
   - `M`:  Boolean masks indicating which Y values are valid (non-NaN)
   - `I`:  Index tuples (start_idx, end_idx, y_idx) for tracking window alignment

   This follows the standard EXAMM training format.

3. **Scaling**
   Windows and targets are normalized *in-place* using the previously fitted mean/scale,
   ensuring identical preprocessing across training and inference pipelines.

**Output**
The `prepare(csv_path)` function returns:
- feature list  
- scaled input windows X  
- scaled next-step targets Y  
- validity masks M  
- index mapping I  

This module is typically used before:
- EXAMM model training,
- EXAMM error extraction,
- OC-SVM feature preparation,
- DBSCAN prototype generation.

"""

import numpy as np, pandas as pd, joblib, json
from pathlib import Path

WIN, STEP = 30, 5

def load_feats_scaler():
    d = joblib.load("artifacts/scalers/standardizer.pkl")
    feats = json.loads(Path("artifacts/features/selected_features.json").read_text())
    return feats, np.array(d["mean"]), np.array(d["scale"])

def to_windows(df: pd.DataFrame, feats):
    A = df[feats].values  # (T, F)
    T, F = A.shape
    Xs, Ys, masks, idxs = [], [], [], []
    for end in range(WIN-1, T-1, STEP):
        x = A[end-WIN+1:end+1]             # (WIN, F)
        y_idx = end + 1                    # t+1
        if y_idx >= T: break
        y = A[y_idx]                       # (F,)
        mask = ~np.isnan(y)
        Xs.append(x); Ys.append(y); masks.append(mask); idxs.append((end-WIN+1, end, y_idx))
    return np.array(Xs), np.array(Ys), np.array(masks), np.array(idxs)

def scale_inplace(arr, mean, scale):
    return (arr - mean) / (scale + 1e-12)

def prepare(csv_path: str):
    feats, mean, scale = load_feats_scaler()
    df = pd.read_csv(csv_path).select_dtypes(include=[np.number])
    df = df[feats]
    X, Y, M, I = to_windows(df, feats)
    X = scale_inplace(X, mean, scale)
    Y = scale_inplace(Y, mean, scale)
    return feats, X, Y, M, I
