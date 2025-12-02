# make_windows.py
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
