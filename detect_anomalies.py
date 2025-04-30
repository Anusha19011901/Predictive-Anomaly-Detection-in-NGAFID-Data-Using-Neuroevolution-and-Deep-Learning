#!/usr/bin/env python3
import os
import sys

# --- DEBUG: confirm this file & cwd ---
print(f"=== Running {__file__} ===", flush=True)
print(f"=== CWD {os.getcwd()} ===", flush=True)

# Use non‐interactive backend so script never blocks on show()
import matplotlib
matplotlib.use('Agg')

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

from typing import List, Generator, Tuple
import numpy.typing as npt

# === CONFIG ===
WINDOW_SIZE: int = 30
STEP_SIZE: int = 25
BEFORE_DIR: str = "dataset/before"
COLUMNS_TO_USE: List[str] = [
    'AltMSL', 'E1 RPM', 'E1 FFlow', 'E1 CHT1',
    'E1 EGT1', 'NormAc', 'IAS'
]
ANOMALY_THRESHOLD: float = 0.3  # >30% of window = anomaly

# === Reuse trained scaler and OC-SVM models from training ===
from ocsvm_pipeline2 import scaler, svms

def load_all_csvs(folder: str) -> pd.DataFrame:
    """Loads and concatenates all CSV files from a given folder."""
    all_files: List[str] = glob.glob(os.path.join(folder, "*.csv"))
    dfs: List[pd.DataFrame] = []
    for file in all_files:
        df: pd.DataFrame = pd.read_csv(file, skiprows=2, low_memory=False)
        df.columns = df.columns.str.strip()
        df.replace('', np.nan, inplace=True)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def generate_windows(
    data: npt.NDArray[np.float_],
    window_size: int,
    step_size: int
) -> Generator[Tuple[int, npt.NDArray[np.float_]], None, None]:
    """Generates sliding windows from time series data."""
    for start in range(0, len(data) - window_size + 1, step_size):
        yield start, data[start:start + window_size]

# === Preprocessing Step ===
df: pd.DataFrame = load_all_csvs(BEFORE_DIR)
df.dropna(subset=COLUMNS_TO_USE, inplace=True)
df[COLUMNS_TO_USE] = df[COLUMNS_TO_USE].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=COLUMNS_TO_USE, inplace=True)

X_before: npt.NDArray[np.float_] = df[COLUMNS_TO_USE].astype(float).values
X_before_scaled: npt.NDArray[np.float_] = scaler.transform(X_before)

# === Anomaly Detection ===
anomaly_flags: List[bool] = []
window_starts: List[int] = []
scores: List[float] = []
feature_means: List[npt.NDArray[np.float_]] = []
pca_vectors: List[npt.NDArray[np.float_]] = []

for i, (start, window) in enumerate(generate_windows(X_before_scaled, WINDOW_SIZE, STEP_SIZE)):
    model: OneClassSVM = svms[i % len(svms)]
    preds: npt.NDArray[np.int_] = model.predict(window)
    score: float = model.decision_function(window).mean()
    ratio: float = np.mean(preds == -1)
    is_anomaly: bool = ratio > ANOMALY_THRESHOLD

    anomaly_flags.append(is_anomaly)
    window_starts.append(start)
    scores.append(score)
    feature_means.append(window.mean(axis=0))
    pca_vectors.append(window.flatten())

# === Console Report: first anomaly & CWD ===
if True in anomaly_flags:
    first_anom_idx: int = window_starts[anomaly_flags.index(True)]
    print(f"\n✅ First anomaly at window start index: {first_anom_idx}", flush=True)
else:
    print("\n✉️ No anomalies detected in BEFORE dataset.", flush=True)
print("Working dir:", os.getcwd(), flush=True)

# === Dump windows to CSV before plotting ===
out_base = os.path.abspath("exact_data")
normal_dir = os.path.join(out_base, "normal")
anomaly_dir = os.path.join(out_base, "anomaly")
print(f"📂 Creating dirs:\n  {normal_dir}\n  {anomaly_dir}", flush=True)
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(anomaly_dir, exist_ok=True)

for flag, start in zip(anomaly_flags, window_starts):
    subdf = df.iloc[start : start + WINDOW_SIZE][COLUMNS_TO_USE].copy()
    subdf['anomaly_flag'] = int(flag)
    subdir = anomaly_dir if flag else normal_dir
    out_path = os.path.join(subdir, f"window_{start}.csv")
    print(f"Writing: {out_path}", flush=True)
    subdf.to_csv(out_path, index=False)
print("✅ Dump complete", flush=True)

# === Plot 1: Anomaly flags over time ===
plt.figure(figsize=(10, 4))
plt.plot(window_starts, anomaly_flags, marker='o')
plt.title("Anomaly Detection on BEFORE Data (OC-SVM)")
plt.xlabel("Window Start Index")
plt.ylabel("Anomaly Detected")
plt.yticks([0, 1], ["Normal", "Anomaly"])
plt.grid(True)
plt.tight_layout()
plt.savefig("anomaly_flags.png")
plt.close()

# === Plot 2: OC-SVM decision scores ===
plt.figure(figsize=(10, 4))
plt.plot(window_starts, scores, marker='o')
plt.axhline(0, linestyle='--')
plt.title("OC-SVM Decision Function Over Time")
plt.xlabel("Window Start Index")
plt.ylabel("Decision Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("decision_scores.png")
plt.close()

# === Plot 3: Feature mean heatmap per window ===
heat_df: pd.DataFrame = pd.DataFrame(feature_means, columns=COLUMNS_TO_USE)
plt.figure(figsize=(10, 6))
sns.heatmap(heat_df.T, cmap='coolwarm', xticklabels=STEP_SIZE)
plt.title("Feature Behavior Across Windows")
plt.xlabel("Window #")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_heatmap.png")
plt.close()

# === Plot 4: PCA projection of windows ===
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_vectors)
colors = ['red' if flag else 'blue' for flag in anomaly_flags]
plt.figure(figsize=(8, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.6)
plt.title("PCA of Sliding Windows (Red = Anomaly)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_windows.png")
plt.close()

print("✅ All done.", flush=True)
