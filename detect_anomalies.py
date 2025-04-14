import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import List, Generator, Tuple
import numpy.typing as npt

# === CONFIG ===
WINDOW_SIZE: int = 30
STEP_SIZE: int = 25
BEFORE_DIR: str = "dataset/before"
COLUMNS_TO_USE: List[str] = ['AltMSL', 'E1 RPM', 'E1 FFlow', 'E1 CHT1', 'E1 EGT1', 'NormAc', 'IAS']
ANOMALY_THRESHOLD: float = 0.3  # >30% of window = anomaly

# === Reuse trained scaler and OC-SVM models from training ===
from ocsvm_pipeline2 import scaler, svms

def load_all_csvs(folder: str) -> pd.DataFrame:
    """
    Loads and concatenates all CSV files from a given folder.

    Args:
        folder (str): Path to the directory containing flight CSVs.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all flight data.
    """
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
    """
    Generates sliding windows from time series data.

    Args:
        data (np.ndarray): Full time series data (scaled).
        window_size (int): Length of each window.
        step_size (int): Step size between windows.

    Yields:
        Tuple[int, np.ndarray]: (start index, window array)
    """
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

# === Plot 1: Anomaly flags over time ===
plt.figure(figsize=(10, 4))
plt.plot(window_starts, anomaly_flags, marker='o', color='crimson')
plt.title("Anomaly Detection on BEFORE Data (OC-SVM)")
plt.xlabel("Window Start Index")
plt.ylabel("Anomaly Detected")
plt.yticks([0, 1], ["Normal", "Anomaly"])
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 2: OC-SVM decision scores ===
plt.figure(figsize=(10, 4))
plt.plot(window_starts, scores, marker='o')
plt.axhline(0, color='red', linestyle='--')
plt.title("OC-SVM Decision Function Over Time")
plt.xlabel("Window Start Index")
plt.ylabel("Decision Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 3: Feature mean heatmap per window ===
heat_df: pd.DataFrame = pd.DataFrame(feature_means, columns=COLUMNS_TO_USE)
plt.figure(figsize=(10, 6))
sns.heatmap(heat_df.T, cmap='coolwarm', xticklabels=25)
plt.title("Feature Behavior Across Windows")
plt.xlabel("Window")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# === Plot 4: PCA projection of windows ===
pca: PCA = PCA(n_components=2)
pca_result: npt.NDArray[np.float_] = pca.fit_transform(pca_vectors)
colors: List[str] = ['red' if flag else 'blue' for flag in anomaly_flags]
plt.figure(figsize=(8, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.6)
plt.title("PCA of Sliding Windows (Red = Anomaly)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Console Report: When was the first anomaly detected? ===
if True in anomaly_flags:
    first_anomaly: int = window_starts[anomaly_flags.index(True)]
    print(f"\n✅ Anomaly first detected at window starting index: {first_anomaly}")
else:
    print("\n✉️ No anomalies detected in BEFORE dataset.")
