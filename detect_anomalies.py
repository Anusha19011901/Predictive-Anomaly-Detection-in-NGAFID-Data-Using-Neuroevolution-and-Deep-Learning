import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# === CONFIG ===
WINDOW_SIZE = 30
STEP_SIZE = 25
BEFORE_DIR = "dataset/before"
COLUMNS_TO_USE = ['AltMSL', 'E1 RPM', 'E1 FFlow', 'E1 CHT1', 'E1 EGT1', 'NormAc', 'IAS']
ANOMALY_THRESHOLD = 0.3  # >30% of window = anomaly

# === Reuse your scaler and SVMs from after training ===
from ocsvm_pipeline2 import scaler, svms

# === Load and clean BEFORE data ===
def load_all_csvs(folder):
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    dfs = []
    for file in all_files:
        df = pd.read_csv(file, skiprows=2, low_memory=False)
        df.columns = df.columns.str.strip()
        df.replace('', np.nan, inplace=True)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def generate_windows(data, window_size, step_size):
    for start in range(0, len(data) - window_size + 1, step_size):
        yield start, data[start:start + window_size]

# === Preprocess ===
df = load_all_csvs(BEFORE_DIR)
df.dropna(subset=COLUMNS_TO_USE, inplace=True)
df[COLUMNS_TO_USE] = df[COLUMNS_TO_USE].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=COLUMNS_TO_USE, inplace=True)
X_before = df[COLUMNS_TO_USE].astype(float).values
X_before_scaled = scaler.transform(X_before)

# === Run anomaly detection ===
anomaly_flags = []
window_starts = []
scores = []
feature_means = []
pca_vectors = []

for i, (start, window) in enumerate(generate_windows(X_before_scaled, WINDOW_SIZE, STEP_SIZE)):
    model = svms[i % len(svms)]
    preds = model.predict(window)
    score = model.decision_function(window).mean()
    ratio = np.mean(preds == -1)
    is_anomaly = ratio > ANOMALY_THRESHOLD

    anomaly_flags.append(is_anomaly)
    window_starts.append(start)
    scores.append(score)
    feature_means.append(window.mean(axis=0))
    pca_vectors.append(window.flatten())

# === Plot: Anomaly flags ===
plt.figure(figsize=(10, 4))
plt.plot(window_starts, anomaly_flags, marker='o', color='crimson')
plt.title("Anomaly Detection on BEFORE Data (OC-SVM)")
plt.xlabel("Window Start Index")
plt.ylabel("Anomaly Detected")
plt.yticks([0, 1], ["Normal", "Anomaly"])
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot: Decision function score ===
plt.figure(figsize=(10, 4))
plt.plot(window_starts, scores, marker='o')
plt.axhline(0, color='red', linestyle='--')
plt.title("OC-SVM Decision Function Over Time")
plt.xlabel("Window Start Index")
plt.ylabel("Decision Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot: Heatmap of feature window means ===
heat_df = pd.DataFrame(feature_means, columns=COLUMNS_TO_USE)
plt.figure(figsize=(10, 6))
sns.heatmap(heat_df.T, cmap='coolwarm', xticklabels=25)
plt.title("Feature Behavior Across Windows")
plt.xlabel("Window")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# === PCA: Project window vectors ===
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
plt.show()

# === Optional: Early anomaly detection ===
if True in anomaly_flags:
    first_anomaly = window_starts[anomaly_flags.index(True)]
    print(f"\n\u2705 Anomaly first detected at window starting index: {first_anomaly}")
else:
    print("\n\u2709\ufe0f No anomalies detected in BEFORE dataset.")
