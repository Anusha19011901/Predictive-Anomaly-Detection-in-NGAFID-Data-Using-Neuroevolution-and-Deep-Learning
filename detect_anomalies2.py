import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import joblib

# -----------------------------
# PARAMETERS
# -----------------------------
WINDOW_SIZE = 30
STEP_SIZE = 25
ANOMALY_THRESHOLD = 0.3

COLUMNS_TO_USE = [
    "AltMSL", "E1 RPM", "E1 FFlow", "E1 CHT1", "E1 EGT1", "NormAc", "IAS"
]

MODEL_PATH = os.path.join("outputs", "ocsvm_models.pkl")

# -----------------------------
# Load trained models
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ No trained models at {MODEL_PATH}. Run ocsvm_pipeline2.py first.")
svms = joblib.load(MODEL_PATH)
print(f"✅ Loaded {len(svms)} trained window-based SVM models")

# -----------------------------
# Sliding window helper
# -----------------------------
def sliding_windows(df, window_size, step_size):
    windows = []
    indices = []
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start + window_size]
        windows.append(window.values)
        indices.append(start)
    return np.array(windows), indices

# -----------------------------
# Detection routine per file
# -----------------------------
def detect_on_file(file_path, label):
    df = pd.read_csv(file_path, skiprows=2)
    df.columns = df.columns.str.strip()

    # Clean numeric data
    df = df[COLUMNS_TO_USE].dropna()
    df[COLUMNS_TO_USE] = df[COLUMNS_TO_USE].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=COLUMNS_TO_USE, inplace=True)

    # Create windows and flatten (30x7 = 210 features)
    windows, indices = sliding_windows(df, WINDOW_SIZE, STEP_SIZE)
    flat_windows = [w.flatten() for w in windows]

    preds, scores = [], []
    for i, flat_window in enumerate(flat_windows):
        if i < len(svms):
            preds.append(svms[i].predict([flat_window])[0])
            scores.append(svms[i].decision_function([flat_window])[0])
        else:  # if more windows than models, reuse last
            preds.append(svms[-1].predict([flat_window])[0])
            scores.append(svms[-1].decision_function([flat_window])[0])

    preds = np.array(preds)
    scores = np.array(scores)
    anomaly_flags = (preds == -1).astype(int)
    anomaly_percent = np.mean(anomaly_flags)
    is_anomalous = anomaly_percent > ANOMALY_THRESHOLD

    base = os.path.basename(file_path)
    prefix = label

    # 1. Anomaly flags timeline
    plt.figure(figsize=(10, 4))
    plt.plot(indices, anomaly_flags, marker="o", linestyle="--")
    plt.title(f"Anomaly Flags - {base}")
    plt.xlabel("Window Start Index")
    plt.ylabel("Anomaly Flag (1=Anomaly)")
    plt.savefig(f"outputs/{prefix}_anomaly_flags_{base}.png")
    plt.close()

    # 2. Decision scores
    plt.figure(figsize=(10, 4))
    plt.plot(indices, scores, marker="o")
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"SVM Decision Scores - {base}")
    plt.xlabel("Window Start Index")
    plt.ylabel("Decision Score")
    plt.savefig(f"outputs/{prefix}_decision_scores_{base}.png")
    plt.close()

    # 3. Heatmap of feature correlations
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Feature Correlation Heatmap - {base}")
    plt.savefig(f"outputs/{prefix}_feature_heatmap_{base}.png")
    plt.close()

    # 4. Raw time series
    plt.figure(figsize=(12, 6))
    for col in COLUMNS_TO_USE:
        plt.plot(df.index, df[col], label=col)
    plt.legend()
    plt.title(f"Raw Time Series - {base}")
    plt.xlabel("Time (row index)")
    plt.ylabel("Sensor Values")
    plt.savefig(f"outputs/{prefix}_timeseries_{base}.png")
    plt.close()

    print(f"✅ Finished {file_path} | Anomaly %: {anomaly_percent:.2f} | Flagged: {is_anomalous}")

    return flat_windows, anomaly_flags, label, df


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    before_files = [
        "dataset/before/open_2017_05_08_close_2017_05_08_flight_Fixed Wing_N550ND_before_3_189762.csv",
        "dataset/before/open_2017_05_08_close_2017_05_08_flight_Fixed Wing_N550ND_before_4_189791.csv"
    ]
    after_files = [
        "dataset/after/open_2017_05_08_close_2017_05_08_flight_Fixed Wing_N550ND_after_0_189675.csv",
        "dataset/after/open_2017_05_08_close_2017_05_08_flight_Fixed Wing_N550ND_after_1_189756.csv"
    ]

    all_windows, all_labels = [], []
    df_store = []  # keep dfs for later comparison

    print("🔎 Running anomaly detection on BEFORE flights (3 & 4)...")
    for f in before_files:
        w, flags, lbl, df = detect_on_file(f, "before")
        all_windows.extend(w)
        all_labels.extend(["before"] * len(w))
        df_store.append((lbl, df))

    print("\n✅ Running anomaly detection on AFTER flights (0 & 1)...")
    for f in after_files:
        w, flags, lbl, df = detect_on_file(f, "after")
        all_windows.extend(w)
        all_labels.extend(["after"] * len(w))
        df_store.append((lbl, df))

    # -----------------------------
    # 5. Joint PCA comparison
    # -----------------------------
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_windows)
    colors = ["red" if l == "before" else "blue" for l in all_labels]

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.6)
    plt.title("Joint PCA Projection - Before (red) vs After (blue)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("outputs/joint_pca_before_vs_after.png")
    plt.close()
    print("📊 Saved joint PCA comparison plot")

    # -----------------------------
    # 6. PCA feature contribution analysis
    # -----------------------------
    loadings = pca.components_.T  # shape (210, 2)
    feature_contrib = {col: 0 for col in COLUMNS_TO_USE}
    for i, col in enumerate(COLUMNS_TO_USE * WINDOW_SIZE):
        feature_contrib[col] += abs(loadings[i, 0]) + abs(loadings[i, 1])
    sorted_features = sorted(feature_contrib.items(), key=lambda x: x[1], reverse=True)

    # Bar chart of top features
    top_feats = sorted_features[:5]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=[f for f, _ in top_feats], y=[v for _, v in top_feats], palette="viridis")
    plt.title("Top Features Contributing to PCA Separation")
    plt.ylabel("Contribution (abs loading)")
    plt.savefig("outputs/pca_feature_contributions.png")
    plt.close()
    print("📊 Saved PCA feature contribution bar chart")

    # -----------------------------
    # 7. Boxplots + Avg Time Series for top features
    # -----------------------------
    df_all = pd.DataFrame(all_windows, columns=[f"{c}_t{t}" for t in range(WINDOW_SIZE) for c in COLUMNS_TO_USE])
    df_all["label"] = all_labels

    for feature, _ in top_feats:
        cols = [c for c in df_all.columns if feature in c]
        df_all[f"{feature}_avg"] = df_all[cols].mean(axis=1)

        # Boxplot
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="label", y=f"{feature}_avg", data=df_all, palette={"before": "red", "after": "blue"})
        plt.title(f"{feature} Distribution (Before vs After)")
        plt.savefig(f"outputs/{feature}_boxplot.png")
        plt.close()

        # Average time series across windows
        avg_before = df_all[df_all["label"] == "before"][cols].mean().values
        avg_after = df_all[df_all["label"] == "after"][cols].mean().values

        plt.figure(figsize=(8, 4))
        plt.plot(range(WINDOW_SIZE), avg_before, "r-", label="Before")
        plt.plot(range(WINDOW_SIZE), avg_after, "b-", label="After")
        plt.title(f"Average {feature} over subsequences")
        plt.xlabel("Timestep in window (0–29)")
        plt.ylabel(feature)
        plt.legend()
        plt.savefig(f"outputs/{feature}_avg_timeseries.png")
        plt.close()

    print("📊 Saved boxplots and avg time series for top PCA-contributing features")
