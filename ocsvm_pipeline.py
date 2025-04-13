import os
import glob
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# CONFIG
DATA_PATH = "/Users/iyashi/Downloads/c_37/"
WINDOW_SIZE = 100
STEP_SIZE = 50

def load_csv_files():
    files = glob.glob(DATA_PATH + "*.csv")
    after = [f for f in files if '_after_' in f]
    before = [f for f in files if '_before_' in f]
    return after, before

def create_windows(df, window_size=100, step_size=50):
    windows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start+window_size]
        windows.append(window.values)
    return windows

def extract_windows(file_list):
    X = []
    for file in file_list:
        try:
            df = pd.read_csv(file, low_memory=False)
            df = df.apply(pd.to_numeric, errors='coerce')  # convert everything to numeric
            df = df.dropna(axis=1, how='any')  # drop bad columns

            if df.shape[1] == 0:
                print(f"[WARN] Skipping {file}: no usable numeric columns after coercion.")
                continue

            if len(df) >= WINDOW_SIZE:
                windows = create_windows(df, WINDOW_SIZE, STEP_SIZE)
                for win in windows:
                    if win.shape[1] == df.shape[1]:  # check window isn't malformed
                        X.append(win)
            else:
                print(f"[WARN] Skipping {file}: too short ({len(df)} rows).")
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")
    return np.array(X)

def flatten_and_scale(X, scaler=None):
    # Flatten 2D time-series windows to 1D
    X_flat = X.reshape(X.shape[0], -1)
    if X_flat.shape[1] == 0:
        raise ValueError(f"[FATAL] Flattened input has 0 features! Check preprocessing.")
    if scaler is None:
        scaler = StandardScaler()
        X_flat = scaler.fit_transform(X_flat)
    else:
        X_flat = scaler.transform(X_flat)
    return X_flat, scaler

def main():
    after_files, before_files = load_csv_files()
    #TEMPORARY TESTING SUBSET
    after_files = after_files[:100]  # temporary testing subset


    print("Loading post-maintenance data...")
    X_post_raw = extract_windows(after_files)
    print(f"[INFO] Raw post-maintenance shape: {X_post_raw.shape}")
    if X_post_raw.size == 0:
        print("[FATAL] No valid windows extracted from post-maintenance data.")
        return
    X_post, scaler = flatten_and_scale(X_post_raw)

    print("Training OC-SVM...")
    model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    model.fit(X_post)

    print("Loading pre-maintenance data...")
    X_pre_raw = extract_windows(before_files)
    X_pre, _ = flatten_and_scale(X_pre_raw, scaler)

    print("Predicting...")
    y_true = np.array([-1] * len(X_pre) + [1] * len(X_post))
    y_pred = model.predict(np.concatenate((X_pre, X_post)))

    print("Results:")
    print(classification_report(y_true, y_pred, labels=[1, -1]))
    print(f"[DEBUG] First good file shape: {X_post_raw[0].shape}" if len(X_post_raw) > 0 else "[DEBUG] No good post windows.")


if __name__ == "__main__":
    main()
