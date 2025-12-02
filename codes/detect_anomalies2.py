import os
import re
import glob
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
BEFORE_DIR = "dataset/before"
AFTER_DIR = "dataset/after"      # for comparing after_0/after_1
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(OUT_DIR, "detections")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Sensitivity bias:
# Lower threshold for before_3 / before_4 to be more sensitive
ANOM_RATE_THRESH_DEFAULT = 0.30
ANOM_RATE_THRESH_BEFORE34 = 0.20

PLOT_FIGS = True  # turn off if you just want CSV outputs


# -----------------------------
# HELPERS
# -----------------------------
def read_meta() -> dict:
    meta_path = os.path.join(OUT_DIR, "ocsvm_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("ocsvm_meta.json not found. Train first with ocsvm_train_fixed.py.")
    with open(meta_path, "r") as f:
        return json.load(f)

def read_and_clean(path: str, columns: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2)
    df.columns = df.columns.str.strip()
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {os.path.basename(path)}: {sorted(missing)}")
    sub = df[columns].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
    return sub

def sliding_windows(arr: np.ndarray, window: int, step: int) -> Tuple[np.ndarray, List[int]]:
    starts = list(range(0, len(arr) - window + 1, step))
    if not starts:
        return np.empty((0, window * arr.shape[1])), []
    w = np.stack([arr[s:s+window, :] for s in starts], axis=0)  # [N, W, F]
    flat = w.reshape(w.shape[0], -1)
    return flat, starts

def sliding_windows_padded(arr: np.ndarray, window: int, step: int) -> Tuple[np.ndarray, List[int]]:
    """
    For very short subsequences (e.g., before_4 with < window rows):
    build at least one window by padding with the last row.
    """
    if len(arr) == 0:
        return np.empty((0, window * arr.shape[1])), []
    if len(arr) < window:
        # single start = 0; pad last row
        chunk = arr.copy()
        pad_rows = window - len(arr)
        pad = np.repeat(chunk[-1:, :], pad_rows, axis=0)
        chunk = np.vstack([chunk, pad])
        return chunk.reshape(1, -1), [0]
    # normal case
    return sliding_windows(arr, window, step)


def choose_threshold(filename: str) -> float:
    """Lower anomaly rate threshold for before_3 / before_4."""
    base = os.path.basename(filename)
    if re.search(r"_before_(3|4)_", base):
        return ANOM_RATE_THRESH_BEFORE34
    return ANOM_RATE_THRESH_DEFAULT


def analyze_file(
    path: str,
    ocsvm,
    scaler,
    meta: dict
) -> Dict:
    cols = meta["columns"]
    W = int(meta["window_size"])
    S = int(meta["step_size"])

    df = read_and_clean(path, cols)
    # Scale per-feature as in training
    scaled = scaler.transform(df.values.astype(float))

    # Create flattened windows (with padding if too short)
    Xw, starts = sliding_windows_padded(scaled, W, S)
    if Xw.size == 0:
        return {
            "file": os.path.basename(path),
            "n_windows": 0,
            "anomaly_rate": None,
            "first_anom_start": None,
            "threshold_used": choose_threshold(path),
            "file_flagged": None,
            "csv": None,
            "plots": [],
        }

    preds = ocsvm.predict(Xw)  # +1 normal, -1 anomaly
    scores = ocsvm.decision_function(Xw).ravel()
    flags = (preds == -1).astype(int)

    anom_rate = float(flags.mean())
    first_anom_start = int(starts[int(np.argmax(flags))]) if anom_rate > 0 else None
    thr = choose_threshold(path)
    file_flagged = bool(anom_rate > thr)

    # Save per-file CSV of window scores
    name = os.path.basename(path).replace(".csv", "")
    out_csv = os.path.join(RESULTS_DIR, f"{name}_window_scores.csv")
    pd.DataFrame({
        "window_start": starts,
        "decision_score": scores,
        "is_anomaly": flags
    }).to_csv(out_csv, index=False)

    plots = []
    if PLOT_FIGS:
        # Anomaly flags
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(starts, flags, marker="o")
        plt.yticks([0,1])
        plt.title(f"Anomaly Flags — {name}")
        plt.xlabel("Window Start Index")
        plt.ylabel("Anomaly (1) / Normal (0)")
        plt.tight_layout()
        p1 = os.path.join(RESULTS_DIR, f"{name}_flags.png")
        plt.savefig(p1); plt.close()
        plots.append(p1)

        # Decision scores
        plt.figure(figsize=(10, 4))
        plt.plot(starts, scores, marker="o")
        plt.axhline(0.0, linestyle="--")
        plt.title(f"OC-SVM Decision Scores — {name}")
        plt.xlabel("Window Start Index")
        plt.ylabel("Decision Function")
        plt.tight_layout()
        p2 = os.path.join(RESULTS_DIR, f"{name}_scores.png")
        plt.savefig(p2); plt.close()
        plots.append(p2)

        # Correlation heatmap (raw, unscaled for interpretability)
        corr = df[cols].corr().values
        plt.figure(figsize=(6, 5))
        plt.imshow(corr, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
        plt.yticks(range(len(cols)), cols)
        plt.title(f"Feature Correlation — {name}")
        plt.tight_layout()
        p3 = os.path.join(RESULTS_DIR, f"{name}_corr.png")
        plt.savefig(p3); plt.close()
        plots.append(p3)

    return {
        "file": os.path.basename(path),
        "n_windows": int(len(starts)),
        "anomaly_rate": anom_rate,
        "first_anom_start": first_anom_start,
        "threshold_used": thr,
        "file_flagged": file_flagged,
        "csv": out_csv,
        "plots": plots,
    }


if __name__ == "__main__":
    # Load artifacts
    model_path = os.path.join(OUT_DIR, "ocsvm_model.pkl")
    scaler_path = os.path.join(OUT_DIR, "scaler.pkl")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError("Model/scaler not found. Run ocsvm_train_fixed.py first.")

    ocsvm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta = read_meta()

    # Targets: before_3, before_4, after_0, after_1 (if present)
    before_files = sorted(glob.glob(os.path.join(BEFORE_DIR, "*.csv")))
    target_before = [f for f in before_files if re.search(r"_before_(3|4)_", os.path.basename(f))]

    after_files = sorted(glob.glob(os.path.join(AFTER_DIR, "*.csv")))
    target_after01 = [f for f in after_files if re.search(r"_after_(0|1)_", os.path.basename(f))]

    targets = target_before + target_after01
    if not targets:
        # fallback to a couple of files so script still runs
        targets = (before_files[:2] if len(before_files) >= 2 else before_files) + target_after01

    results = [analyze_file(p, ocsvm, scaler, meta) for p in targets]

    # Summary CSV
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)

    print("✅ Detection complete.")
    print(f"Summary: {summary_path}")
    for r in results:
        print(f"- {r['file']}: windows={r['n_windows']}  anom_rate={r['anomaly_rate']}  "
              f"thr={r['threshold_used']}  flagged={r['file_flagged']}  first_idx={r['first_anom_start']}")
