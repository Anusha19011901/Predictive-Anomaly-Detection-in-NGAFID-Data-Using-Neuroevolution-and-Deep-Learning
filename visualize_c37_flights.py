#!/usr/bin/env python3
"""
Whole-flight overviews (shaded anomaly windows) using raw c_37 flights.

- Robust header handling (spaces/underscores/case) via canonical renaming
- Trains ONE OC-SVM on AFTER flights (per-feature StandardScaler)
- Scores BEFORE flights, shades anomalous windows, shows decision curve
- Uses fast CSV path (C engine), falls back to Python engine only if needed
- Caps rows read for training + caps number of AFTER files
- Hard filtering of NaN/Inf values at multiple stages
- Saves PNGs to outputs/c37_vis

Run (30 plots, modest train size):
  python visualize_c37_flights.py \
    --c37_dir "/Users/iyashi/Downloads/c_37" \
    --max_plots 30 \
    --max_after_files 30 \
    --train_windows_per_after 60

Process everything (not recommended on laptop):
  --max_plots -1
"""

import os, re, glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib

# -----------------------------
# Global defaults / paths
# -----------------------------
WINDOW_SIZE = 30
STEP_SIZE   = 5
TOPK_PLOT   = 6

# Reasonable laptop defaults
DEFAULT_MAX_AFTER_FILES       = 30
DEFAULT_TRAIN_WINDOWS_PER_AFT = 60   # ~60*5 + 25 ≈ 325 rows/file

def rows_needed_for_windows(nw, W=WINDOW_SIZE, S=STEP_SIZE):
    # rows ~= windows*STEP + (WINDOW - STEP)
    return int(nw * S + (W - S))

OUT_ROOT   = Path("outputs")
MODEL_DIR  = OUT_ROOT / "c37_ocsvm"
PLOTS_DIR  = OUT_ROOT / "c37_vis"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Preferred labels after normalization
PREFERRED_FEATURES = [
    "AltMSL","AltB","IAS","TAS","VSpd","VSpdG","Pitch","Roll","HDG","LatAc","NormAc",
    "E1 RPM","E1 FFlow","E1 OilT","E1 OilP","E1 CHT1","E1 CHT2","E1 CHT4","E1 EGT1","E1 EGT2","E1 EGT4",
    "FQtyL","FQtyR","volt1","volt2","amp1","amp2","Latitude","Longitude","BaroA","MagVar","COM1","COM2","CRS",
]

CANDIDATE_COLS = [
    "AltMSL","AltB","IAS","TAS","VSpd","VSpdG",
    "E1 RPM","E1_RPM",
    "E1_FFlow","E1 FFlow",
    "E1_CHT1","E1 CHT1","E1_CHT2","E1 CHT2","E1_CHT4","E1 CHT4",
    "E1_EGT1","E1 EGT1","E1_EGT2","E1 EGT2","E1_EGT4","E1 EGT4",
    "NormAc","LatAc","Pitch","Roll","HDG"
]

# -----------------------------
# File listing
# -----------------------------
def list_flights(folder: str, kind: str) -> List[str]:
    return sorted([p for p in glob.glob(os.path.join(folder, f"*_{kind}_*.csv"))])

# -----------------------------
# Header canonicalization & robust CSV reader
# -----------------------------
_CANON_CACHE: Dict[str, str] = {}

def _canon(name: str) -> str:
    if name in _CANON_CACHE:
        return _CANON_CACHE[name]
    key = re.sub(r"[^0-9a-zA-Z]+", "", str(name)).lower()
    _CANON_CACHE[name] = key
    return key

def sniff_header_row(path: str, max_lines: int = 250) -> Tuple[int, List[str]]:
    candidates = {"altmsl","altb","ias","rpm","fflow","cht","egt",
                  "pitch","roll","normac","latac","hdg","latitude","longitude","tas","vspd","vspdg"}
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > max_lines:
                break
            line = line.strip()
            if not line or line.count(",") < 5:
                continue
            cols = [c.strip() for c in line.split(",")]
            can_set = {_canon(c) for c in cols}
            if len(candidates & can_set) >= 2:
                return i, cols
    # fallback: first line
    with open(path, "r", errors="ignore") as f:
        first = f.readline().strip()
    cols = [c.strip() for c in first.split(",")]
    return 0, cols

def _build_rename_map(actual_cols: List[str], preferred: List[str]) -> Dict[str, str]:
    can_to_actual: Dict[str, str] = {}
    for c in actual_cols:
        can = _canon(c)
        if can not in can_to_actual:
            can_to_actual[can] = c
    rename_map: Dict[str, str] = {}
    for p in preferred:
        can_p = _canon(p)
        if can_p in can_to_actual:
            rename_map[can_to_actual[can_p]] = p
    return rename_map

def read_ngafid_csv(path: str,
                    usecols: Optional[List[str]] = None,
                    nrows: Optional[int] = None) -> pd.DataFrame:
    hdr_idx, _ = sniff_header_row(path)
    # fast C engine first
    try:
        df = pd.read_csv(path, header=hdr_idx, engine="c", nrows=nrows)
        df.columns = df.columns.str.strip()
    except Exception:
        # tolerant Python engine fallback
        df = pd.read_csv(path, header=hdr_idx, engine="python", on_bad_lines="skip", nrows=nrows)
        df.columns = df.columns.str.strip()

    preferred = usecols if usecols else PREFERRED_FEATURES
    rename_map = _build_rename_map(list(df.columns), preferred)
    df = df.rename(columns=rename_map)

    if usecols is not None:
        keep = [c for c in usecols if c in df.columns]
        if not keep:
            raise ValueError(f"No requested columns found in {os.path.basename(path)}")
        df = df[keep]

    # numeric + finite
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")  # drop any row with NaN/Inf
    return df

# -----------------------------
# Feature picking (header-only)
# -----------------------------
def pick_feature_set(files: List[str]) -> List[str]:
    if not files:
        raise RuntimeError("No files to infer feature set from.")
    counts: Dict[str,int] = {}
    for f in files:
        _, cols = sniff_header_row(f)
        for c in cols:
            counts[c] = counts.get(c, 0) + 1
    n = len(files)
    commonish = [c for c,v in counts.items() if v >= max(2, n//2)]
    bad_tokens = {"date","time","ident","mode","enum","bool","record"}
    numericish = [c for c in commonish if not any(t in _canon(c) for t in bad_tokens)]
    if not numericish:
        numericish = commonish

    canon_to_actual = { _canon(c): c for c in numericish }
    chosen = []
    for cand in CANDIDATE_COLS:
        key = _canon(cand)
        if key in canon_to_actual:
            chosen.append(canon_to_actual[key])
    if not chosen:
        chosen = numericish
    return list(dict.fromkeys(chosen))

# -----------------------------
# Helpers: windows & display
# -----------------------------
def sliding_windows_from_scaled(X: np.ndarray, W: int, S: int) -> Tuple[np.ndarray, List[int]]:
    starts = list(range(0, len(X) - W + 1, S))
    if not starts:
        return np.empty((0, W * X.shape[1])), []
    win = np.stack([X[s:s+W,:] for s in starts], axis=0)  # [N, W, F]
    flat = win.reshape(win.shape[0], -1)
    return flat, starts

def variance_rank_cols(df: pd.DataFrame, k: int) -> List[str]:
    v = df.var(numeric_only=True).sort_values(ascending=False)
    return [c for c in v.index[:k]]

def window_idx_to_span(wi: int, W=WINDOW_SIZE, S=STEP_SIZE) -> Tuple[int,int]:
    s = wi * S
    return s, s + W - 1

# -----------------------------
# Train OC-SVM on AFTER flights
# -----------------------------
def train_model(c37_dir: str,
                force_retrain: bool = False,
                max_after_files: int = DEFAULT_MAX_AFTER_FILES,
                train_windows_per_after: int = DEFAULT_TRAIN_WINDOWS_PER_AFT):
    model_p = MODEL_DIR / "ocsvm.pkl"
    scaler_p = MODEL_DIR / "scaler.pkl"
    meta_p   = MODEL_DIR / "meta.json"

    if model_p.exists() and scaler_p.exists() and meta_p.exists() and not force_retrain:
        ocsvm = joblib.load(model_p)
        scaler = joblib.load(scaler_p)
        meta = pd.read_json(meta_p, typ="series").to_dict()
        return ocsvm, scaler, meta

    after_files_all = list_flights(c37_dir, "after")
    if not after_files_all:
        raise RuntimeError("No AFTER files found in the c_37 folder.")

    after_files = after_files_all[:max_after_files]
    print(f"\n[TRAIN] Using up to {max_after_files} AFTER files out of {len(after_files_all)} total.")

    cols = pick_feature_set(after_files)
    if len(cols) < 4:
        raise RuntimeError(f"Too few usable columns inferred: {cols}")

    print(f"[TRAIN] Feature count: {len(cols)}")
    print(f"[TRAIN] Example features: {cols[:10]}{' ...' if len(cols)>10 else ''}")

    dfs = []
    per_file_rows = rows_needed_for_windows(train_windows_per_after, WINDOW_SIZE, STEP_SIZE)
    for f in after_files:
        try:
            print(f"[TRAIN] Reading {os.path.basename(f)} (≤{per_file_rows} rows)")
            df = read_ngafid_csv(f, usecols=cols, nrows=per_file_rows)
            if len(df) >= WINDOW_SIZE:
                dfs.append(df)
            else:
                print("  [skip] Not enough clean rows for one window.")
        except Exception as e:
            print(f"  [warn] {os.path.basename(f)}: {e}")

    if not dfs:
        raise RuntimeError("No AFTER files have enough rows for windowing.")

    all_after = pd.concat(dfs, axis=0, ignore_index=True)

    # Extra safety: finite filtering before scaling
    finite_mask = np.isfinite(all_after.values).all(axis=1)
    if finite_mask.sum() < len(all_after):
        print(f"[TRAIN] Dropped {len(all_after) - int(finite_mask.sum())} non-finite rows pre-scaling.")
    all_after = all_after.loc[finite_mask].reset_index(drop=True)

    # Scale per-feature
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_after.values.astype(np.float64))

    # Safety: drop non-finite rows after scaling (should be none)
    finite_mask2 = np.isfinite(all_scaled).all(axis=1)
    if finite_mask2.sum() < all_scaled.shape[0]:
        print(f"[TRAIN] Dropped {all_scaled.shape[0] - int(finite_mask2.sum())} non-finite rows post-scaling.")
    all_scaled = all_scaled[finite_mask2]

    # Build flattened windows per file
    X_list = []
    cursor = 0
    for df in dfs:
        n = len(df)
        block = all_scaled[cursor:cursor+n, :]
        cursor += n
        if block.shape[0] < WINDOW_SIZE:
            continue
        flat, _ = sliding_windows_from_scaled(block, WINDOW_SIZE, STEP_SIZE)
        if flat.size:
            # Final guard: ensure no NaN/Inf in windows from this file
            good = np.isfinite(flat).all(axis=1)
            if good.sum() < flat.shape[0]:
                print(f"  [warn] Dropped {flat.shape[0]-int(good.sum())} windows with non-finite values in {df.shape[1]}-D.")
            flat = flat[good]
            if flat.size:
                X_list.append(flat)

    if not X_list:
        raise RuntimeError("Empty training set after windowing/cleaning.")

    X_train = np.vstack(X_list)

    # Last safety: replace any residual NaNs/Infs just in case (should be none)
    if not np.isfinite(X_train).all():
        print("[TRAIN] np.nan_to_num applied to X_train (unexpected non-finite values found).")
        X_train = np.nan_to_num(X_train, copy=False, posinf=0.0, neginf=0.0)

    print(f"[TRAIN] Total windows: {X_train.shape[0]}   Dim: {X_train.shape[1]}")

    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(X_train)

    # Persist
    joblib.dump(ocsvm, model_p)
    joblib.dump(scaler, scaler_p)
    pd.Series({
        "columns": cols,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "max_after_files": int(max_after_files),
        "train_windows_per_after": int(train_windows_per_after)
    }).to_json(meta_p)

    return ocsvm, scaler, {
        "columns": cols,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE
    }

# -----------------------------
# Plot a whole BEFORE flight
# -----------------------------
def plot_flight(csv_path: str,
                ocsvm: OneClassSVM,
                scaler: StandardScaler,
                meta: Dict,
                max_features: int = TOPK_PLOT) -> str:
    name = Path(csv_path).stem
    print(f"[PLOT] {name}")
    raw  = read_ngafid_csv(csv_path, usecols=meta["columns"])

    if len(raw) < WINDOW_SIZE:
        raise RuntimeError(f"{name}: too short for a window.")

    # Score sliding windows
    X = scaler.transform(raw.values.astype(np.float64))
    # guard against non-finite rows (shouldn’t happen here)
    if not np.isfinite(X).all():
        good_rows = np.isfinite(X).all(axis=1)
        raw = raw.iloc[good_rows].reset_index(drop=True)
        X   = X[good_rows]

    Xw, starts = sliding_windows_from_scaled(X, meta["window_size"], meta["step_size"])
    flags = np.zeros(len(starts), dtype=int)
    scores = np.zeros(len(starts), dtype=float)
    if Xw.size:
        good = np.isfinite(Xw).all(axis=1)
        Xw = np.nan_to_num(Xw[good], copy=False, posinf=0.0, neginf=0.0)
        preds = ocsvm.predict(Xw)
        flags = (preds == -1).astype(int)
        scores = ocsvm.decision_function(Xw).ravel()

    # Choose display features by raw variance (more interpretable)
    if raw.shape[1] > max_features:
        disp_cols = variance_rank_cols(raw, max_features)
    else:
        disp_cols = list(raw.columns)

    T = len(raw)
    spans = [window_idx_to_span(i, meta["window_size"], meta["step_size"]) for i in range(len(starts))]

    n_panels = len(disp_cols) + 1
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, max(2.5*len(disp_cols)+2, 6)),
        sharex=True,
        gridspec_kw={"height_ratios":[1]*len(disp_cols)+[0.6]}
    )
    if n_panels == 2:
        axes = np.array(axes)

    x = np.arange(T)
    for i, c in enumerate(disp_cols):
        ax = axes[i]
        y = pd.to_numeric(raw[c], errors="coerce").to_numpy()
        ax.plot(x, y, lw=1.2, label=c)
        for (s, e), f in zip(spans, flags):
            if f == 1:
                ax.axvspan(s, min(e, T-1), color="tab:red", alpha=0.22)
        ax.set_ylabel(c)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    centers = [min((s+e)//2, T-1) for (s,e) in spans]
    ax = axes[-1]
    if len(centers):
        ax.plot(centers, scores, marker="o", lw=1.0)
        ax.axhline(0.0, ls="--", lw=1.0)
        for j,(cx,f) in enumerate(zip(centers, flags)):
            if f == 1:
                ax.plot(cx, scores[j], "o", ms=5, color="tab:red")
    ax.set_ylabel("OC-SVM score")
    ax.set_xlabel("t (row index)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Flight: {name}  |  shaded = anomaly windows ({meta['window_size']}/{meta['step_size']})")

    fig.tight_layout()
    out_png = str(PLOTS_DIR / f"{name}_overview.png")
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_png

# -----------------------------
# Main
# -----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--c37_dir", required=True, help="Path to /Users/iyashi/Downloads/c_37")
    ap.add_argument("--max_plots", type=int, default=20, help="-1 = all BEFORE flights; else cap")
    ap.add_argument("--retrain", action="store_true", help="Force retraining on c_37 AFTER flights")
    ap.add_argument("--max_after_files", type=int, default=DEFAULT_MAX_AFTER_FILES,
                    help="Cap number of AFTER files used for training (default 30)")
    ap.add_argument("--train_windows_per_after", type=int, default=DEFAULT_TRAIN_WINDOWS_PER_AFT,
                    help="Cap training windows per AFTER file (default 60)")
    args = ap.parse_args()

    print("==== c_37 whole-flight visualization ====")
    print("Folder     :", args.c37_dir)
    print("Max plots  :", args.max_plots)
    print("Retrain    :", args.retrain)
    print("W/Step     :", WINDOW_SIZE, STEP_SIZE)
    print("Train cap  :", f"{args.train_windows_per_after} windows/file, "
                         f"{rows_needed_for_windows(args.train_windows_per_after)} rows/file")
    print("After cap  :", f"{args.max_after_files} files")
    print("Out plots  :", str(PLOTS_DIR.resolve()))
    print("=========================================")

    ocsvm, scaler, meta = train_model(
        args.c37_dir,
        force_retrain=args.retrain,
        max_after_files=args.max_after_files,
        train_windows_per_after=args.train_windows_per_after
    )

    before_files = list_flights(args.c37_dir, "before")
    if not before_files:
        raise RuntimeError("No BEFORE flights found in the c_37 folder.")

    if args.max_plots != -1:
        before_files = before_files[:max(1, args.max_plots)]

    saved = []
    for i, f in enumerate(before_files, 1):
        try:
            out = plot_flight(f, ocsvm, scaler, meta, max_features=TOPK_PLOT)
            saved.append(out)
            print(f"[{i:02d}/{len(before_files)}] Saved: {out}")
        except Exception as e:
            print(f"[warn] {Path(f).name}: {e}")

    print(f"\n✅ Done. Saved {len(saved)} plot(s) to: {PLOTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
