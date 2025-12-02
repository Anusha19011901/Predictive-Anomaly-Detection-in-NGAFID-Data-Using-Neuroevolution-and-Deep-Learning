#!/usr/bin/env python3
# dbscan_boxes.py
# Explain windows using DBSCAN "prototype boxes" saved by sweep/fit stage.
# - Robust NGAFID CSV parsing (headered or headerless).
# - Per-timestep scaling (7 features) -> flatten to 210.
# - If prototypes npz includes PCA, apply SAME PCA before matching.
# - Match by nearest centroid in the (PCA-or-raw) space; count "box" violations.
#
# Commands:
#   explain        : run on a folder of NGAFID flights (e.g., BEFORE)
#   explain_error  : run on exact_data/anomaly windows (30 x 7 CSVs)

import os, re, glob, argparse, json
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# NGAFID template & helpers
# -----------------------------
NGAFID_HEADER_TEMPLATE = [
    "Lcl Date","Lcl Time","UTCOfst","AtvWpt","Latitude","Longitude","AltB","BaroA","AltMSL","OAT",
    "IAS","GndSpd","VSpd","Pitch","Roll","LatAc","NormAc","HDG","TRK","volt1","volt2","amp1","amp2",
    "FQtyL","FQtyR","E1 FFlow","E1 OilT","E1 OilP","E1 RPM","E1 CHT1","E1 CHT2","E1 CHT3","E1 CHT4",
    "E1 EGT1","E1 EGT2","E1 EGT3","E1 EGT4","AltGPS","TAS","HSIS","CRS","NAV1","NAV2","COM1","COM2",
    "HCDI","VCDI","WndSpd","WndDr","WptDst","WptBrg","MagVar","AfcsOn","RollM","PitchM","RollC","PichC",
    "VSpdG","GPSfix","HAL","VAL","HPLwas","HPLfd","VPLwas"
]
NGAFID_POS = {name: idx for idx, name in enumerate(NGAFID_HEADER_TEMPLATE)}

def canon(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _all_look_numeric(names: List[str]) -> bool:
    for x in names:
        try:
            float(str(x))
        except Exception:
            return False
    return True

def align_columns(df: pd.DataFrame, desired: List[str]) -> List[str]:
    m = {canon(c): c for c in df.columns}
    got, miss = [], []
    for d in desired:
        key = canon(d)
        if key in m: got.append(m[key])
        else: miss.append(d)
    if miss:
        ex = [str(c) for c in df.columns[:30]]
        raise KeyError(f"Missing columns {miss}. Available example: {ex}")
    return got

def _load_template_header(template_dir: Optional[str]) -> List[str]:
    if template_dir and os.path.isdir(template_dir):
        for p in sorted(glob.glob(os.path.join(template_dir, "*.csv"))):
            try:
                df = pd.read_csv(p, skiprows=2, nrows=1)
                cols = [str(c).strip() for c in df.columns]
                if not _all_look_numeric(cols):
                    return cols
            except Exception:
                pass
    return NGAFID_HEADER_TEMPLATE[:]

def read_ngafid(path: str, cols: List[str],
                template_dir: Optional[str]=None,
                force_template: bool=False) -> pd.DataFrame:
    """
    Robust NGAFID reader:
      - Try skiprows=2 with headers + name-align
      - If fails OR force_template: headerless with template assignment
      - Final fallback: pick by NGAFID positions
    """
    if not force_template:
        try:
            df = pd.read_csv(path, skiprows=2)
            df.columns = df.columns.str.strip()
            use = align_columns(df, cols)
            sub = df[use].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
            sub.columns = cols
            return sub
        except Exception:
            pass

    # headerless fallback
    df_raw = pd.read_csv(path, header=None, skiprows=2)
    templ = _load_template_header(template_dir)
    m = df_raw.shape[1]
    header = templ[:m] if len(templ) >= m else templ + [f"Dummy_{i}" for i in range(m - len(templ))]
    header = [str(x).strip() for x in header]
    df_raw.columns = header
    try:
        use = align_columns(df_raw, cols)
        sub = df_raw[use].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
        sub.columns = cols
        return sub
    except Exception:
        miss = [c for c in cols if c not in NGAFID_POS]
        if miss:
            raise
        idxs = [NGAFID_POS[c] for c in cols]
        if any(i >= m for i in idxs):
            raise ValueError(f"{os.path.basename(path)} has {m} cols; requested {idxs}")
        sub = df_raw.iloc[:, idxs].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
        sub.columns = cols
        return sub

def list_any_files(folder: str, pat: str="*.csv") -> List[str]:
    return sorted(glob.glob(os.path.join(folder, pat)))

# -----------------------------
# Windowing & scaling
# -----------------------------
def sliding_windows(A: np.ndarray, w: int, step: int) -> Tuple[np.ndarray, List[int]]:
    starts = list(range(0, len(A) - w + 1, step))
    if not starts:
        return np.empty((0, w, A.shape[1])), []
    W = np.stack([A[s:s+w, :] for s in starts], axis=0)  # [N,W,F]
    return W, starts

def transform_windows_for_scaler(win3d: np.ndarray, scaler, window: int, n_feats: int) -> np.ndarray:
    """
    If scaler fit on F (7): scale per timestep then flatten.
    If scaler fit on W*F (210): flatten then scale.
    """
    if win3d.size == 0:
        return np.empty((0, window * n_feats))
    n, w, f = win3d.shape
    want = getattr(scaler, "n_features_in_", None)
    if want == f:
        X2 = win3d.reshape(-1, f)
        X2s = scaler.transform(X2)
        return X2s.reshape(n, w, f).reshape(n, w*f)
    elif want == w * f:
        Xflat = win3d.reshape(n, w*f)
        return scaler.transform(Xflat)
    else:
        raise ValueError(f"Scaler expects {want}, but W*F={w*f} and F={f}")

# -----------------------------
# Prototypes & PCA handling
# -----------------------------
class Proto:
    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=True)
        self.centroids = z["centroids"]            # [K, Dp] (Dp = PCA dims or 210)
        self.halfwidths = z["halfwidths"]          # [K, Dp]
        self.columns = list(z["columns"])
        self.window_size = int(z["window_size"][0])
        self.step_size = int(z["step_size"][0])
        # Optional PCA payload
        self.has_pca = ("pca_components" in z) and ("pca_mean" in z)
        if self.has_pca:
            self.pca_components = z["pca_components"]   # [Dp, D]
            self.pca_mean = z["pca_mean"].ravel()       # [D]
            # Optional flags/attrs if present; safe defaults otherwise
            self.pca_whiten = bool(z["pca_whiten"][0]) if "pca_whiten" in z and z["pca_whiten"].size else False
        else:
            self.pca_components = None
            self.pca_mean = None
            self.pca_whiten = False

    def project(self, Xflat: np.ndarray) -> np.ndarray:
        """Project Xflat to prototype space (apply PCA if prototypes were saved with PCA)."""
        if not self.has_pca:
            return Xflat
        # Center then project
        Xc = Xflat - self.pca_mean[None, :]
        return Xc @ self.pca_components.T  # [N, Dp]

# -----------------------------
# Distance & violation helpers
# -----------------------------
def nearest_centroid(x: np.ndarray, centroids: np.ndarray) -> Tuple[int, float]:
    diffs = centroids - x[None, :]
    d2 = np.sum(diffs * diffs, axis=1)
    k = int(np.argmin(d2))
    return k, float(np.sqrt(d2[k]))

def count_violations(x: np.ndarray, c: np.ndarray, h: np.ndarray) -> Tuple[int, float]:
    """
    Count how many dims of x fall outside [c-h, c+h] and sum normalized severity over ALL dims.
    This is generic (works in raw or PCA space). Returns (count_total, severity_total).
    """
    lo, hi = c - h, c + h
    below = x < lo
    above = x > hi
    cnt = int(np.count_nonzero(below | above))
    # normalized severity: sum |x - c| / (h + eps)
    hw = np.maximum(h, 1e-8)
    sev = float(np.sum(np.abs(x - c) / hw))
    return cnt, sev

# -----------------------------
# Explain BEFORE folder (windowize & match)
# -----------------------------
def explain_folder(
    in_dir: str,
    scaler_path: str,
    prototypes_path: str,
    out_csv: str,
    columns: List[str],
    window_size: int,
    step_size: int,
    template_dir: Optional[str],
    force_template: bool
):
    P = Proto(prototypes_path)
    if window_size != P.window_size or step_size != P.step_size:
        raise SystemExit(f"Window/step mismatch with prototypes ({P.window_size},{P.step_size}).")

    scaler = joblib.load(scaler_path)

    files = list_any_files(in_dir, "*.csv")
    all_rows = []
    for f in files:
        try:
            df = read_ngafid(f, columns, template_dir, force_template)
            X = df.values.astype(float)
            win, starts = sliding_windows(X, window_size, step_size)    # [N, W, F]
            Xflat = transform_windows_for_scaler(win, scaler, window_size, len(columns))  # [N, 210]
            if Xflat.size == 0:
                continue
            Xp = P.project(Xflat)  # [N, Dp]

            for w_idx, x in enumerate(Xp):
                k, d = nearest_centroid(x, P.centroids)
                c = P.centroids[k]
                h = P.halfwidths[k]
                vcnt, vsev = count_violations(x, c, h)

                all_rows.append({
                    "file": os.path.basename(f),
                    "window_idx": int(w_idx),
                    "start_idx": int(starts[w_idx]),
                    "end_idx": int(starts[w_idx] + window_size - 1),
                    "prototype_id": int(k),
                    "nearest_dist": float(d),
                    "viol_count_total": int(vcnt),
                    "viol_sev_total": float(vsev),
                })
        except Exception as e:
            print(f"[WARN] {os.path.basename(f)}: {e}")

    if not all_rows:
        print("No windows explained (empty).")
        return

    out_df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Wrote explanations for {len(out_df)} windows → {out_csv}")

# -----------------------------
# Explain ERROR windows (exact_data/anomaly window_*.csv)
# -----------------------------
def _read_error_window_matrix(path: str, cols: List[str]) -> np.ndarray:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # If headerless numeric-like, assign desired cols
    if _all_look_numeric(list(df.columns)):
        if df.shape[1] != len(cols):
            raise ValueError(f"{os.path.basename(path)} has {df.shape[1]} cols, expected {len(cols)}")
        df.columns = cols
        return df.apply(pd.to_numeric, errors="coerce").values.astype(float)
    use = align_columns(df, cols)
    return df[use].apply(pd.to_numeric, errors="coerce").values.astype(float)

def explain_error_dir(
    err_dir: str,
    scaler_path: str,
    prototypes_path: str,
    out_csv: str,
    columns: List[str],
    window_size: int
):
    P = Proto(prototypes_path)
    if window_size != P.window_size:
        raise SystemExit(f"Window mismatch with prototypes ({P.window_size}).")

    scaler = joblib.load(scaler_path)

    files = list_any_files(err_dir, "window_*.csv")
    rows = []
    for f in files:
        try:
            M = _read_error_window_matrix(f, columns)      # [W, F]
            if M.shape[0] != window_size:
                # tolerate minor mismatches by clipping/padding if needed
                if M.shape[0] < window_size:
                    # pad last row
                    pad = np.repeat(M[-1:, :], window_size - M.shape[0], axis=0)
                    M = np.vstack([M, pad])
                else:
                    M = M[:window_size, :]
            Ms = scaler.transform(M.astype(float))         # per-timestep scaling (F=7)
            xflat = Ms.reshape(1, -1)                      # [1, 210]
            xp = P.project(xflat)                          # [1, Dp]
            k, d = nearest_centroid(xp[0], P.centroids)
            c = P.centroids[k]
            h = P.halfwidths[k]
            vcnt, vsev = count_violations(xp[0], c, h)
            rows.append({
                "file": os.path.basename(f),
                "prototype_id": int(k),
                "nearest_dist": float(d),
                "viol_count_total": int(vcnt),
                "viol_sev_total": float(vsev),
            })
        except Exception as e:
            print(f"[WARN] {os.path.basename(f)}: {e}")

    if not rows:
        print("No error windows explained.")
        return
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote ERROR explanations for {len(df)} windows → {out_csv}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # BEFORE folder
    ap_e = sub.add_parser("explain", help="Explain windowed flights in a folder (e.g., BEFORE).")
    ap_e.add_argument("--input_dir", required=True)
    ap_e.add_argument("--scaler_path", required=True)
    ap_e.add_argument("--prototypes_path", required=True)
    ap_e.add_argument("--out_csv", default="outputs/dbscan/explanations_before.csv")
    ap_e.add_argument("--columns", nargs="+", default=["AltMSL","E1 RPM","E1 FFlow","E1 CHT1","E1 EGT1","NormAc","IAS"])
    ap_e.add_argument("--window_size", type=int, default=30)
    ap_e.add_argument("--step_size", type=int, default=25)
    ap_e.add_argument("--template_dir", default="dataset/after_examm2")
    ap_e.add_argument("--force_template", action="store_true")

    # ERROR folder (exact windows)
    ap_x = sub.add_parser("explain_error", help="Explain pre-extracted error windows (exact_data/anomaly).")
    ap_x.add_argument("--error_dir", required=True)
    ap_x.add_argument("--scaler_path", required=True)
    ap_x.add_argument("--prototypes_path", required=True)
    ap_x.add_argument("--out_csv", default="outputs/dbscan/explanations_error.csv")
    ap_x.add_argument("--columns", nargs="+", default=["AltMSL","E1 RPM","E1 FFlow","E1 CHT1","E1 EGT1","NormAc","IAS"])
    ap_x.add_argument("--window_size", type=int, default=30)

    args = ap.parse_args()
    if args.cmd == "explain":
        explain_folder(
            in_dir=args.input_dir,
            scaler_path=args.scaler_path,
            prototypes_path=args.prototypes_path,
            out_csv=args.out_csv,
            columns=args.columns,
            window_size=args.window_size,
            step_size=args.step_size,
            template_dir=args.template_dir,
            force_template=args.force_template
        )
    elif args.cmd == "explain_error":
        explain_error_dir(
            err_dir=args.error_dir,
            scaler_path=args.scaler_path,
            prototypes_path=args.prototypes_path,
            out_csv=args.out_csv,
            columns=args.columns,
            window_size=args.window_size
        )

if __name__ == "__main__":
    main()
