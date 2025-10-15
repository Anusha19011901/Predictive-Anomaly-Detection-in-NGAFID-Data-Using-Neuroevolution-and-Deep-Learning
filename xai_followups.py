#!/usr/bin/env python3
# xai_followups.py
# Extra analyses for OC-SVM + Prototype Boxes (RAW & ERROR).

import os, argparse, glob, json, re
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import joblib

# NGAFID header template (order matters)
NGAFID_HEADER_TEMPLATE = [
    "Lcl Date","Lcl Time","UTCOfst","AtvWpt","Latitude","Longitude","AltB","BaroA","AltMSL","OAT",
    "IAS","GndSpd","VSpd","Pitch","Roll","LatAc","NormAc","HDG","TRK","volt1","volt2","amp1","amp2",
    "FQtyL","FQtyR","E1 FFlow","E1 OilT","E1 OilP","E1 RPM","E1 CHT1","E1 CHT2","E1 CHT3","E1 CHT4",
    "E1 EGT1","E1 EGT2","E1 EGT3","E1 EGT4","AltGPS","TAS","HSIS","CRS","NAV1","NAV2","COM1","COM2",
    "HCDI","VCDI","WndSpd","WndDr","WptDst","WptBrg","MagVar","AfcsOn","RollM","PitchM","RollC","PichC",
    "VSpdG","GPSfix","HAL","VAL","HPLwas","HPLfd","VPLwas"
]
NGAFID_POS = {name: idx for idx, name in enumerate(NGAFID_HEADER_TEMPLATE)}

# Defaults
RAW_EXPL_CSV_DEF   = "outputs/prototype_explanations_before.csv"
ERR_EXPL_CSV_DEF   = "outputs/prototype_explanations_errors.csv"
RAW_PROTO_DEF      = "outputs/prototypes.npz"
ERR_PROTO_DEF      = "outputs/prototypes_errors.npz"
RAW_SCALER_DEF     = "outputs/scaler.pkl"
ERR_SCALER_DEF     = "outputs/scalers/error_scaler.pkl"
OCSVM_MODEL_DEF    = "outputs/ocsvm_model.pkl"

AFTER_DIR_DEF      = "dataset/after"
BEFORE_DIR_DEF     = "dataset/before_examm2"
ERR_NORMAL_DIR_DEF = "exact_data/normal"
ERR_ANOM_DIR_DEF   = "exact_data/anomaly"

COLS_DEF = ["AltMSL","E1 RPM","E1 FFlow","E1 CHT1","E1 EGT1","NormAc","IAS"]
W_DEF = 30
STEP_DEF = 25

# -------------------------
# Helpers
# -------------------------
def canon(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def align_columns(df: pd.DataFrame, desired_cols: List[str]) -> List[str]:
    df_map = {canon(c): c for c in df.columns}
    resolved, missing = [], []
    for want in desired_cols:
        key = canon(want)
        if key in df_map:
            resolved.append(df_map[key])
        else:
            missing.append(want)
    if missing:
        avail = ", ".join(map(str, df.columns[:30]))
        raise KeyError(f"Could not find columns {missing}. Available example columns: {avail}")
    return resolved

def _all_look_numeric(names: List[str]) -> bool:
    for s in names:
        try: float(str(s))
        except Exception: return False
    return True

def _load_template_header(template_dir: str) -> List[str]:
    cand = sorted(glob.glob(os.path.join(template_dir, "*.csv")))
    for p in cand:
        try:
            df = pd.read_csv(p, skiprows=2, nrows=1)
            cols = [str(c).strip() for c in df.columns]
            if not _all_look_numeric(cols):  # real headers
                return cols
        except Exception:
            pass
    return NGAFID_HEADER_TEMPLATE[:]

def list_any_files(folder, pat="*.csv"):
    return sorted(glob.glob(os.path.join(folder, pat)))

def read_ngafid_clean(path: str, cols: List[str], template_dir: str) -> pd.DataFrame:
    # try NGAFID (skip 2 meta rows)
    df = pd.read_csv(path, skiprows=2)
    df.columns = df.columns.str.strip()
    try:
        use_cols = align_columns(df, cols)
        sub = df[use_cols].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
        sub.columns = cols
        return sub
    except KeyError:
        # headerless: assign template
        df_raw = pd.read_csv(path, header=None)
        templ = _load_template_header(template_dir)
        m = df_raw.shape[1]
        header = [str(x).strip() for x in (templ[:m] if len(templ) >= m else templ + [f"Dummy_{i}" for i in range(m-len(templ))])]
        df_raw.columns = header
        try:
            use_cols = align_columns(df_raw, cols)
            sub = df_raw[use_cols].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
            sub.columns = cols
            return sub
        except KeyError:
            # final fallback: by NGAFID positions
            missing_in_template = [c for c in cols if c not in NGAFID_POS]
            if missing_in_template:
                raise KeyError(f"Columns not in NGAFID template: {missing_in_template}")
            idxs = [NGAFID_POS[c] for c in cols]
            too_wide = [c for c, idx in zip(cols, idxs) if idx >= m]
            if too_wide:
                raise ValueError(f"File has {m} cols, some requested map past end: {too_wide}")
            sub = df_raw.iloc[:, idxs].replace("", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
            sub.columns = cols
            return sub

def sliding_windows(arr2d: np.ndarray, w: int, step: int) -> Tuple[np.ndarray, List[int]]:
    starts = list(range(0, len(arr2d) - w + 1, step))
    if not starts: return np.empty((0, w, arr2d.shape[1])), []
    win = np.stack([arr2d[s:s+w, :] for s in starts], axis=0)
    return win, starts

def flatten_windows(win: np.ndarray) -> np.ndarray:
    if win.size == 0: return np.empty((0,0))
    n, w, f = win.shape
    return win.reshape(n, w*f)

def transform_windows_for_scaler(win_3d: np.ndarray, scaler, window: int, n_feats: int) -> np.ndarray:
    if win_3d.size == 0: return np.empty((0, window * n_feats))
    n, w, f = win_3d.shape
    want = getattr(scaler, "n_features_in_", None)
    if want == f:
        X2 = win_3d.reshape(-1, f)
        X2s = scaler.transform(X2)
        return X2s.reshape(n, w, f).reshape(n, w*f)
    elif want == w * f:
        Xflat = win_3d.reshape(n, w*f)
        return scaler.transform(Xflat)
    else:
        raise ValueError(f"Scaler expects {want} features, but window*F={w*f} and F={f}.")

def read_error_window_matrix(path: str, cols: List[str]) -> np.ndarray:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if _all_look_numeric(list(df.columns)):
        if df.shape[1] != len(cols):
            raise ValueError(f"{os.path.basename(path)} has {df.shape[1]} cols, expected {len(cols)}.")
        df.columns = cols
        return df.apply(pd.to_numeric, errors="coerce").values.astype(float)
    use_cols = align_columns(df, cols)
    return df[use_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)

def read_error_window_vector(path: str, cols: List[str]) -> np.ndarray:
    return read_error_window_matrix(path, cols).reshape(-1)

def load_expl_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "window_idx" not in df.columns:
        df["window_idx"] = df.groupby("file").cumcount()
    return df.sort_values(["file","window_idx"]).reset_index(drop=True)

def nearest_dists_to_proto(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    d2 = ((X[:,None,:] - centroids[None,:,:])**2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))

# -------------------------
# sync: scatter + AUC
# -------------------------
def cmd_sync(args):
    print("[sync] before_dir:", args.before_dir)
    print("[sync] err_anom_dir:", args.err_anom_dir)
    print("[sync] raw_proto exists:", os.path.exists(args.raw_proto))
    print("[sync] err_proto exists:", os.path.exists(args.err_proto))
    print("[sync] raw_scaler:", args.raw_scaler)
    print("[sync] err_scaler:", args.err_scaler)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    ocsvm      = joblib.load(args.ocsvm_model)
    scaler_raw = joblib.load(args.raw_scaler)
    scaler_err = joblib.load(args.err_scaler)
    proto_raw  = np.load(args.raw_proto, allow_pickle=True) if os.path.exists(args.raw_proto) else None
    proto_err  = np.load(args.err_proto, allow_pickle=True)

    RAW = pd.DataFrame()
    ERR = pd.DataFrame()

    # RAW (BEFORE windows) using RAW prototypes/scaler frame
    raw_points = []
    if args.before_dir and os.path.isdir(args.before_dir) and proto_raw is not None:
        cent_raw = proto_raw["centroids"]
        files = list_any_files(args.before_dir, "*.csv")
        for f in files:
            try:
                df = read_ngafid_clean(f, args.columns, args.after_template_dir)
                X = df.values.astype(float)
                win, _ = sliding_windows(X, args.window, args.step)
                Xflat_scaled = transform_windows_for_scaler(win, scaler_raw, args.window, len(args.columns))
                if Xflat_scaled.size == 0: continue
                dists  = nearest_dists_to_proto(Xflat_scaled, cent_raw)
                scores = ocsvm.decision_function(Xflat_scaled).ravel()
                lab    = np.zeros_like(scores)
                raw_points.append(pd.DataFrame({"nearest_dist": dists, "ocsvm_score": scores, "label": lab}))
            except Exception as e:
                print(f"[sync][RAW] skip {os.path.basename(f)} → {e}")
        if raw_points:
            RAW = pd.concat(raw_points, axis=0, ignore_index=True)

    # ERROR (exact_data/anomaly windows) using ERR prototypes; OC-SVM expects RAW scaling
    cent_err = proto_err["centroids"]
    err_files = list_any_files(args.err_anom_dir, "window_*.csv")
    Xerr_for_proto, Xerr_for_ocsvm = [], []
    for f in err_files:
        try:
            M = read_error_window_matrix(f, args.columns)              # [W,F]
            Ms_err = scaler_err.transform(M.astype(float))             # frame for err prototypes
            Xerr_for_proto.append(Ms_err.reshape(-1))                  # (W*F,)

            Ms_raw = scaler_raw.transform(M.astype(float))             # frame for OCSVM
            Xerr_for_ocsvm.append(Ms_raw.reshape(-1))
        except Exception as e:
            print(f"[sync][ERR] skip {os.path.basename(f)} → {e}")

    if Xerr_for_proto:
        Xp = np.vstack(Xerr_for_proto)
        Xo = np.vstack(Xerr_for_ocsvm)
        dists_e  = nearest_dists_to_proto(Xp, cent_err)
        scores_e = ocsvm.decision_function(Xo).ravel()
        ERR = pd.DataFrame({"nearest_dist": dists_e, "ocsvm_score": scores_e, "label": np.ones_like(scores_e)})

    print("[sync] RAW rows:", 0 if RAW.empty else len(RAW))
    print("[sync] ERR rows:", 0 if ERR.empty else len(ERR))

    BOTH = pd.concat([RAW, ERR], axis=0, ignore_index=True)
    if BOTH.empty:
        print("[sync] No data to plot — check inputs.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if not RAW.empty:
        ax.scatter(RAW["nearest_dist"], RAW["ocsvm_score"], s=8, alpha=0.35, label="RAW")
    if not ERR.empty:
        ax.scatter(ERR["nearest_dist"], ERR["ocsvm_score"], s=8, alpha=0.35, label="ERROR")
    ax.set_xlabel("prototype nearest_dist")
    ax.set_ylabel("OC-SVM decision score")
    ax.set_title("OC-SVM score vs Prototype distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sync_scatter.png"), dpi=200)
    plt.close(fig)

    auc = None
    if not RAW.empty and not ERR.empty:
        y = BOTH["label"].values
        s = BOTH["nearest_dist"].values
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y, s)
    print(f"AUC (nearest_dist separates ERROR vs RAW): {auc if auc is not None else 'N/A'}")
    with open(os.path.join(outdir, "sync_metrics.json"), "w") as f:
        json.dump({
            "auc_nearest_dist_ERR_vs_RAW": None if auc is None else float(auc),
            "n_raw": int(len(RAW)),
            "n_err": int(len(ERR))
        }, f, indent=2)

# -------------------------
# narratives: ERROR groups by prototype
# -------------------------
def cmd_narratives(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    dfE = load_expl_csv(args.err_expl_csv)
    if "prototype_id" not in dfE.columns:
        raise SystemExit("ERROR explanations csv missing 'prototype_id'")

    files = list_any_files(args.err_anom_dir, "window_*.csv")
    ts_map: Dict[str, np.ndarray] = {}
    for f in files:
        base = os.path.basename(f)
        try:
            arr = read_error_window_matrix(f, args.columns)
            ts_map[base] = arr
        except Exception as e:
            print(f"[narratives] skip {base} → {e}")

    proto = np.load(args.err_proto, allow_pickle=True)
    centroids = proto["centroids"]

    for pid, g in dfE.groupby("prototype_id"):
        pid_int = int(pid)
        series = []
        for _, row in g.iterrows():
            base = row["file"]
            if base in ts_map:
                series.append(ts_map[base])
        if not series:
            continue

        S = np.stack(series, axis=0)   # [N,W,F]
        meanS = S.mean(axis=0)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        for fi, col in enumerate(args.columns):
            ax.plot(meanS[:, fi], label=col, linewidth=2)
        ax.set_title(f"ERROR narratives — prototype {pid_int}: mean sensor shapes (N={S.shape[0]})")
        ax.set_xlabel("t within window (0..29)")
        ax.set_ylabel("scaled value (relative)")
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"narrative_proto_{pid_int}_mean_shapes.png"), dpi=200)
        plt.close(fig)

        cent = centroids[pid_int]
        V = S.reshape(S.shape[0], -1)
        d = np.sqrt(((V - cent.reshape(1, -1))**2).sum(axis=1))
        idx = np.argsort(d)[:min(args.exemplars, len(d))]
        for rank, i in enumerate(idx, start=1):
            fig2, ax2 = plt.subplots(figsize=(8, 4.8))
            for fi, col in enumerate(args.columns):
                ax2.plot(S[i, :, fi], label=col)
            ax2.set_title(f"Prototype {pid_int} exemplar #{rank} (dist={d[i]:.2f})")
            ax2.set_xlabel("t within window")
            ax2.set_ylabel("scaled value")
            ax2.legend(ncol=4, fontsize=8)
            fig2.tight_layout()
            fig2.savefig(os.path.join(outdir, f"narrative_proto_{pid_int}_exemplar_{rank}.png"), dpi=200)
            plt.close(fig2)

# -------------------------
# timeline: overlay & shade anomalous windows
# -------------------------
def cmd_timeline(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    base = os.path.basename(args.flight_csv)
    df = read_ngafid_clean(args.flight_csv, args.columns, args.after_template_dir)

    dfR = load_expl_csv(args.raw_expl_csv)
    subR = dfR[dfR["file"] == base].copy()
    if subR.empty:
        raise SystemExit(f"No RAW explanations rows for file: {base}")

    thr = np.percentile(dfR["nearest_dist"], args.threshold_pct)
    if "start_idx" in subR.columns:
        starts = subR["start_idx"].values.tolist()
    else:
        starts = (subR["window_idx"].values * args.step).tolist()
    flagged = set([s for s, nd in zip(starts, subR["nearest_dist"].values) if nd >= thr])

    t = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.plot(t, df["AltMSL"].values, label="AltMSL")
    ax.plot(t, df["IAS"].values, label="IAS")
    ax.plot(t, df["E1 RPM"].values, label="E1 RPM")
    ax.set_title(f"Timeline with shaded anomalous windows (≥ P{args.threshold_pct} by nearest_dist)\n{base}")
    ax.set_xlabel("t"); ax.set_ylabel("value")
    for s in sorted(flagged):
        ax.axvspan(s, s + args.window - 1, color="red", alpha=0.12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "timeline_overlay.png"), dpi=200)
    plt.close(fig)
    print(f"✓ Saved timeline overlay → {os.path.join(outdir,'timeline_overlay.png')} (thr={thr:.3f}, flagged={len(flagged)} windows)")

# -------------------------
# calibrate: thresholds on RAW, eval on ERROR
# -------------------------
def cmd_calibrate(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    raw_df = load_expl_csv(args.raw_expl_csv)
    err_df = load_expl_csv(args.err_expl_csv)

    metrics = []
    def eval_metric(raw_vals, err_vals, name):
        raw_vals = np.asarray(raw_vals, dtype=float)
        err_vals = np.asarray(err_vals, dtype=float)
        thr95 = np.percentile(raw_vals, 95.0)
        thr99 = np.percentile(raw_vals, 99.0)
        for thr, lbl in [(thr95, "P95"), (thr99, "P99")]:
            y_true = np.concatenate([np.zeros_like(raw_vals, dtype=int),
                                     np.ones_like(err_vals, dtype=int)])
            y_pred = np.concatenate([(raw_vals >= thr).astype(int),
                                     (err_vals >= thr).astype(int)])
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            metrics.append({
                "signal": name, "threshold": lbl, "thr_value": float(thr),
                "precision": float(prec), "recall": float(rec), "f1": float(f1)
            })

    eval_metric(raw_df["nearest_dist"].values, err_df["nearest_dist"].values, "nearest_dist")

    viol_counts_cols = [c for c in raw_df.columns if c.startswith("viol_count_")]
    viol_sev_cols    = [c for c in raw_df.columns if c.startswith("viol_sev_")]
    if viol_counts_cols:
        eval_metric(raw_df[viol_counts_cols].sum(axis=1).values,
                    err_df[viol_counts_cols].sum(axis=1).values, "viol_count_sum")
    if viol_sev_cols:
        eval_metric(raw_df[viol_sev_cols].sum(axis=1).values,
                    err_df[viol_sev_cols].sum(axis=1).values, "viol_sev_sum")

    mt = pd.DataFrame(metrics)
    mt.to_csv(os.path.join(outdir, "calibration_metrics.csv"), index=False)
    print(mt.to_string(index=False))
    print(f"✓ Saved metrics → {os.path.join(outdir,'calibration_metrics.csv')}")

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_sync = sub.add_parser("sync", help="OC-SVM score vs prototype distance; AUC RAW vs ERROR.")
    ap_sync.add_argument("--ocsvm_model", default=OCSVM_MODEL_DEF)
    ap_sync.add_argument("--raw_scaler", default=RAW_SCALER_DEF)
    ap_sync.add_argument("--err_scaler", default=ERR_SCALER_DEF)
    ap_sync.add_argument("--raw_proto", default=RAW_PROTO_DEF)
    ap_sync.add_argument("--err_proto", default=ERR_PROTO_DEF)
    ap_sync.add_argument("--before_dir", default=BEFORE_DIR_DEF)
    ap_sync.add_argument("--err_anom_dir", default=ERR_ANOM_DIR_DEF)
    ap_sync.add_argument("--columns", nargs="+", default=COLS_DEF)
    ap_sync.add_argument("--window", type=int, default=W_DEF)
    ap_sync.add_argument("--step", type=int, default=STEP_DEF)
    ap_sync.add_argument("--outdir", default="outputs/sync")
    ap_sync.add_argument("--after_template_dir", default="dataset/after_examm2",
                         help="Folder with one AFTER file that has proper headers (used for headerless BEFORE files).")

    ap_n = sub.add_parser("narratives", help="ERROR per-prototype narratives (mean shapes + exemplars).")
    ap_n.add_argument("--err_expl_csv", default=ERR_EXPL_CSV_DEF)
    ap_n.add_argument("--err_anom_dir", default=ERR_ANOM_DIR_DEF)
    ap_n.add_argument("--err_proto", default=ERR_PROTO_DEF)
    ap_n.add_argument("--columns", nargs="+", default=COLS_DEF)
    ap_n.add_argument("--exemplars", type=int, default=3)
    ap_n.add_argument("--outdir", default="outputs/narratives")

    ap_t = sub.add_parser("timeline", help="Overlay AltMSL/IAS/RPM with shaded anomalous windows for one RAW flight.")
    ap_t.add_argument("--flight_csv", required=True)
    ap_t.add_argument("--raw_expl_csv", default=RAW_EXPL_CSV_DEF)
    ap_t.add_argument("--columns", nargs="+", default=COLS_DEF)
    ap_t.add_argument("--window", type=int, default=W_DEF)
    ap_t.add_argument("--step", type=int, default=STEP_DEF)
    ap_t.add_argument("--threshold_pct", type=float, default=95.0)
    ap_t.add_argument("--outdir", default="outputs/timeline")
    ap_t.add_argument("--after_template_dir", default="dataset/after_examm2")

    ap_c = sub.add_parser("calibrate", help="Set RAW thresholds and report precision/recall on ERROR.")
    ap_c.add_argument("--raw_expl_csv", default=RAW_EXPL_CSV_DEF)
    ap_c.add_argument("--err_expl_csv", default=ERR_EXPL_CSV_DEF)
    ap_c.add_argument("--outdir", default="outputs/calibration")

    args = ap.parse_args()
    if args.cmd == "sync":
        cmd_sync(args)
    elif args.cmd == "narratives":
        cmd_narratives(args)
    elif args.cmd == "timeline":
        cmd_timeline(args)
    elif args.cmd == "calibrate":
        cmd_calibrate(args)

if __name__ == "__main__":
    main()
