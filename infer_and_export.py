import os, json, joblib, pandas as pd, numpy as np
from pathlib import Path

NGAFID_ROOT = Path(os.getcwd())
ART_DIR = NGAFID_ROOT / "artifacts"
PRED_DIR = ART_DIR / "examm_run" / "preds"
BEFORE_DIR = NGAFID_ROOT / "dataset" / "before_examm2"

PER_T = ART_DIR / "errors" / "per_timestep"
PER_W = ART_DIR / "errors" / "per_window"
PER_T.mkdir(parents=True, exist_ok=True)
PER_W.mkdir(parents=True, exist_ok=True)

feats = json.loads((ART_DIR/"features"/"selected_features.json").read_text())
scaler = joblib.load(ART_DIR/"scalers"/"standardizer.pkl")
f_order = scaler["features"] if "features" in scaler else feats
feats = [f for f in f_order if f in feats]

def load_preds(pred_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def load_before_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[[c for c in feats if c in df.columns]]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def window_indices(T, win=30, step=5):
    idx = []
    for end in range(win-1, T-1, step):
        tplus = end+1
        if tplus >= T: break
        idx.append((end-win+1, end, tplus))
    return idx

def main():
    ocsvm_rows = []
    rank_rows = []
    pred_files = sorted(PRED_DIR.glob("*_predictions.csv"))
    if not pred_files:
        raise SystemExit(f"No prediction files in {PRED_DIR}")

    for pf in pred_files:
        base = pf.name.replace("_predictions.csv", "")
        bef_csv = BEFORE_DIR / (base + ".csv")
        if not bef_csv.exists():
            print(f"skip: no matching before csv for {pf.name}")
            continue

        df_pred = load_preds(pf)
        df_true = load_before_csv(bef_csv)

        # Try to identify prediction columns:
        # Case A: columns named exactly as features (pred-only)
        # Case B: columns like "<feat>_pred" or "pred_<feat>"
        pred_cols = None
        if set(feats).issubset(df_pred.columns):
            pred_cols = feats
            Yhat = df_pred[feats].copy()
        else:
            suffix_cols = [c for c in df_pred.columns if c.endswith("_pred")]
            prefix_cols = [c for c in df_pred.columns if c.startswith("pred_")]
            map_suf = {c[:-5]: c for c in suffix_cols}
            map_pre = {c[5:]: c for c in prefix_cols}
            if set(feats).issubset(map_suf.keys()):
                pred_cols = [map_suf[f] for f in feats]
                Yhat = df_pred[pred_cols].copy()
                Yhat.columns = feats
            elif set(feats).issubset(map_pre.keys()):
                pred_cols = [map_pre[f] for f in feats]
                Yhat = df_pred[pred_cols].copy()
                Yhat.columns = feats
            else:
                # Fallback: assume df_pred has only predictions, same order as feats
                Yhat = df_pred.iloc[:, :len(feats)].copy()
                Yhat.columns = feats

        # Align with true t+1
        Ytrue = df_true[feats].shift(-1)
        N = min(len(Yhat), len(Ytrue))
        Yhat = Yhat.iloc[:N].reset_index(drop=True)
        Ytrue = Ytrue.iloc[:N].reset_index(drop=True)

        # Errors at t+1
        ERR = Yhat - Ytrue
        ERR = ERR.replace([np.inf, -np.inf], np.nan)

        # Per-timestep export (one file per subseq)
        df_t = ERR.copy()
        df_t.insert(0, "t_end", np.arange(N)-1)  # placeholder; recompute with windowing
        df_t.insert(0, "window_idx", np.arange(N))
        df_t.insert(0, "subseq_id", base)
        df_t.insert(0, "flight_id", base)
        df_t.columns = ["flight_id","subseq_id","window_idx","t_end"] + [f"err_{c}" for c in feats]
        df_t.to_csv(PER_T / f"{base}_per_timestep_errors.csv", index=False)

        # Per-window aggregation using OC-SVM windowing (30 len, step 5)
        idxs = window_indices(T=len(df_true), win=30, step=5)
        for w, (s,e,tplus) in enumerate(idxs):
            if tplus >= len(ERR): break
            row = {
                "flight_id": base,
                "subseq_id": base,
                "window_idx": w,
                "start_idx": s,
                "end_idx": e
            }
            abs_err = ERR.iloc[tplus].abs()
            for f in feats:
                v = abs_err.get(f, np.nan)
                row[f"mae_{f}"] = float(v) if pd.notna(v) else np.nan
            ocsvm_rows.append(row)

            denom = float(abs_err.sum()) + 1e-8
            contrib = abs_err / denom
            order = contrib.sort_values(ascending=False).index.tolist()[:5]
            rr = {"flight_id": base, "subseq_id": base, "window_idx": w}
            for k, f in enumerate(order, 1):
                rr[f"top{k}_feature"] = f
                rr[f"top{k}_contrib"] = float(contrib.loc[f])
            rank_rows.append(rr)

    if ocsvm_rows:
        pd.DataFrame(ocsvm_rows).to_csv(PER_W/"ocsvm_input.csv", index=False)
    if rank_rows:
        pd.DataFrame(rank_rows).to_csv(PER_W/"window_feature_rankings.csv", index=False)
    print("DONE")

if __name__ == "__main__":
    main()
