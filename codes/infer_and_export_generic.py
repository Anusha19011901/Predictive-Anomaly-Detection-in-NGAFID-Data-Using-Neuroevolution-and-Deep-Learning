import os, json, joblib, pandas as pd, numpy as np
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--pred_dir", required=True)
ap.add_argument("--truth_dir", required=True)
ap.add_argument("--tag", required=True)
args = ap.parse_args()

ROOT = Path(os.getcwd())
ART = ROOT / "artifacts"
PER_T = ART / "errors" / "per_timestep"
PER_W = ART / "errors" / "per_window"
PER_T.mkdir(parents=True, exist_ok=True)
PER_W.mkdir(parents=True, exist_ok=True)

feats = json.loads((ART/"features"/"selected_features.json").read_text())
scaler = joblib.load(ART/"scalers"/"standardizer.pkl")
if isinstance(scaler, dict) and "features" in scaler:
    feats = [f for f in scaler["features"] if f in feats]

pred_dir = Path(args.pred_dir)
truth_dir = Path(args.truth_dir)

def window_indices(T, win=30, step=5):
    idx=[]
    for end in range(win-1, T-1, step):
        tplus=end+1
        if tplus>=T: break
        idx.append((end-win+1,end,tplus))
    return idx

ocsvm_rows=[]; rank_rows=[]
for pf in sorted(pred_dir.glob("*_predictions.csv")):
    base = pf.name.replace("_predictions.csv","")
    tf = truth_dir / f"{base}.csv"
    if not tf.exists(): 
        print(f"skip: no truth for {pf.name}")
        continue
    P = pd.read_csv(pf, low_memory=False)
    T = pd.read_csv(tf, low_memory=False)
    T = T[[c for c in feats if c in T.columns]].apply(pd.to_numeric, errors="coerce")
    if set(feats).issubset(P.columns):
        Yh = P[feats].copy()
    else:
        suf = {c[:-5]:c for c in P.columns if c.endswith("_pred")}
        pre = {c[5:]:c for c in P.columns if c.startswith("pred_")}
        if set(feats).issubset(suf.keys()):
            Yh = P[[suf[f] for f in feats]].copy(); Yh.columns=feats
        elif set(feats).issubset(pre.keys()):
            Yh = P[[pre[f] for f in feats]].copy(); Yh.columns=feats
        else:
            Yh = P.iloc[:, :len(feats)].copy(); Yh.columns=feats
    Ytrue = T[feats].shift(-1)
    n = min(len(Yh), len(Ytrue))
    Yh = Yh.iloc[:n].reset_index(drop=True)
    Ytrue = Ytrue.iloc[:n].reset_index(drop=True)
    ERR = (Yh - Ytrue).replace([np.inf,-np.inf], np.nan)

    df_t = ERR.copy()
    df_t.insert(0,"t_end", np.arange(n)-1)
    df_t.insert(0,"window_idx", np.arange(n))
    df_t.insert(0,"subseq_id", base)
    df_t.insert(0,"flight_id", base)
    df_t.columns = ["flight_id","subseq_id","window_idx","t_end"] + [f"err_{c}" for c in feats]
    (PER_T / f"{base}_{args.tag}_per_timestep_errors.csv").write_text(df_t.to_csv(index=False))

    idxs = window_indices(T=len(T), win=30, step=5)
    for w,(s,e,tplus) in enumerate(idxs):
        if tplus>=len(ERR): break
        abs_e = ERR.iloc[tplus].abs()
        row={"flight_id":base,"subseq_id":base,"window_idx":w,"start_idx":s,"end_idx":e}
        for f in feats:
            v = abs_e.get(f, np.nan)
            row[f"mae_{f}"]= float(v) if pd.notna(v) else np.nan
        ocsvm_rows.append(row)
        denom = float(abs_e.sum())+1e-8
        contrib = abs_e/denom
        order = contrib.sort_values(ascending=False).index.tolist()[:5]
        rr={"flight_id":base,"subseq_id":base,"window_idx":w}
        for k,f in enumerate(order,1):
            rr[f"top{k}_feature"]=f
            rr[f"top{k}_contrib"]=float(contrib.loc[f])
        rank_rows.append(rr)

pd.DataFrame(ocsvm_rows).to_csv(PER_W/f"ocsvm_input_{args.tag}.csv", index=False)
pd.DataFrame(rank_rows).to_csv(PER_W/f"window_feature_rankings_{args.tag}.csv", index=False)
print("DONE", args.tag)
