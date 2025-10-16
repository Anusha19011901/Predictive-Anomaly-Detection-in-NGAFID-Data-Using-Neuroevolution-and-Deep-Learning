#!/usr/bin/env python3
# make_postcards.py — small PNGs per flagged window

import os, argparse, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

NEEDED = ["AltMSL","IAS","E1 RPM"]
def read_flight(path):
    import pandas as pd, re
    df = pd.read_csv(path, skiprows=2)
    df.columns = df.columns.str.strip()
    if all([_looks_num(c) for c in df.columns]):
        raw = pd.read_csv(path, header=None)
        from demo_portfolio import NGAFID_HEADER_TEMPLATE  # reuse if in same folder; else paste here
        m = raw.shape[1]
        cols = (NGAFID_HEADER_TEMPLATE + [f"Dummy_{i}" for i in range(max(0, m-len(NGAFID_HEADER_TEMPLATE)))])[:m]
        raw.columns = cols
        return raw
    return df
def _looks_num(s):
    try: float(str(s)); return True
    except: return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before_dir", default="dataset/before_examm2")
    ap.add_argument("--raw_expl_csv", default="outputs/prototype_explanations_before.csv")
    ap.add_argument("--outdir", default="outputs/postcards")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--threshold_pct", type=float, default=95.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    expl = pd.read_csv(args.raw_expl_csv)
    if "window_idx" not in expl.columns:
        expl["window_idx"] = expl.groupby("file").cumcount()
    thr = np.percentile(expl["nearest_dist"].values, args.threshold_pct)

    for name, sub in expl.groupby("file"):
        sub = sub.copy()
        sub["flag"] = sub["nearest_dist"] >= thr
        if sub["flag"].sum() == 0:
            continue
        flight_path = os.path.join(args.before_dir, name)
        if not os.path.exists(flight_path): 
            continue
        df = read_flight(flight_path)
        for _, r in sub[sub["flag"]==1].iterrows():
            s = int(r["start_idx"]) if "start_idx" in sub.columns else int(r["window_idx"])*args.step
            e = s + args.window
            clip = df.iloc[s:e].copy()
            # best-effort column names
            for col in NEEDED:
                if col not in clip.columns: 
                    continue
            fig = plt.figure(figsize=(6,3.6), constrained_layout=True)
            gs = fig.add_gridspec(2,2, height_ratios=[2,1])
            ax1 = fig.add_subplot(gs[0,:])
            # timeline
            for col in ["AltMSL","IAS","E1 RPM"]:
                if col in clip.columns:
                    ax1.plot(clip[col].values, label=col, linewidth=1.6)
            ax1.set_title(f"{name} — window {int(r['window_idx'])} (nearest_dist={r['nearest_dist']:.1f})")
            ax1.legend(ncol=3, fontsize=8)
            ax1.set_xticks([])

            # bar: violations
            ax2 = fig.add_subplot(gs[1,:])
            sev_cols = [c for c in r.index if c.startswith("viol_sev_")]
            cnt_cols = [c for c in r.index if c.startswith("viol_count_")]
            if sev_cols:
                items = [(c.replace("viol_sev_",""), float(r[c])) for c in sev_cols]
            elif cnt_cols:
                items = [(c.replace("viol_count_",""), float(r[c])) for c in cnt_cols]
            else:
                items=[]
            items.sort(key=lambda x: x[1], reverse=True)
            names, vals = zip(*items) if items else ([],[])
            ax2.bar(names, vals)
            ax2.set_ylabel("violation")
            ax2.set_xticklabels(names, rotation=30, ha="right")
            out = os.path.join(args.outdir, f"window_{Path(name).stem}_{int(r['window_idx'])}.png")
            fig.savefig(out, dpi=180)
            plt.close(fig)
            print("saved", out)

if __name__ == "__main__":
    main()
