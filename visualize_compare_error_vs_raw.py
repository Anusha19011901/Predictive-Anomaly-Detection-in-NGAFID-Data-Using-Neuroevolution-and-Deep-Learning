import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_expl(path):
    df = pd.read_csv(path)
    # minimal checks
    need = {"nearest_dist"}
    miss = need - set(df.columns)
    if miss: raise ValueError(f"{path} missing columns: {miss}")
    # optional columns
    if "file" not in df.columns: df["file"] = "UNKNOWN"
    if "window_idx" not in df.columns:
        # make a dense index per file
        df["window_idx"] = df.groupby("file").cumcount()
    return df.sort_values(["file","window_idx"]).reset_index(drop=True)

def get_sensor_cols(df):
    vc = [c for c in df.columns if c.startswith("viol_count_")]
    vs = [c for c in df.columns if c.startswith("viol_sev_")]
    sensors = [c.replace("viol_count_","") for c in vc]
    return sensors, vc, vs

def robust_limits(x, lo=2.0, hi=98.0):
    x = np.asarray(x)
    return np.percentile(x, lo), np.percentile(x, hi)

def heatmap_from_df(df, title, out, rowsize=120, robust=(2,98)):
    # collapse by window_idx irrespective of file (works for error-mode & raw)
    vals = df.sort_values("window_idx")["nearest_dist"].to_numpy().astype(float)
    if len(vals)==0: return
    vmin, vmax = robust_limits(vals, *robust)

    if rowsize and rowsize>0:
        W = len(vals); R = int(np.ceil(W/rowsize))
        M = np.full((R, rowsize), np.nan)
        for i in range(R):
            s=i*rowsize; e=min((i+1)*rowsize, W)
            M[i,0:(e-s)] = vals[s:e]
        ylabels = [f"{i*rowsize}–{min((i+1)*rowsize-1,W-1)}" for i in range(R)]
        figsize = (min(18, 2+rowsize*0.06), 1.5 + R*0.45)
    else:
        M = vals.reshape(1,-1)
        ylabels = ["all"]
        figsize = (min(18, 2+M.shape[1]*0.05), 3.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("nearest_dist")
    ax.set_yticks(np.arange(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("window_idx")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

def overlay_dists(raw_df, err_df, out):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(9,4))
    sns.histplot(raw_df["nearest_dist"], ax=ax, stat="density", bins=40, alpha=0.35, label="RAW")
    sns.histplot(err_df["nearest_dist"], ax=ax, stat="density", bins=40, alpha=0.35, label="ERROR")
    sns.kdeplot(raw_df["nearest_dist"], ax=ax, lw=2, label="RAW KDE")
    sns.kdeplot(err_df["nearest_dist"], ax=ax, lw=2, label="ERROR KDE")
    ax.set_xlabel("nearest_dist"); ax.set_ylabel("density")
    ax.set_title("Nearest distance distribution: RAW vs ERROR")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

def sensor_compare_bars(raw_df, err_df, out_prefix, top=10):
    sensors, vc_cols, vs_cols = get_sensor_cols(err_df if len(get_sensor_cols(err_df)[0])>0 else raw_df)
    if not sensors:
        print("No per-sensor violation columns found; skipping sensor bars.")
        return
    # align to common sensor set present in both
    sensors_raw, vc_raw, vs_raw = get_sensor_cols(raw_df)
    sensors_err, vc_err, vs_err = get_sensor_cols(err_df)
    common = sorted(list(set(sensors_raw) & set(sensors_err)))
    if not common:
        print("No common sensor columns; skipping sensor bars.")
        return

    vc_raw_map = {s:f"viol_count_{s}" for s in sensors_raw}
    vs_raw_map = {s:f"viol_sev_{s}"  for s in sensors_raw}
    vc_err_map = {s:f"viol_count_{s}" for s in sensors_err}
    vs_err_map = {s:f"viol_sev_{s}"  for s in sensors_err}

    mean_counts_raw = pd.Series({s: raw_df[vc_raw_map[s]].mean() for s in common})
    mean_counts_err = pd.Series({s: err_df[vc_err_map[s]].mean() for s in common})
    mean_sev_raw    = pd.Series({s: raw_df[vs_raw_map[s]].mean() for s in common})
    mean_sev_err    = pd.Series({s: err_df[vs_err_map[s]].mean() for s in common})

    # pick top by |Δ severity|
    delta_sev = (mean_sev_err - mean_sev_raw).abs().sort_values(ascending=False)
    top_s = list(delta_sev.head(top).index)

    # counts
    fig1, ax1 = plt.subplots(figsize=(max(8, 0.6*len(top_s)), 4))
    idx = np.arange(len(top_s))
    ax1.bar(idx-0.18, mean_counts_raw[top_s].values, width=0.36, label="RAW")
    ax1.bar(idx+0.18, mean_counts_err[top_s].values, width=0.36, label="ERROR")
    ax1.set_xticks(idx); ax1.set_xticklabels(top_s, rotation=30, ha="right")
    ax1.set_ylabel("mean violation count (0–30)")
    ax1.set_title("Per-sensor mean count (top by |Δ severity|)")
    ax1.legend()
    fig1.tight_layout(); fig1.savefig(out_prefix+"_counts.png", dpi=200); plt.close(fig1)

    # severity
    fig2, ax2 = plt.subplots(figsize=(max(8, 0.6*len(top_s)), 4))
    ax2.bar(idx-0.18, mean_sev_raw[top_s].values, width=0.36, label="RAW")
    ax2.bar(idx+0.18, mean_sev_err[top_s].values, width=0.36, label="ERROR")
    ax2.set_xticks(idx); ax2.set_xticklabels(top_s, rotation=30, ha="right")
    ax2.set_ylabel("mean violation severity (normalized sum)")
    ax2.set_title("Per-sensor mean severity (top by |Δ severity|)")
    ax2.legend()
    fig2.tight_layout(); fig2.savefig(out_prefix+"_severity.png", dpi=200); plt.close(fig2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_raw", required=True, help="prototype_explanations (RAW/non-error)")
    ap.add_argument("--csv_err", required=True, help="prototype_explanations (ERROR/error-mode)")
    ap.add_argument("--outdir", default="outputs/compare")
    ap.add_argument("--rowsize", type=int, default=120)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    raw_df = load_expl(args.csv_raw)
    err_df = load_expl(args.csv_err)

    # 1) heatmaps (two panels)
    tmp1 = os.path.join(args.outdir, "tmp_raw_heat.png")
    tmp2 = os.path.join(args.outdir, "tmp_err_heat.png")
    heatmap_from_df(raw_df, f"RAW: nearest_dist (rowsize={args.rowsize})", tmp1, rowsize=args.rowsize)
    heatmap_from_df(err_df, f"ERROR: nearest_dist (rowsize={args.rowsize})", tmp2, rowsize=args.rowsize)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    img1 = plt.imread(tmp1); axes[0].imshow(img1); axes[0].axis("off")
    img2 = plt.imread(tmp2); axes[1].imshow(img2); axes[1].axis("off")
    fig.suptitle("RAW vs ERROR — nearest_dist heatmaps", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(os.path.join(args.outdir, "side_by_side_heatmaps.png"), dpi=200)
    plt.close(fig)
    try: os.remove(tmp1); os.remove(tmp2)
    except: pass

    # 2) distributions overlay
    overlay_dists(raw_df, err_df, os.path.join(args.outdir, "nearest_dist_distribution.png"))

    # 3) per-sensor comparison (mean count & severity)
    sensor_compare_bars(raw_df, err_df, os.path.join(args.outdir, "sensor_compare"), top=10)

if __name__ == "__main__":
    main()
