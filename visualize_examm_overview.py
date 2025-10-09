#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCORES_CSV   = "outputs/ocsvm_examm_only/before_scores.csv"
TOPK_CSV     = "outputs/ocsvm_examm_only/before_topk_contributors.csv"
ZSCORES_PQ   = "outputs/ocsvm_examm_only/before_zscores.parquet"   # optional
BEFORE_DIR   = "dataset/before_examm2"
OUT_DIR      = "outputs/vis_examm"
os.makedirs(OUT_DIR, exist_ok=True)

def pick_features_auto(df: pd.DataFrame, max_k: int = 6) -> list[str]:
    drop_like = ("flight_id","subseq_id","window_idx")
    numeric = [c for c in df.columns
               if c not in drop_like and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors='coerce'))]
    if not numeric:
        raise ValueError("No numeric columns found in flight CSV.")
    vars_ = df[numeric].var(numeric_only=True).sort_values(ascending=False)
    return list(vars_.index[:max_k])

def window_idx_to_span(widx: int, window: int, step: int) -> tuple[int,int]:
    start = widx * step
    end   = start + window - 1
    return start, end

def maybe_load_zl2(zscores_path: str, flight_id: str) -> pd.DataFrame | None:
    if not os.path.exists(zscores_path):
        return None
    try:
        z = pd.read_parquet(zscores_path)
    except Exception:
        return None
    need = {"flight_id","subseq_id","window_idx"}
    if not need.issubset(z.columns):
        return None
    zf = z[z["flight_id"] == flight_id].copy()
    if zf.empty:
        return None
    mae_cols = [c for c in zf.columns if str(c).startswith("mae_")]
    if not mae_cols:
        return None
    zf["examm_z_l2"] = np.sqrt((zf[mae_cols].values**2).sum(axis=1))
    return zf[["subseq_id","window_idx","examm_z_l2"]]

def make_plot(
    flight_csv: str,
    scores_df: pd.DataFrame,
    contrib_df: pd.DataFrame,
    features: list[str] | None,
    normalize: bool,
    zscores_path: str | None,
    window_size: int,
    step_size: int
):
    name = Path(flight_csv).stem
    raw = pd.read_csv(flight_csv, low_memory=False).reset_index(drop=True)

    # Choose features
    if features:
        feats = [c for c in features if c in raw.columns]
        if not feats:
            raise ValueError(f"None of the requested features exist in {name}.")
    else:
        feats = pick_features_auto(raw, max_k=6)

    # Build anomaly mask & score series from EXAMM-only OC-SVM outputs
    wrows = scores_df[scores_df["flight_id"] == name].copy()
    if wrows.empty:
        raise ValueError(f"No rows found for flight_id={name} in {SCORES_CSV}")
    spans  = [window_idx_to_span(int(wi), window_size, step_size) for wi in wrows["window_idx"]]
    flags  = wrows["anomaly_flag"].to_numpy().astype(int)
    scores = wrows["ocsvm_score"].to_numpy().astype(float)
    subseqs = wrows["subseq_id"].tolist()
    win_idx = wrows["window_idx"].astype(int).tolist()

    # Optional EXAMM z-L2 (requires --save_before_zscores in the scorer)
    zl2 = None
    if zscores_path:
        zl2 = maybe_load_zl2(zscores_path, name)
        if zl2 is not None:
            zl2 = zl2.sort_values(["subseq_id","window_idx"])

    # ---------- FIGURE LAYOUT (tighter + cleaner) ----------
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    extra_panels = 1 + (1 if zl2 is not None else 0)    # score + optional z-L2
    n_panels = len(feats) + extra_panels

    # adaptive height; compact inter-panel spacing
    fig_h = max(1.8 * len(feats) + (1.2 * extra_panels) + 0.5, 7)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(12.5, fig_h), sharex=True,
        gridspec_kw={"height_ratios": [1.0]*len(feats) + [0.6]*extra_panels}
    )
    if n_panels == 2:
        axes = np.array(axes)
    # reduce whitespace between panels + outside margins (room for legend on right)
    plt.subplots_adjust(left=0.07, right=0.86, top=0.96, bottom=0.08, hspace=0.08)

    # axes helpers
    def beautify_axis(ax):
        for s in ("top","right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, alpha=0.15, linewidth=0.8)
        ax.margins(x=0)  # trims inner whitespace left/right

    T = len(raw)
    x = np.arange(T)

    # Optional z-normalization for display
    disp = raw.copy()
    ylabels = []
    if normalize:
        for c in feats:
            v = pd.to_numeric(disp[c], errors='coerce')
            mu = np.nanmean(v); sd = np.nanstd(v) or 1.0
            disp[c] = (v - mu) / sd
            ylabels.append(f"{c} (z)")
    else:
        ylabels = feats

    # Plot each raw feature with anomaly shading
    for i, c in enumerate(feats):
        ax = axes[i]
        y = pd.to_numeric(disp[c], errors='coerce').to_numpy()
        ax.plot(x, y, linewidth=1.2, label=c, zorder=2)
        for (s,e), f in zip(spans, flags):
            if f == 1:
                ax.axvspan(s, min(e, T-1), alpha=0.16, color="#1f77b4", zorder=1)
        ax.set_ylabel(ylabels[i])
        # tidier ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        beautify_axis(ax)
        ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.85)

    # Build centers for per-window series
    centers = [min((s+e)//2, T-1) for (s,e) in spans]

    # Optional EXAMM z-L2 strip
    panel_idx = len(feats)
    if zl2 is not None:
        ax = axes[panel_idx]
        zl2_sorted = zl2.sort_values(["subseq_id","window_idx"])
        z_centers = centers[:len(zl2_sorted)]
        ax.plot(z_centers, zl2_sorted["examm_z_l2"].to_numpy(), linewidth=1.0, zorder=2)
        for (cx, f) in zip(z_centers, flags[:len(z_centers)]):
            if f == 1:
                ax.axvline(cx, ymin=0.0, ymax=1.0, alpha=0.06, color="k", zorder=1)
        ax.set_ylabel("EXAMM z-L2")
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        beautify_axis(ax)
        panel_idx += 1

    # --- Color anomalous dots by Top-1 EXAMM contributor + compact legend outside ---
    contrib_rows = contrib_df[contrib_df["flight_id"] == name].copy()
    top1 = {}
    if not contrib_rows.empty:
        for _, r in contrib_rows.iterrows():
            k = (r.get("subseq_id"), int(r.get("window_idx", -1)))
            ftr = r.get("top1_feature")
            if pd.notna(k[0]) and k[1] >= 0 and isinstance(ftr, str):
                top1[k] = ftr

    import itertools
    feature_names = sorted(set(top1.values()))
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', []))
    feat2color = {f: next(color_cycle) for f in feature_names}

    # OC-SVM score strip
    ax = axes[panel_idx]
    ax.plot(centers, scores, linewidth=1.0, color="#1f77b4", zorder=1)
    ax.axhline(0.0, linestyle="--", linewidth=0.9, color="gray")
    for cx, sc, f, ss, wi in zip(centers, scores, flags, subseqs, win_idx):
        if f == 1:
            key = (ss, wi)
            feat = top1.get(key, None)
            ax.scatter(cx, sc, s=18,
                       color=feat2color.get(feat, "#d62728"),
                       zorder=3, edgecolors="white", linewidths=0.3)
    ax.set_ylabel("OC-SVM score\n(higher = more normal)")
    ax.set_xlabel("t (row index)")
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    beautify_axis(ax)
    ax.set_title(f"EXAMM-only OC-SVM — {name}  |  shaded = anomaly windows ({window_size}/{step_size})")

    # compact legend outside on the right
    if feature_names:
        handles = [plt.Line2D([0],[0], marker='o', linestyle='None',
                   label=feat, markerfacecolor=feat2color[feat], markeredgecolor="white",
                   markeredgewidth=0.3) for feat in feature_names]
        ncols = min(2, max(1, len(handles)//10 + (len(handles)%10>0)))
        leg = ax.legend(handles=handles[:40],  # safety cap if many features
                        title="Top-1 contributor (anomalies)",
                        loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        borderaxespad=0., frameon=True, fancybox=True, framealpha=0.92,
                        ncol=ncols)
        leg._legend_box.align = "left"

    # Save with explicit white background + tight bbox
    out_png = os.path.join(OUT_DIR, f"{name}_examm_overview.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Text summary (unchanged)
    if not contrib_rows.empty:
        stack = []
        for _, r in contrib_rows.iterrows():
            for j in range(1, 6):
                fcol = f"top{j}_feature"; zcol = f"top{j}_z"
                if fcol in r and zcol in r and pd.notna(r[fcol]) and pd.notna(r[zcol]):
                    stack.append((r[fcol], abs(float(r[zcol]))))
        if stack:
            agg = {}
            for f, z in stack:
                agg[f] = agg.get(f, 0.0) + z
            top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\nTop contributors (sum |z| across anomalies) for {name}:")
            for f, w in top:
                print(f"  - {f}: {w:.3f}")
    print(f"✅ Saved: {out_png}")

def main():
    ap = argparse.ArgumentParser(description="Visualize EXAMM-only OC-SVM with anomaly shading over raw signals.")
    ap.add_argument("--scores_csv", default=SCORES_CSV)
    ap.add_argument("--topk_csv",   default=TOPK_CSV)
    ap.add_argument("--before_dir", default=BEFORE_DIR)
    ap.add_argument("--zscores_parquet", default=ZSCORES_PQ, help="Optional parquet produced with --save_before_zscores")
    ap.add_argument("--flight_id",  default=None, help="Filename stem under before_examm2 to visualize.")
    ap.add_argument("--features",   default=None, help="Comma-separated list of raw columns to plot.")
    ap.add_argument("--normalize",  action="store_true", help="Z-normalize plotted raw features for display.")
    ap.add_argument("--window_size", type=int, default=30)
    ap.add_argument("--step_size",   type=int, default=5)
    args = ap.parse_args()

    scores = pd.read_csv(args.scores_csv)
    contrib = pd.read_csv(args.topk_csv)

    # Pick a flight (or first available)
    if args.flight_id is None:
        flight_id = scores["flight_id"].iloc[0]
    else:
        flight_id = args.flight_id

    # locate raw BEFORE csv
    flight_csv = os.path.join(args.before_dir, f"{flight_id}.csv")
    if not os.path.exists(flight_csv):
        candidates = list(Path(args.before_dir).glob(f"{flight_id}*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Could not find BEFORE CSV for {flight_id} in {args.before_dir}")
        flight_csv = str(candidates[0])

    feats = None
    if args.features:
        feats = [x.strip() for x in args.features.split(",") if x.strip()]

    make_plot(
        flight_csv, scores, contrib, feats,
        normalize=args.normalize,
        zscores_path=args.zscores_parquet,
        window_size=args.window_size,
        step_size=args.step_size
    )

if __name__ == "__main__":
    main()
