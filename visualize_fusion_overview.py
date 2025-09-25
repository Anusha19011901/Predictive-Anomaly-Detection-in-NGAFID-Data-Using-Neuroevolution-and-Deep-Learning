#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FUSION_SCORES = "outputs/fusion/before_scores.csv"
FUSION_TOPK   = "outputs/fusion/before_topk_contributors.csv"
BEFORE_DIR    = "dataset/before_examm2"
OUT_DIR       = "outputs/vis_fusion"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 30
STEP_SIZE   = 5

def pick_features_auto(df: pd.DataFrame, max_k: int = 6) -> list[str]:
    # drop clearly non-sensor cols if present
    drop_like = ("flight_id","subseq_id","window_idx")
    numeric = [c for c in df.columns
               if c not in drop_like and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors='coerce'))]
    if not numeric:
        raise ValueError("No numeric columns found in flight CSV.")
    # rank by variance to get informative curves
    vars_ = df[numeric].var(numeric_only=True).sort_values(ascending=False)
    return list(vars_.index[:max_k])

def window_idx_to_span(widx: int, window: int = WINDOW_SIZE, step: int = STEP_SIZE) -> tuple[int,int]:
    start = widx * step
    end   = start + window - 1
    return start, end

def make_plot(flight_csv: str,
              scores_df: pd.DataFrame,
              contrib_df: pd.DataFrame,
              features: list[str] | None,
              normalize: bool = False):
    name = Path(flight_csv).stem
    raw = pd.read_csv(flight_csv, low_memory=False).reset_index(drop=True)

    # Choose features
    if features:
        feats = [c for c in features if c in raw.columns]
        if not feats:
            raise ValueError(f"None of the requested features exist in {name}.")
    else:
        feats = pick_features_auto(raw, max_k=6)

    # Build anomaly mask over the time axis using window_idx & STEP/WINDOW
    wrows = scores_df[scores_df["flight_id"] == name].copy()
    if wrows.empty:
        raise ValueError(f"No fusion rows found for flight_id={name} in {FUSION_SCORES}")
    # Precompute spans
    spans = [window_idx_to_span(int(wi)) for wi in wrows["window_idx"]]
    flags = wrows["anomaly_flag"].to_numpy().astype(int)
    scores = wrows["ocsvm_score"].to_numpy().astype(float)

    # Create figure
    n_panels = len(feats) + 1  # features + score strip
    fig_h = max(2.5 * len(feats) + 2, 6)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, fig_h), sharex=True,
                             gridspec_kw={"height_ratios": [1.0]*len(feats) + [0.6]})
    if n_panels == 2:
        axes = np.array(axes)

    T = len(raw)
    x = np.arange(T)

    # Optional z-normalization (per feature) to overlay scales nicely
    normed = raw.copy()
    ylabels = []
    if normalize:
        for c in feats:
            v = pd.to_numeric(normed[c], errors='coerce')
            mu = np.nanmean(v); sd = np.nanstd(v) or 1.0
            normed[c] = (v - mu) / sd
            ylabels.append(f"{c} (z)")
    else:
        ylabels = feats

    # Plot each feature with anomaly shading
    for i, c in enumerate(feats):
        ax = axes[i]
        y = pd.to_numeric(normed[c], errors='coerce').to_numpy()
        ax.plot(x, y, linewidth=1.2, label=c)
        # Shade anomaly windows
        for (s,e), f in zip(spans, flags):
            if f == 1:
                ax.axvspan(s, min(e, T-1), alpha=0.22, color="tab:red")
        ax.set_ylabel(ylabels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    # Score strip
    ax = axes[-1]
    # Place score at each window center for reference
    centers = [min((s+e)//2, T-1) for (s,e) in spans]
    ax.plot(centers, scores, marker='o', linewidth=1.0)
    ax.axhline(0.0, linestyle='--', linewidth=1.0)
    # Mark anomalies as red dots
    for (cx, f) in zip(centers, flags):
        if f == 1:
            ax.plot(cx, scores[centers.index(cx)], marker='o', markersize=5, color='tab:red')
    ax.set_ylabel("OC-SVM score")
    ax.set_xlabel("t (row index)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Flight: {name}  |  shaded = anomaly windows ({WINDOW_SIZE}/{STEP_SIZE})")

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{name}_fusion_overview.png")
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Aggregate top contributors over anomalies for a quick textual summary
    contrib_rows = contrib_df[contrib_df["flight_id"] == name].copy()
    if not contrib_rows.empty:
        cols = [c for c in contrib_rows.columns if c.startswith("top")]
        # count feature hits weighted by |z|
        contrib_stack = []
        for _, r in contrib_rows.iterrows():
            for j in range(1, 6):
                fcol = f"top{j}_feature"; zcol = f"top{j}_z"
                if fcol in r and zcol in r and pd.notna(r[fcol]) and pd.notna(r[zcol]):
                    contrib_stack.append((r[fcol], abs(float(r[zcol]))))
        if contrib_stack:
            agg = {}
            for f, z in contrib_stack:
                agg[f] = agg.get(f, 0.0) + z
            top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\nTop contributors for {name}:")
            for f, w in top:
                print(f"  - {f}: {w:.3f}")
        else:
            print(f"\nNo contributor rows found for {name}.")
    else:
        print(f"\nNo contributor file rows for {name}.")

    print(f"✅ Saved: {out_png}")

def main():
    ap = argparse.ArgumentParser(description="Visualize full-flight with anomaly shading from fusion OC-SVM.")
    ap.add_argument("--scores_csv", default=FUSION_SCORES)
    ap.add_argument("--topk_csv",   default=FUSION_TOPK)
    ap.add_argument("--before_dir", default=BEFORE_DIR)
    ap.add_argument("--flight_id",  default=None,
                    help="Filename stem in before_examm2 to visualize (default: first one found).")
    ap.add_argument("--features",   default=None,
                    help="Comma-separated list to force features (must exist in flight).")
    ap.add_argument("--normalize",  action="store_true", help="Z-normalize selected features for display.")
    args = ap.parse_args()

    scores = pd.read_csv(args.scores_csv)
    contrib = pd.read_csv(args.topk_csv)

    # Pick a flight
    if args.flight_id is None:
        # use the first flight_id present in scores
        first = scores["flight_id"].iloc[0]
        flight_id = first
    else:
        flight_id = args.flight_id

    flight_csv = os.path.join(args.before_dir, f"{flight_id}.csv")
    if not os.path.exists(flight_csv):
        # try to locate by stem
        candidates = list(Path(args.before_dir).glob(f"{flight_id}*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Could not find BEFORE CSV for {flight_id} under {args.before_dir}")
        flight_csv = str(candidates[0])

    feats = None
    if args.features:
        feats = [x.strip() for x in args.features.split(",") if x.strip()]

    make_plot(flight_csv, scores, contrib, feats, normalize=args.normalize)

if __name__ == "__main__":
    main()
