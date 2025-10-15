#!/usr/bin/env python3
"""
EXAMM-only OC-SVM visualization with selective per-feature shading.

- Shading is drawn in a panel only when that panel's feature is a Top-K contributor
  for that specific (flagged) window. Default K=1 (Top-1 only).
- Panels are auto-selected and ordered by how often the feature is Top-1 across
  anomalous windows (descending). Control with --top_features and --min_top1.
- Colors are consistent: legend colors match panel shading.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCORES_CSV = "outputs/ocsvm_examm_only/before_scores.csv"
TOPK_CSV   = "outputs/ocsvm_examm_only/before_topk_contributors.csv"
ZSCORES_PQ = "outputs/ocsvm_examm_only/before_zscores.parquet"   # optional (unused panel here)
BEFORE_DIR = "dataset/before_examm2"

# Save new outputs into a separate folder to avoid overwriting older figures
OUT_DIR = "outputs/vis_examm_top1"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- helpers ----------
def pick_features_auto(df: pd.DataFrame, max_k: int = 6) -> list[str]:
    drop_like = ("flight_id", "subseq_id", "window_idx")
    numeric = [
        c for c in df.columns
        if c not in drop_like and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors='coerce'))
    ]
    if not numeric:
        raise ValueError("No numeric columns found in flight CSV.")
    vars_ = df[numeric].var(numeric_only=True).sort_values(ascending=False)
    return list(vars_.index[:max_k])

def window_idx_to_span(widx: int, window: int, step: int) -> tuple[int, int]:
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
    step_size: int,
    top_features: int = 10,
    select_by: str = "top1",
    shade_topk: int = 1,
    min_top1: int = 1,
    print_summary: bool = False,
    limit_legend: int = 48,
):
    import re, itertools
    from collections import Counter

    def canon(s: str) -> str:
        # canonicalize: mae_AltB ↔ AltB ↔ alt_b ↔ ALT B
        return re.sub(r'[\s_\-/]+', '', str(s).strip().lower())

    def map_mae_to_raw(mae_name: str, raw_cols: list[str]) -> str | None:
        """Return the best matching raw column for a mae_* feature name."""
        base = str(mae_name).replace("mae_", "", 1)
        c_base = canon(base)
        # exact canonical match
        canon_map = {canon(c): c for c in raw_cols}
        if c_base in canon_map:
            return canon_map[c_base]
        # heuristic: endswith/startswith (covers e.g., VSpd vs VSpdG)
        for c in raw_cols:
            cb = canon(c)
            if cb.endswith(c_base) or c_base.endswith(cb):
                return c
        # a few known aliases (extend as needed)
        aliases = {
            "vspd":   ["vspdg", "vvi", "verticalspeed"],
            "gndspd": ["groundspeed", "gs"],
            "latac":  ["laac", "latacc", "lataccel"],
            "longac": ["longacc", "longaccel"],
        }
        for key, vals in aliases.items():
            if c_base == key or c_base in vals:
                for c in raw_cols:
                    if canon(c) in [key] + vals:
                        return c
        return None

    name = Path(flight_csv).stem
    raw = pd.read_csv(flight_csv, low_memory=False).reset_index(drop=True)

    # --- OC-SVM outputs for this flight
    wrows = scores_df[scores_df["flight_id"] == name].copy()
    if wrows.empty:
        raise ValueError(f"No rows found for flight_id={name} in {SCORES_CSV}")
    spans   = [window_idx_to_span(int(wi), window_size, step_size) for wi in wrows["window_idx"]]
    flags   = wrows["anomaly_flag"].to_numpy().astype(int)
    scores  = wrows["ocsvm_score"].to_numpy().astype(float)
    subseqs = wrows["subseq_id"].tolist()
    win_idx = wrows["window_idx"].astype(int).tolist()

    # --- build Top-K contributor lookup per window
    contrib_rows = contrib_df[contrib_df["flight_id"] == name].copy()
    topk_map: dict[tuple, list[str]] = {}   # (subseq_id, window_idx) -> [feat1, feat2, ...]
    if not contrib_rows.empty:
        for _, r in contrib_rows.iterrows():
            k = (r.get("subseq_id"), int(r.get("window_idx", -1)))
            feats = []
            j = 1
            while f"top{j}_feature" in r:
                ftr = r.get(f"top{j}_feature")
                if isinstance(ftr, str) and pd.notna(ftr):
                    feats.append(ftr)
                j += 1
            if k[0] is not None and k[1] >= 0 and feats:
                topk_map[k] = feats

    # --- color map: consistent across panels & score dots
    uniq_feats = sorted({f for L in topk_map.values() for f in L})
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', []))
    feat2color = {f: next(color_cycle) for f in uniq_feats}

    # --- which raw panels to show (ordered by Top-1 frequency on anomalies)
    raw_cols = list(raw.columns)
    if features:
        feats = [c for c in features if c in raw_cols]
        if not feats:
            raise ValueError(f"None of the requested features exist in {name}.")
        top1_count_for_panel = {f: 0 for f in feats}
    else:
        if select_by == "top1" and topk_map:
            # Build keys aligned to rows; then keep only anomalous ones
            keys = list(zip(subseqs, win_idx))
            anom_keys = { keys[i] for i, f in enumerate(flags) if f == 1 }

            # Count Top-1 only on anomalous windows
            top1_list = [L[0] for k, L in topk_map.items() if L and k in anom_keys]
            counts = Counter(top1_list)

            ordered = []
            raw2mae = {}
            for mae_feat, cnt in counts.most_common():
                if cnt < min_top1:
                    continue
                raw_match = map_mae_to_raw(mae_feat, raw_cols)
                if raw_match is not None and raw_match not in [f for f, _, _ in ordered]:
                    ordered.append((raw_match, cnt, mae_feat))
                    raw2mae[raw_match] = mae_feat
                if len(ordered) >= top_features:
                    break

            if not ordered:
                feats = pick_features_auto(raw, max_k=top_features)
                top1_count_for_panel = {f: 0 for f in feats}
            else:
                feats = [f for f, _, _ in ordered]
                top1_count_for_panel = {f: cnt for f, cnt, _ in ordered}
        else:
            feats = pick_features_auto(raw, max_k=top_features)
            top1_count_for_panel = {f: 0 for f in feats}

    # order panels by Top-1 count desc (then name)
    feats.sort(key=lambda f: (-top1_count_for_panel.get(f, 0), f))

    # ---------- FIGURE LAYOUT ----------
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
    n_extra = 1  # score panel
    fig_h = max(1.7 * len(feats) + 1.2 * n_extra + 0.5, 7)
    fig, axes = plt.subplots(
        len(feats) + n_extra, 1, figsize=(12.5, fig_h), sharex=True,
        gridspec_kw={"height_ratios": [1.0]*len(feats) + [0.6]*n_extra}
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    plt.subplots_adjust(left=0.07, right=0.86, top=0.96, bottom=0.08, hspace=0.08)

    def beautify_axis(ax):
        for s in ("top","right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, alpha=0.15, linewidth=0.8)
        ax.margins(x=0)

    T = len(raw)
    x = np.arange(T)

    # Optional z-normalization for display
    disp = raw.copy()
    if normalize:
        for c in feats:
            v = pd.to_numeric(disp[c], errors='coerce')
            mu = np.nanmean(v); sd = np.nanstd(v) or 1.0
            disp[c] = (v - mu) / sd

    # --- per-feature panels with selective shading in the feature’s legend color
    def label_for(c: str) -> str:
        if select_by == "top1":
            return f"{c} (n={top1_count_for_panel.get(c,0)})"
        return f"{c} (z)" if normalize else c

    def canon(s: str) -> str:
        return re.sub(r'[\s_\-/]+', '', str(s).strip().lower())

    for i, c in enumerate(feats):
        ax = axes[i]
        y = pd.to_numeric(disp[c], errors='coerce').to_numpy()
        ax.plot(x, y, linewidth=1.2, color="#1f77b4", zorder=2, label=c)

        c_can = canon(c)
        for (s, e), fl, ss, wi in zip(spans, flags, subseqs, win_idx):
            if fl != 1:
                continue
            feats_this_win = topk_map.get((ss, wi), [])
            if not feats_this_win:
                continue
            feats_this_win = feats_this_win[:max(1, shade_topk)]

            # match by canonicalized name; shade in that feature's color
            shade = None
            for mae_feat in feats_this_win:
                if canon(str(mae_feat).replace("mae_", "", 1)) == c_can:
                    shade = feat2color.get(mae_feat, "#1f77b4")
                    break
            if shade is not None:
                ax.axvspan(s, min(e, T-1), alpha=0.18, color=shade, zorder=1)

        ax.set_ylabel(label_for(c))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        beautify_axis(ax)
        ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.85)

    # --- OC-SVM score panel with colored anomaly dots
    centers = [min((s+e)//2, T-1) for (s,e) in spans]
    ax = axes[-1]
    ax.plot(centers, scores, linewidth=1.0, color="#1f77b4", zorder=1)
    ax.axhline(0.0, linestyle="--", linewidth=0.9, color="gray")

    for cx, sc, fl, ss, wi in zip(centers, scores, flags, subseqs, win_idx):
        if fl == 1:
            L = topk_map.get((ss, wi), [])
            if not L:
                continue
            feat = L[0]  # Top-1 for color key
            ax.scatter(cx, sc, s=18, color=feat2color.get(feat, "#d62728"),
                       zorder=3, edgecolors="white", linewidths=0.3)

    ax.set_ylabel("OC-SVM score\n(higher = more normal)")
    ax.set_xlabel("t (row index)")
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    beautify_axis(ax)
    ax.set_title(f"EXAMM-only OC-SVM — {name} | shaded only when panel feature is Top-{shade_topk} contributor ({window_size}/{step_size})")

    # Legend outside on the right, ordered by Top-1 frequency on anomalies
    top1_counts = Counter([L[0] for L in topk_map.values() if L])
    legend_feats = [f for f, _ in top1_counts.most_common()]
    if legend_feats:
        handles = [plt.Line2D([0],[0], marker='o', linestyle='None',
                   label=f"{f} (n={top1_counts.get(f,0)})",
                   markerfacecolor=feat2color.get(f, "#1f77b4"), markeredgecolor="white",
                   markeredgewidth=0.3) for f in legend_feats]
        ncols = min(2, max(1, len(handles)//12 + (len(handles)%12>0)))
        leg = ax.legend(handles=handles[:limit_legend],
                        title="Top-1 contributor (anomalies)",
                        loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        borderaxespad=0., frameon=True, fancybox=True, framealpha=0.92,
                        ncol=ncols)
        leg._legend_box.align = "left"

    out_png = os.path.join(OUT_DIR, f"{name}_examm_overview_top1.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Optional textual summary: sum|z| across anomalies (decimal because it's z-magnitude)
    if print_summary and not contrib_rows.empty:
        contrib_stack = []
        for _, r in contrib_rows.iterrows():
            for j in range(1, 6):
                fcol = f"top{j}_feature"; zcol = f"top{j}_z"
                if fcol in r and zcol in r and pd.notna(r[fcol]) and pd.notna(r[zcol]):
                    contrib_stack.append((r[fcol], abs(float(r[zcol]))))
        if contrib_stack:
            agg = {}
            for ftr, zabs in contrib_stack:
                agg[ftr] = agg.get(ftr, 0.0) + zabs
            top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:20]
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

    # New controls:
    ap.add_argument("--top_features", type=int, default=10,
                    help="Number of panels to auto-select (by Top-1 frequency on anomalies). Ignored if --features is set.")
    ap.add_argument("--select_by", choices=["top1","variance"], default="top1",
                    help="Auto-panel selection: 'top1' = most frequent Top-1 contributors; 'variance' = highest raw variance.")
    ap.add_argument("--shade_topk", type=int, default=1,
                    help="Shade if feature is among the Top-K contributors for the window (1 = only Top-1).")
    ap.add_argument("--min_top1", type=int, default=1,
                    help="Require at least this many Top-1 hits for a feature to be auto-selected.")
    ap.add_argument("--print_summary", action="store_true",
                    help="Print per-feature contribution summary (sum|z| and Top-1 counts).")
    ap.add_argument("--limit_legend", type=int, default=48,
                    help="Max number of legend entries to show.")

    args = ap.parse_args()

    scores = pd.read_csv(args.scores_csv)
    contrib = pd.read_csv(args.topk_csv)

    # Pick a flight
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
        step_size=args.step_size,
        top_features=args.top_features,
        select_by=args.select_by,
        shade_topk=args.shade_topk,
        min_top1=args.min_top1,
        print_summary=args.print_summary,
        limit_legend=args.limit_legend,
    )


if __name__ == "__main__":
    main()
