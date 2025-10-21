#!/usr/bin/env python3
"""
EXAMM-only OC-SVM visualization (Top-N, weighted ordering, tidy legend, coverage report).

- Shading in a panel only when that panel's feature is among the Top-K contributors
  for that (flagged) window. Default K=3 via --shade_topk.
- Panels are auto-selected and ordered by weighted Top-N hits on anomalous windows:
    weight = sum(1/rank) across windows where the feature is in the top-K.
- Panel labels show Top-1 count (n=...) and Top-K hits (k=...).
- Legend rendered as a bottom strip (multi-column).
- Prints coverage report: cumulative share of |z| captured by Top-k contributors (k=1..5).
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCORES_CSV = "outputs/ocsvm_examm_only/before_scores.csv"
TOPK_CSV   = "outputs/ocsvm_examm_only/before_topk_contributors.csv"
BEFORE_DIR = "dataset/before_examm2"

OUT_DIR = "outputs/vis_examm_top1"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
def pick_features_auto(df: pd.DataFrame, max_k: int = 6) -> list[str]:
    drop_like = ("flight_id", "subseq_id", "window_idx")
    numeric = [c for c in df.columns
               if c not in drop_like and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors='coerce'))]
    if not numeric:
        raise ValueError("No numeric columns found in flight CSV.")
    vars_ = df[numeric].var(numeric_only=True).sort_values(ascending=False)
    return list(vars_.index[:max_k])

def window_idx_to_span(widx: int, window: int, step: int) -> tuple[int, int]:
    start = widx * step
    end   = start + window - 1
    return start, end

def canon(s: str) -> str:
    import re
    return re.sub(r'[\s_\-/]+', '', str(s).strip().lower())

def map_mae_to_raw(mae_name: str, raw_cols: list[str]) -> str | None:
    """Best-effort map mae_* name to a raw column."""
    base = str(mae_name).replace("mae_", "", 1)
    c_base = canon(base)
    cmap = {canon(c): c for c in raw_cols}
    if c_base in cmap: return cmap[c_base]
    for c in raw_cols:
        cb = canon(c)
        if cb.endswith(c_base) or c_base.endswith(cb):
            return c
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

def build_topk_map(contrib_rows: pd.DataFrame) -> dict[tuple, list[str]]:
    """(subseq_id, window_idx) -> [top1, top2, ...]"""
    topk_map = {}
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
    return topk_map

def coverage_report(contrib_rows: pd.DataFrame) -> list[float]:
    """
    For k=1..5, compute cumulative share of |z| captured by Top-k contributors.
    Uses columns top{j}_z (absolute z magnitudes).
    """
    if contrib_rows.empty:
        return []
    ks = range(1, 6)
    totals = []
    for _, r in contrib_rows.iterrows():
        zvals = []
        j = 1
        while f"top{j}_z" in r:
            z = r.get(f"top{j}_z")
            if pd.notna(z):
                zvals.append(abs(float(z)))
            j += 1
        if zvals:
            totals.append(zvals)
    if not totals:
        return []

    totals = [np.array(v) for v in totals]
    denom = sum(v.sum() for v in totals) or 1.0
    shares = []
    for k in ks:
        num = sum(v[:min(k, len(v))].sum() for v in totals)
        shares.append(num / denom)
    return shares

# ---------- main plot ----------
def make_plot(
    flight_csv: str,
    scores_df: pd.DataFrame,
    contrib_df: pd.DataFrame,
    features: list[str] | None,
    normalize: bool,
    window_size: int,
    step_size: int,
    top_features: int,
    select_by: str,
    shade_topk: int,
    min_top1: int,
    print_summary: bool,
    limit_legend: int,
):
    import itertools
    from collections import Counter

    name = Path(flight_csv).stem
    raw = pd.read_csv(flight_csv, low_memory=False).reset_index(drop=True)

    # OC-SVM rows
    wrows = scores_df[scores_df["flight_id"] == name].copy()
    if wrows.empty:
        raise ValueError(f"No rows found for flight_id={name} in {SCORES_CSV}")
    spans   = [window_idx_to_span(int(wi), window_size, step_size) for wi in wrows["window_idx"]]
    flags   = wrows["anomaly_flag"].to_numpy().astype(int)
    scores  = wrows["ocsvm_score"].to_numpy().astype(float)
    subseqs = wrows["subseq_id"].tolist()
    win_idx = wrows["window_idx"].astype(int).tolist()
    keys    = list(zip(subseqs, win_idx))
    anom_keys = {keys[i] for i, f in enumerate(flags) if f == 1}

    # Top-k contributors
    contrib_rows = contrib_df[contrib_df["flight_id"] == name].copy()
    topk_map = build_topk_map(contrib_rows)

    # Colors for EXAMM channels (consistent for spans + score dots)
    uniq_feats = sorted({f for L in topk_map.values() for f in L})
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', []))
    feat2color = {f: next(color_cycle) for f in uniq_feats}

    # Choose raw panels
    raw_cols = list(raw.columns)
    if features:
        feats = [c for c in features if c in raw_cols]
        if not feats:
            raise ValueError(f"None of the requested features exist in {name}.")
        top1_count_for_panel = {f: 0 for f in feats}
        topk_hits_for_panel  = {f: 0 for f in feats}
        weighted_hits        = {f: 0.0 for f in feats}
    else:
        if select_by == "top1" and topk_map:
            # count Top-1 ONLY on anomalous windows
            top1_list = [L[0] for k, L in topk_map.items() if L and k in anom_keys]
            top1_counts = Counter(top1_list)

            # also compute weighted hits and top-k hits (for labels)
            weights = {}
            topk_hits = {}
            for k, L in topk_map.items():
                if k not in anom_keys:  # anomalies only
                    continue
                for rank, f_examm in enumerate(L[:max(1, shade_topk)], start=1):
                    weights[f_examm] = weights.get(f_examm, 0.0) + 1.0 / rank
                    topk_hits[f_examm] = topk_hits.get(f_examm, 0) + 1

            # Map EXAMM to raw, order by weighted hits
            ordered = []
            for ex_feat, _ in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
                raw_match = map_mae_to_raw(ex_feat, raw_cols)
                if raw_match and raw_match not in [f for f, _ in ordered]:
                    ordered.append((raw_match, ex_feat))
                if len(ordered) >= top_features:
                    break

            if not ordered:
                feats = pick_features_auto(raw, max_k=top_features)
                top1_count_for_panel = {f: 0 for f in feats}
                topk_hits_for_panel  = {f: 0 for f in feats}
                weighted_hits        = {f: 0.0 for f in feats}
            else:
                feats = [f for f, _ in ordered]
                # fill metrics for labels/order
                top1_count_for_panel = {map_mae_to_raw(f, raw_cols) or f: top1_counts.get(f, 0) for f in uniq_feats}
                top1_count_for_panel = {f: top1_count_for_panel.get(f, 0) for f in feats}
                topk_hits_for_panel  = {map_mae_to_raw(f, raw_cols) or f: topk_hits.get(f, 0) for f in uniq_feats}
                topk_hits_for_panel  = {f: topk_hits_for_panel.get(f, 0) for f in feats}
                weighted_hits        = {map_mae_to_raw(f, raw_cols) or f: weights.get(f, 0.0) for f in uniq_feats}
                weighted_hits        = {f: weighted_hits.get(f, 0.0) for f in feats}
        else:
            feats = pick_features_auto(raw, max_k=top_features)
            top1_count_for_panel = {f: 0 for f in feats}
            topk_hits_for_panel  = {f: 0 for f in feats}
            weighted_hits        = {f: 0.0 for f in feats}

    # Order panels by weighted hits desc, then by Top-1 count
    feats.sort(key=lambda f: (-weighted_hits.get(f, 0.0), -top1_count_for_panel.get(f, 0), f))

    # ---- layout
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
    n_extra = 1  # score panel
    fig_h = max(1.7 * len(feats) + 1.2 * n_extra + 0.5, 7)
    fig, axes = plt.subplots(len(feats) + n_extra, 1, figsize=(12.5, fig_h), sharex=True,
                             gridspec_kw={"height_ratios": [1.0]*len(feats) + [0.6]*n_extra})
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    # leave space at bottom for legend strip
    plt.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.14, hspace=0.08)

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

    # Per-feature panels with Top-K selective shading + alpha ladder by rank
    alpha_by_rank = {1: 0.22, 2: 0.14, 3: 0.09, 4: 0.06, 5: 0.04}
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
            # if this panel's feature is in Top-K, shade with alpha by rank
            rank_found = None
            shade_color = None
            for rank, mae_feat in enumerate(feats_this_win, start=1):
                if canon(str(mae_feat).replace("mae_", "", 1)) == c_can:
                    rank_found = rank
                    shade_color = feat2color.get(mae_feat, "#1f77b4")
                    break
            if rank_found is not None:
                ax.axvspan(s, min(e, T-1), alpha=alpha_by_rank.get(rank_found, 0.05),
                           color=shade_color, zorder=1)

        lbl = f"{c} (n={top1_count_for_panel.get(c,0)}, k={topk_hits_for_panel.get(c,0)})" if select_by=="top1" \
              else (f"{c} (z)" if normalize else c)
        ax.set_ylabel(lbl)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        beautify_axis(ax)
        ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.85)

    # Score strip with Top-1 colored dots
    centers = [min((s+e)//2, T-1) for (s,e) in spans]
    ax = axes[-1]
    ax.plot(centers, scores, linewidth=1.0, color="#1f77b4", zorder=1)
    ax.axhline(0.0, linestyle="--", linewidth=0.9, color="gray")
    for cx, sc, fl, ss, wi in zip(centers, scores, flags, subseqs, win_idx):
        if fl == 1:
            L = topk_map.get((ss, wi), [])
            if L:
                feat = L[0]
                ax.scatter(cx, sc, s=18, color=feat2color.get(feat, "#d62728"),
                           zorder=3, edgecolors="white", linewidths=0.3)
    ax.set_ylabel("OC-SVM score\n(higher = more normal)")
    ax.set_xlabel("t (row index)")
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    beautify_axis(ax)
    ax.set_title(f"EXAMM-only OC-SVM — {name} | shaded when panel feature is Top-{shade_topk} ({window_size}/{step_size})")

    # Bottom legend strip (multi-column), sorted by Top-1 freq
    top1_counts = Counter([L[0] for L in topk_map.values() if L])
    legend_feats = [f for f, _ in top1_counts.most_common()]
    if legend_feats:
        handles = [plt.Line2D([0],[0], marker='o', linestyle='None',
                   label=f"{f} (n={top1_counts.get(f,0)})",
                   markerfacecolor=feat2color.get(f, "#1f77b4"), markeredgecolor="white",
                   markeredgewidth=0.3) for f in legend_feats]
        ncols = 4
        fig.legend(handles=handles[:limit_legend],
                   title="Top-1 contributor (anomalies)",
                   loc="lower center", bbox_to_anchor=(0.5, 0.02),
                   ncol=ncols, frameon=True, fancybox=True, framealpha=0.92)

    out_png = os.path.join(OUT_DIR, f"{name}_examm_overview_topn.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Coverage report + optional sum|z| table
    cov = coverage_report(contrib_rows)
    if cov:
        cov_str = ", ".join(f"{x:.2f}" for x in cov)
        # automated "pick N" suggestion: argmin of diff after elbow (simple)
        suggested_k = 1 + np.argmax(np.diff([0.0]+cov) < 0.03)
        print(f"Top-k |z|-coverage (k=1..5): [{cov_str}]  → consider k={max(3, min(5, suggested_k))}")
    if print_summary and not contrib_rows.empty:
        # sum|z| table (same semantics as before)
        stack = []
        for _, r in contrib_rows.iterrows():
            for j in range(1, 6):
                fcol = f"top{j}_feature"; zcol = f"top{j}_z"
                if fcol in r and zcol in r and pd.notna(r[fcol]) and pd.notna(r[zcol]):
                    stack.append((r[fcol], abs(float(r[zcol]))))
        if stack:
            agg = {}
            for ftr, zabs in stack:
                agg[ftr] = agg.get(ftr, 0.0) + zabs
            top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"\nTop contributors (sum |z| across anomalies) for {name}:")
            for f, w in top:
                print(f"  - {f}: {w:.3f}")

    print(f"✅ Saved: {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Visualize EXAMM-only OC-SVM with Top-N selective shading.")
    ap.add_argument("--scores_csv", default=SCORES_CSV)
    ap.add_argument("--topk_csv",   default=TOPK_CSV)
    ap.add_argument("--before_dir", default=BEFORE_DIR)
    ap.add_argument("--flight_id",  default=None, help="Filename stem in before_examm2 to visualize.")
    ap.add_argument("--features",   default=None, help="Comma-separated raw columns to plot (overrides auto-select).")
    ap.add_argument("--normalize",  action="store_true", help="Z-normalize plotted raw features for display.")
    ap.add_argument("--window_size", type=int, default=30)
    ap.add_argument("--step_size",   type=int, default=5)

    # Top-N / selection
    ap.add_argument("--top_features", type=int, default=10,
                    help="Panels to auto-select; ignored if --features is set.")
    ap.add_argument("--select_by", choices=["top1","variance"], default="top1",
                    help="Auto selection: 'top1' uses Top-N stats; 'variance' uses raw variance.")
    ap.add_argument("--shade_topk", type=int, default=3,
                    help="Shade if feature is among Top-K contributors (1=only Top-1).")
    ap.add_argument("--min_top1", type=int, default=1,
                    help="Require at least this many Top-1 hits for a feature to be selected.")
    ap.add_argument("--print_summary", action="store_true",
                    help="Also print sum|z| table in terminal.")
    ap.add_argument("--limit_legend", type=int, default=48,
                    help="Max legend entries to show in bottom strip.")

    args = ap.parse_args()
    scores = pd.read_csv(args.scores_csv)
    contrib = pd.read_csv(args.topk_csv)

    flight_id = args.flight_id or scores["flight_id"].iloc[0]
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
