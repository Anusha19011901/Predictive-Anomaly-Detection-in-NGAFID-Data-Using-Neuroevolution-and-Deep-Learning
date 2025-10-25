#!/usr/bin/env python3
import os, argparse
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCORES_CSV = "outputs/ocsvm_examm_only/before_scores.csv"
TOPK_CSV   = "outputs/ocsvm_examm_only/before_topk_contributors.csv"
BEFORE_DIR = "dataset/before_examm2"
OUT_DIR    = "outputs/vis_examm_top1"
os.makedirs(OUT_DIR, exist_ok=True)

def canon(s: str) -> str:
    import re
    return re.sub(r'[\s_\-/]+', '', str(s).strip().lower())

def map_mae_to_raw(mae_name: str, raw_cols: list[str]) -> str | None:
    base = str(mae_name).replace("mae_", "", 1)
    c_base = canon(base)
    cmap = {canon(c): c for c in raw_cols}
    if c_base in cmap: return cmap[c_base]
    for c in raw_cols:
        cb = canon(c)
        if cb.endswith(c_base) or c_base.endswith(cb):
            return c
    aliases = {
        "vspd": ["vspdg", "vvi", "verticalspeed"],
        "gndspd": ["groundspeed", "gs"],
        "latac": ["laac", "latacc", "lataccel"],
        "longac": ["longacc", "longaccel"],
    }
    for key, vals in aliases.items():
        if c_base == key or c_base in vals:
            for c in raw_cols:
                if canon(c) in [key] + vals:
                    return c
    return None

def build_topk_map(contrib_rows: pd.DataFrame) -> dict[tuple, list[str]]:
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

def shapes_for_k(k: int, selected_raw, spans, flags, subseq, widx, topk_map, feat2color):
    # alpha ladder by rank
    alpha = {1: 0.22, 2: 0.14, 3: 0.09, 4: 0.06, 5: 0.04}
    shapes = []
    for i, c in enumerate(selected_raw, start=1):
        c_can = canon(c)
        for (s, e), fl, ss, wi in zip(spans, flags, subseq, widx):
            if fl != 1:
                continue
            L = topk_map.get((ss, wi), [])
            if not L:
                continue
            Lk = L[:k]
            hit_rank = None
            color = '#1f77b4'
            for rank, f_ex in enumerate(Lk, start=1):
                if canon(str(f_ex).replace("mae_", "", 1)) == c_can:
                    hit_rank = rank
                    color = feat2color.get(f_ex, color)
                    break
            if hit_rank is None:
                continue

            # xref uses shared x-axis ("x"); yref must be "y domain" for row 1, "y2 domain" for row 2, etc.
            yref = "y domain" if i == 1 else f"y{i} domain"

            shapes.append(dict(
                type="rect",
                xref="x",
                yref=yref,
                x0=s, x1=e,
                y0=0, y1=1,              # fill full panel height
                fillcolor=color,
                opacity=alpha.get(hit_rank, 0.05),
                line_width=0,
                layer="below"
            ))
    return shapes


def make_html(flight_id: str, scores_df: pd.DataFrame, contrib_df: pd.DataFrame,
              before_dir: str, top_features: int, min_top1: int,
              window_size: int = 30, step_size: int = 5):

    # Load raw flight
    flight_csv = os.path.join(before_dir, f"{flight_id}.csv")
    if not os.path.exists(flight_csv):
        cand = list(Path(before_dir).glob(f"{flight_id}*.csv"))
        if not cand:
            raise FileNotFoundError(f"Raw CSV not found for {flight_id}")
        flight_csv = str(cand[0])
    raw = pd.read_csv(flight_csv, low_memory=False).reset_index(drop=True)
    T = len(raw); x = np.arange(T)

    # OC-SVM per-window
    wrows = scores_df[scores_df["flight_id"] == flight_id].copy()
    if wrows.empty:
        raise ValueError(f"No rows in scores for {flight_id}")
    spans  = [ (int(w)*step_size, int(w)*step_size + window_size - 1)
               for w in wrows["window_idx"] ]
    flags  = wrows["anomaly_flag"].to_numpy().astype(int)
    scores = wrows["ocsvm_score"].to_numpy().astype(float)
    subseq = wrows["subseq_id"].tolist()
    widx   = wrows["window_idx"].astype(int).tolist()
    keys   = list(zip(subseq, widx))
    anom_keys = {keys[i] for i,f in enumerate(flags) if f==1}

    # Top-k contributors
    contrib_rows = contrib_df[contrib_df["flight_id"] == flight_id].copy()
    topk_map = build_topk_map(contrib_rows)

    # Select features by weighted hits (k=3 heuristic)
    from collections import Counter
    weights = {}
    top1_counts = Counter([L[0] for k, L in topk_map.items() if L and k in anom_keys])
    for k_, L in topk_map.items():
        if k_ not in anom_keys: continue
        for rank, f_examm in enumerate(L[:3], start=1):
            weights[f_examm] = weights.get(f_examm, 0.0) + 1.0/rank

    ordered = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    selected_raw = []
    for ex_feat, _ in ordered:
        m = map_mae_to_raw(ex_feat, list(raw.columns))
        if m and m not in selected_raw:
            selected_raw.append(m)
        if len(selected_raw) >= top_features:
            break
    if not selected_raw:
        selected_raw = list(raw.columns[:min(10, raw.shape[1])])

    # Color map for EXAMM feat names
    base_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                   '#8c564b','#e377c2',"#d78f8f",'#bcbd22','#17becf']
    ex_feat_list = sorted({L[0] for L in topk_map.values() if L})
    feat2color = {f: base_colors[i % len(base_colors)] for i, f in enumerate(ex_feat_list)}

    # Figure
    rows = len(selected_raw) + 1
    fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    vertical_spacing=0.02)

    # Raw traces
    for i, c in enumerate(selected_raw, start=1):
        fig.add_trace(
            go.Scattergl(
                x=centers, y=scores, mode='lines',
                name="OC-SVM", showlegend=False,
                hovertemplate="<b>Score</b><br>t=%{x}<br>value=%{y}<extra></extra>"
            ),
            row=rows, col=1
        )
        fig.update_yaxes(title_text="OC-SVM score (higher = more normal)", row=rows, col=1)



    # Score trace + colored anomaly dots (Top-1)
    centers = [min((s+e)//2, T-1) for (s,e) in spans]
    fig.add_trace(go.Scattergl(x=centers, y=scores, mode='lines',
                               name="OC-SVM", showlegend=False), row=rows, col=1)
    dot_x, dot_y, dot_color = [], [], []
    for cx, sc, fl, ss, wi in zip(centers, scores, flags, subseq, widx):
        if fl == 1:
            L = topk_map.get((ss, wi), [])
            if L:
                dot_x.append(cx); dot_y.append(sc)
                dot_color.append(feat2color.get(L[0], '#d62728'))
    fig.add_trace(go.Scattergl(x=dot_x, y=dot_y, mode='markers',
                               marker=dict(size=6, line=dict(width=0.5, color='white'),
                                           color=dot_color),
                               name="Anomaly (Top-1 colored)", showlegend=False), row=rows, col=1)

    # Initial shapes for k=3
    fig.update_layout(shapes=shapes_for_k(3, selected_raw, spans, flags, subseq, widx, topk_map, feat2color))

    # Slider for Top-N (1..5)
    steps = []
    for k in range(1, 6):
        steps.append(dict(
            method="relayout",
            args=[{"shapes": shapes_for_k(k, selected_raw, spans, flags, subseq, widx, topk_map, feat2color)}],
            label=str(k),
        ))
    fig.update_layout(
        sliders=[dict(active=2, currentvalue={"prefix": "Top-N = "},
                      steps=steps, x=0.5, y=1.08, xanchor="center", yanchor="top")],
        updatemenus=[dict(type="buttons", direction="right",
                          x=0.5, y=1.14, xanchor="center", yanchor="top",
                          buttons=[dict(method="relayout",
                                        args=[{"shapes": shapes_for_k(3, selected_raw, spans, flags, subseq, widx, topk_map, feat2color)}],
                                        label="Reset k=3")],
                          showactive=False)]
    )

    # Look & rangeslider
    for i in range(1, rows+1):
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.15)", row=i, col=1)
    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        height=250*rows, width=1400, margin=dict(l=70, r=20, t=110, b=60),
        title=f"EXAMM-only OC-SVM — {flight_id} (interactive)",
        hovermode="x unified", template="plotly_white"
    )

    out_html = os.path.join(OUT_DIR, f"{flight_id}_interactive.html")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"✅ Saved interactive HTML: {out_html}")

def main():
    ap = argparse.ArgumentParser(description="Interactive EXAMM-only OC-SVM visualization (Plotly).")
    ap.add_argument("--scores_csv", default=SCORES_CSV)
    ap.add_argument("--topk_csv",   default=TOPK_CSV)
    ap.add_argument("--before_dir", default=BEFORE_DIR)
    ap.add_argument("--flight_id",  default=None)
    ap.add_argument("--top_features", type=int, default=10)
    ap.add_argument("--min_top1", type=int, default=1)
    ap.add_argument("--window_size", type=int, default=30)
    ap.add_argument("--step_size",   type=int, default=5)
    args = ap.parse_args()

    scores = pd.read_csv(args.scores_csv)
    contrib = pd.read_csv(args.topk_csv)
    flight_id = args.flight_id or scores["flight_id"].iloc[0]

    make_html(flight_id, scores, contrib, args.before_dir,
              args.top_features, args.min_top1,
              window_size=args.window_size, step_size=args.step_size)

if __name__ == "__main__":
    main()
