#!/usr/bin/env python3
# demo_portfolio.py
# Build a small offline HTML demo (Plotly) for airline-friendly viewing.

import os, glob, json, argparse
from typing import List, Dict
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

COLS = ["AltMSL","IAS","E1 RPM"]   # show these on timeline

def load_expl(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "window_idx" not in df.columns:
        df["window_idx"] = df.groupby("file").cumcount()
    # robust columns
    if "nearest_dist" not in df.columns:
        raise SystemExit("explanations csv missing 'nearest_dist'")
    return df

def robust_read_ngafid(path: str) -> pd.DataFrame:
    # Try normal NGAFID (skiprows=2); if headers numeric, re-read raw and assume first row is data header → make it headerless
    try:
        df = pd.read_csv(path, skiprows=2)
    except Exception:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    # If all look numeric, treat file as headerless and re-read with header=None
    def _numy(names):
        ok=0
        for c in names:
            try: float(str(c)); ok+=1
            except: pass
        return ok==len(names)
    if _numy(df.columns):
        df = pd.read_csv(path, header=None)
        # we won’t rely on names for timeline; we’ll try to locate desired columns by heuristics later if needed
        # but many of your “before_examm2” have proper NGAFID headers after skiprows=2, so this branch is rare
    return df

def align_columns(df: pd.DataFrame, want: List[str]) -> pd.DataFrame:
    # simple name match; if missing, bail with an informative message
    df_cols = {c.lower().replace(" ",""): c for c in df.columns}
    resolved = []
    for w in want:
        k = w.lower().replace(" ","")
        if k in df_cols:
            resolved.append(df_cols[k])
        else:
            raise KeyError(f"Column '{w}' not found in {list(df.columns)[:20]}")
    out = df[resolved].apply(pd.to_numeric, errors="coerce").dropna()
    out.columns = want
    return out

def make_timeline_figure(df: pd.DataFrame, flagged_spans: List[tuple]):
    t = np.arange(len(df))
    traces = []
    for c in COLS:
        if c in df.columns:
            traces.append(go.Scatter(x=t, y=df[c], name=c, mode="lines"))
    shapes = []
    for s,e in flagged_spans:
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=s, x1=e, y0=0, y1=1,
                           fillcolor="rgba(255,0,0,0.12)", line=dict(width=0)))
    layout = go.Layout(title="Timeline (AltMSL / IAS / E1 RPM) with flagged windows",
                       xaxis=dict(title="t (sample index)"),
                       yaxis=dict(title="value"),
                       shapes=shapes, legend=dict(orientation="h"))
    return go.Figure(data=traces, layout=layout)

def make_heatmap(df_expl_f: pd.DataFrame):
    z = df_expl_f.set_index("window_idx")["nearest_dist"].reindex(range(df_expl_f["window_idx"].max()+1)).fillna(0).values[None,:]
    hm = go.Heatmap(z=z, colorscale="Viridis",
                    colorbar=dict(title="nearest_dist"),
                    showscale=True)
    layout = go.Layout(title="nearest_dist per window (hover for value)",
                       xaxis=dict(title="window_idx"),
                       yaxis=dict(title=""), height=250)
    return go.Figure(data=[hm], layout=layout)

def top_windows(df_expl_f: pd.DataFrame, n=5) -> pd.DataFrame:
    return df_expl_f.sort_values("nearest_dist", ascending=False).head(n).copy()

def make_contrib_bar(row: pd.Series, sensors_full: List[str]):
    # Build counts & severity per sensor for a single row
    counts, sevs, labels = [], [], []
    for s in sensors_full:
        c_col = f"viol_count_{s}"
        v_col = f"viol_sev_{s}"
        if c_col in row.index and v_col in row.index:
            counts.append(row[c_col]); sevs.append(row[v_col]); labels.append(s)
    if not labels:
        return None

    bar1 = go.Bar(x=labels, y=counts, name="violation count (0..W)", opacity=0.9)
    bar2 = go.Bar(x=labels, y=sevs, name="violation severity (sum over W)", opacity=0.9)
    layout = go.Layout(title=f"Per-sensor contributions for window {int(row['window_idx'])}",
                       barmode="group", xaxis=dict(tickangle=-30))
    return go.Figure(data=[bar1, bar2], layout=layout)

def latlon_scatter(df_full: pd.DataFrame, df_expl_f: pd.DataFrame, step: int, window: int):
    # Build nearest_dist per sample index by expanding window values (approximate: set value at window start)
    nn = len(df_full)
    nd_series = np.zeros(nn)
    if "window_idx" in df_expl_f.columns:
        for _, r in df_expl_f.iterrows():
            start = int(r.get("start_idx", r["window_idx"]*step))
            if 0 <= start < nn:
                nd_series[start] = r["nearest_dist"]
    # Try to find Lat/Lon columns robustly
    lat_col = next((c for c in df_full.columns if c.lower().startswith("lat")), None)
    lon_col = next((c for c in df_full.columns if c.lower().startswith("long")), None)
    if lat_col is None or lon_col is None:
        return None
    trace = go.Scattergl(
        x=df_full[lon_col], y=df_full[lat_col],
        mode="markers", marker=dict(size=5, color=nd_series, colorscale="Viridis", showscale=True, colorbar=dict(title="nearest_dist")),
        text=[f"t={i}, nd={nd_series[i]:.2f}" for i in range(nn)],
        name="Lat/Lon by nearest_dist"
    )
    layout = go.Layout(title="Lat / Lon colored by nearest_dist (no basemap)", xaxis=dict(title="Longitude"), yaxis=dict(title="Latitude"))
    return go.Figure(data=[trace], layout=layout)

def build_flight_page(outdir_f: str, flight_csv: str, df_expl: pd.DataFrame, thr_pct: float, sensors_all: List[str], step: int, window: int):
    os.makedirs(outdir_f, exist_ok=True)
    base = os.path.basename(flight_csv)

    df_full = robust_read_ngafid(flight_csv)
    try:
        df_show = align_columns(df_full, COLS)
    except Exception:
        # If timeline columns missing, just use what exists
        keep = [c for c in COLS if c in df_full.columns]
        df_show = df_full[keep].apply(pd.to_numeric, errors="coerce").dropna()

    df_f = df_expl[df_expl["file"] == base].copy()
    if df_f.empty:
        # make an empty page with a note
        html = f"<h2>{base}</h2><p>No explanation rows for this flight.</p>"
        with open(os.path.join(outdir_f,"index.html"),"w") as f: f.write(html)
        return

    # Threshold from global RAW distribution
    thr = np.percentile(df_expl["nearest_dist"].values, thr_pct)

    # flagged spans
    spans = []
    for _, r in df_f.iterrows():
        s = int(r.get("start_idx", r["window_idx"]*step))
        e = s + window - 1
        if r["nearest_dist"] >= thr:
            spans.append((s,e))

    # Figures
    fig_tl = make_timeline_figure(df_show, spans)
    fig_hm = make_heatmap(df_f)

    # Lat/Lon scatter (optional)
    fig_ll = latlon_scatter(df_full, df_f, step, window)
    ll_html = plot(fig_ll, include_plotlyjs=False, output_type="div") if fig_ll is not None else "<p><i>Lat/Lon not available.</i></p>"

    # Top windows dropdown (build first one as default)
    top = top_windows(df_f, n=5)
    contrib_divs = []
    for _, r in top.iterrows():
        fig_c = make_contrib_bar(r, sensors_all)
        if fig_c is not None:
            contrib_divs.append((int(r["window_idx"]), plot(fig_c, include_plotlyjs=False, output_type="div")))
    contrib_html = "".join([f"<h3>Window {wid}</h3>{div}" for wid,div in contrib_divs]) or "<p><i>Violation columns not present.</i></p>"

    # Assemble page
    tl_html = plot(fig_tl, include_plotlyjs=False, output_type="div")
    hm_html = plot(fig_hm, include_plotlyjs=False, output_type="div")

    page = f"""
<!doctype html><html><head>
<meta charset="utf-8" />
<title>{base}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style> body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; }}
.card {{ border:1px solid #ddd; border-radius:10px; padding:14px; margin-bottom:18px; box-shadow:0 1px 4px rgba(0,0,0,0.06); }}
h1,h2,h3 {{ margin:6px 0 10px 0; }}
small {{ color:#666; }}
</style>
</head><body>
<h1>{base}</h1>
<small>Threshold = RAW P{thr_pct:.0f} (nearest_dist ≥ {thr:.2f})</small>

<div class="card">
  <h2>Timeline</h2>
  {tl_html}
</div>

<div class="card">
  <h2>Nearest distance per window</h2>
  {hm_html}
</div>

<div class="card">
  <h2>Top anomalous windows — sensor contributions</h2>
  {contrib_html}
</div>

<div class="card">
  <h2>Lat / Lon colored by nearest_dist</h2>
  {ll_html}
</div>

<p><a href="../index.html">← back to gallery</a></p>
</body></html>
"""
    with open(os.path.join(outdir_f,"index.html"),"w") as f:
        f.write(page)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before_dir", default="dataset/before_examm2")
    ap.add_argument("--raw_expl_csv", default="outputs/prototype_explanations_before.csv")
    ap.add_argument("--outdir", default="outputs/demo")
    ap.add_argument("--threshold_pct", type=float, default=95.0)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--sensors", nargs="+", default=["AltMSL","E1 RPM","E1 FFlow","E1 CHT1","E1 EGT1","NormAc","IAS"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_expl = load_expl(args.raw_expl_csv)

    flights = sorted(glob.glob(os.path.join(args.before_dir,"*.csv")))
    if not flights:
        raise SystemExit(f"No flights found in {args.before_dir}")

    # Build per-flight pages + collect gallery rows
    rows = []
    for f in flights:
        base = os.path.basename(f)
        df_f = df_expl[df_expl["file"] == base]
        if df_f.empty:
            n_win = 0; n_flag = 0
        else:
            n_win = len(df_f)
            thr = np.percentile(df_expl["nearest_dist"].values, args.threshold_pct)
            n_flag = int((df_f["nearest_dist"] >= thr).sum())
        rows.append({"file": base, "n_windows": n_win, "n_flagged": n_flag})
        flight_dir = os.path.join(args.outdir, base.replace(".csv",""))
        build_flight_page(flight_dir, f, df_expl, args.threshold_pct, args.sensors, args.step, args.window)

    gal = pd.DataFrame(rows)
    gal["flag_rate_%"] = (100.0 * gal["n_flagged"] / np.maximum(gal["n_windows"],1)).round(1)

    # Gallery HTML
    cards = []
    for _, r in gal.iterrows():
        link = r["file"].replace(".csv","") + "/index.html"
        cards.append(f"""
<div class="card">
  <h3>{r["file"]}</h3>
  <div>Windows: <b>{int(r["n_windows"])}</b> &nbsp; | &nbsp; Flagged: <b>{int(r["n_flagged"])}</b> &nbsp; | &nbsp; Rate: <b>{r["flag_rate_%"]}%</b></div>
  <div style="margin-top:8px;"><a href="{link}">Open</a></div>
</div>
""")
    html = f"""<!doctype html><html><head>
<meta charset="utf-8" />
<title>NGAFID Anomaly Demo</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style> body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; }}
h1 {{ margin-bottom: 12px; }}
.card {{ border:1px solid #ddd; border-radius:10px; padding:14px; margin:10px 0; box-shadow:0 1px 4px rgba(0,0,0,0.06); }}
</style>
</head><body>
<h1>NGAFID Anomaly Detection — Demo Gallery</h1>
<p>Threshold = RAW P{args.threshold_pct:.0f} (nearest_dist). Click a flight to explore.</p>
{"".join(cards)}
</body></html>"""
    with open(os.path.join(args.outdir,"index.html"),"w") as f:
        f.write(html)

    # Save gallery data too (optional)
    gal.to_csv(os.path.join(args.outdir,"gallery_summary.csv"), index=False)
    print(f"✓ Wrote demo to {args.outdir}/index.html with {len(gal)} flights.")

if __name__ == "__main__":
    main()
