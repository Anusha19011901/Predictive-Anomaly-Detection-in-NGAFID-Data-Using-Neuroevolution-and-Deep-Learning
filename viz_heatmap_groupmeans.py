# viz_heatmap_groupmeans.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_long", default="outputs/ocsvm_examm_only/before_groupmeans_long.csv")
    ap.add_argument("--flight_id", default=None, help="Filter to one flight_id; if omitted, first flight is used.")
    ap.add_argument("--top_features", type=int, default=30, help="Show top-K features by mean |z|.")
    ap.add_argument("--outfile", default=None, help="If set, save PNG here instead of showing.")
    args = ap.parse_args()

    gm = pd.read_csv(args.gm_long)
    if gm.empty:
        raise SystemExit("No rows in before_groupmeans_long.csv")

    # pick a flight
    flight = args.flight_id or gm["flight_id"].iloc[0]
    gm_f = gm[gm["flight_id"] == flight].copy()
    if gm_f.empty:
        raise SystemExit(f"No rows for flight_id={flight}")

    # pivot: rows=feature, cols=subseq_id
    M = gm_f.pivot(index="feature", columns="subseq_id", values="z_mean").sort_index()
    # order features by mean |z| (descending) and keep top-K
    feat_order = (M.abs().mean(axis=1)).sort_values(ascending=False).index
    if args.top_features is not None and args.top_features > 0:
        M = M.loc[feat_order[:args.top_features]]

    # shorten X labels: subseq_id -> S1..Sn and print legend
    subseqs = M.columns.tolist()
    xshort = {subseq: f"S{i+1}" for i, subseq in enumerate(subseqs)}
    M.columns = [xshort[c] for c in subseqs]

    # figure size scaled to content (caps to avoid huge canvases)
    nrows, ncols = M.shape
    width  = min(16, max(6, 1.0 * ncols + 4))      # columns drive width
    height = min(18, max(5, 0.35 * nrows + 2))     # rows drive height

    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(M.values, aspect="auto", interpolation="nearest")

    # ticks and fonts
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(M.index, fontsize=8)
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(M.columns, rotation=0, ha="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean z (EXAMM mae)")

    ax.set_title(f"EXAMM z-mean heatmap (aggregated) — flight {flight}", fontsize=11)
    ax.set_xlabel("subseq_id (short labels)")
    ax.set_ylabel("feature")

    # tighter layout without crushing the axes
    plt.tight_layout()

    # Print legend mapping short -> full subseq_id for your notes/slide
    print("\nLegend (X labels):")
    for short, full in xshort.items():
        pass
    for full in subseqs:
        print(f"  {xshort[full]} = {full}")

    if args.outfile:
        Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.outfile, dpi=200, bbox_inches="tight")
        print(f"\nSaved figure to {args.outfile}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
