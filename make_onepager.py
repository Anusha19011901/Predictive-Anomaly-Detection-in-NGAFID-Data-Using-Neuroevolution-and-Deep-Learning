#!/usr/bin/env python3
# make_onepager.py — a simple one-page PDF summary

import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BLURB = ("We train an OC-SVM on healthy flight windows and learn prototype “boxes” from K-Means.\n"
         "During inference, each window’s distance to the nearest prototype and per-sensor box violations\n"
         "flag anomalies and explain which channels deviated (e.g., AltMSL, IAS, RPM).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_csv", default="outputs/calibration/calibration_metrics.csv")
    ap.add_argument("--timeline_png", default="outputs/timeline/timeline_overlay.png")
    ap.add_argument("--contributors_png", default=None, help="Optional: a per-window bar image; else nearest_dist distribution text.")
    ap.add_argument("--out", default="outputs/demo/onepager.pdf")
    ap.add_argument("--title", default="NGAFID Anomaly Detection — One-Pager")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    calib = pd.read_csv(args.calib_csv)

    with PdfPages(args.out) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        ax0 = fig.add_axes([0.08, 0.86, 0.84, 0.1]); ax0.axis("off")
        ax0.text(0, 0.6, args.title, fontsize=18, weight="bold")
        ax0.text(0, 0.05, BLURB, fontsize=10)

        # left: timeline
        ax1 = fig.add_axes([0.08, 0.54, 0.84, 0.26]); ax1.axis("off")
        if os.path.exists(args.timeline_png):
            img = plt.imread(args.timeline_png)
            ax1.imshow(img); ax1.set_title("Example timeline with shaded anomalies (≥ P95 nearest_dist)")

        # right/bottom: contributors or placeholder
        ax2 = fig.add_axes([0.08, 0.24, 0.84, 0.26]); ax2.axis("off")
        if args.contributors_png and os.path.exists(args.contributors_png):
            img2 = plt.imread(args.contributors_png)
            ax2.imshow(img2); ax2.set_title("Per-sensor contributions (example)")

        # table: calibration metrics
        ax3 = fig.add_axes([0.08, 0.05, 0.84, 0.16]); ax3.axis("off")
        show = calib.copy()
        show["thr_value"] = show["thr_value"].round(2)
        show["precision"] = show["precision"].round(3)
        show["recall"] = show["recall"].round(3)
        show["f1"] = show["f1"].round(3)
        tbl = ax3.table(cellText=show.values, colLabels=show.columns, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        tbl.scale(1, 1.2)
        ax3.set_title("Calibration (RAW-based thresholds applied to ERROR)")

        pdf.savefig(fig, dpi=200)
        plt.close(fig)

    print("✓ wrote", args.out)

if __name__ == "__main__":
    main()
