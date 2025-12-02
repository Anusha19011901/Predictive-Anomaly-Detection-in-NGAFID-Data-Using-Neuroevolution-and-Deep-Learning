#!/usr/bin/env python3
"""
Dump per-cluster medoids using the explanations CSV.

Definition: medoid = member with minimum `nearest_dist` within each prototype_id.

It expects window filenames in the explanations CSV (e.g., window_12345.csv) to
exist inside a --windows_dir. If not found, you can add --recursive to search
subfolders.

Outputs:
- <out_dir>/medoids/proto_<k>.csv             (raw time series for medoid k)
- <out_dir>/medoids/proto_<k>.png             (optional plot if --plot)
- <out_dir>/medoids_summary.csv               (which window was chosen + stats)
"""

import argparse
import os
from pathlib import Path
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def find_file(base_dir: Path, filename: str, recursive: bool) -> Path | None:
    p = base_dir / filename
    if p.exists():
        return p
    if not recursive:
        return None
    # slow fallback: walk to find first match
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return Path(root) / filename
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explanations_csv", required=True,
                    help="CSV produced by explanations step (with columns: file, prototype_id, nearest_dist, ...)")
    ap.add_argument("--windows_dir", required=True,
                    help="Directory that contains the window_*.csv files referenced in the explanations CSV.")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write medoid CSVs and summary.")
    ap.add_argument("--plot", action="store_true",
                    help="Also save a simple PNG plot per medoid (requires matplotlib).")
    ap.add_argument("--recursive", action="store_true",
                    help="If set, search subfolders of --windows_dir when a file isn't found at top-level.")
    args = ap.parse_args()

    exp_csv = Path(args.explanations_csv)
    windows_dir = Path(args.windows_dir)
    out_dir = Path(args.out_dir)
    med_dir = out_dir / "medoids"
    med_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(exp_csv)
    required_cols = {"file", "prototype_id", "nearest_dist"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: Missing required columns in {exp_csv}: {missing}")

    # choose medoid per prototype_id = argmin nearest_dist
    # in case of ties, take the first occurrence
    df_sorted = df.sort_values(["prototype_id", "nearest_dist"], ascending=[True, True])
    medoids = df_sorted.groupby("prototype_id", as_index=False).first()

    summary_rows = []
    for _, row in medoids.iterrows():
        pid = int(row["prototype_id"])
        fname = str(row["file"])
        nearest_dist = float(row["nearest_dist"])
        viol_count = int(row.get("viol_count_total", 0))
        viol_sev = float(row.get("viol_sev_total", 0.0))

        fpath = find_file(windows_dir, Path(fname).name, recursive=args.recursive)
        if fpath is None:
            print(f"[warn] window file not found for proto {pid}: {fname} (looked in {windows_dir})")
            summary_rows.append({
                "prototype_id": pid,
                "file": fname,
                "found_path": "",
                "nearest_dist": nearest_dist,
                "viol_count_total": viol_count,
                "viol_sev_total": viol_sev,
                "status": "MISSING"
            })
            continue

        try:
            wdf = pd.read_csv(fpath)
        except Exception as e:
            print(f"[warn] failed reading {fpath}: {e}")
            summary_rows.append({
                "prototype_id": pid,
                "file": fname,
                "found_path": str(fpath),
                "nearest_dist": nearest_dist,
                "viol_count_total": viol_count,
                "viol_sev_total": viol_sev,
                "status": f"READ_ERROR:{e}"
            })
            continue

        # Save the medoid time series
        out_csv = med_dir / f"proto_{pid}.csv"
        wdf.to_csv(out_csv, index=False)

        # Optional plot
        if args.plot:
            if plt is None:
                print("[warn] matplotlib not available; skipping plots.")
            else:
                try:
                    plt.figure()
                    # Plot each column except obvious non-numeric indices if present
                    cols = [c for c in wdf.columns if pd.api.types.is_numeric_dtype(wdf[c])]
                    if not cols:
                        cols = [c for c in wdf.columns if c.lower() not in {"time", "t", "index"}]
                    for c in cols:
                        plt.plot(wdf[c], label=str(c))
                    plt.title(f"Prototype {pid} • {Path(fname).name}")
                    plt.xlabel("timestep")
                    plt.ylabel("value")
                    plt.legend(loc="best", fontsize=8)
                    plt.tight_layout()
                    plt.savefig(med_dir / f"proto_{pid}.png", dpi=160)
                    plt.close()
                except Exception as e:
                    print(f"[warn] plotting failed for proto {pid}: {e}")

        summary_rows.append({
            "prototype_id": pid,
            "file": fname,
            "found_path": str(fpath),
            "nearest_dist": nearest_dist,
            "viol_count_total": viol_count,
            "viol_sev_total": viol_sev,
            "status": "OK"
        })

    summary = pd.DataFrame(summary_rows).sort_values("prototype_id")
    summary_out = out_dir / "medoids_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"Saved medoids summary → {summary_out}")
    print(f"Medoid CSVs in        → {med_dir}")
    if args.plot and plt is not None:
        print(f"Medoid PNGs in        → {med_dir}")

if __name__ == "__main__":
    main()
