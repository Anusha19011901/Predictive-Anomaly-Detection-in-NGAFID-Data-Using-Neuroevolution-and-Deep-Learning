#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

CSV = "outputs/dbscan_eps2.1_run/explanations_error_eps4.0.csv"  # <-- adjust if needed
OUT = Path("outputs/dbscan_eps2.1_run/analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# Basic sanity
print("Rows:", len(df))
print("Prototype IDs:", sorted(df["prototype_id"].unique()))
print()

# Distribution by prototype
dist = (df.groupby("prototype_id")
          .agg(n=("file","count"),
               mean_dist=("nearest_dist","mean"),
               mean_viol_cnt=("viol_count_total","mean"),
               mean_viol_sev=("viol_sev_total","mean"))
          .sort_values("n", ascending=False))
dist.to_csv(OUT/"distribution_by_prototype.csv")
print("Saved:", OUT/"distribution_by_prototype.csv")

# Top anomalies (by severity and by distance)
top_sev = df.sort_values("viol_sev_total", ascending=False).head(50)
top_dst = df.sort_values("nearest_dist", ascending=False).head(50)
top_cnt = df.sort_values("viol_count_total", ascending=False).head(50)

top_sev.to_csv(OUT/"top50_by_severity.csv", index=False)
top_dst.to_csv(OUT/"top50_by_distance.csv", index=False)
top_cnt.to_csv(OUT/"top50_by_violation_count.csv", index=False)

print("Saved:", OUT/"top50_by_severity.csv")
print("Saved:", OUT/"top50_by_distance.csv")
print("Saved:", OUT/"top50_by_violation_count.csv")

# Per-prototype anomaly exemplars
# (grab top 5 highest-severity windows per prototype)
exemplars = (df.sort_values(["prototype_id","viol_sev_total"], ascending=[True, False])
               .groupby("prototype_id")
               .head(5))
exemplars.to_csv(OUT/"exemplars_per_prototype.csv", index=False)
print("Saved:", OUT/"exemplars_per_prototype.csv")
