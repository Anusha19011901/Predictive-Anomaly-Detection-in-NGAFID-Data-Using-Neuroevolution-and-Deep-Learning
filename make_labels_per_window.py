#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# 1) Windows → prototype mapping
win_labels_path = Path("outputs/labels/labels_window.csv")
win_df = pd.read_csv(win_labels_path)
# expect columns: ['file', 'subsequence_id', 'prototype_id', ... ]

# 2) DBSCAN prototype → cluster label
# pick the eps you care about (example: eps=1.0)
assign_path = Path("outputs/dbscan/assignments_eps1.0.csv")
assign_df = pd.read_csv(assign_path)

# add prototype_id as the index (0,1,2,...) so we can join
assign_df = assign_df.reset_index().rename(
    columns={"index": "prototype_id", "label": "dbscan_label"}
)

# 3) Join: window → prototype → dbscan_label
merged = win_df.merge(
    assign_df[["prototype_id", "dbscan_label"]],
    on="prototype_id",
    how="left"
)

# sanity check
print(merged[["file", "prototype_id", "dbscan_label"]].head())

# 4) Save in the place your diagnostics script expects
out_dir = Path("outputs/dbscan_eps2.1_run")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "labels_per_window.csv"
merged.to_csv(out_path, index=False)
print(f"wrote {out_path}")
