# viz_top_contributors.py
import pandas as pd
import matplotlib.pyplot as plt

contrib = pd.read_csv("outputs/ocsvm_examm_only/before_topk_contributors.csv")
row = contrib.iloc[0]  # choose a specific window you care about

pairs = []
j = 1
while f"top{j}_feature" in row:
    pairs.append((row[f"top{j}_feature"], abs(float(row[f"top{j}_z"]))))
    j += 1

features, mags = zip(*pairs)
plt.figure(figsize=(7,3))
plt.barh(features[::-1], mags[::-1])
plt.title(f"Top |z| contributors — {row.flight_id} / {row.subseq_id} / w{row.window_idx}")
plt.xlabel("|z| (standardized EXAMM error)")
plt.tight_layout()
plt.show()
