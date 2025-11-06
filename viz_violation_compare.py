# save as viz_violation_compare.py and run: python3 viz_violation_compare.py
import glob, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

anom = pd.concat([pd.read_parquet(p) for p in glob.glob("outputs_alt/explanations/*.parquet")], ignore_index=True)
norm = pd.concat([pd.read_parquet(p) for p in glob.glob("outputs_alt/explanations_normal/*.parquet")], ignore_index=True)

anom["label"] = 1
norm["label"] = 0
df = pd.concat([anom, norm], ignore_index=True)

print("Counts:", df["label"].value_counts().to_dict())
print("Violation summary (by class):\n", df.groupby("label")["violation_L1"].describe())

# AUROC / AUPRC using violation_L1 as anomaly score
y = df["label"].values
s = df["violation_L1"].values
print("AUROC =", roc_auc_score(y, s))
print("AUPRC =", average_precision_score(y, s))

# Simple visualization
plt.figure(figsize=(6,4))
for lab, col in [(0,"tab:blue"), (1,"tab:orange")]:
    sub = df.loc[df.label==lab, "violation_L1"].clip(upper=np.percentile(df["violation_L1"], 99))
    plt.hist(sub, bins=50, alpha=0.5, label=("normal" if lab==0 else "anomaly"))
plt.xlabel("violation_L1"); plt.ylabel("count"); plt.legend(); plt.tight_layout()
plt.savefig("outputs_alt/viz/violation_hist.png", dpi=180)
