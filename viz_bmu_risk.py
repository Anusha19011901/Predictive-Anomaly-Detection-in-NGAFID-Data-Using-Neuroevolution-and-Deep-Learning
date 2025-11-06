# python3 viz_bmu_risk.py
import glob, pandas as pd, numpy as np, matplotlib.pyplot as plt

def load_dir(dir_, label):
    df = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{dir_}/*.parquet")], ignore_index=True)
    df["label"] = label
    return df

anom = load_dir("outputs_alt/explanations", 1)
norm = load_dir("outputs_alt/explanations_normal", 0)
df = pd.concat([anom, norm], ignore_index=True)

summary = df.groupby("prototype_id").agg(
    n=("label","size"),
    n_anom=("label","sum"),
    mean_violation=("violation_L1","mean"),
)
summary["anom_rate"] = summary["n_anom"] / summary["n"]
summary = summary.sort_values("anom_rate", ascending=False)
print(summary.head(15))

plt.figure(figsize=(8,3))
plt.plot(summary.index, summary["anom_rate"], ".", ms=6)
plt.xlabel("prototype_id"); plt.ylabel("anomaly rate"); plt.tight_layout()
plt.savefig("outputs_alt/viz/bmu_anom_rate.png", dpi=180)
