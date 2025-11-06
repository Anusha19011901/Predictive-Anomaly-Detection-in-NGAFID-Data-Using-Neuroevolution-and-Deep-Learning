# python3 viz_timeline_one.py exact_data/anomaly/window_2000.csv
import sys, pandas as pd, matplotlib.pyplot as plt

parq = "outputs_alt/explanations/" + sys.argv[1].split("/")[-1].replace(".csv","_explain.parquet")
df = pd.read_parquet(parq)
plt.figure(figsize=(10,2.5))
plt.plot(df["start"], df["prototype_id"], lw=1)
plt.xlabel("row index (window start)"); plt.ylabel("prototype_id")
plt.title("SOM prototype timeline: " + parq.split("/")[-1])
plt.tight_layout()
plt.savefig("outputs_alt/viz/" + parq.split("/")[-1].replace(".parquet","_timeline.png"), dpi=180)
