# viz_timeline.py
import pandas as pd
import matplotlib.pyplot as plt

scores = pd.read_csv("outputs/ocsvm_examm_only/before_scores.csv")

# pick one flight/subseq to show (or loop)
f = scores['flight_id'].iloc[0]
s = scores.loc[scores['flight_id']==f, 'subseq_id'].iloc[0]
sub = scores[(scores.flight_id==f) & (scores.subseq_id==s)].sort_values("window_idx")

plt.figure(figsize=(10,4))
plt.plot(sub["window_idx"], sub["ocsvm_score"], lw=1.5)
anom = sub[sub["anomaly_flag"]==1]
plt.scatter(anom["window_idx"], anom["ocsvm_score"], marker="x")
plt.title(f"OC-SVM score timeline — {f} / {s}")
plt.xlabel("window_idx")
plt.ylabel("ocsvm_score (higher = more normal)")
plt.tight_layout()
plt.show()
