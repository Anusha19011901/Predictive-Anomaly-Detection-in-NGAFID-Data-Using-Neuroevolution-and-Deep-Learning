import os, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib

# --------- config (change if your paths differ) ----------
NPZ_PATH   = "outputs_alt/after_windows_alt.npz"              # your training windows
SCALER_PKL = "outputs_alt/scaler_alt.pkl"
SOM_NPZ    = "outputs_alt/som/prototypes_som_5x5_alt.npz"     # <-- update if your file name differs
EXPL_ANOM  = "outputs_alt/explanations/*.parquet"             # anomaly explanations (optional)
EXPL_NORM  = "outputs_alt/explanations_normal/*.parquet"      # normal explanations (optional)
VIZ_DIR    = "outputs_alt/viz"

SENSORS = ["AltMSL","IAS","NormAc","E1 RPM","E1 FFlow","E1 CHT1","E1 EGT1"]
WIN = 30  # you trained with --win 30
# ---------------------------------------------------------

os.makedirs(VIZ_DIR, exist_ok=True)

# --- load training set + scale ---
D = np.load(NPZ_PATH)
X = D["X"]                                   # (n_windows, WIN*len(SENSORS))
scaler = joblib.load(SCALER_PKL)
Xs = scaler.transform(X)

# --- load SOM prototypes, infer grid ---
S = np.load(SOM_NPZ)
h = int(S["h"]); w = int(S["w"]); d = int(S["d"])
protos = S["weights"].reshape(h*w, d)   # (M*M, dim)
M = h
assert M*M == protos.shape[0], "Prototypes not square grid?"


# --- BMU mapping for training windows ---
d2 = ((Xs[:,None,:] - protos[None,:,:])**2).sum(-1)
bmu = d2.argmin(axis=1)

# --- hits per prototype + heatmap ---
hits = np.bincount(bmu, minlength=protos.shape[0]).reshape(M, M)

plt.figure(figsize=(6,6))
im = plt.imshow(hits, cmap="viridis", origin="upper")
plt.title(f"SOM Hit Heatmap ({M}×{M})")
plt.xlabel("column"); plt.ylabel("row")
plt.colorbar(im, label="# training windows mapped")
plt.tight_layout()
heatmap_path = f"{VIZ_DIR}/hitmap_{M}x{M}.png"
plt.savefig(heatmap_path, dpi=220)
plt.close()
print("Saved:", heatmap_path)

# --- table of prototypes by hits (descending) ---
hit_series = pd.Series(hits.ravel(), name="hit_count")
hit_df = hit_series.rename_axis("prototype_id").reset_index().sort_values("hit_count", ascending=False)
hit_csv = f"{VIZ_DIR}/prototype_hits_desc.csv"
hit_df.to_csv(hit_csv, index=False)
print("Saved:", hit_csv)
print("Top 10 prototypes by hits:\n", hit_df.head(10).to_string(index=False))

# --- bar chart of hits (sorted) ---
plt.figure(figsize=(10,4))
plt.bar(np.arange(len(hit_df)), hit_df["hit_count"].values)
plt.title("Prototype Hit Counts (sorted desc)")
plt.xlabel("ranked prototype"); plt.ylabel("hit count")
plt.tight_layout()
bar_path = f"{VIZ_DIR}/prototype_hits_bar.png"
plt.savefig(bar_path, dpi=220)
plt.close()
print("Saved:", bar_path)

# --- prototype gallery: time×sensor heatmaps for top-K prototypes ---
# reshape prototypes to (WIN, n_sensors)
n_sens = len(SENSORS)
assert d == WIN*n_sens, f"Dim mismatch: d={d} vs WIN*n_sens={WIN*n_sens}"
P = protos.reshape(M*M, WIN, n_sens)


K = min(12, M*M)  # show up to 12 most-used prototypes
top_ids = hit_df.head(K)["prototype_id"].tolist()

cols = 4
rows = int(np.ceil(K/cols))
fig, axes = plt.subplots(rows, cols, figsize=(3.4*cols, 2.8*rows), squeeze=False)

for ax, pid in zip(axes.ravel(), top_ids):
    H = P[pid]                                  # (WIN, n_sensors)
    # z-score per sensor column for contrast (across time)
    Hz = (H - H.mean(axis=0, keepdims=True)) / (H.std(axis=0, keepdims=True) + 1e-9)
    im = ax.imshow(Hz.T, aspect="auto", origin="lower")
    ax.set_title(f"proto {pid} (hits={int(hit_series[pid])})", fontsize=9)
    ax.set_yticks(range(n_sens)); ax.set_yticklabels(SENSORS, fontsize=7)
    ax.set_xticks([0, WIN//2, WIN-1]); ax.set_xticklabels([0, WIN//2, WIN-1], fontsize=7)
for ax in axes.ravel()[len(top_ids):]:
    ax.axis("off")
fig.suptitle("Top prototypes (time × sensor, z-scored per sensor)")
plt.tight_layout()
gallery_path = f"{VIZ_DIR}/prototype_gallery_top{K}.png"
plt.savefig(gallery_path, dpi=220)
plt.close()
print("Saved:", gallery_path)

# --- OPTIONAL: use explanations to compute risk per prototype ---
def load_parquets(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print("Skip (read error):", f, e)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

# Expect columns from som_boxes.py explain: ['prototype_id','violation_L1','label', ...]
E_anom = load_parquets(EXPL_ANOM)
E_norm = load_parquets(EXPL_NORM)
E = None
if E_anom is not None and E_norm is not None:
    E = pd.concat([E_anom.assign(source="anom"),
                   E_norm.assign(source="norm")], ignore_index=True)
elif E_anom is not None:
    E = E_anom.assign(source="anom")
elif E_norm is not None:
    E = E_norm.assign(source="norm")

if E is not None and "prototype_id" in E.columns:
    # If no 'label' column, synthesize from source
    if "label" not in E.columns:
        E["label"] = (E["source"]=="anom").astype(int)

    grp = E.groupby("prototype_id").agg(
        n=("prototype_id","size"),
        n_anom=("label","sum"),
        mean_violation=("violation_L1","mean")
    ).reset_index()
    grp["anom_rate"] = grp["n_anom"] / grp["n"]

    # save table
    risk_csv = f"{VIZ_DIR}/prototype_risk_table.csv"
    grp.sort_values("anom_rate", ascending=False).to_csv(risk_csv, index=False)
    print("Saved:", risk_csv)

    # bar: anomaly rate per prototype (only those observed)
    plt.figure(figsize=(10,4))
    gsort = grp.sort_values("anom_rate", ascending=False)
    plt.bar(range(len(gsort)), gsort["anom_rate"].values)
    plt.title("Prototype anomaly rate (observed in explanations)")
    plt.xlabel("ranked prototype"); plt.ylabel("anomaly rate")
    plt.tight_layout()
    out1 = f"{VIZ_DIR}/prototype_anom_rate_bar.png"
    plt.savefig(out1, dpi=220); plt.close(); print("Saved:", out1)

    # bar: mean violation per prototype
    plt.figure(figsize=(10,4))
    gsort2 = grp.sort_values("mean_violation", ascending=False)
    plt.bar(range(len(gsort2)), gsort2["mean_violation"].values)
    plt.title("Prototype mean L1 violation (observed in explanations)")
    plt.xlabel("ranked prototype"); plt.ylabel("mean violation")
    plt.tight_layout()
    out2 = f"{VIZ_DIR}/prototype_mean_violation_bar.png"
    plt.savefig(out2, dpi=220); plt.close(); print("Saved:", out2)
else:
    print("No explanations found (or missing columns); skipping risk/violation plots.")
