#!/usr/bin/env python3
"""
@file noise_windows_diagnostics.py
@brief Comprehensive diagnostics for DBSCAN noise windows (label = -1 by default).

This script analyzes windows assigned to the *noise* cluster in DBSCAN runs.  
Noise windows often correspond to anomalous or rarely observed behaviors, and this tool
summarizes, ranks, and visualizes their statistical deviation from normal (clustered) windows.

It provides a full diagnostic package including feature extraction, z-scores, embeddings,
top-feature rankings, raw trace visualization, and exportable tables — enabling a deeper
understanding of what makes these windows “anomalous.”

-------------------------------------------------------------------------------
Core Workflow
-------------------------------------------------------------------------------
1. **Load Window Labels**
   - Reads a CSV mapping window identifiers → DBSCAN labels.
   - Automatically detects label and filename/id columns.
   - Matches label rows to actual window CSV files via filename stems.

2. **Build Per-Window Feature Vectors**
   For each window CSV:
   - Remove time-like columns (regex-based)
   - Drop sparse features (< min_non_nan_ratio)
   - Impute remaining NaNs with per-feature median
   - Compute summary statistics per feature:
         mean, std, min, max
   - Creates a flattened per-window descriptor vector.

3. **Z-Score Normalization**
   - Clustered windows (label != noise_label) define the *baseline distribution*
   - Noise windows are z-scored relative to this baseline across all features
   - If no clustered windows exist, use all windows as fallback baseline

4. **Ranking Important Noise Features**
   - Computes mean |z| across noise windows
   - Selects Top-K highest-deviation features
   - Generates:
       • bar plot of feature importance
       • heatmap of top-K z-scores
       • histograms comparing noise vs clustered distributions

5. **Projection / Embedding Visualizations**
   - PCA 2-D embedding of all windows, highlighting noise
   - UMAP projection (if installed)

6. **Raw Signal Trace Export (optional)**
   - Saves example time-series plots of selected noise windows
   - Uses the most variant numeric columns to highlight unusual dynamics

7. **Exports**
   All results are written under `out_dir`:
   - figs/
       • topk_noise_bar.png  
       • noise_heatmap_topk.png  
       • windows_pca.png  
       • windows_umap.png (optional)  
       • top_feature_distributions.png  
       • trace_*.png  
   - tables/
       • noise_windows_summary.csv  
   - matrices/
       • window_features.parquet  
       • window_features_z.parquet  
   - README_noise_diagnostics.txt summarizing inputs & outputs

-------------------------------------------------------------------------------
Intended Use
-------------------------------------------------------------------------------
Use this script to:
- Investigate DBSCAN noise windows after clustering
- Identify unusual sensor behaviors
- Support explainability for anomaly detection pipelines
- Diagnose mislabeled or structurally different windows
- Select stable features for hybrid anomaly labeling

This tool is typically run after a DBSCAN sweep and before final OC-SVM / hybrid
anomaly scoring, providing interpretability and insight into anomalous flight behavior.
"""

# -*- coding: utf-8 -*-

import argparse
import sys
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optional UMAP
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ----------------- CLI -----------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Diagnostics for DBSCAN 'noise' windows (label = -1 by default)."
    )
    ap.add_argument("--windows_dir", required=True, help="Directory with per-window CSVs")
    ap.add_argument("--labels_csv", required=True, help="CSV mapping windows -> labels")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--noise_label", type=int, default=-1, help="Label used for noise")
    ap.add_argument("--topk", type=int, default=20, help="Top-K features for visuals")
    ap.add_argument("--max_windows", type=int, default=100000,
                    help="Max windows to process (safety)")
    ap.add_argument("--id_col", default=None,
                    help="Explicit id/filename column in labels CSV")
    ap.add_argument("--label_col", default=None,
                    help="Explicit label column (e.g., dbscan_label)")
    ap.add_argument("--time_cols_regex", default=r"^(time|timestamp)$",
                    help="Regex for time-like columns to drop")
    ap.add_argument("--min_non_nan_ratio", type=float, default=0.7,
                    help="Keep feature if >= this non-NaN ratio per window")
    ap.add_argument("--save_sample_traces", action="store_true",
                    help="Save a few raw signal traces for noise windows")
    ap.add_argument("--sample_traces_n", type=int, default=12,
                    help="How many noise windows to render as traces")
    return ap.parse_args()


# ----------------- IO helpers -----------------

def ensure_out_dirs(base: Path):
    figs = base / "figs"
    tables = base / "tables"
    mats = base / "matrices"
    for p in (base, figs, tables, mats):
        p.mkdir(parents=True, exist_ok=True)
    return figs, tables, mats


def load_labels(labels_csv: Path, id_col=None, label_col=None):
    df = pd.read_csv(labels_csv)

    # label column
    cand_label_cols = [label_col] if label_col else [
        "label", "cluster", "prototype_id", "proto", "dbscan_label"
    ]
    label_c = next((c for c in cand_label_cols if c in df.columns), None)
    if label_c is None:
        raise ValueError(f"Could not find label column in {labels_csv}. "
                         f"Tried: {cand_label_cols}")

    # id/filename column
    cand_id_cols = [id_col] if id_col else [
        "window_id", "filename", "file", "path", "window_csv", "name", "id"
    ]
    id_c = next((c for c in cand_id_cols if c in df.columns), None)
    if id_c is None:
        df = df.reset_index(names="index_id")
        id_c = "index_id"
        warnings.warn("No id/filename column found; using row index as id.")

    def stem_from_val(val):
        s = str(val)
        s = s.replace("\\", "/").split("/")[-1]
        return s[:-4] if s.lower().endswith(".csv") else s

    df["__id_raw__"] = df[id_c].astype(str)
    df["__stem__"] = df["__id_raw__"].map(stem_from_val)
    return df, id_c, label_c


def index_window_files(windows_dir: Path, max_files=10**7):
    files = []
    for p in windows_dir.rglob("*.csv"):
        files.append(p)
        if len(files) >= max_files:
            break

    file_df = pd.DataFrame({"path": files})

    # ALWAYS convert to pure Python strings (even if path or stem is weird)
    file_df["stem"] = file_df["path"].apply(lambda p: str(Path(p).stem))
    file_df["stem"] = file_df["stem"].astype(str)   # <--- THE FIX

    file_df["stem_lc"] = file_df["stem"].str.lower()
    return file_df



def match_labels_to_files(labels_df: pd.DataFrame, file_df: pd.DataFrame):
    L = labels_df.copy()
    L["__stem_lc__"] = L["__stem__"].str.lower()

    merged = L.merge(file_df, left_on="__stem_lc__", right_on="stem_lc", how="left")
    unmatched = merged[merged["path"].isna()]

    if len(unmatched) > 0:
        warnings.warn(f"{len(unmatched)} label rows did not match any file by exact stem. "
                      "Trying a simple substring heuristic...")
        stems = file_df["stem_lc"].tolist()
        paths = file_df["path"].tolist()

        def fuzzy_find(st):
            st = str(st)
            for s, p in zip(stems, paths):
                if st in s or s in st:
                    return p
            return np.nan

        idx = merged["path"].isna()
        merged.loc[idx, "path"] = merged.loc[idx, "__stem_lc__"].map(fuzzy_find)

    still = merged[merged["path"].isna()]
    if len(still) > 0:
        warnings.warn(f"{len(still)} label rows could not be matched to CSVs; dropping.")
        merged = merged[~merged["path"].isna()].copy()

    return merged


# ----------------- Feature building -----------------

def summarize_window_csv(csv_path: Path, time_cols_regex, min_non_nan_ratio=0.7):
    try:
        W = pd.read_csv(csv_path)
    except Exception as e:
        return None, f"read_error: {e}"

    time_cols = [c for c in W.columns
                 if re.match(time_cols_regex, str(c), flags=re.I)]
    W = W.drop(columns=time_cols, errors="ignore")

    num_cols = W.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return None, "no_numeric_columns"

    Wn = W[num_cols]
    keep = (Wn.notna().sum(axis=0) / max(1, len(Wn))) >= min_non_nan_ratio
    Wn = Wn.loc[:, keep]
    if Wn.shape[1] == 0:
        return None, "all_numeric_cols_sparse"

    Wn = Wn.fillna(Wn.median(numeric_only=True))

    feats = {}
    means = Wn.mean(axis=0)
    stds = Wn.std(axis=0, ddof=0).replace(0, 1e-12)
    mins = Wn.min(axis=0)
    maxs = Wn.max(axis=0)

    for c in Wn.columns:
        feats[f"{c}__mean"] = float(means[c])
        feats[f"{c}__std"] = float(stds[c])
        feats[f"{c}__min"] = float(mins[c])
        feats[f"{c}__max"] = float(maxs[c])

    return pd.Series(feats), None


def build_window_matrix(merged_df: pd.DataFrame, label_col: str,
                        time_cols_regex: str, min_non_nan_ratio: float,
                        max_windows: int):
    rows = []
    errors = []

    for _, row in merged_df.head(max_windows).iterrows():
        vec, err = summarize_window_csv(row["path"], time_cols_regex, min_non_nan_ratio)
        if vec is None:
            errors.append((row["path"], err))
            continue

        rec = {
            "window_stem": row["stem"],
            "label": row[label_col],
            "path": str(row["path"]),
        }
        rows.append(pd.concat([pd.Series(rec), vec]))

    if errors:
        print(f"[INFO] Skipped {len(errors)} windows due to issues. First 5:",
              file=sys.stderr)
        for p, e in errors[:5]:
            print(f" - {p}: {e}", file=sys.stderr)

    if not rows:
        raise RuntimeError("No valid windows summarized. "
                           "Check windows_dir, labels mapping, or CSV contents.")

    M = pd.DataFrame(rows)
    meta_cols = ["window_stem", "label", "path"]
    feat_cols = [c for c in M.columns if c not in meta_cols]
    return M, meta_cols, feat_cols


# ----------------- Z-scores + ranking -----------------

def compute_zscores(M: pd.DataFrame, feat_cols, label_col="label", noise_label=-1):
    """
    Compute z-scores per feature.

    If there are clustered windows (label != noise_label), use them as the
    baseline. If *all* windows are noise, fall back to using all windows
    as the baseline instead of crashing.
    """
    clustered = M[M[label_col] != noise_label]

    if clustered.empty:
        # No clustered windows – everything is noise.
        # Use all windows as the baseline so we can still compute z-scores.
        print("[INFO] No clustered windows found; using ALL windows as baseline "
              "for z-scores.", file=sys.stderr)
        base = M
    else:
        base = clustered

    mu = base[feat_cols].mean(axis=0)
    sd = base[feat_cols].std(axis=0, ddof=0).replace(0, 1e-12)

    Z = (M[feat_cols] - mu) / sd
    Z = Z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Mz = pd.concat(
        [M[["window_stem", "label", "path"]].reset_index(drop=True),
         Z.reset_index(drop=True)],
        axis=1,
    )
    return Mz, mu, sd



def topk_features_by_abs_z(noise_Z: pd.DataFrame, feat_cols, topk=20):
    mean_abs = noise_Z[feat_cols].abs().mean(axis=0).sort_values(ascending=False)
    return mean_abs.head(topk)


# ----------------- Plots -----------------

def plot_topk_bar(figs_dir: Path, mean_abs_series: pd.Series,
                  title="Top-K features by |z| in noise",
                  fname="topk_noise_bar.png"):
    plt.figure(figsize=(10, max(4, 0.35 * len(mean_abs_series))))
    mean_abs_series.iloc[::-1].plot(kind="barh")
    plt.xlabel("Mean |z| (noise windows)")
    plt.title(title)
    plt.tight_layout()
    out = figs_dir / fname
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_heatmap(figs_dir: Path, noise_Z: pd.DataFrame, sel_feats,
                 fname="noise_heatmap_topk.png"):
    if noise_Z.empty or not sel_feats:
        return None

    Z = noise_Z[sel_feats].copy()
    row_order = Z.abs().mean(axis=1).sort_values(ascending=False).index
    Z = Z.loc[row_order]

    plt.figure(figsize=(min(18, 1.1 * len(sel_feats)),
                        min(12, 0.25 * len(Z))))
    im = plt.imshow(Z.values, aspect="auto", cmap="coolwarm",
                    vmin=-3, vmax=3)
    plt.colorbar(im, fraction=0.025)
    plt.yticks(range(len(Z)), Z.index, fontsize=6)
    plt.xticks(range(len(sel_feats)), sel_feats, rotation=90)
    plt.title("Noise windows heatmap (z-scores, clamped to ±3)")
    plt.tight_layout()
    out = figs_dir / fname
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_embeddings(figs_dir: Path, M: pd.DataFrame, feat_cols, labels,
                    noise_label=-1, fname_prefix="windows"):
    X = StandardScaler().fit_transform(M[feat_cols].values)
    is_noise = (labels == noise_label)

    def scatter_plot(X2, kind):
        plt.figure(figsize=(7, 6))
        plt.scatter(X2[~is_noise, 0], X2[~is_noise, 1], s=8,
                    alpha=0.5, label="clustered")
        plt.scatter(X2[is_noise, 0], X2[is_noise, 1], s=10,
                    alpha=0.9, label="noise")
        plt.legend()
        plt.title(f"{kind} embedding of windows (noise highlighted)")
        plt.tight_layout()
        outp = figs_dir / f"{fname_prefix}_{kind.lower()}.png"
        plt.savefig(outp, dpi=200)
        plt.close()
        return outp

    pca = PCA(n_components=2, random_state=0)
    P = pca.fit_transform(X)
    pca_out = scatter_plot(P, "PCA")

    umap_out = None
    if HAS_UMAP:
        um = umap.UMAP(n_components=2, random_state=0,
                       n_neighbors=15, min_dist=0.1)
        U = um.fit_transform(X)
        umap_out = scatter_plot(U, "UMAP")

    return pca_out, umap_out


def plot_feature_distributions(figs_dir: Path, M: pd.DataFrame, feat_cols,
                               label_col="label", noise_label=-1,
                               top_feats=None,
                               fname="top_feature_distributions.png"):
    if not top_feats:
        return None
    feats = list(top_feats[:6])

    plt.figure(figsize=(max(10, 3 * len(feats)), 4))
    for i, f in enumerate(feats, 1):
        plt.subplot(1, len(feats), i)
        x_noise = M[M[label_col] == noise_label][f].values
        x_ok = M[M[label_col] != noise_label][f].values
        bins = 30
        plt.hist(x_ok, bins=bins, alpha=0.5, label="clustered")
        plt.hist(x_noise, bins=bins, alpha=0.8, label="noise")
        plt.title(f)
        if i == 1:
            plt.legend()
    plt.tight_layout()
    out = figs_dir / fname
    plt.savefig(out, dpi=200)
    plt.close()
    return out


# ----------------- Exports -----------------

def export_noise_table(tables_dir: Path, noise_Z: pd.DataFrame,
                       feat_cols, topk=3):
    rows = []
    for _, row in noise_Z.iterrows():
        z = row[feat_cols]
        top_feats = z.abs().sort_values(ascending=False).head(topk)
        rec = {
            "window_stem": row["window_stem"],
            "path": row["path"],
            "mean_abs_z": float(z.abs().mean()),
        }
        for j, (fname, val) in enumerate(top_feats.items(), 1):
            rec[f"top{j}_feature"] = fname
            rec[f"top{j}_z"] = float(val)
        rows.append(rec)

    out_df = pd.DataFrame(rows).sort_values("mean_abs_z", ascending=False)
    out_path = tables_dir / "noise_windows_summary.csv"
    out_df.to_csv(out_path, index=False)
    return out_path, out_df


def maybe_save_sample_traces(figs_dir: Path, noise_paths: pd.Series,
                             sample_n: int, time_cols_regex: str):
    sample = noise_paths.drop_duplicates().head(sample_n)
    saved = []

    for p in sample:
        try:
            W = pd.read_csv(p)
        except Exception:
            continue

        x = None
        time_cols = [c for c in W.columns
                     if re.match(time_cols_regex, str(c), flags=re.I)]
        if time_cols:
            x = W[time_cols[0]].values

        num_cols = W.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            continue

        var_order = W[num_cols].var(axis=0).sort_values(
            ascending=False).index.tolist()
        sel = var_order[:8]

        plt.figure(figsize=(10, 6))
        for c in sel:
            y = W[c].values
            if x is None:
                plt.plot(y, alpha=0.9, linewidth=1)
            else:
                plt.plot(x, y, alpha=0.9, linewidth=1)
        plt.title(Path(p).stem)
        plt.xlabel("time" if x is not None else "row")
        plt.ylabel("value")
        plt.tight_layout()
        out = figs_dir / f"trace_{Path(p).stem}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        saved.append(out)

    return saved


# ----------------- Main -----------------

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    figs_dir, tables_dir, mats_dir = ensure_out_dirs(out_dir)

    labels_df, id_c, label_c = load_labels(Path(args.labels_csv),
                                           args.id_col, args.label_col)

    windows_dir = Path(args.windows_dir)

    if args.id_col is not None:
        # 🚀 Fast path: we already know which column has the filename (e.g., "file")
        # Treat that column as the CSV filename and build paths directly.
        labels_df["path"] = labels_df[args.id_col].apply(
            lambda s: windows_dir / str(s)
        )
        labels_df["stem"] = labels_df["path"].apply(lambda p: Path(p).stem)
        merged = labels_df
    else:
        # Fallback: index directory and use stem-based matching
        file_df = index_window_files(windows_dir)
        merged = match_labels_to_files(labels_df, file_df)

    if merged.empty:
        raise RuntimeError("No label/file matches found. "
                           "Check --windows_dir and --labels_csv mapping.")


    M, meta_cols, feat_cols = build_window_matrix(
        merged, label_col=label_c,
        time_cols_regex=args.time_cols_regex,
        min_non_nan_ratio=args.min_non_nan_ratio,
        max_windows=args.max_windows,
    )

    Mz, mu, sd = compute_zscores(M, feat_cols,
                                 label_col="label",
                                 noise_label=args.noise_label)

    # Save matrices
    M.to_parquet(mats_dir / "window_features.parquet", index=False)
    Mz.to_parquet(mats_dir / "window_features_z.parquet", index=False)

    noise_Z = Mz[Mz["label"] == args.noise_label].copy()
    if noise_Z.empty:
        print("[INFO] No noise windows found (no rows with noise_label).",
              file=sys.stderr)
        sys.exit(0)

    topk_series = topk_features_by_abs_z(noise_Z, feat_cols, topk=args.topk)

    plot_topk_bar(figs_dir, topk_series,
                  title="Top features by |z| among noise windows",
                  fname="topk_noise_bar.png")
    plot_heatmap(figs_dir, noise_Z,
                 sel_feats=list(topk_series.index),
                 fname="noise_heatmap_topk.png")

    plot_embeddings(figs_dir, M, feat_cols,
                    labels=M["label"].values,
                    noise_label=args.noise_label,
                    fname_prefix="windows")

    plot_feature_distributions(figs_dir, M, feat_cols,
                               label_col="label",
                               noise_label=args.noise_label,
                               top_feats=list(topk_series.index),
                               fname="top_feature_distributions.png")

    summary_csv, _ = export_noise_table(tables_dir, noise_Z, feat_cols, topk=3)

    traces_saved = []
    if args.save_sample_traces:
        traces_saved = maybe_save_sample_traces(
            figs_dir, noise_Z["path"], args.sample_traces_n,
            args.time_cols_regex
        )

    # Tiny README
    readme_path = out_dir / "README_noise_diagnostics.txt"
    with open(readme_path, "w") as f:
        f.write(
            f"Noise Windows Diagnostics\n\n"
            f"Inputs:\n"
            f"- windows_dir: {args.windows_dir}\n"
            f"- labels_csv : {args.labels_csv}\n"
            f"- noise_label: {args.noise_label}\n\n"
            f"Key outputs (in {out_dir}):\n"
            f"- figs/topk_noise_bar.png\n"
            f"- figs/noise_heatmap_topk.png\n"
            f"- figs/windows_pca.png\n"
            f"- figs/windows_umap.png (if umap available)\n"
            f"- figs/top_feature_distributions.png\n"
            f"- tables/noise_windows_summary.csv\n"
            f"- matrices/window_features.parquet\n"
            f"- matrices/window_features_z.parquet\n"
        )

    print("\n=== DONE ===")
    print(f"- Summary table: {summary_csv}")
    print(f"- Figures: {figs_dir}")
    print(f"- Matrices: {mats_dir}")
    if traces_saved:
        print(f"- Saved {len(traces_saved)} sample trace PNGs")


if __name__ == "__main__":
    main()
