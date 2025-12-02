#!/usr/bin/env python3
import argparse
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbscan_labels", required=True)
    ap.add_argument("--examm_errors", required=True)
    ap.add_argument("--ocsvm_scores", required=True)
    ap.add_argument("--out_csv", required=True)
    return ap.parse_args()

def main():
    args = parse_args()

    # Load
    df_db = pd.read_csv(args.dbscan_labels)
    df_db["filename"] = df_db["filename"].astype(str)

    df_ex = pd.read_csv(args.examm_errors)
    df_ex.rename(columns={"file":"filename"}, inplace=True)

    df_oc = pd.read_csv(args.ocsvm_scores)

    # Merge on window_idx
    df = df_db.merge(df_oc, on="window_idx", how="left")

    # Merge EXAMM
    df = df.merge(df_ex, on="filename", how="left")

    # ------------------------------
    # HYBRID LABELING
    # ------------------------------
    
    # EXAMM flag
    df["examm_flag"] = (
        (df["viol_count_total"].fillna(0) >= 5) |
        (df["viol_sev_total"].fillna(0) >= 10)
    ).astype(int)

    # OcSVM flag
    df["ocsvm_flag"] = (df["ocsvm_score"] < 0).astype(int)

    # DBSCAN flag (only if supported by ocsvm/examm)
    df["dbscan_flag"] = (
        (df["dbscan_label"] == -1) &
        ((df["ocsvm_flag"] == 1) | (df["examm_flag"] == 1))
    ).astype(int)

    # Final hybrid label
    df["hybrid_label"] = (
        (df["examm_flag"] == 1) |
        (df["ocsvm_flag"] == 1) |
        (df["dbscan_flag"] == 1)
    ).astype(int)

    # Save
    df.to_csv(args.out_csv, index=False)
    print("[OK] Wrote hybrid supervised dataset →", args.out_csv)
    print(df["hybrid_label"].value_counts())

if __name__ == "__main__":
    main()
