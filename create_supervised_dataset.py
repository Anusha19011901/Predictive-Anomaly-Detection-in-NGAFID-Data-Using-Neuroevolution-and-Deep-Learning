#!/usr/bin/env python3
import argparse
import pandas as pd

##
# @brief Parse command-line arguments for generating hybrid anomaly labels.
#
# This function defines the required input CSVs:
# - DBSCAN labels
# - EXAMM error metrics
# - OC-SVM scores
# and the output CSV to write the final hybrid labels.
#
# @return argparse.Namespace Parsed command-line arguments.
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbscan_labels", required=True,
                    help="CSV file with DBSCAN cluster labels per window.")
    ap.add_argument("--examm_errors", required=True,
                    help="CSV containing EXAMM-derived violation counts/severity.")
    ap.add_argument("--ocsvm_scores", required=True,
                    help="CSV containing OC-SVM anomaly scores per window.")
    ap.add_argument("--out_csv", required=True,
                    help="Output CSV path for hybrid anomaly labels.")
    return ap.parse_args()


##
# @brief Main entry point for creating hybrid anomaly labels.
#
# This script loads DBSCAN, EXAMM, and OC-SVM outputs and merges them
# by window index and filename to produce a unified hybrid anomaly label.
#
# Hybrid labeling rules:
# - @b examm_flag = 1 if EXAMM violation counts/severity exceed thresholds.
# - @b ocsvm_flag = 1 if OC-SVM score < 0 (outlier).
# - @b dbscan_flag = 1 only if DBSCAN marks noise (-1) AND either EXAMM or OC-SVM agrees.
# - @b hybrid_label = 1 if any of the above flags activate.
#
# The output CSV contains all merged fields and the new labels.
#
# @return None
def main():
    args = parse_args()

    # ------------------------------
    # Load input files
    # ------------------------------
    df_db = pd.read_csv(args.dbscan_labels)
    df_db["filename"] = df_db["filename"].astype(str)

    df_ex = pd.read_csv(args.examm_errors)
    df_ex.rename(columns={"file": "filename"}, inplace=True)

    df_oc = pd.read_csv(args.ocsvm_scores)

    # ------------------------------
    # Merge DBSCAN + OC-SVM on window index
    # ------------------------------
    df = df_db.merge(df_oc, on="window_idx", how="left")

    # Merge EXAMM using filename
    df = df.merge(df_ex, on="filename", how="left")

    # ------------------------------
    # HYBRID LABELING
    # ------------------------------

    # EXAMM-based anomaly flag
    df["examm_flag"] = (
        (df["viol_count_total"].fillna(0) >= 5) |
        (df["viol_sev_total"].fillna(0) >= 10)
    ).astype(int)

    # OC-SVM-based anomaly flag
    df["ocsvm_flag"] = (df["ocsvm_score"] < 0).astype(int)

    # DBSCAN-based anomaly flag
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

    # ------------------------------
    # Save output CSV
    # ------------------------------
    df.to_csv(args.out_csv, index=False)
    print("[OK] Wrote hybrid supervised dataset →", args.out_csv)
    print(df["hybrid_label"].value_counts())


if __name__ == "__main__":
    main()
