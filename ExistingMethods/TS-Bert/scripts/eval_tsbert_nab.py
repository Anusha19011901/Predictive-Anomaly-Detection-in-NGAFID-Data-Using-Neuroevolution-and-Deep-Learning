import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

import matplotlib.pyplot as plt


def load_scores(csv_path):
    df = pd.read_csv(csv_path)

    # Try to infer column names
    # Adjust these if your infer_export wrote different names
    label_col_candidates = ["label", "y_true", "target"]
    score_col_candidates = ["prob", "score", "logit"]

    label_col = next((c for c in label_col_candidates if c in df.columns), None)
    score_col = next((c for c in score_col_candidates if c in df.columns), None)

    if label_col is None:
        raise ValueError(f"Could not find label column in {df.columns}")
    if score_col is None:
        raise ValueError(f"Could not find score column in {df.columns}")

    y_true = df[label_col].values.astype(int)
    scores = df[score_col].values.astype(float)

    print(f"Using label column: {label_col}")
    print(f"Using score column: {score_col}")
    print(f"y_true distribution: {np.bincount(y_true)}")
    return y_true, scores, df


def compute_metrics(y_true, scores):
    from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f1s)

    best_f1 = f1s[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.3

    # F1, precision, recall @ 0.5
    preds05 = (scores >= 0.5).astype(int)
    f1_05 = f1_score(y_true, preds05)
    precision_05 = precision_score(y_true, preds05, zero_division=0)
    recall_05 = recall_score(y_true, preds05, zero_division=0)

    metrics = {
        "AUROC": auroc,
        "AUPRC": auprc,

        "Best_F1": best_f1,
        "Best_precision": best_precision,
        "Best_recall": best_recall,
        "Best_threshold": best_thresh,

        "F1_at_0.5": f1_05,
        "Precision_at_0.5": precision_05,
        "Recall_at_0.5": recall_05,
    }
    return metrics, (precisions, recalls)



def plot_curves(y_true, scores, outdir):
    os.makedirs(outdir, exist_ok=True)

    # ROC
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("TS-BERT ROC (NAB test)")
    plt.grid(True)
    roc_path = os.path.join(outdir, "tsbert_nab_test_roc.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    # PR
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("TS-BERT Precision-Recall (NAB test)")
    plt.grid(True)
    pr_path = os.path.join(outdir, "tsbert_nab_test_pr.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()

    return roc_path, pr_path


if __name__ == "__main__":
    csv_path = "ExistingMethods/TS-Bert/outputs/tsbert_scores_nab.csv"
    outdir = "ExistingMethods/TS-Bert/outputs"

    y_true, scores, df = load_scores(csv_path)
    metrics, (precisions, recalls) = compute_metrics(y_true, scores)
    roc_path, pr_path = plot_curves(y_true, scores, outdir)

    print("\n=== TS-BERT on NAB (test) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"\nROC curve saved to: {roc_path}")
    print(f"PR curve saved to:  {pr_path}")
