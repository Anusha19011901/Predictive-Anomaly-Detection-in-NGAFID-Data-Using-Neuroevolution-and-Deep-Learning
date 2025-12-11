

# **README.md**

```markdown
# 🚀 Predictive Anomaly Detection in NGAFID Data Using Neuroevolution + Machine Learning

This project develops a full anomaly detection pipeline for general aviation telemetry using **EXAMM recurrent neural networks**, **forecast-error modeling**, **OC-SVM**, and **DBSCAN prototype-based explainability**.  
The system compares *before vs. after maintenance* behavior to detect early signs of mechanical faults in NGAFID flight data.

---

# 🎯 Objective

Transform raw telemetry into **interpretable anomaly signatures** by:

- Using a pretrained EXAMM RNN to compute **t+1 forecast errors**  
- Converting each time window into an **error vector**  
- Normalizing, scoring, and clustering windows in error space  
- Explaining anomalies using **prototype envelopes**  
- Evaluating performance with the **Numenta Anomaly Benchmark (NAB)**

This pipeline identifies high-risk subsequences *long before* failures occur.

---

# 📁 Repository Structure

```

Predictive-Anomaly-Detection-in-NGAFID-Data-Using-Neuroevolution-and-Deep-Learning/
│
├── dataset/
│   ├── before/
│   ├── after/
│   ├── before_examm2/
│   └── after_examm2/
│
├── artifacts/
│   └── errors/per_window/
│
├── exact_data/anomaly/      # EXAMM-generated 30-step error windows
│
├── outputs/
│   ├── ocsvm_examm_only/
│   ├── dbscan_eps2.1_run/
│   ├── proto_diagnostics/
│   ├── noise_diagnostics/
│   ├── dbscan_vis/
│   └── supervised_dataset_windows.csv
│
└── scripts/
└── run_everything.sh    # MASTER PIPELINE SCRIPT

````

---

# ⚡ Running the Full Pipeline

The *entire* EXAMM → Error Modeling → OC-SVM → DBSCAN → Explainability → NAB pipeline is automated in:

### **scripts/run_everything.sh**

From the repository root:

```bash
chmod +x scripts/run_everything.sh
bash scripts/run_everything.sh
````

This will execute all stages:

| Step | Description                                          |
| ---- | ---------------------------------------------------- |
| 0    | Generate EXAMM MAE tables (per-window error vectors) |
| 1    | Run OC-SVM on EXAMM errors                           |
| 2    | Build DBSCAN-based labels per window                 |
| 3    | Generate prototype-box explanations                  |
| 4    | Build hybrid supervised dataset                      |
| 5    | Prototype diagnostics (cluster-level analysis)       |
| 6    | Noise diagnostics (outlier analysis)                 |
| 7    | DBSCAN cluster visualizations                        |
| 8    | Apply DBSCAN explanations to BEFORE flights          |
| 9    | NAB benchmark evaluation                             |

All outputs appear in the `outputs/` directory.

---

# 🧠 Methodology Overview

## **1. Forecast Error Modeling with EXAMM**

A pretrained EXAMM RNN produces **next-step predictions**:

[
\hat{x}*{t+1} = f(x*{t-W+1:t})
]

Each window forms an **error matrix**:

[
e_t = |x_{t+1} - \hat{x}_{t+1}|
]

These errors capture deviations from expected flight dynamics.

---

## **2. Z-Normalization of Errors**

Z-scores are computed **using BEFORE flights only**, forming a stable baseline of normal error magnitude.

---

## **3. OC-SVM on Error Vectors**

OC-SVM learns a boundary of normality using z-normalized EXAMM error windows.

Outputs:

* A continuous anomaly score
* Top contributing error dimensions
* Drift relative to BEFORE baseline

(OC-SVM is used for *characterization*, not filtering.)

---

## **4. Density-Based Clustering (DBSCAN)**

DBSCAN clusters windows with similar error shapes:

* Dense clusters → **consistent error patterns**
* Label = -1 → **true outlier windows**

Cluster prototypes are computed as mean error signatures.

---

## **5. Prototype-Box Explanations**

For each prototype:

* Compute per-feature percentile envelopes (e.g., 5–95%)
* Identify violations when a window exceeds bounds
* Summarize anomaly severity with:

  * `viol_count_total`
  * `viol_sev_total`
  * Nearest prototype ID

Produces human-interpretable anomaly reasons.

---

## **6. Noise Diagnostics**

DBSCAN noise windows undergo deeper inspection:

* Top features by |z|
* Heatmaps for distortion severity
* PCA and UMAP embeddings
* Raw trace plots for representative windows

---

## **7. NAB Benchmark**

Evaluates final anomaly scoring using:

* ROC-AUC
* PR-AUC
* Threshold sweeps (best F1, F2, K anomalies)
* Early-detection penalty metrics

---

# 🧪 Supervised Dataset Generation

The script produces:

```
outputs/supervised_dataset_windows.csv
```

This unified dataset includes:

* EXAMM z-normalized error features
* DBSCAN cluster labels
* Prototype violation metrics
* OC-SVM anomaly scores
* Hybrid anomaly label

Can be used for:

* Random Forests
* Gradient Boosting
* Neural anomaly classifiers
* SHAP-based feature attribution

---

# ⚙️ Installation & Environment

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Ensure **Python 3.10–3.11** is used.

---

# 👩‍💻 Authors

* **Anusha Seshadri**
* **Iyashi Pal**
* **Dr. Travis Desell** — Faculty Advisor

---

