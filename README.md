#  Predictive Anomaly Detection in NGAFID Data Using Machine Learning

This project detects early signs of mechanical anomalies in general aviation flights using subsequence-based anomaly detection. By analyzing sensor data collected before and after maintenance events, we aim to flag problematic flight segments **before** failures occur.

>  **Core Technique**: One-Class SVM trained only on healthy (post-maintenance) flight subsequences  
>  **Dataset**: NGAFID — segmented into `before/` and `after/` maintenance flight subsequences  
>  **Input Format**: Each flight subsequence is a `.csv` with multivariate sensor time series

---

##  Objective

To predict anomalies **before** aircraft maintenance occurs by training models on **after-maintenance (healthy)** data and testing them on **before-maintenance (unseen)** subsequences. The goal is **early detection** of faults in multivariate sensor logs using time-windowed machine learning.

---

##  Quick Start 


# 1. Clone the repo
```bash
git clone https://github.com/Anusha19011901/Predictive-Anomaly-Detection-in-NGAFID-Data-Using-Neuroevolution-and-Deep-Learning.git
cd Predictive-Anomaly-Detection-in-NGAFID-Data-Using-Neuroevolution-and-Deep-Learning
```
# 2. Set up Python virtual environment
```bash
# Mac/Linux
python3 -m venv myenv
source myenv/bin/activate
```
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate
```
# 3. Install dependencies
```bash
pip install -r requirements.txt
```

# 4. Visualize a single flight
Feel free to test it with any of the datasets given in the repo
```bash
python explore_flight_data.py
```

# 5. Train OC-SVM on AFTER flights
```bash
python ocsvm_pipeline2.py
```
# 6. Detect anomalies in BEFORE flights
```bash
python detect_anomalies.py
```

---

##  Folder Structure

```
dataset/
│
├── before/    # Flight subsequences BEFORE maintenance
└── after/     # Flight subsequences AFTER maintenance

explore_flight_data.py    # Visualization of a single flight
ocsvm_pipeline2.py        # Train One-Class SVM model on AFTER flights
detect_anomalies.py       # Use trained models on BEFORE flights
```

---

## 🔍 What Each Script Does

### 🔎 explore_flight_data.py
Visualizes one flight file (edit the `csv_path` inside). Shows:
- Altitude, RPM, Fuel Flow, 3D Path, CHT/EGT, Acceleration, Pitch/Roll
- Correlation heatmap and more

Useful for understanding sensor behavior and data quality.

---

###  ocsvm_pipeline2.py
Trains One-Class SVM models on healthy (after-maintenance) flight subsequences:
- Cleans sensor data
- Uses sliding windows (default 30 rows, 25 step)
- Trains OC-SVM on each window
- Stores models in memory for later use

---

###  detect_anomalies.py
Uses the trained OC-SVM models to scan BEFORE-maintenance flights:
- Same windowing
- Flags any window with >30% anomalies
- Shows:
  - Binary anomaly flags over time
  - SVM decision score curve
  - Heatmap of feature behavior
  - PCA projection of windows

Outputs:
```
 Anomaly first detected at window starting index: 325
```
Check outputs folder for expected output diagrams

---

## ⚙️ Customize the Pipeline

Change these values in `ocsvm_pipeline2.py` and `detect_anomalies.py`:

| Setting              | Default | Description                                 |
|---------------------|---------|---------------------------------------------|
| `WINDOW_SIZE`       | 30      | Number of rows per sliding window           |
| `STEP_SIZE`         | 25      | Step size between windows                   |
| `ANOMALY_THRESHOLD` | 0.3     | % of anomalies in a window to be flagged    |
| `COLUMNS_TO_USE`    | 7 cols  | AltMSL, RPM, Fuel Flow, CHT, EGT, Acc, IAS  |

---

##  What’s Next?

🔄 **EXAMM Integration** (coming soon):  
Evolve custom RNNs (LSTM, GRU) for deeper anomaly detection with attention and explainability.

 **Deep Learning Baselines** (optional):  
Compare OC-SVM with Anomaly Transformer, TranAD, TS-BERT (code in `models/` folder).

---

## 👩‍💻 Authors

- **Anusha Seshadri**   
- **Iyashi Pal**   
- **Travis Desell** — Faculty Advisor  

---

