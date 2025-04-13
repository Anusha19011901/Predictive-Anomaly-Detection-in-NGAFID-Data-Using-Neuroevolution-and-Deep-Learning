import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import glob

# Parameters
WINDOW_SIZE = 30
STEP_SIZE = 25
AFTER_DIR = "dataset/after"
COLUMNS_TO_USE = ['AltMSL', 'E1 RPM', 'E1 FFlow', 'E1 CHT1', 'E1 EGT1', 'NormAc', 'IAS']

# Load all CSVs from the after folder
def load_all_csvs(folder):
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    dfs = []
    for file in all_files:
        df = pd.read_csv(file, skiprows=2)
        df.columns = df.columns.str.strip()
        df.replace('', np.nan, inplace=True)  # Turn empty strings into NaN
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Sliding window generator
def generate_windows(data, window_size, step_size):
    for start in range(0, len(data) - window_size + 1, step_size):
        yield data[start:start + window_size]

# Load and preprocess
df = load_all_csvs(AFTER_DIR)
# Step 1: Drop blank strings already marked as NaN
df.dropna(subset=COLUMNS_TO_USE, inplace=True)

# Step 2: Coerce any leftover junk strings to NaN
df[COLUMNS_TO_USE] = df[COLUMNS_TO_USE].apply(pd.to_numeric, errors='coerce')

# Step 3: Drop any new NaNs created from failed conversion
df.dropna(subset=COLUMNS_TO_USE, inplace=True)

# Step 4: Now safe to convert to float
X = df[COLUMNS_TO_USE].astype(float).values


# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train One-Class SVM on each window
svms = []
for i, window in enumerate(generate_windows(X_scaled, WINDOW_SIZE, STEP_SIZE)):
    model = OneClassSVM(gamma='auto', nu=0.01)
    model.fit(window)
    svms.append(model)
    print(f"Trained One-Class SVM on window {i} ({i * STEP_SIZE}–{i * STEP_SIZE + WINDOW_SIZE})")

print(f"Total models trained: {len(svms)}")
