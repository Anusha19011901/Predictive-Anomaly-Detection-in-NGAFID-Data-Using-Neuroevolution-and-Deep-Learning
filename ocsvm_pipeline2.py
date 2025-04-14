import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import glob
from typing import List, Generator
import numpy.typing as npt

# Parameters
WINDOW_SIZE: int = 30
STEP_SIZE: int = 25
AFTER_DIR: str = "dataset/after"
COLUMNS_TO_USE: List[str] = ['AltMSL', 'E1 RPM', 'E1 FFlow', 'E1 CHT1', 'E1 EGT1', 'NormAc', 'IAS']

# Load all CSVs from the after folder
def load_all_csvs(folder: str) -> pd.DataFrame:
    all_files: List[str] = glob.glob(os.path.join(folder, "*.csv"))
    dfs: List[pd.DataFrame] = []
    for file in all_files:
        df: pd.DataFrame = pd.read_csv(file, skiprows=2)
        df.columns = df.columns.str.strip()
        df.replace('', np.nan, inplace=True)  # Turn empty strings into NaN
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Sliding window generator
def generate_windows(
    data: npt.NDArray[np.float_], 
    window_size: int, 
    step_size: int
) -> Generator[npt.NDArray[np.float_], None, None]:
    for start in range(0, len(data) - window_size + 1, step_size):
        yield data[start:start + window_size]

# Load and preprocess
df: pd.DataFrame = load_all_csvs(AFTER_DIR)

# Step 1: Drop blank strings already marked as NaN
df.dropna(subset=COLUMNS_TO_USE, inplace=True)

# Step 2: Coerce any leftover junk strings to NaN
df[COLUMNS_TO_USE] = df[COLUMNS_TO_USE].apply(pd.to_numeric, errors='coerce')

# Step 3: Drop any new NaNs created from failed conversion
df.dropna(subset=COLUMNS_TO_USE, inplace=True)

# Step 4: Now safe to convert to float
X: npt.NDArray[np.float_] = df[COLUMNS_TO_USE].astype(float).values

# Scale
scaler: StandardScaler = StandardScaler()
X_scaled: npt.NDArray[np.float_] = scaler.fit_transform(X)

# Train One-Class SVM on each window
svms: List[OneClassSVM] = []
for i, window in enumerate(generate_windows(X_scaled, WINDOW_SIZE, STEP_SIZE)):
    model: OneClassSVM = OneClassSVM(gamma='auto', nu=0.01)
    model.fit(window)
    svms.append(model)
    print(f"Trained One-Class SVM on window {i} ({i * STEP_SIZE}–{i * STEP_SIZE + WINDOW_SIZE})")

print(f"Total models trained: {len(svms)}")
