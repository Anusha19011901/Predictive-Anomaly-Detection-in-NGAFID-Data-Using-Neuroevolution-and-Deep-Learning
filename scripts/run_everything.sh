#!/usr/bin/env bash
set -euo pipefail

echo "======================================="
echo "   NGAFID EXAMM + OCSVM FULL PIPELINE"
echo "      (BEFORE + AFTER, DBSCAN, NAB)"
echo "======================================="

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# -------------------- PATHS --------------------
AFTER_RAW="dataset/after"
BEFORE_RAW="dataset/before"
ERROR_WINDOWS="exact_data/anomaly"

EXAMM_AFTER_ERRORS="artifacts/errors/per_window/ocsvm_input_after.csv"
EXAMM_BEFORE_ERRORS="artifacts/errors/per_window/ocsvm_input_before.csv"

DBSCAN_DIR="outputs/dbscan_eps2.1_run"
SCALER="outputs/scaler.pkl"
OCSVM_EXAMM_DIR="outputs/ocsvm_examm_only"

PROTOS_NPZ="${DBSCAN_DIR}/prototypes_dbscan_eps4.0_min5_p95.npz"
DBSCAN_ASSIGN="${DBSCAN_DIR}/assignments_dbscan_eps4.0_min5.csv"

echo "[INFO] Using repo root: $REPO_ROOT"
echo

# ============================================================
# STEP 0 — GENERATE EXAMM MAE TABLES
# ============================================================

echo "[STEP 0] Generating EXAMM MAE feature tables..."
python3 generate_examm_mae_windows.py
echo "   ✔ EXAMM MAE tables ready."
echo

# ============================================================
# STEP 1 — OCSVM ON EXAMM FORECAST ERRORS
# ============================================================

echo "[STEP 1] OC-SVM (EXAMM-only error features)..."
mkdir -p "${OCSVM_EXAMM_DIR}"

python3 ocsvm_examm_only.py \
    --after_errors  "${EXAMM_AFTER_ERRORS}" \
    --before_errors "${EXAMM_BEFORE_ERRORS}" \
    --out_dir       "${OCSVM_EXAMM_DIR}"

echo "   ✔ OC-SVM EXAMM-only → ${OCSVM_EXAMM_DIR}/before_scores.csv"
echo

# ============================================================
# STEP 2 — BUILD labels_per_window.csv
# ============================================================

echo "[STEP 2] labels_per_window.csv generation..."
python3 make_labels_per_window_fixed.py \
    --windows_dir "${ERROR_WINDOWS}" \
    --assignments_csv "${DBSCAN_ASSIGN}" \
    --out_csv "${DBSCAN_DIR}/labels_per_window.csv"

echo "   ✔ labels_per_window.csv written"
echo

# ============================================================
# STEP 3 — PROTOTYPE-BOX EXPLANATIONS FOR ERROR WINDOWS
# ============================================================

echo "[STEP 3] Prototype-box explanations for ERROR windows..."

python3 dbscan_boxes.py explain_error \
    --error_dir "${ERROR_WINDOWS}" \
    --scaler_path "${SCALER}" \
    --prototypes_path "${PROTOS_NPZ}" \
    --out_csv "${DBSCAN_DIR}/explanations_error_eps4.0.csv" \
    --columns AltMSL "E1 RPM" "E1 FFlow" "E1 CHT1" "E1 EGT1" NormAc IAS \
    --window_size 30

echo "   ✔ explanations_error_eps4.0.csv created."
echo

# ============================================================
# PATCH FILE FOR VISUALIZATION — remove old wrong file
# ============================================================

echo "[PATCH] Ensuring correct columns for Step 7..."

# Remove outdated / wrong file if it exists
rm -f "${DBSCAN_DIR}/explanations_error.csv" || true

# Patch column names inside explanations_error_eps4.0.csv
python3 - << 'EOF'
import pandas as pd

src = "outputs/dbscan_eps2.1_run/explanations_error_eps4.0.csv"
df = pd.read_csv(src)

if "prototype_id" in df.columns and "cluster_id" not in df.columns:
    df = df.rename(columns={"prototype_id": "cluster_id"})

df.to_csv(src, index=False)
print("✔ Patched correct file →", src)
EOF

echo

# ============================================================
# STEP 4 — BUILD HYBRID SUPERVISED DATASET
# ============================================================

echo "[STEP 4] Building supervised_dataset_windows.csv..."

python3 create_supervised_dataset.py \
  --dbscan_labels "${DBSCAN_DIR}/labels_per_window.csv" \
  --examm_errors "${DBSCAN_DIR}/explanations_error_eps4.0.csv" \
  --ocsvm_scores "${OCSVM_EXAMM_DIR}/before_scores.csv" \
  --out_csv outputs/supervised_dataset_windows.csv

echo "   ✔ Supervised dataset ready."
echo

# ============================================================
# STEP 5 — PROTOTYPE DIAGNOSTICS
# ============================================================

echo "[STEP 5] Prototype diagnostics..."

mkdir -p outputs/proto_diagnostics

python3 prototype_diagnostics.py \
  --windows_dir "${ERROR_WINDOWS}" \
  --exemplars_csv "${DBSCAN_DIR}/analysis/exemplars_per_prototype.csv" \
  --distribution_csv "${DBSCAN_DIR}/analysis/distribution_by_prototype.csv" \
  --scores_csv "${DBSCAN_DIR}/analysis/exemplars_per_prototype.csv" \
  --out_dir outputs/proto_diagnostics \
  --small_cluster_thresh 20

echo "   ✔ Prototype diagnostics saved."
echo

# ============================================================
# STEP 6 — NOISE DIAGNOSTICS
# ============================================================

echo "[STEP 6] Noise diagnostics..."

mkdir -p outputs/noise_diagnostics

python3 noise_windows_diagnostics.py \
  --windows_dir "${ERROR_WINDOWS}" \
  --labels_csv "${DBSCAN_DIR}/labels_per_window.csv" \
  --label_col dbscan_label \
  --id_col filename \
  --noise_label -1 \
  --out_dir outputs/noise_diagnostics \
  --topk 20 \
  --save_sample_traces

echo "   ✔ Noise diagnostics complete."
echo

# ============================================================
# STEP 7 — DBSCAN VISUALIZATIONS (FIXED)
# ============================================================

echo "[STEP 7] DBSCAN visualizations..."

mkdir -p outputs/dbscan_vis

python3 dbscan_vis.py \
    --error_expl_csv "${DBSCAN_DIR}/explanations_error_eps4.0.csv" \
    --error_dir "${ERROR_WINDOWS}" \
    --prototypes_path "${PROTOS_NPZ}" \
    --columns AltMSL "E1 RPM" "E1 FFlow" "E1 CHT1" "E1 EGT1" NormAc IAS \
    --exemplars 3 \
    --out_dir outputs/dbscan_vis

echo "   ✔ Visualizations saved."
echo

# ============================================================
# STEP 8 — BEFORE FLIGHT PROTOTYPE-BOX EXPLANATIONS
# ============================================================

echo "[STEP 8] BEFORE flight prototype-box explanations..."

python3 dbscan_boxes.py explain \
    --input_dir "${BEFORE_RAW}" \
    --scaler_path "${SCALER}" \
    --prototypes_path "${PROTOS_NPZ}" \
    --template_dir dataset/after_examm2 \
    --window_size 30 \
    --step_size 25 \
    --columns AltMSL "E1 RPM" "E1 FFlow" "E1 CHT1" "E1 EGT1" NormAc IAS \
    --out_csv "${DBSCAN_DIR}/explanations_before_eps4.0.csv"

echo "   ✔ BEFORE explanations generated."
echo

# ============================================================
# STEP 9 — NAB BENCHMARK
# ============================================================

echo "[STEP 9] NAB benchmark..."

python3 external/NAB/scripts/run_nab_dbscan_boxes_tuned.py || \
  echo "[WARN] NAB scoring optional — skipping errors."

echo
echo "======================================="
echo "        PIPELINE COMPLETE 🎉"
echo "======================================="
echo "Supervised dataset:   outputs/supervised_dataset_windows.csv"
echo "OCSVM EXAMM-only:     outputs/ocsvm_examm_only/"
echo "DBSCAN outputs:       ${DBSCAN_DIR}"
echo "======================================="

