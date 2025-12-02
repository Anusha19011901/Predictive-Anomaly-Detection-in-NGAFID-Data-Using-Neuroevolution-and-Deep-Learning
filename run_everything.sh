#!/usr/bin/env bash
set -euo pipefail

echo "============================"
echo " NGAFID FULL PIPELINE RE-RUN"
echo "============================"

###########################################
# CONFIG — MODIFY PATHS ONLY IF NEEDED
###########################################

AFTER_RAW="dataset/after"
BEFORE_RAW="dataset/before"
ERROR_WINDOWS="exact_data/anomaly"              # window_*.csv
EXAMM_AFTER_ERRORS="artifacts/errors/per_window/ocsvm_input_after.csv"
EXAMM_BEFORE_ERRORS="artifacts/errors/per_window/ocsvm_input_before.csv"

DBSCAN_DIR="outputs/dbscan_eps2.1_run"
SCALER="outputs/scaler.pkl"
OCSVM_EXAMM_DIR="outputs/ocsvm_examm_only"

PROTOS_NPZ="${DBSCAN_DIR}/prototypes_dbscan_eps4.0_min5_p95.npz"
DBSCAN_ASSIGN="${DBSCAN_DIR}/assignments_dbscan_eps4.0_min5.csv"

###########################################
# 0 — RUN OC-SVM on EXAMM ERROR FEATURES
###########################################

echo "[STEP 0] OC-SVM (EXAMM-only error features)..."

mkdir -p ${OCSVM_EXAMM_DIR}

python3 ocsvm_examm_only.py \
    --after_errors  ${EXAMM_AFTER_ERRORS} \
    --before_errors ${EXAMM_BEFORE_ERRORS} \
    --out_dir       ${OCSVM_EXAMM_DIR}

echo "   ✔ OC-SVM EXAMM-only → before_scores.csv"

###########################################
# 1 — FIXED LABELS FOR WINDOWS
###########################################

echo "[STEP 1] labels_per_window.csv generation..."

python3 make_labels_per_window_fixed.py \
    --windows_dir ${ERROR_WINDOWS} \
    --assignments_csv ${DBSCAN_ASSIGN} \
    --out_csv ${DBSCAN_DIR}/labels_per_window.csv

echo "   ✔ Fixed labels_per_window.csv created."

###########################################
# 2 — EXPLAIN ERROR WINDOWS USING PROTOTYPE BOXES
###########################################

echo "[STEP 2] Prototype-box explanations for ERROR windows..."

python3 dbscan_boxes.py explain_error \
    --error_dir ${ERROR_WINDOWS} \
    --scaler_path ${SCALER} \
    --prototypes_path ${PROTOS_NPZ} \
    --out_csv ${DBSCAN_DIR}/explanations_error_eps4.0.csv \
    --columns AltMSL "E1 RPM" "E1 FFlow" "E1 CHT1" "E1 EGT1" NormAc IAS \
    --window_size 30

echo "   ✔ explanations_error_eps4.0.csv"

###########################################
# 3 — PROTOTYPE DIAGNOSTICS
###########################################

echo "[STEP 3] Prototype diagnostics..."

mkdir -p outputs/proto_diagnostics

python3 prototype_diagnostics.py \
  --windows_dir ${ERROR_WINDOWS} \
  --exemplars_csv ${DBSCAN_DIR}/analysis/exemplars_per_prototype.csv \
  --distribution_csv ${DBSCAN_DIR}/analysis/distribution_by_prototype.csv \
  --scores_csv ${DBSCAN_DIR}/analysis/exemplars_per_prototype.csv \
  --out_dir outputs/proto_diagnostics \
  --small_cluster_thresh 20

echo "   ✔ Prototype diagnostics ready."

###########################################
# 4 — NOISE DIAGNOSTICS (DBSCAN NOISE = -1)
###########################################

echo "[STEP 4] Noise diagnostics..."

mkdir -p outputs/noise_diagnostics

python3 noise_windows_diagnostics.py \
  --windows_dir ${ERROR_WINDOWS} \
  --labels_csv ${DBSCAN_DIR}/labels_per_window.csv \
  --label_col dbscan_label \
  --id_col filename \
  --noise_label -1 \
  --out_dir outputs/noise_diagnostics \
  --topk 20 \
  --save_sample_traces

echo "   ✔ Noise diagnostics complete."

###########################################
# 5 — DBSCAN VISUALIZATIONS (MEAN SHAPES + EXEMPLARS)
###########################################

echo "[STEP 5] DBSCAN VISUALIZATIONS..."

mkdir -p outputs/dbscan_vis

python3 dbscan_vis.py \
    --error_expl_csv ${DBSCAN_DIR}/explanations_error_eps4.0.csv \
    --error_dir ${ERROR_WINDOWS} \
    --prototypes_path ${PROTOS_NPZ} \
    --columns AltMSL "E1 RPM" "E1 FFlow" "E1 CHT1" "E1 EGT1" NormAc IAS \
    --exemplars 3 \
    --out_dir outputs/dbscan_vis

echo "   ✔ Mean cluster shapes / exemplars saved."

###########################################
# 6 — BEFORE FLIGHTS EXPLANATIONS (DBSCAN BOXES)
###########################################

echo "[STEP 6] BEFORE flight DBSCAN explanations..."

python3 dbscan_boxes.py explain \
    --input_dir ${BEFORE_RAW} \
    --scaler_path outputs/scaler.pkl \
    --prototypes_path ${PROTOS_NPZ} \
    --template_dir dataset/after_examm2 \
    --window_size 30 \
    --step_size 25 \
    --columns AltMSL "E1 RPM" "E1 FFlow" "E1 CHT1" "E1 EGT1" NormAc IAS \
    --out_csv ${DBSCAN_DIR}/explanations_before_eps4.0.csv

echo "   ✔ BEFORE explanations saved."

###########################################
# 7 — NAB BENCHMARK (DBSCAN BOXES)
###########################################

echo "[STEP 7] NAB benchmark..."

python3 external/NAB/scripts/run_nab_dbscan_boxes_tuned.py

echo "   ✔ NAB scores saved (AUCs, threshold metrics)."

###########################################
# 8 — SUPERVISED HYBRID DATASET (WINDOW-LEVEL)
###########################################

echo "[STEP 8] Creating supervised dataset..."

python3 create_supervised_dataset.py \
    --dbscan_labels  ${DBSCAN_DIR}/labels_per_window.csv \
    --examm_errors   ${DBSCAN_DIR}/explanations_error_eps4.0.csv \
    --ocsvm_scores   ${OCSVM_EXAMM_DIR}/before_scores.csv \
    --out_csv        outputs/supervised_dataset_windows.csv

echo "   ✔ Supervised dataset ready: outputs/supervised_dataset_windows.csv"

echo "============================"
echo "     PIPELINE COMPLETE"
echo "============================"
