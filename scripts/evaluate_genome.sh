#!/usr/bin/env bash

###############################################
#   Evaluate a Pre-Trained EXAMM Genome
###############################################

# ---- MODIFY IF NEEDED ----
EXACT_DIR="$HOME/exact"
GENOME_PATH="rnn_genome_5424.bin"
WINDOWS_DIR="exact_data/before_windows"     # your BEFORE window folder
OUT_DIR="outputs/eval_best"
###############################################

echo "======================================="
echo "      EVALUATING PRETRAINED GENOME"
echo "======================================="

# 1) Verify genome exists
if [[ ! -f "$GENOME_PATH" ]]; then
    echo "[ERROR] Genome file not found: $GENOME_PATH"
    exit 1
fi
echo "[OK] Genome found: $GENOME_PATH"

# 2) Validate EXAMM repo
if [[ ! -d "$EXACT_DIR" ]]; then
    echo "[ERROR] EXAMM repo not found at: $EXACT_DIR"
    exit 1
fi

# 3) Check rnn_eval exists
if [[ ! -f "$EXACT_DIR/build/rnn_eval" ]]; then
    echo "[ERROR] Missing rnn_eval inside EXAMM build."
    echo "Run:  cd ~/exact && ./build_examm_auto.sh"
    exit 1
fi
echo "[OK] Found rnn_eval binary."

# 4) Create output directory
mkdir -p "$OUT_DIR"

# 5) Run evaluation
echo "======================================="
echo "       RUNNING rnn_eval ON WINDOWS"
echo "======================================="

"$EXACT_DIR/build/rnn_eval" \
    --model "$GENOME_PATH" \
    --input_directory "$WINDOWS_DIR" \
    --output_directory "$OUT_DIR" \
    --print_mse true \
    --print_predictions false \
    --print_raw_outputs false

if [[ $? -ne 0 ]]; then
    echo "[ERROR] rnn_eval failed."
    exit 1
fi

echo "======================================="
echo "    EVALUATION COMPLETE!"
echo "  Results saved to: $OUT_DIR"
echo "======================================="
