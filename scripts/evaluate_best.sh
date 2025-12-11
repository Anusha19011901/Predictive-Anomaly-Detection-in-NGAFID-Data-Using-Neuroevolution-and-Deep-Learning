#!/usr/bin/env bash
set -e

echo "======================================="
echo "     EXAMM BEST GENOME EVALUATION"
echo "======================================="

EXACT_DIR="$HOME/exact"
EVAL_BIN="$EXACT_DIR/build/rnn_examples/evaluate_rnn"
GENOME="rnn_genome_5424.bin"

# ✅ Use the EXAMM-ready BEFORE data, not raw NGAFID
TEST_DIR="dataset/before_examm2"

OUT_DIR="artifacts/evaluation_output"
mkdir -p "$OUT_DIR"

echo "[INFO] Using evaluator:"
echo "       $EVAL_BIN"
echo "[INFO] Genome file:"
echo "       $GENOME"
echo "[INFO] Test files:"
echo "       $TEST_DIR/*.csv"

"$EVAL_BIN" \
    --genome_file "$GENOME" \
    --testing_filenames "$TEST_DIR"/*.csv \
    --input_parameter_names  "AltMSL" "E1 RPM" "E1 FFlow" "E1 CHT1" "E1 EGT1" "NormAc" "IAS" \
    --output_parameter_names "AltMSL" "E1 RPM" "E1 FFlow" "E1 CHT1" "E1 EGT1" "NormAc" "IAS" \
    --time_offset 1 \
    --train_sequence_length 30 \
    --std_message_level info \
    --file_message_level info \
    --error_message_level warning \
    --output_directory "$OUT_DIR"

echo "======================================="
echo "     DONE! Results in $OUT_DIR"
echo "======================================="
