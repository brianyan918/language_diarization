#!/usr/bin/env bash
# run_all_lder_scoring.sh
# Wrapper to run score_lder.py or score_lder_detection.py for multiple test sets.
# 
# Usage:
#   ./scripts/run_all_lder_scoring.sh <model_type> <pred_prefix> [model_name] [extra_args...]
#
# Arguments:
#   model_type: "frame" or "detection"
#     - frame: uses score_lder.py with --vocab data/vocab_102.txt
#     - detection: uses score_lder_detection.py (no vocab needed)
#   pred_prefix: path prefix for predictions (will append test set name pattern)
#     Example: /data/.../exp/runs/inf_langdiar_whisper_multi/all_indomain.det
#   model_name: optional model suffix used in filenames (default: langdiar_whisper_multi)
#     Example: langdiar_mms_multi
#   extra_args: additional flags passed to the scoring script (e.g., --collar 0.25)
#
# Examples:
#   # Detection model
#   ./scripts/run_all_lder_scoring.sh detection /path/to/all_indomain.det --include_fa
#
#   # Frame model
#   ./scripts/run_all_lder_scoring.sh frame /path/to/all_indomain --include_fa --collar 0.25
#
#   # MMS model name
#   ./scripts/run_all_lder_scoring.sh detection /path/to/inf_langdiar_mms_multi mms --include_fa

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model_type> <pred_prefix> [model_name] [extra_args...]" >&2
    echo "  model_type: 'frame' or 'detection'" >&2
    echo "  pred_prefix: path prefix to predictions (without .on.v3_<dataset>)" >&2
    echo "  model_name: optional, e.g., 'langdiar_mms_multi' (default: langdiar_whisper_multi)" >&2
    exit 1
fi

MODEL_TYPE="$1"
PRED_PREFIX="$2"
shift 2

# Optional model name override
MODEL_NAME="langdiar_whisper_multi"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    if [[ "$1" == langdiar_* ]]; then
        MODEL_NAME="$1"
        shift
    elif [[ "$1" == "mms" ]]; then
        MODEL_NAME="langdiar_mms_multi"
        shift
    elif [[ "$1" == "whisper" ]]; then
        MODEL_NAME="langdiar_whisper_multi"
        shift
    fi
fi

# Remaining args passed to scorer
EXTRA_ARGS=("$@")

# Validate model type
if [[ "$MODEL_TYPE" != "frame" && "$MODEL_TYPE" != "detection" ]]; then
    echo "Error: model_type must be 'frame' or 'detection', got: $MODEL_TYPE" >&2
    exit 1
fi

# Datasets to score (same as run_all_scoring.sh)
datasets=(csfl_read xtts_test seame_sge seame_man mucs_hin mucs_ben arzen)
# datasets=(mucs_hin)

# Vocab file (only for frame model)
VOCAB_FILE="data/vocab_102.txt"

echo "Running LDER scoring for datasets: ${datasets[*]}"
echo "Model type: $MODEL_TYPE"
echo "Prediction prefix: $PRED_PREFIX"
echo "Model name: $MODEL_NAME"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo ""

for ds in "${datasets[@]}"; do
    # Build the full prediction path
    # Pattern: <pred_prefix>.on.v3_<dataset>/langdiar_whisper_multi.*.jsonl
    pred_pattern="${PRED_PREFIX}.on.v3_${ds}/${MODEL_NAME}.*.jsonl"
    pred_dir="${PRED_PREFIX}.on.v3_${ds}"
    
    echo "---"
    echo "Dataset: $ds"
    echo "Prediction pattern: $pred_pattern"
    
    # Check if any files match
    shopt -s nullglob
    pred_files=($pred_pattern)
    shopt -u nullglob
    
    if [ ${#pred_files[@]} -eq 0 ]; then
        echo "[SKIP] No prediction files found matching: $pred_pattern" >&2
        continue
    fi
    
    echo "Found ${#pred_files[@]} prediction file(s)"
    
    # Generate output JSON filename based on model name and dataset
    output_json="${pred_dir}/${MODEL_NAME}.lder_scores.json"
    
    # Select scoring script based on model type
    if [ "$MODEL_TYPE" == "frame" ]; then
        if [ ! -f "$VOCAB_FILE" ]; then
            echo "[ERROR] Vocab file not found: $VOCAB_FILE" >&2
            exit 1
        fi
        echo "Running: python scripts/score_lder.py --input_jsonl_glob \"$pred_pattern\" --vocab $VOCAB_FILE --output_json \"$output_json\" ${EXTRA_ARGS[*]}"
        python scripts/score_lder.py --input_jsonl_glob "$pred_pattern" --vocab "$VOCAB_FILE" --output_json "$output_json" "${EXTRA_ARGS[@]}"
    else
        echo "Running: python scripts/score_lder_detection.py --input_jsonl_glob \"$pred_pattern\" --output_json \"$output_json\" ${EXTRA_ARGS[*]}"
        python scripts/score_lder_detection.py --input_jsonl_glob "$pred_pattern" --output_json "$output_json" "${EXTRA_ARGS[@]}"
    fi
    
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[ERROR] LDER scoring failed for dataset $ds (rc=$rc); continuing to next dataset" >&2
        continue
    fi
    echo "Output saved to: $output_json"
    echo ""
done

echo "All done. Review the output above for results and any skipped datasets."
