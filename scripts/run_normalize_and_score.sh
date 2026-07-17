#!/bin/bash

# Batch script to normalize segments and score CER for all datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR/../asr_exp/base_v3-turbo/pred-lid"

# List of datasets
DATASETS=(
    "arzen"
    "csfl_read"
    "mucs_ben"
    "mucs_hin"
    "seame_devman"
    "seame_devsge"
)

echo "Running normalization and CER scoring for all datasets..."
echo

for dataset in "${DATASETS[@]}"; do
    dataset_path="$BASE_DIR/$dataset"
    
    if [ ! -f "$dataset_path/out.jsonl" ]; then
        echo "Skipping $dataset: out.jsonl not found"
        continue
    fi
    
    echo "Processing: $dataset"
    
    # Normalize segments
    python "$SCRIPT_DIR/normalize_segments_with_whisper.py" \
        --input_jsonl "$dataset_path/out.jsonl" \
        --output_jsonl "$dataset_path/norm_out.jsonl"
    
    # Score CER with normalized text
    python "$SCRIPT_DIR/score_cer_confusions_jsonl.py" \
        -i "$dataset_path/norm_out.jsonl" \
        --exclude_langs slk-eng tel-eng \
        --ignore_ws \
        --lower \
        --remove_punct \
        --out_json "$dataset_path/norm_cer.jsonl" \
        --ref_field whisper_normalized_text \
        --hyp_field whisper_pred_text
    
    echo "✓ Completed: $dataset"
    echo
done

echo "All datasets processed successfully!"
