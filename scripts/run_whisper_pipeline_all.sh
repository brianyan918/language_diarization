#!/bin/bash
# run_whisper_pipeline_all.sh
#
# Run whisper_pipeline.py and score_jaccard.py on all test sets
#
# Usage:
#   ./scripts/run_whisper_pipeline_all.sh                 # Run normally
#   ./scripts/run_whisper_pipeline_all.sh --dry_run       # Print commands without running

# set -e

# Parse arguments
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry_run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            # exit 1
            ;;
    esac
done

# Base paths
BASE_ASR_EXP="/data/group_data/swl/old_home/byan/lang_diar/asr_exp/base_v3-turbo/pred-lid"
BASE_LANGDIAR="/data/group_data/swl/old_home/byan/lang_diar/model/spoken-language-diarization/exp/runs/inf_langdiar_whisper_multi"
VOCAB="data/vocab_102.txt"
FASTTEXT_MODEL="lid.176.bin"
UROMAN="uroman/uroman/"
FASTTEXT_ENV="other_envs/fasttext/activate_python.sh"

# List of datasets to process
# Format: "asr_name:langdiar_name"
# If no colon, both names are the same
DATASETS=(
    # "arzen:arzen"
    "csfl_read:csfl_read"
    "mucs_ben:mucs_ben"
    "mucs_hin:mucs_hin"
    "seame_devman:seame_man"
    "seame_devsge:seame_sge"
)

echo "Checking files and preparing commands..."
echo ""

# First pass: check all files exist
FILES_OK=1
for dataset_pair in "${DATASETS[@]}"; do
    # Parse dataset pair (asr_name:langdiar_name)
    IFS=':' read -r asr_name langdiar_name <<< "$dataset_pair"
    if [ -z "$langdiar_name" ]; then
        langdiar_name="$asr_name"
    fi
    
    ASR_DIR="$BASE_ASR_EXP/$asr_name"
    LANGDIAR_DIR="$BASE_LANGDIAR/whispermedium_frame_indomain.on.v3_$langdiar_name"
    
    WHISPER_JSONL="$ASR_DIR/out.jsonl"
    REFERENCE_JSONL="$LANGDIAR_DIR/langdiar_whisper_multi.0.jsonl"
    
    echo "Checking: $asr_name (-> $langdiar_name)"
    
    if [ ! -f "$WHISPER_JSONL" ]; then
        echo "  ✗ MISSING: $WHISPER_JSONL"
        FILES_OK=0
    else
        echo "  ✓ Found: $WHISPER_JSONL"
    fi
    
    if [ ! -f "$REFERENCE_JSONL" ]; then
        echo "  ✗ MISSING: $REFERENCE_JSONL"
        FILES_OK=0
    else
        echo "  ✓ Found: $REFERENCE_JSONL"
    fi
    
    if [ ! -f "$VOCAB" ]; then
        echo "  ✗ MISSING: $VOCAB"
        FILES_OK=0
    else
        echo "  ✓ Found: $VOCAB"
    fi
    
    if [ ! -f "$FASTTEXT_MODEL" ]; then
        echo "  ✗ MISSING: $FASTTEXT_MODEL"
        FILES_OK=0
    else
        echo "  ✓ Found: $FASTTEXT_MODEL"
    fi
    
    echo ""
done

if [ $FILES_OK -eq 0 ]; then
    echo "ERROR: Some required files are missing!"
    # exit 1
fi

echo "========================================"
echo "All required files found!"
echo "========================================"
echo ""

if [ $DRY_RUN -eq 1 ]; then
    echo "========================================"
    echo "DRY RUN - Commands that would be executed:"
    echo "========================================"
    echo ""
    
    for dataset in "${DATASETS[@]}"; do
        ASR_DIR="$BASE_ASR_EXP/$dataset"
        LANGDIAR_DIR="$BASE_LANGDIAR/whispermedium_frame_indomain.on.v3_$dataset"
        
        WHISPER_JSONL="$ASR_DIR/out.jsonl"
        REFERENCE_JSONL="$LANGDIAR_DIR/langdiar_whisper_multi.0.jsonl"
        PIPELINE_OUTPUT="$ASR_DIR/whisper_pipeline.jsonl"
        JACCARD_OUTPUT="$ASR_DIR/whisper_pipeline.jaccard.json"
        
        echo "# Dataset: $dataset"
        echo "python scripts/whisper_pipeline.py \\"
        echo "    --whisper_jsonl '$WHISPER_JSONL' \\"
        echo "    --reference_jsonl '$REFERENCE_JSONL' \\"
        echo "    --output '$PIPELINE_OUTPUT' \\"
        echo "    --fasttext_model '$FASTTEXT_MODEL' \\"
        echo "    --uroman '$UROMAN' \\"
        echo "    --fasttext_env '$FASTTEXT_ENV'"
        echo ""
        echo "python scripts/score_jaccard.py \\"
        echo "    --input_jsonl '$PIPELINE_OUTPUT' \\"
        echo "    --vocab '$VOCAB' \\"
        echo "    --output_json '$JACCARD_OUTPUT'"
        echo ""
    done
    # exit 0
fi

# Normal execution
SUCCESSFUL=0
FAILED=0

for dataset_pair in "${DATASETS[@]}"; do
    # Parse dataset pair
    IFS=':' read -r asr_name langdiar_name <<< "$dataset_pair"
    if [ -z "$langdiar_name" ]; then
        langdiar_name="$asr_name"
    fi
    
    echo "========================================"
    echo "Processing: $asr_name -> $langdiar_name"
    echo "========================================"
    
    # Set paths based on dataset
    ASR_DIR="$BASE_ASR_EXP/$asr_name"
    LANGDIAR_DIR="$BASE_LANGDIAR/whispermedium_frame_indomain.on.v3_$langdiar_name"
    
    WHISPER_JSONL="$ASR_DIR/out.jsonl"
    REFERENCE_JSONL="$LANGDIAR_DIR/langdiar_whisper_multi.0.jsonl"
    PIPELINE_OUTPUT="$ASR_DIR/whisper_pipeline.jsonl"
    JACCARD_OUTPUT="$ASR_DIR/whisper_pipeline.jaccard.json"
    
    # Run whisper_pipeline.py
    echo "Running whisper_pipeline.py..."
    if python scripts/whisper_pipeline.py \
        --whisper_jsonl "$WHISPER_JSONL" \
        --reference_jsonl "$REFERENCE_JSONL" \
        --output "$PIPELINE_OUTPUT" \
        --fasttext_model "$FASTTEXT_MODEL" \
        --uroman "$UROMAN" \
        --fasttext_env "$FASTTEXT_ENV"; then
        echo "✓ whisper_pipeline.py succeeded"
    else
        echo "✗ whisper_pipeline.py failed"
        ((FAILED++))
        echo ""
        continue
    fi
    
    # Run score_jaccard.py
    echo "Running score_jaccard.py..."
    if python scripts/score_jaccard.py \
        --input_jsonl "$PIPELINE_OUTPUT" \
        --vocab "$VOCAB" \
        --output_json "$JACCARD_OUTPUT"; then
        echo "✓ score_jaccard.py succeeded"
        ((SUCCESSFUL++))
    else
        echo "✗ score_jaccard.py failed"
        ((FAILED++))
    fi
    
    echo ""
done

# Print summary
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Total datasets: ${#DATASETS[@]}"
echo "Successful: $SUCCESSFUL"
echo "Failed: $FAILED"
echo ""

if [ $SUCCESSFUL -gt 0 ]; then
    echo "========================================"
    echo "JER SCORES"
    echo "========================================"
    echo ""
    
    for dataset_pair in "${DATASETS[@]}"; do
        # Parse dataset pair
        IFS=':' read -r asr_name langdiar_name <<< "$dataset_pair"
        if [ -z "$langdiar_name" ]; then
            langdiar_name="$asr_name"
        fi
        
        ASR_DIR="$BASE_ASR_EXP/$asr_name"
        JACCARD_OUTPUT="$ASR_DIR/whisper_pipeline.jaccard.json"
        
        if [ -f "$JACCARD_OUTPUT" ]; then
            jer=$(python -c "import json; d=json.load(open('$JACCARD_OUTPUT')); print(d.get('global', {}).get('jer', 'N/A'))" 2>/dev/null || echo "N/A")
            jac=$(python -c "import json; d=json.load(open('$JACCARD_OUTPUT')); print(d.get('global', {}).get('jaccard_mean', 'N/A'))" 2>/dev/null || echo "N/A")
            utts=$(python -c "import json; d=json.load(open('$JACCARD_OUTPUT')); print(d.get('global', {}).get('utts', 'N/A'))" 2>/dev/null || echo "N/A")
            
            printf "%-20s JER=%8s  Jaccard=%8s  (utts=%s)\n" "$asr_name" "$jer" "$jac" "$utts"
        fi
    done
fi
