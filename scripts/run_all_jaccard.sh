#!/bin/bash
# run_all_jaccard.sh
#
# Automatically discover and score all JSONL files in multiple experiment directories.
# Runs on all directories matching a given prefix.
#
# Usage:
#   ./scripts/run_all_jaccard.sh \
#     --prefix /data/group_data/swl/old_home/byan/lang_diar/model/spoken-language-diarization/exp/runs/inf_langdiar_whisper_multi/whispermedium_frame_indomain.on.v3_ \
#     --collar 0.25 \
#     --exclude_langs "slk,tel"
#
# Or with dry-run:
#   ./scripts/run_all_jaccard.sh \
#     --prefix /path/to/prefix \
#     --dry_run

# set -e

# Default values
VOCAB="data/vocab_102.txt"
COLLAR=0.0
EXCLUDE_LANGS="slk,tel"
PATTERN="*.jsonl"
DRY_RUN=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --vocab)
            VOCAB="$2"
            shift 2
            ;;
        --collar)
            COLLAR="$2"
            shift 2
            ;;
        --exclude_langs)
            EXCLUDE_LANGS="$2"
            shift 2
            ;;
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
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

# Validate required arguments
if [ -z "$PREFIX" ]; then
    echo "ERROR: --prefix is required"
    # exit 1
fi

if [ -z "$VOCAB" ]; then
    echo "ERROR: --vocab is required"
    # exit 1
fi

if [ ! -f "$VOCAB" ]; then
    echo "ERROR: Vocab file not found: $VOCAB"
    # exit 1
fi

echo "Prefix: $PREFIX"
echo ""

# Find all directories matching the prefix
BASE_DIR=$(dirname "$PREFIX")
PREFIX_NAME=$(basename "$PREFIX")

if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: Base directory does not exist: $BASE_DIR"
    # exit 1
fi

EXP_DIRS=($(find "$BASE_DIR" -maxdepth 1 -type d -name "${PREFIX_NAME}*" | sort))

if [ ${#EXP_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No directories found matching prefix: $PREFIX"
    # exit 1
fi

echo "Found ${#EXP_DIRS[@]} experiment directory(ies):"
for dir in "${EXP_DIRS[@]}"; do
    echo "  - $(basename $dir)"
done
echo ""

# Process each experiment directory
TOTAL_SUCCESSFUL=0
TOTAL_FAILED=0
ALL_RESULTS=()

for EXP_DIR in "${EXP_DIRS[@]}"; do
    echo "========================================"
    echo "PROCESSING: $(basename $EXP_DIR)"
    echo "========================================"
    echo ""
    
    # Find all JSONL files in this experiment directory (recursive search)
    JSONL_FILES=($(find "$EXP_DIR" -name "$PATTERN" -type f -not -path "*/.*" | sort))
    
    if [ ${#JSONL_FILES[@]} -eq 0 ]; then
        echo "WARNING: No JSONL files found matching pattern '$PATTERN' in $EXP_DIR"
        echo ""
        continue
    fi

    echo "Found ${#JSONL_FILES[@]} JSONL file(s):"
    for file in "${JSONL_FILES[@]}"; do
        echo "  - $(basename $file)"
    done
    echo ""

    if [ $DRY_RUN -eq 1 ]; then
        echo "[DRY RUN] Commands that would be run:"
        for jsonl_path in "${JSONL_FILES[@]}"; do
            base_name=$(basename "$jsonl_path" .jsonl)
            output_dir=$(dirname "$jsonl_path")
            output_json="$output_dir/${base_name}.jaccard.json"
            
            cmd="python scripts/score_jaccard.py --input_jsonl $jsonl_path --vocab $VOCAB --collar $COLLAR"
            if [ -n "$EXCLUDE_LANGS" ]; then
                cmd="$cmd --exclude_langs \"$EXCLUDE_LANGS\""
            fi
            cmd="$cmd --output_json $output_json"
            
            echo "$cmd"
        done
        echo ""
        continue
    fi

    # Process each JSONL file
    for jsonl_path in "${JSONL_FILES[@]}"; do
        base_name=$(basename "$jsonl_path" .jsonl)
        output_dir=$(dirname "$jsonl_path")
        output_json="$output_dir/${base_name}.jaccard.json"
        
        echo "Processing: $(basename $jsonl_path)"
        echo "Output: $output_json"
        
        cmd="python scripts/score_jaccard.py --input_jsonl $jsonl_path --vocab $VOCAB --collar $COLLAR"
        
        # Only apply language exclusion for csfl_read dataset
        if [ -n "$EXCLUDE_LANGS" ] && [[ "$EXP_DIR" == *"csfl_read"* ]]; then
            cmd="$cmd --exclude_langs $EXCLUDE_LANGS"
        fi
        cmd="$cmd --output_json $output_json"
        
        # Capture both stdout and stderr
        error_output=$(eval "$cmd" 2>&1)
        cmd_exit_code=$?
        
        if [ $cmd_exit_code -eq 0 ]; then
            # Verify output file was actually created
            if [ -f "$output_json" ]; then
                ((TOTAL_SUCCESSFUL++))
                echo "SUCCESS: Output saved to $output_json"
                
                # Store result for summary
                jer=$(python -c "import json; d=json.load(open('$output_json')); print(d.get('global', {}).get('jer', 'N/A'))")
                jac=$(python -c "import json; d=json.load(open('$output_json')); print(d.get('global', {}).get('jaccard_mean', 'N/A'))")
                utts=$(python -c "import json; d=json.load(open('$output_json')); print(d.get('global', {}).get('utts', 'N/A'))")
                ref_speech=$(python -c "import json; d=json.load(open('$output_json')); print(d.get('global', {}).get('ref_speech', 'N/A'))")
                
                ALL_RESULTS+=("$(printf '%-60s JER=%8s  Jaccard=%8s  (utts=%4s, ref_speech=%10s)' "$(basename $EXP_DIR)/$base_name" "$jer" "$jac" "$utts" "$ref_speech")")
            else
                ((TOTAL_FAILED++))
                echo "ERROR: Command executed (exit code 0) but output file was not created"
                echo "Output: $output_json"
                echo "Command output:"
                echo "$error_output"
                ls -la "$(dirname $output_json)" | grep -E "jaccard|jsonl" || echo "No matching files found in directory"
            fi
        else
            ((TOTAL_FAILED++))
            echo "ERROR: Command failed with exit code $cmd_exit_code"
            echo "Input: $jsonl_path"
            echo "Output: $output_json"
            echo "Command was: $cmd"
            echo "Command output:"
            echo "$error_output"
        fi
    done
    
    echo ""
done

# Print summary
if [ $DRY_RUN -eq 0 ]; then
    echo "========================================"
    echo "OVERALL SUMMARY"
    echo "========================================"
    echo "Total experiment directories: ${#EXP_DIRS[@]}"
    echo "Total JSONL files successful: $TOTAL_SUCCESSFUL"
    echo "Total JSONL files failed: $TOTAL_FAILED"
    echo ""
    
    if [ $TOTAL_SUCCESSFUL -gt 0 ]; then
        echo "========================================"
        echo "JER SCORES"
        echo "========================================"
        for result in "${ALL_RESULTS[@]}"; do
            echo "$result"
        done
    fi
fi

