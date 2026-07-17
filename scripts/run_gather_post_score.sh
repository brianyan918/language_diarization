#!/usr/bin/env bash
# set -euo pipefail

# -----------------------------------------
# Usage:
#
#   bash run_asr_postprocess.sh \
#       --dicow_root <path_to_wer_dir> \
#       --ref_manifest <path_to_ref_manifest> \
#       [--ignore_id_prefix]
#
# Example:
#
#   bash run_asr_postprocess.sh \
#       --dicow_root asr_exp/.../4800/wer/ \
#       --ref_manifest data/cs-fleurs/read/test/diar/manifest.json
#
#   bash run_asr_postprocess.sh \
#       --dicow_root asr_exp/.../4800/wer/ \
#       --ref_manifest data/cs-fleurs/read/test/diar/manifest.json \
#       --ignore_id_prefix
# -----------------------------------------

# -------- Parse arguments --------
IGNORE_ID_PREFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dicow_root)
            DICOW_WER_ROOT="$2"
            shift 2
            ;;
        --ref_manifest)
            REF_MANIFEST="$2"
            shift 2
            ;;
        --ignore_id_prefix)
            IGNORE_ID_PREFIX="--ignore_id_prefix"
            shift 1
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# -------- Required args --------
if [[ -z "${DICOW_WER_ROOT:-}" ]]; then
    echo "ERROR: --dicow_root is required"
    exit 1
fi

if [[ -z "${REF_MANIFEST:-}" ]]; then
    echo "ERROR: --ref_manifest is required"
    exit 1
fi

if [[ ! -d "${DICOW_WER_ROOT}" ]]; then
    echo "ERROR: dicow_root does not exist: ${DICOW_WER_ROOT}"
    exit 1
fi

if [[ ! -f "${REF_MANIFEST}" ]]; then
    echo "ERROR: ref_manifest does not exist: ${REF_MANIFEST}"
    exit 1
fi

# -------- Derive OUT_DIR --------
DICOW_WER_ROOT="${DICOW_WER_ROOT%/}"

if [[ "$(basename "${DICOW_WER_ROOT}")" != "wer" ]]; then
    echo "ERROR: dicow_root must point to a 'wer' directory"
    echo "Given: ${DICOW_WER_ROOT}"
    exit 1
fi

OUT_DIR="$(dirname "${DICOW_WER_ROOT}")"

GATHERED_JSONL="${OUT_DIR}/output.jsonl"
COMBINED_JSONL="${OUT_DIR}/combined_output.jsonl"
MERGED_JSONL="${OUT_DIR}/combined_output_w-hyp.jsonl"
CER_JSON="${OUT_DIR}/cer.jsonl"

echo "========== CONFIG =========="
echo "DICOW_WER_ROOT : ${DICOW_WER_ROOT}"
echo "OUT_DIR        : ${OUT_DIR}"
echo "REF_MANIFEST   : ${REF_MANIFEST}"
echo "============================"
echo

# -------- Step 1: Gather --------
echo "== [1/4] Gathering DiCoW outputs =="
python scripts/gather_dicow_outputs_v2.py \
    --root "${DICOW_WER_ROOT}" \
    --out "${GATHERED_JSONL}"

# -------- Step 2: Combine --------
echo "== [2/4] Combining segments =="
python scripts/combine_dicow_segments.py \
    -i "${GATHERED_JSONL}" \
    -o "${COMBINED_JSONL}"

# -------- Step 3: Merge hyp with ref --------
echo "== [3/4] Merging hypothesis with reference =="
MERGE_CMD=(python scripts/merge_hyp_w_ref.py \
    --source "${REF_MANIFEST}" \
    --target "${COMBINED_JSONL}" \
    --output "${MERGED_JSONL}")
if [[ -n "${IGNORE_ID_PREFIX}" ]]; then
    MERGE_CMD+=("${IGNORE_ID_PREFIX}")
fi
"${MERGE_CMD[@]}"

# -------- Step 4: Score CER --------
echo "== [4/4] Scoring CER =="
SCORE_CMD=(python scripts/score_cer_confusions_jsonl.py \
    -i "${MERGED_JSONL}" \
    --exclude_langs slk-eng tel-eng \
    --ignore_ws \
    --remove_punct \
    --lower \
    --out_json "${CER_JSON}" \
    --ref_field text \
    --hyp_field hyp_text)
if [[ -n "${IGNORE_ID_PREFIX}" ]]; then
    SCORE_CMD+=("--ignore_id_prefix")
fi
"${SCORE_CMD[@]}"

echo
echo "✅ Done."
echo "Final merged output: ${MERGED_JSONL}"
echo "CER results saved to: ${CER_JSON}"
