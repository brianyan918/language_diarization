#!/usr/bin/env bash
# Note: avoid `set -euo pipefail` because `srun` can cause the script to exit unexpectedly
# when run inside some job schedulers. Keep the shell more permissive so long-running
# scoring calls aren't terminated by a single failure.

# run_all_scoring.sh
# Wrapper to run `scripts/run_gather_post_score.sh` for multiple test sets.
# Edit the `ref_map` entries below to point to the correct reference manifests on disk.
# Usage:
#   ./scripts/run_all_scoring.sh [dicow_prefix] [shard] [-- extra args]
# or override pattern/extra args:
#   DICOW_PATTERN="asr_exp/.../test_{dataset}_.../0/wer/" ./scripts/run_all_scoring.sh --extra-arg

BASE_DIR="."

# Pattern used to build the dicow_root path for each dataset. {dataset} will be replaced.
# You can provide a prefix in three ways:
# 1) As the first positional argument to this script: ./scripts/run_all_scoring.sh <dicow_prefix> [shard] [-- extra args]
# 2) Via env var: DICOW_PREFIX=/path/to/prefix ./scripts/run_all_scoring.sh
# 3) Or the default prefix below will be used.
DEFAULT_PREFIX="asr_exp/cs_decode_ft_whisper_check_v2"

# If first arg looks like a prefix (not an option), consume it.
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    DICOW_PREFIX="$1"
    shift
else
    DICOW_PREFIX="${DICOW_PREFIX:-$DEFAULT_PREFIX}"
fi

# ensure no trailing slash on prefix
DICOW_PREFIX="${DICOW_PREFIX%/}"

# Optional shard index (defaults to 0). Can be provided as the next positional arg or via env var.
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    DICOW_SHARD="$1"
    shift
else
    DICOW_SHARD="${DICOW_SHARD:-0}"
fi

# Build pattern (replace {dataset} later)
DICOW_PATTERN="${DICOW_PREFIX}/test_{dataset}_cuts_with_lang_speakers_test_v4/${DICOW_SHARD}/wer/"
# DICOW_PATTERN="${DICOW_PREFIX}/test_{dataset}_cuts_with_lang_speakers_test_v4_sl/${DICOW_SHARD}/wer/"
# DICOW_PATTERN="${DICOW_PREFIX}/test_{dataset}_cuts_with_lang_speakers_test_v4_remove/${DICOW_SHARD}/wer/"

# Datasets to run
# datasets=(csfl0 csfl1 csfl2 csfl3 csfl4 csfl5 csfl6 csfl7 csfl8 csfl9 csfl10)
datasets=(csfl5)
# datasets=(csfl_merge025)

# Mapping: dataset -> ref_manifest path
# PLEASE EDIT these paths to the correct manifest on your machine before running.
declare -A ref_map
ref_map[csfl0]="/data/group_data/swl/old_home/byan/lang_diar/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json"
ref_map[csfl1]="/data/group_data/swl/old_home/byan/lang_diar/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json"
ref_map[csfl2]="/data/group_data/swl/old_home/byan/lang_diar/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json"
ref_map[csfl3]="/data/group_data/swl/old_home/byan/lang_diar/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json"
ref_map[csfl4]="/data/group_data/swl/old_home/byan/lang_diar/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json"
ref_map[csfl5]="/data/group_data/swl/old_home/byan/lang_diar/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json"
ref_map[csfl6]="/data/group_data/swl/old_home/byan/lang_diar/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json"
ref_map[csfl7]="/data/group_data/swl/old_home/byan/lang_diar/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json"
ref_map[csfl8]="/data/group_data/swl/old_home/byan/lang_diar/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json"
ref_map[csfl9]="/data/group_data/swl/old_home/byan/lang_diar/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json"
ref_map[csfl10]="/data/group_data/swl/old_home/byan/lang_diar/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json"


# Optional: allow overriding the dicow pattern via env var
if [ -n "${DICOW_PATTERN:-}" ]; then
    DICOW_PATTERN="$DICOW_PATTERN"
fi

EXTRA_ARGS=()
while (("$#")); do
    EXTRA_ARGS+=("$1")
    shift
done

echo "Running scoring for datasets: ${datasets[*]}"

for ds in "${datasets[@]}"; do
    ref_manifest="${ref_map[$ds]:-}"
    if [ -z "${ref_manifest}" ]; then
        echo "[SKIP] No ref_manifest configured for dataset: ${ds}. Update script mapping first." >&2
        continue
    fi
    if [ ! -f "${ref_manifest}" ]; then
        echo "[SKIP] ref_manifest file not found for ${ds}: ${ref_manifest}" >&2
        continue
    fi

    dicow_root=${DICOW_PATTERN//\{dataset\}/$ds}
    # If you already have a specific subdir, you can set an env var override DICOW_ROOT_<DATASET>=...
    override_var="DICOW_ROOT_${ds^^}"
    if [ -n "${!override_var:-}" ]; then
        dicow_root="${!override_var}"
    fi

    echo "---"
    echo "Dataset: $ds"
    echo "dicow_root: $dicow_root"
    echo "ref_manifest: $ref_manifest"

    if [ ! -d "$dicow_root" ]; then
        echo "[SKIP] dicow_root does not exist: $dicow_root" >&2
        continue
    fi

    # Call your existing scoring wrapper. Use `bash` to avoid requiring the executable bit.
    bash ./scripts/run_gather_post_score.sh --dicow_root "$dicow_root" --ref_manifest "$ref_manifest" "${EXTRA_ARGS[@]}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[ERROR] scoring failed for dataset $ds (rc=$rc); continuing to next dataset" >&2
        continue
    fi

done

echo "All done. Review the output for any skipped datasets or errors."
