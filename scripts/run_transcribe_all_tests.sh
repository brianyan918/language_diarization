#!/bin/bash

# Transcribe segments for all test sets

echo "Running transcription for all test sets..."

# csfl
echo "Processing csfl..."
python scripts/transcribe_segments.py --input_jsonl data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json --output_dir debug/seg/csfl

# seame_man
echo "Processing seame_man..."
python scripts/transcribe_segments.py --input_jsonl data/seame/devman/diar/manifest3.jsonl/manifest.json --output_dir debug/seg/seame_man

# seame_sge
echo "Processing seame_sge..."
python scripts/transcribe_segments.py --input_jsonl data/seame/devsge/diar/manifest3.jsonl/manifest.json --output_dir debug/seg/seame_sge

# arzen
echo "Processing arzen..."
python scripts/transcribe_segments.py --input_jsonl data/arzen/test/diar/manifest3.jsonl/manifest.json --output_dir debug/seg/arzen

# mucs_hin
echo "Processing mucs_hin..."
python scripts/transcribe_segments.py --input_jsonl data/mucs/hin-eng/test/diar/manifest3.jsonl/manifest.json --output_dir debug/seg/mucs_hin

# mucs_ben
echo "Processing mucs_ben..."
python scripts/transcribe_segments.py --input_jsonl data/mucs/ben-eng/test/diar/manifest3.jsonl/manifest.json --output_dir debug/seg/mucs_ben

echo "All transcription jobs completed!"
