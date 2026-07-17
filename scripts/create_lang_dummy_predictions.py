#!/usr/bin/env python3
"""
create_lang_dummy_predictions.py

Create dummy predictions where all segments are predicted as a single language.
Two variants: 100% matrix language (non-English) and 100% English.

Usage:
  python scripts/create_lang_dummy_predictions.py \
    --input langdiar_whisper_multi.0.jsonl \
    --vocab data/vocab_102.txt \
    --matrix_lang spa \
    --output_matrix langdiar_whisper_multi.0.dummy_100pct_matrix.jsonl \
    --output_english langdiar_whisper_multi.0.dummy_100pct_english.jsonl
"""

import argparse
import json
from typing import Dict, List


def load_vocab_id_token(path: str) -> Dict[str, int]:
    """Parse vocab lines like: '2 eng' -> {'eng': 2}"""
    vocab: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Bad vocab line at {path}:{ln} (expected '<id> <token>'): {line}")
            idx = int(parts[0])
            tok = parts[1]
            vocab[tok] = idx
    if not vocab:
        raise ValueError(f"Empty vocab parsed from: {path}")
    return vocab


def infer_matrix_language(input_jsonl: str, vocab: Dict[str, int], per_utt: bool = False) -> str:
    """
    Infer matrix language by finding the most common non-English language in reference segments.
    
    If per_utt=True, infer per utterance and return the most common across all utterances.
    This is useful for CSFL where matrix language may vary per utterance.
    """
    from collections import Counter
    
    lang_counts = Counter()
    utt_matrix_langs = []
    
    with open(input_jsonl, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                utt_key = next(iter(obj.keys()))
                entry = obj[utt_key]
                
                passthrough = entry.get("passthrough", {})
                seg_langs = passthrough.get("segment_langs", [])
                
                if per_utt:
                    # Count non-English languages in this utterance
                    utt_counts = Counter(lang for lang in seg_langs if lang != "eng")
                    if utt_counts:
                        utt_matrix = utt_counts.most_common(1)[0][0]
                        utt_matrix_langs.append(utt_matrix)
                
                # Also accumulate globally
                for lang in seg_langs:
                    if lang != "eng":  # Exclude English
                        lang_counts[lang] += 1
            except Exception:
                continue
    
    if per_utt and utt_matrix_langs:
        # Return most common matrix language across utterances
        global_counts = Counter(utt_matrix_langs)
        matrix_lang = global_counts.most_common(1)[0][0]
    elif lang_counts:
        matrix_lang = lang_counts.most_common(1)[0][0]
    else:
        raise ValueError("Could not infer matrix language: no non-English segments found")
    
    return matrix_lang


def create_dummy_predictions_per_utt_matrix(
    input_jsonl: str,
    output_matrix_jsonl: str,
    output_english_jsonl: str,
    vocab: Dict[str, int],
) -> None:
    """
    Read JSONL file and create two sets of dummy predictions:
    1. Matrix lang: For each utterance, all segments labeled as the most common non-English language in that utterance
    2. English: All segments labeled as English
    """
    from collections import Counter
    
    eng_label_id = vocab["eng"]
    
    with open(input_jsonl, "r", encoding="utf-8") as f_in, \
         open(output_matrix_jsonl, "w", encoding="utf-8") as f_matrix, \
         open(output_english_jsonl, "w", encoding="utf-8") as f_eng:
        
        total_lines = 0
        skipped_lines = 0
        
        for ln, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            
            total_lines += 1
            
            try:
                obj = json.loads(line)
                
                # Get the single key in the object
                if not isinstance(obj, dict) or len(obj) != 1:
                    raise ValueError("Expected single key per line")
                
                utt_key = next(iter(obj.keys()))
                entry = obj[utt_key]
                
                # Extract reference from passthrough
                passthrough = entry.get("passthrough", {})
                seg_ts = passthrough.get("segment_timestamps", [])
                seg_langs = passthrough.get("segment_langs", [])
                
                if not seg_ts:
                    raise ValueError("No segment_timestamps in passthrough")
                
                if len(seg_ts) != len(seg_langs):
                    raise ValueError(f"Mismatch: {len(seg_ts)} timestamps vs {len(seg_langs)} langs")
                
                # Infer matrix language for this utterance: most common non-English language
                lang_counts = Counter(lang for lang in seg_langs if lang != "eng")
                if lang_counts:
                    matrix_lang = lang_counts.most_common(1)[0][0]
                    matrix_label_id = vocab[matrix_lang]
                else:
                    # If no non-English languages, skip this utterance
                    skipped_lines += 1
                    continue
                
                # Create matrix language predictions
                matrix_pred: List[dict] = []
                for (start, end) in seg_ts:
                    matrix_pred.append({
                        "start": float(start),
                        "end": float(end),
                        "label": matrix_label_id,
                        "score": None
                    })
                
                # Create English predictions
                eng_pred: List[dict] = []
                for (start, end) in seg_ts:
                    eng_pred.append({
                        "start": float(start),
                        "end": float(end),
                        "label": eng_label_id,
                        "score": None
                    })
                
                # Write both variants
                matrix_obj = json.loads(json.dumps(obj))  # Deep copy
                matrix_obj[utt_key]["pred"] = matrix_pred
                f_matrix.write(json.dumps(matrix_obj) + "\n")
                
                eng_obj = json.loads(json.dumps(obj))  # Deep copy
                eng_obj[utt_key]["pred"] = eng_pred
                f_eng.write(json.dumps(eng_obj) + "\n")
                
            except Exception as e:
                print(f"WARNING: Skipping line {ln}: {e}")
                skipped_lines += 1
                continue
        
        print(f"Processed {total_lines} lines")
        print(f"Skipped {skipped_lines} lines")
        print(f"Matrix language output written to: {output_matrix_jsonl}")
        print(f"English output written to: {output_english_jsonl}")


def create_dummy_predictions(
    input_jsonl: str,
    output_jsonl: str,
    vocab: Dict[str, int],
    target_lang: str,
) -> None:
    """
    Read JSONL file and create dummy predictions where all segments are the target language.
    """
    if target_lang not in vocab:
        raise KeyError(f"Target language '{target_lang}' not in vocab")
    
    target_label_id = vocab[target_lang]
    
    with open(input_jsonl, "r", encoding="utf-8") as f_in, \
         open(output_jsonl, "w", encoding="utf-8") as f_out:
        
        total_lines = 0
        skipped_lines = 0
        
        for ln, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            
            total_lines += 1
            
            try:
                obj = json.loads(line)
                
                # Get the single key in the object
                if not isinstance(obj, dict) or len(obj) != 1:
                    raise ValueError("Expected single key per line")
                
                utt_key = next(iter(obj.keys()))
                entry = obj[utt_key]
                
                # Extract reference from passthrough to get segment boundaries
                passthrough = entry.get("passthrough", {})
                seg_ts = passthrough.get("segment_timestamps", [])
                
                if not seg_ts:
                    raise ValueError("No segment_timestamps in passthrough")
                
                # Create dummy predictions: all segments labeled as target_lang
                dummy_pred: List[dict] = []
                for (start, end) in seg_ts:
                    dummy_pred.append({
                        "start": float(start),
                        "end": float(end),
                        "label": target_label_id,
                        "score": None
                    })
                
                # Replace pred with dummy predictions
                entry["pred"] = dummy_pred
                
                # Write to output
                f_out.write(json.dumps(obj) + "\n")
                
            except Exception as e:
                print(f"WARNING: Skipping line {ln}: {e}")
                skipped_lines += 1
                continue
        
        print(f"[{target_lang}] Processed {total_lines} lines")
        print(f"[{target_lang}] Skipped {skipped_lines} lines")
        print(f"[{target_lang}] Output written to: {output_jsonl}")


def main():
    ap = argparse.ArgumentParser(
        description="Create dummy predictions with 100%% single language."
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with predictions.",
    )
    ap.add_argument(
        "--vocab",
        required=True,
        help="Vocabulary file mapping language codes to IDs.",
    )
    ap.add_argument(
        "--matrix_lang",
        default=None,
        help="Matrix language code (non-English) for first dummy. If not provided, will be inferred.",
    )
    ap.add_argument(
        "--output_matrix",
        required=True,
        help="Output JSONL file with 100%% matrix language predictions.",
    )
    ap.add_argument(
        "--output_english",
        required=True,
        help="Output JSONL file with 100%% English predictions.",
    )
    ap.add_argument(
        "--per_utt_infer",
        action="store_true",
        help="Infer matrix language per utterance (useful for code-switching datasets like CSFL).",
    )
    args = ap.parse_args()
    
    # Load vocabulary
    vocab = load_vocab_id_token(args.vocab)
    
    # Validate English exists in vocab
    if "eng" not in vocab:
        raise KeyError("English 'eng' not found in vocab")
    
    if args.per_utt_infer:
        print("Creating per-utterance matrix language dummy predictions:")
        print("  Each utterance uses its own most common non-English language as matrix")
        print("  English (100%%): eng")
        print()
        create_dummy_predictions_per_utt_matrix(args.input, args.output_matrix, args.output_english, vocab)
    else:
        # Global matrix language inference
        if args.matrix_lang is None:
            print("Matrix language not specified, inferring from reference segments...")
            args.matrix_lang = infer_matrix_language(args.input, vocab, per_utt=False)
            print(f"  Inferred matrix language: {args.matrix_lang}")
        
        # Validate languages exist in vocab
        if args.matrix_lang not in vocab:
            raise KeyError(f"Matrix language '{args.matrix_lang}' not found in vocab")
        
        print(f"Creating dummy predictions:")
        print(f"  Matrix language (100%%): {args.matrix_lang}")
        print(f"  English (100%%): eng")
        print()
        
        # Create both variants
        create_dummy_predictions(args.input, args.output_matrix, vocab, args.matrix_lang)
        print()
        create_dummy_predictions(args.input, args.output_english, vocab, "eng")


if __name__ == "__main__":
    main()
