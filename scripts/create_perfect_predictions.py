#!/usr/bin/env python3
"""
create_perfect_predictions.py

Read a JSONL predictions file and create a new JSONL where predictions
perfectly match the reference (segment_timestamps and segment_langs from passthrough).

Usage:
  python scripts/create_perfect_predictions.py \
    --input langdiar_whisper_multi.0.jsonl \
    --vocab data/vocab_102.txt \
    --output langdiar_whisper_multi.0.perfect.jsonl
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


def create_perfect_predictions(
    input_jsonl: str,
    output_jsonl: str,
    vocab: Dict[str, int],
) -> None:
    """
    Read JSONL file and create perfect predictions based on reference.
    """
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
                
                # Extract reference from passthrough
                passthrough = entry.get("passthrough", {})
                seg_ts = passthrough.get("segment_timestamps", [])
                seg_langs = passthrough.get("segment_langs", [])
                
                if len(seg_ts) != len(seg_langs):
                    raise ValueError(f"Mismatch: {len(seg_ts)} timestamps vs {len(seg_langs)} langs")
                
                # Create perfect predictions from reference
                perfect_pred: List[dict] = []
                for (start, end), lang in zip(seg_ts, seg_langs):
                    if lang not in vocab:
                        raise KeyError(f"Language '{lang}' not in vocab")
                    
                    label_id = vocab[lang]
                    perfect_pred.append({
                        "start": float(start),
                        "end": float(end),
                        "label": label_id,
                        "score": None
                    })
                
                # Replace pred with perfect predictions
                entry["pred"] = perfect_pred
                
                # Write to output
                f_out.write(json.dumps(obj) + "\n")
                
            except Exception as e:
                print(f"WARNING: Skipping line {ln}: {e}")
                skipped_lines += 1
                continue
        
        print(f"Processed {total_lines} lines")
        print(f"Skipped {skipped_lines} lines")
        print(f"Output written to: {output_jsonl}")


def main():
    ap = argparse.ArgumentParser(
        description="Create perfect predictions from reference segments."
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
        "--output",
        required=True,
        help="Output JSONL file with perfect predictions.",
    )
    args = ap.parse_args()
    
    # Load vocabulary
    vocab = load_vocab_id_token(args.vocab)
    
    # Create perfect predictions
    create_perfect_predictions(args.input, args.output, vocab)


if __name__ == "__main__":
    main()
