#!/usr/bin/env python3
"""
analyze_hyp_overlaps.py

Analyze duration of overlap time in hypothesized segments.

Input: JSONL with structure:
{
  "utterance_id": {
    "hyp": [
      {"session_id": "...", "start_time": X, "end_time": Y, "speaker": "...", ...},
      ...
    ],
    "overlaps": {
      "hyp": [
        "hyp overlap in id=session_id: seg0 [start,end] overlaps seg1 [start,end]",
        ...
      ],
      "ref": [...]
    },
    "language": "ara"  # or inferred from session_id
  }
}

Metrics:
- Overlap percentage = total_overlap_duration / total_utterance_duration
- Per-language breakdown using the "language" attribute
- Per-utterance report

Usage:
  python analyze_hyp_overlaps.py --input_jsonl predictions.jsonl \
    --output_jsonl overlaps.jsonl --output_report report.txt \
    --exclude_langs eng,ara --ignore_id_prefix
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def strip_id_prefix(utt_id: str) -> str:
    """Strip prefix like 'test-0-' from utterance ID."""
    parts = utt_id.split("-")
    # Remove numeric prefixes at the start (e.g., "test", "0", "1", etc.)
    while parts and (parts[0].isdigit() or parts[0] in ("test", "train", "dev", "val")):
        parts.pop(0)
    return "-".join(parts)


def extract_language_from_id(session_id: str, ignore_prefix: bool = False) -> Optional[str]:
    """
    Extract language code from session_id.
    
    Examples:
      "test-0-ara_1667_BT" -> "ara"
      "ara_1667_BT" -> "ara"
    """
    if ignore_prefix:
        session_id = strip_id_prefix(session_id)
    
    # Extract first 3-letter language code
    match = re.match(r"([a-z]{3})", session_id)
    if match:
        return match.group(1)
    return None


def parse_overlap_string(overlap_str: str) -> Tuple[float, float]:
    """
    Parse overlap string like:
    'hyp overlap in id=test-0-ara_1667_BT: seg0 [0.000000,6.640000] overlaps seg3 [6.600000,8.580000]'
    
    Returns the duration of overlap (intersection of the two time intervals).
    """
    # Find all [start,end] patterns
    pattern = r'\[(\d+\.?\d*),(\d+\.?\d*)\]'
    matches = re.findall(pattern, overlap_str)
    
    if len(matches) < 2:
        return 0.0
    
    # Extract the two time intervals
    start1, end1 = float(matches[0][0]), float(matches[0][1])
    start2, end2 = float(matches[1][0]), float(matches[1][1])
    
    # Calculate overlap
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_duration = max(0.0, overlap_end - overlap_start)
    
    return overlap_duration


def calculate_utterance_duration(hyp_segments: List[dict]) -> float:
    """Calculate total duration from min start_time to max end_time."""
    if not hyp_segments:
        return 0.0
    
    min_start = min(seg["start_time"] for seg in hyp_segments)
    max_end = max(seg["end_time"] for seg in hyp_segments)
    
    return max(0.0, max_end - min_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True, help="Input JSONL file with overlap data.")
    ap.add_argument("--output_jsonl", help="Optional output JSONL with per-utterance metrics.")
    ap.add_argument("--output_report", help="Optional output text report.")
    ap.add_argument("--exclude_langs", type=str, default="", help="Comma-separated languages to exclude.")
    ap.add_argument("--ignore_id_prefix", action="store_true", help="Strip prefixes from session_id when extracting language.")
    args = ap.parse_args()
    
    # Parse excluded languages
    excluded_langs = set(lang.strip() for lang in args.exclude_langs.split(",") if lang.strip())
    
    # Accumulators
    global_overlap_duration = 0.0
    global_utterance_duration = 0.0
    global_utts = 0
    
    # Per-language breakdown
    overlap_by_lang: Dict[str, float] = defaultdict(float)
    duration_by_lang: Dict[str, float] = defaultdict(float)
    utts_by_lang: Dict[str, int] = defaultdict(int)
    
    # Per-utterance results
    per_utt_results: List[dict] = []
    
    # Read input
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                
                # Get utterance ID and language
                utt_id = entry.get("id", f"utt_{ln}")
                lang_field = entry.get("language")
                
                # Extract hyp segments and overlaps
                hyp_segments = entry.get("hyp", [])
                overlaps_obj = entry.get("overlaps", {})
                hyp_overlaps = overlaps_obj.get("hyp", [])
                
                if not hyp_segments or not hyp_overlaps:
                    continue
                
                # Calculate total overlap duration
                overlap_duration = sum(parse_overlap_string(ov) for ov in hyp_overlaps)
                
                # Calculate total utterance duration
                utterance_duration = calculate_utterance_duration(hyp_segments)
                
                if utterance_duration <= 0:
                    continue
                
                # Calculate overlap percentage
                overlap_pct = (overlap_duration / utterance_duration) * 100 if utterance_duration > 0 else 0.0
                
                # Infer language from "language" field or session_id
                lang = None
                if lang_field:
                    # Extract language from lang_field like "test-0-ara-eng" -> extract first 3-char lang
                    lang_parts = lang_field.split("-")
                    if args.ignore_id_prefix:
                        # Remove prefixes like "test", "0"
                        while lang_parts and (lang_parts[0].isdigit() or lang_parts[0] in ("test", "train", "dev", "val")):
                            lang_parts.pop(0)
                    # Get first remaining part that looks like a language code
                    for part in lang_parts:
                        if len(part) == 3 and part.isalpha() and part.islower():
                            lang = part
                            break
                
                # Fallback: extract from session_id
                if not lang:
                    session_id = None
                    for seg in hyp_segments:
                        if "session_id" in seg:
                            session_id = seg["session_id"]
                            break
                    
                    if session_id:
                        lang = extract_language_from_id(session_id, ignore_prefix=args.ignore_id_prefix)
                
                if not lang or lang in excluded_langs:
                    continue
                
                # Accumulate global metrics
                global_overlap_duration += overlap_duration
                global_utterance_duration += utterance_duration
                global_utts += 1
                
                # Accumulate per-language metrics
                overlap_by_lang[lang] += overlap_duration
                duration_by_lang[lang] += utterance_duration
                utts_by_lang[lang] += 1
                
                # Store per-utterance result
                per_utt_results.append({
                    "utt_id": utt_id,
                    "language": lang,
                    "overlap_duration": overlap_duration,
                    "utterance_duration": utterance_duration,
                    "overlap_percentage": overlap_pct,
                    "num_overlaps": len(hyp_overlaps),
                    "num_segments": len(hyp_segments)
                })
                
            except Exception as e:
                print(f"Error processing line {ln}: {e}")
                continue
    
    # Print global summary
    print("=== GLOBAL OVERLAP SUMMARY ===")
    print(f"Utterances: {global_utts}")
    print(f"Total overlap duration: {global_overlap_duration:.3f}s")
    print(f"Total utterance duration: {global_utterance_duration:.3f}s")
    if global_utterance_duration > 0:
        global_overlap_pct = (global_overlap_duration / global_utterance_duration) * 100
        print(f"Overall overlap percentage: {global_overlap_pct:.2f}%")
    else:
        print("No utterances with overlap data found.")
    
    # Print per-language breakdown
    print("\n=== PER-LANGUAGE BREAKDOWN ===")
    for lang in sorted(overlap_by_lang.keys()):
        overlap = overlap_by_lang[lang]
        duration = duration_by_lang[lang]
        utts = utts_by_lang[lang]
        pct = (overlap / duration * 100) if duration > 0 else 0.0
        
        print(f"[{lang}] Utts: {utts:4d}  Overlap: {overlap:7.3f}s  Duration: {duration:8.3f}s  Overlap %: {pct:6.2f}%")
    
    # Write per-utterance JSONL if requested
    if args.output_jsonl:
        # Sort by overlap percentage (worst first)
        per_utt_results.sort(key=lambda x: x["overlap_percentage"], reverse=True)
        
        with open(args.output_jsonl, 'w', encoding='utf-8') as f:
            for result in per_utt_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"\nPer-utterance results saved to: {args.output_jsonl}")
    
    # Write text report if requested
    if args.output_report:
        with open(args.output_report, 'w', encoding='utf-8') as f:
            f.write("=== GLOBAL OVERLAP SUMMARY ===\n")
            f.write(f"Utterances: {global_utts}\n")
            f.write(f"Total overlap duration: {global_overlap_duration:.3f}s\n")
            f.write(f"Total utterance duration: {global_utterance_duration:.3f}s\n")
            if global_utterance_duration > 0:
                global_overlap_pct = (global_overlap_duration / global_utterance_duration) * 100
                f.write(f"Overall overlap percentage: {global_overlap_pct:.2f}%\n")
            else:
                f.write("No utterances with overlap data found.\n")
            
            f.write("\n=== PER-LANGUAGE BREAKDOWN ===\n")
            for lang in sorted(overlap_by_lang.keys()):
                overlap = overlap_by_lang[lang]
                duration = duration_by_lang[lang]
                utts = utts_by_lang[lang]
                pct = (overlap / duration * 100) if duration > 0 else 0.0
                
                f.write(f"[{lang}] Utts: {utts:4d}  Overlap: {overlap:7.3f}s  Duration: {duration:8.3f}s  Overlap %: {pct:6.2f}%\n")
            
            f.write("\n=== TOP 20 UTTERANCES BY OVERLAP % ===\n")
            # Sort by overlap percentage
            sorted_utts = sorted(per_utt_results, key=lambda x: x["overlap_percentage"], reverse=True)
            for result in sorted_utts[:20]:
                f.write(
                    f"{result['utt_id']:40s} "
                    f"[{result['language']}] "
                    f"Overlap: {result['overlap_percentage']:6.2f}% "
                    f"({result['overlap_duration']:7.3f}s / {result['utterance_duration']:8.3f}s) "
                    f"Segs: {result['num_segments']} Overlaps: {result['num_overlaps']}\n"
                )
        
        print(f"Report saved to: {args.output_report}")
    
    # Print easy copy-paste section per language
    print("\n" + "="*80)
    print("EASY COPY-PASTE FORMAT (Overlap % by language, alphabetically sorted)")
    print("="*80 + "\n")
    
    for lang in sorted(overlap_by_lang.keys()):
        overlap = overlap_by_lang[lang]
        duration = duration_by_lang[lang]
        pct = (overlap / duration * 100) if duration > 0 else 0.0
        
        print(f"{pct:.2f}%")


if __name__ == "__main__":
    main()
