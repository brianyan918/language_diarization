
#!/usr/bin/env python3
"""
analyze_overlaps.py

Summarize overlap statistics from a JSONL log file.
- Reports total and per-language overlap counts for 'ref' and 'hyp'.
- Optionally prints detailed overlap segment info.

Usage:
  python analyze_overlaps.py --input log.jsonl [--details]
"""
import argparse
import json
from collections import defaultdict, Counter
import re

def strip_id_prefix(lang):
    # Remove 'test-####-' or similar prefix, keep only language code (e.g., 'ara-eng')
    return re.sub(r'^\w+-\d+-', '', lang)

def get_segment_duration(seg):
    """Extract duration from a segment dict, trying various field names."""
    if not isinstance(seg, dict):
        return 0.0
    # Try common field name patterns
    start = seg.get('start_time') or seg.get('start')
    end = seg.get('end_time') or seg.get('end')
    if start is None or end is None:
        return 0.0
    try:
        return max(0.0, float(end) - float(start))
    except (ValueError, TypeError):
        return 0.0

def parse_overlap_duration(overlap_str):
    """
    Parse overlap duration from a string like:
    "hyp overlap in id=test-104-ara_1790_BT: seg1 [4.800000,5.080000] overlaps seg5 [4.980000,5.280000]"
    
    Returns the duration of the overlap region (intersection of two time ranges).
    """
    if not isinstance(overlap_str, str):
        return 0.0
    
    # Find all time ranges in brackets [start,end]
    import re
    matches = re.findall(r'\[([0-9.]+),([0-9.]+)\]', overlap_str)
    if len(matches) < 2:
        return 0.0
    
    try:
        # Extract the two time ranges
        seg1_start, seg1_end = float(matches[0][0]), float(matches[0][1])
        seg2_start, seg2_end = float(matches[1][0]), float(matches[1][1])
        
        # Compute intersection
        overlap_start = max(seg1_start, seg2_start)
        overlap_end = min(seg1_end, seg2_end)
        
        return max(0.0, overlap_end - overlap_start)
    except (ValueError, IndexError):
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input JSONL log file')
    parser.add_argument('--details', action='store_true', help='Print detailed overlap segment info')
    args = parser.parse_args()

    total = Counter()
    per_lang = defaultdict(Counter)
    per_lang_entries = defaultdict(int)
    per_lang_entries_with_overlap = defaultdict(int)
    per_lang_duration = defaultdict(float)
    total_duration = 0.0
    overlap_segments = defaultdict(lambda: defaultdict(list))  # lang_code -> ref/hyp -> list of segments
    overlap_duration = defaultdict(lambda: defaultdict(float))  # lang_code -> ref/hyp -> total duration

    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            lang = obj.get('language', 'unknown')
            lang_code = strip_id_prefix(lang)
            per_lang_entries[lang_code] += 1
            overlaps = obj.get('overlaps', {})
            overlap_counts = obj.get('overlap_counts', {})
            
            # Try to get total segment duration from all ref segments
            ref_segs = obj.get('ref', [])
            seg_duration = 0.0
            if ref_segs and isinstance(ref_segs, list):
                for seg in ref_segs:
                    seg_duration += get_segment_duration(seg)
            per_lang_duration[lang_code] += seg_duration
            total_duration += seg_duration
            
            for who in ('ref', 'hyp'):
                count = overlap_counts.get(who, 0)
                total[who] += count
                per_lang[lang_code][who] += count
                segs = overlaps.get(who, [])
                if segs:
                    per_lang_entries_with_overlap[lang_code] += 1
                    overlap_segments[lang_code][who].extend(segs)
                    # Sum duration of overlap segments
                    for seg in segs:
                        overlap_duration[lang_code][who] += parse_overlap_duration(seg)

    print('=== Overlap Summary ===')
    print(f"Total overlaps (ref): {total['ref']}")
    print(f"Total overlaps (hyp): {total['hyp']}")
    print(f"Total entries processed: {sum(per_lang_entries.values())}")
    print(f"Entries with overlaps: {sum(per_lang_entries_with_overlap.values())}")
    print(f"Total duration (all languages): {total_duration:.2f}s")
    
    # Calculate global overlap duration
    global_ref_dur = sum(overlap_duration[lang]['ref'] for lang in overlap_duration)
    global_hyp_dur = sum(overlap_duration[lang]['hyp'] for lang in overlap_duration)
    global_total_dur = global_ref_dur + global_hyp_dur
    global_ref_pct = 100.0 * global_ref_dur / total_duration if total_duration > 0 else 0.0
    global_hyp_pct = 100.0 * global_hyp_dur / total_duration if total_duration > 0 else 0.0
    global_total_pct = 100.0 * global_total_dur / total_duration if total_duration > 0 else 0.0
    
    print(f"Total overlap duration (ref): {global_ref_dur:.2f}s ({global_ref_pct:.2f}% of total)")
    print(f"Total overlap duration (hyp): {global_hyp_dur:.2f}s ({global_hyp_pct:.2f}% of total)")
    print(f"Total overlap duration (both): {global_total_dur:.2f}s ({global_total_pct:.2f}% of total)")
    print()
    print('--- Per-language overlap counts ---')
    print(f"{'Language':<20s} {'Entries':>8s} {'Ref Ovlp':>10s} {'Hyp Ovlp':>10s} {'Total':>10s} {'Ref Dur(s)':>12s} {'Ref %':>8s} {'Hyp Dur(s)':>12s} {'Hyp %':>8s}")
    print("-" * 110)
    for lang_code in sorted(per_lang):
        ref_count = per_lang[lang_code]['ref']
        hyp_count = per_lang[lang_code]['hyp']
        total_count = ref_count + hyp_count
        entries = per_lang_entries[lang_code]
        ref_dur = overlap_duration[lang_code]['ref']
        hyp_dur = overlap_duration[lang_code]['hyp']
        lang_total_dur = per_lang_duration[lang_code]
        ref_pct = 100.0 * ref_dur / lang_total_dur if lang_total_dur > 0 else 0.0
        hyp_pct = 100.0 * hyp_dur / lang_total_dur if lang_total_dur > 0 else 0.0
        print(f"{lang_code:<20s} {entries:>8d} {ref_count:>10d} {hyp_count:>10d} {total_count:>10d} {ref_dur:>12.2f} {ref_pct:>7.2f}% {hyp_dur:>12.2f} {hyp_pct:>7.2f}%")
    print()
    if args.details:
        print('--- Overlap segment details (by language) ---')
        for lang_code in sorted(overlap_segments):
            for who in ('ref', 'hyp'):
                segs = overlap_segments[lang_code][who]
                if segs:
                    print(f"{lang_code} [{who}] ({len(segs)} segments):")
                    for seg in segs:
                        print(f"  {seg}")
    else:
        print('--- Example overlap segments per language ---')
        for lang_code in sorted(overlap_segments):
            for who in ('ref', 'hyp'):
                segs = overlap_segments[lang_code][who]
                if segs:
                    print(f"{lang_code} [{who}] ({len(segs)} total segments):")
                    print(f"  Example: {segs[0]}")

if __name__ == '__main__':
    main()
