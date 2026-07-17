#!/usr/bin/env python3
"""
merge_short_segments.py

Merge short audio segments (< threshold) with adjacent segments.

When a segment is below the duration threshold:
1. Merge it with the preceding segment (if exists)
2. Merge it with the following segment (if exists)
3. Combine text, normalized_text, uroman_tokens with spaces
4. Update audio_start_sec, audio_end_sec, duration to span all merged segments
5. Use the language code of adjacent segments (assume they have same lang)

Input JSONL format:
{
  "id": "...",
  "file_name": "...",
  "segments": [
    {"audio_start_sec": ..., "audio_end_sec": ..., "duration": ..., "text": "...", "lang": "ara"},
    {"audio_start_sec": ..., "audio_end_sec": ..., "duration": ..., "text": "...", "lang": "eng"},
    ...
  ],
  ...
}

Output: JSONL with merged segments, preserving all other fields.

Usage:
  python merge_short_segments.py -i input.jsonl -o output.jsonl \
    --min_duration 0.5 --merge_mode both

merge_mode:
  - 'both': merge with both preceding and following (default)
  - 'preceding': merge only with preceding segment
  - 'following': merge only with following segment
"""

import argparse
import json
from typing import List, Dict, Any


def merge_segments(segments: List[Dict[str, Any]], min_duration: float, merge_mode: str = 'both') -> List[Dict[str, Any]]:
    """
    Merge segments with duration < min_duration with adjacent segments.
    
    Args:
        segments: List of segment dicts
        min_duration: Minimum duration threshold in seconds
        merge_mode: 'both', 'preceding', or 'following'
    
    Returns:
        List of merged segments
    """
    if not segments:
        return segments
    
    # Mark segments to skip (will be merged into others)
    skip_indices = set()
    result = []
    
    for i, seg in enumerate(segments):
        # If this segment was already merged into another, skip it
        if i in skip_indices:
            continue
        
        duration = seg.get('duration', 0.0)
        
        # If segment is long enough, keep as-is
        if duration >= min_duration:
            result.append(seg.copy())
            continue
        
        # Segment is too short - merge with adjacent segments
        merged_seg = seg.copy()
        
        # Collect adjacent segments to merge
        segments_to_merge = [merged_seg]
        
        # Get preceding segment if exists and in 'both' or 'preceding' mode
        if (merge_mode in ['both', 'preceding']) and i > 0:
            preceding = segments[i - 1]
            if i - 1 not in skip_indices:
                segments_to_merge.insert(0, preceding)
                skip_indices.add(i - 1)
                # Remove the preceding segment from result if it was just added
                if result and result[-1].get('audio_start_sec') == preceding.get('audio_start_sec'):
                    result.pop()
        
        # Get following segment if exists and in 'both' or 'following' mode
        if (merge_mode in ['both', 'following']) and i < len(segments) - 1:
            following = segments[i + 1]
            segments_to_merge.append(following)
            skip_indices.add(i + 1)
        
        # Merge the collected segments
        merged_seg = _merge_segment_group(segments_to_merge)
        result.append(merged_seg)
    
    return result


def _merge_segment_group(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a group of segments into one.
    
    Combines:
    - text, normalized_text, uroman_tokens: joined with spaces
    - audio_start_sec: from first segment
    - audio_end_sec: from last segment
    - duration: calculated from start/end
    - lang: from adjacent segments (assume they have same lang)
    - other fields: kept from first segment
    """
    if not segments:
        return {}
    
    if len(segments) == 1:
        return segments[0].copy()
    
    # Start with copy of first segment
    merged = segments[0].copy()
    
    # Collect text fields to merge
    text_parts = [safe_str(segments[0].get('text', ''))]
    normalized_parts = [safe_str(segments[0].get('normalized_text', ''))]
    uroman_parts = [safe_str(segments[0].get('uroman_tokens', ''))]
    
    lang_codes = [segments[0].get('lang', 'unknown')]
    
    # Merge with remaining segments
    for seg in segments[1:]:
        text_parts.append(safe_str(seg.get('text', '')))
        normalized_parts.append(safe_str(seg.get('normalized_text', '')))
        uroman_parts.append(safe_str(seg.get('uroman_tokens', '')))
        lang_codes.append(seg.get('lang', 'unknown'))
    
    # Update fields
    merged['text'] = ' '.join(text_parts).strip()
    merged['normalized_text'] = ' '.join(normalized_parts).strip()
    merged['uroman_tokens'] = ' '.join(uroman_parts).strip()
    
    # Update time boundaries
    merged['audio_start_sec'] = segments[0].get('audio_start_sec', 0.0)
    merged['audio_end_sec'] = segments[-1].get('audio_end_sec', 0.0)
    merged['duration'] = merged['audio_end_sec'] - merged['audio_start_sec']
    
    # Language: use the most common lang from adjacent segments
    # If the target segment has different lang from adjacent, use adjacent lang
    # If multiple different langs in adjacent, keep first
    if len(lang_codes) > 1:
        # Get langs of adjacent segments (all except middle if odd number)
        adjacent_langs = [lang_codes[0], lang_codes[-1]]
        # Use the most common language code
        if adjacent_langs[0] == adjacent_langs[1]:
            merged['lang'] = adjacent_langs[0]
        else:
            # If different, use the first adjacent (preceding)
            merged['lang'] = adjacent_langs[0]
    
    return merged


def safe_str(x: Any) -> str:
    """Safe string conversion."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def main():
    ap = argparse.ArgumentParser(
        description="Merge short audio segments with adjacent segments."
    )
    ap.add_argument('-i', '--input', required=True, help='Input JSONL file')
    ap.add_argument('-o', '--output', required=True, help='Output JSONL file')
    ap.add_argument('--min_duration', type=float, default=0.5, help='Minimum duration threshold in seconds (default: 0.5)')
    ap.add_argument('--merge_mode', choices=['both', 'preceding', 'following'], default='both',
                    help='How to merge short segments (default: both)')
    ap.add_argument('--stats', action='store_true', help='Print merge statistics')
    args = ap.parse_args()
    
    # Process records
    total_records = 0
    records_modified = 0
    total_segments_before = 0
    total_segments_after = 0
    total_merged = 0
    
    with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            total_records += 1
            
            segments = record.get('segments', [])
            if not isinstance(segments, list):
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                continue
            
            total_segments_before += len(segments)
            
            # Count short segments before merging
            short_count_before = sum(1 for seg in segments if seg.get('duration', 0.0) < args.min_duration)
            
            # Merge short segments
            merged_segments = merge_segments(segments, args.min_duration, args.merge_mode)
            
            total_segments_after += len(merged_segments)
            merged_count = len(segments) - len(merged_segments)
            total_merged += merged_count
            
            # Update record
            record['segments'] = merged_segments
            
            if merged_count > 0:
                records_modified += 1
                if args.stats:
                    print(f"  {record.get('id')}: merged {merged_count} segments "
                          f"({len(segments)} -> {len(merged_segments)})")
            
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n=== Merge Statistics ===")
    print(f"Total records processed: {total_records}")
    print(f"Records with merges: {records_modified}")
    print(f"Total segments before: {total_segments_before}")
    print(f"Total segments after: {total_segments_after}")
    print(f"Total segments merged: {total_merged}")
    print(f"Merge mode: {args.merge_mode}")
    print(f"Duration threshold: {args.min_duration}s")
    print(f"\nOutput written to: {args.output}")


if __name__ == '__main__':
    main()
