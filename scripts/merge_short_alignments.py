#!/usr/bin/env python3
"""
merge_short_alignments.py

Merge short alignment segments (< threshold) with adjacent segments in prediction data.

Input format (from diarization predictions):
{
  "record_id": {
    "pred": {
      "alignments": [
        {"start": 0.0, "end": 3.58, "label": 3, "score": null},
        {"start": 3.58, "end": 3.88, "label": 2, "score": null},
        ...
      ]
    },
    "passthrough": {
      "utt_id": "...",
      "file_name": "...",
      "segment_langs": ["ara", "eng", "ara", ...],
      ...
    }
  }
}

When a segment is below the duration threshold:
1. Merge it with the preceding segment (if exists)
2. Merge it with the following segment (if exists)
3. Update start, end, and compute new duration
4. Use the language label from adjacent segments (assume they have same label)

Usage:
  python merge_short_alignments.py -i input.jsonl -o output.jsonl \
    --min_duration 0.5 --merge_mode both

merge_mode:
  - 'both': merge with both preceding and following (default)
  - 'preceding': merge only with preceding segment
  - 'following': merge only with following segment
"""

import argparse
import json
from typing import List, Dict, Any


def merge_alignments(alignments: List[Dict[str, Any]], min_duration: float, merge_mode: str = 'both') -> List[Dict[str, Any]]:
    """
    Merge alignments with duration < min_duration with adjacent alignments.
    
    Args:
        alignments: List of alignment dicts with "start", "end", "label", "score"
        min_duration: Minimum duration threshold in seconds
        merge_mode: 'both', 'preceding', or 'following'
    
    Returns:
        List of merged alignments
    """
    if not alignments:
        return alignments
    
    # Mark alignments to skip (will be merged into others)
    skip_indices = set()
    result = []
    
    for i, align in enumerate(alignments):
        # If this alignment was already merged into another, skip it
        if i in skip_indices:
            continue
        
        start = align.get('start', 0.0)
        end = align.get('end', 0.0)
        duration = end - start
        
        # If alignment is long enough, keep as-is
        if duration >= min_duration:
            result.append(align.copy())
            continue
        
        # Alignment is too short - merge with adjacent alignments
        merged_align = align.copy()
        
        # Collect adjacent alignments to merge
        alignments_to_merge = [merged_align]
        
        # Get preceding alignment if exists and in 'both' or 'preceding' mode
        if (merge_mode in ['both', 'preceding']) and i > 0:
            preceding = alignments[i - 1]
            if i - 1 not in skip_indices:
                alignments_to_merge.insert(0, preceding)
                skip_indices.add(i - 1)
                # Remove the preceding alignment from result if it was just added
                if result and result[-1].get('start') == preceding.get('start'):
                    result.pop()
        
        # Get following alignment if exists and in 'both' or 'following' mode
        if (merge_mode in ['both', 'following']) and i < len(alignments) - 1:
            following = alignments[i + 1]
            alignments_to_merge.append(following)
            skip_indices.add(i + 1)
        
        # Merge the collected alignments
        merged_align = _merge_alignment_group(alignments_to_merge)
        result.append(merged_align)
    
    return result


def _merge_alignment_group(alignments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a group of alignments into one.
    
    Updates:
    - start: from first alignment
    - end: from last alignment
    - label: from adjacent alignments (assume they have same label)
    - score: set to null
    """
    if not alignments:
        return {}
    
    if len(alignments) == 1:
        return alignments[0].copy()
    
    # Start with copy of first alignment
    merged = alignments[0].copy()
    
    # Collect labels
    labels = [align.get('label') for align in alignments]
    
    # Update boundaries
    merged['start'] = alignments[0].get('start', 0.0)
    merged['end'] = alignments[-1].get('end', 0.0)
    merged['score'] = None
    
    # Label: use the most common label from adjacent alignments
    # If the target has different label from adjacent, use adjacent label
    if len(labels) > 1:
        # Get labels of adjacent alignments (first and last)
        adjacent_labels = [labels[0], labels[-1]]
        # Use the most common label code
        if adjacent_labels[0] == adjacent_labels[1]:
            merged['label'] = adjacent_labels[0]
        else:
            # If different, use the first adjacent (preceding)
            merged['label'] = adjacent_labels[0]
    
    return merged


def main():
    ap = argparse.ArgumentParser(
        description="Merge short alignment segments with adjacent segments in prediction data."
    )
    ap.add_argument('-i', '--input', required=True, help='Input JSONL file with pred/passthrough structure')
    ap.add_argument('-o', '--output', required=True, help='Output JSONL file')
    ap.add_argument('--min_duration', type=float, default=0.5, help='Minimum duration threshold in seconds (default: 0.5)')
    ap.add_argument('--merge_mode', choices=['both', 'preceding', 'following'], default='both',
                    help='How to merge short segments (default: both)')
    ap.add_argument('--stats', action='store_true', help='Print merge statistics')
    args = ap.parse_args()
    
    # Process records
    total_records = 0
    records_modified = 0
    total_alignments_before = 0
    total_alignments_after = 0
    total_merged = 0
    
    with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            record_dict = json.loads(line)
            
            # The line is a dict with record_id as key
            for record_id, record_data in record_dict.items():
                total_records += 1
                
                # Extract alignments from pred
                try:
                    pred_data = record_data.get('pred', {})
                    
                    # Handle case where pred is a dict with 'alignments' key
                    if isinstance(pred_data, dict) and 'alignments' in pred_data:
                        alignments = pred_data.get('alignments', [])
                    # Handle case where pred is directly a list of alignments
                    elif isinstance(pred_data, list):
                        alignments = pred_data
                    # Handle case where pred doesn't have alignments
                    else:
                        alignments = []
                        
                except (AttributeError, TypeError):
                    # If structure is unexpected, write as-is
                    outfile.write(json.dumps({record_id: record_data}, ensure_ascii=False) + '\n')
                    continue
                
                if not isinstance(alignments, list):
                    outfile.write(json.dumps({record_id: record_data}, ensure_ascii=False) + '\n')
                    continue
                
                total_alignments_before += len(alignments)
                
                # Count short alignments before merging
                short_count_before = sum(1 for align in alignments 
                                        if (align.get('end', 0.0) - align.get('start', 0.0)) < args.min_duration)
                
                # Merge short alignments
                merged_alignments = merge_alignments(alignments, args.min_duration, args.merge_mode)
                
                total_alignments_after += len(merged_alignments)
                merged_count = len(alignments) - len(merged_alignments)
                total_merged += merged_count
                
                # Update record - handle both dict and list pred formats
                if isinstance(pred_data, dict):
                    record_data['pred']['alignments'] = merged_alignments
                elif isinstance(pred_data, list):
                    record_data['pred'] = merged_alignments
                
                if merged_count > 0:
                    records_modified += 1
                    if args.stats:
                        utt_id = record_data.get('passthrough', {}).get('utt_id', record_id)
                        print(f"  {utt_id}: merged {merged_count} alignments "
                              f"({len(alignments)} -> {len(merged_alignments)})")
                
                outfile.write(json.dumps({record_id: record_data}, ensure_ascii=False) + '\n')
    
    print(f"\n=== Merge Statistics ===")
    print(f"Total records processed: {total_records}")
    print(f"Records with merges: {records_modified}")
    print(f"Total alignments before: {total_alignments_before}")
    print(f"Total alignments after: {total_alignments_after}")
    print(f"Total alignments merged: {total_merged}")
    print(f"Merge mode: {args.merge_mode}")
    print(f"Duration threshold: {args.min_duration}s")
    print(f"\nOutput written to: {args.output}")


if __name__ == '__main__':
    main()
