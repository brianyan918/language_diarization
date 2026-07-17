#!/usr/bin/env python3
"""
Find and report on short segments in manifest files.

This script identifies entries with segments shorter than a threshold duration,
helping to understand the impact of short segments on training data.
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

def load_manifest(manifest_path):
    """Load JSON manifest file line by line."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}")
        sys.exit(1)
    
    entries = []
    with open(manifest_path, 'r') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    entries.append((line_no, json.loads(line)))
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON on line {line_no}: {e}")
                    sys.exit(1)
    
    return entries

def find_short_segments(entries, threshold=0.5):
    """Find all entries with segments shorter than threshold."""
    results = []
    
    for line_no, entry in entries:
        segments = entry.get('segments', [])
        short_segs = []
        
        for seg_idx, seg in enumerate(segments):
            duration = seg.get('duration', 0)
            if duration < threshold:
                short_segs.append({
                    'idx': seg_idx,
                    'duration': duration,
                    'text': seg.get('text', ''),
                    'lang': seg.get('lang', '?'),
                    'start': seg.get('audio_start_sec', 0),
                    'end': seg.get('audio_end_sec', 0),
                })
        
        if short_segs:
            results.append({
                'line_no': line_no,
                'entry_id': entry.get('id', '?'),
                'language': entry.get('language', '?'),
                'num_segments': len(segments),
                'short_segments': short_segs,
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Find short segments in manifest files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s -m manifest.json                    # Find all segments < 0.5s
  %(prog)s -m manifest.json -t 0.3             # Find all segments < 0.3s
  %(prog)s -m manifest.json -t 0.5 --stats     # Show summary stats
  %(prog)s -m manifest.json -t 0.5 --export    # Export to CSV
        '''
    )
    parser.add_argument('-m', '--manifest', type=str, required=True,
                        help='Path to manifest JSON file')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Duration threshold in seconds (default: 0.5)')
    parser.add_argument('--stats', action='store_true',
                        help='Show summary statistics only')
    parser.add_argument('--export', type=str, metavar='FILE',
                        help='Export short segment entries to CSV file')
    parser.add_argument('--show-details', action='store_true',
                        help='Show full text of short segments')
    
    args = parser.parse_args()
    
    print(f"Loading manifest: {args.manifest}")
    entries = load_manifest(args.manifest)
    print(f"Loaded {len(entries)} entries\n")
    
    print(f"Searching for segments with duration < {args.threshold}s...")
    results = find_short_segments(entries, threshold=args.threshold)
    
    # Calculate statistics
    total_short_segs = sum(len(r['short_segments']) for r in results)
    
    print(f"Found {len(results)} entries with short segments")
    print(f"Total short segments: {total_short_segs}\n")
    
    if args.stats:
        # Summary stats only
        if results:
            all_durations = []
            for r in results:
                for seg in r['short_segments']:
                    all_durations.append(seg['duration'])
            
            all_durations.sort()
            print(f"Short segment duration statistics:")
            print(f"  Min: {min(all_durations):.4f}s")
            print(f"  Max: {max(all_durations):.4f}s")
            print(f"  Mean: {sum(all_durations)/len(all_durations):.4f}s")
            print(f"  Median: {all_durations[len(all_durations)//2]:.4f}s")
            
            # Distribution
            print(f"\nDuration distribution:")
            buckets = defaultdict(int)
            for d in all_durations:
                bucket = f"{d:.2f}"
                buckets[bucket] += 1
            
            for bucket in sorted(buckets.keys()):
                count = buckets[bucket]
                pct = 100 * count / len(all_durations)
                bar = "█" * int(pct / 2)
                print(f"  {bucket}s: {count:4d} ({pct:5.1f}%) {bar}")
        return
    
    # Detailed output
    if results:
        print("=" * 100)
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Line {result['line_no']}: {result['entry_id']}")
            print(f"    Language: {result['language']}, Total segments: {result['num_segments']}")
            print(f"    Short segments: {len(result['short_segments'])}")
            
            for seg in result['short_segments']:
                text_display = seg['text']
                if not args.show_details and len(text_display) > 40:
                    text_display = text_display[:37] + "..."
                
                print(f"      [{seg['idx']}] {seg['duration']:.4f}s ({seg['lang']:3s}) @ {seg['start']:.4f}-{seg['end']:.4f}s: '{text_display}'")
        
        print("\n" + "=" * 100)
    
    # Export to CSV if requested
    if args.export:
        import csv
        print(f"\nExporting to {args.export}...")
        
        with open(args.export, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['line_no', 'entry_id', 'language', 'segment_idx', 'duration', 'lang', 'text', 'start_sec', 'end_sec'])
            
            for result in results:
                for seg in result['short_segments']:
                    writer.writerow([
                        result['line_no'],
                        result['entry_id'],
                        result['language'],
                        seg['idx'],
                        f"{seg['duration']:.4f}",
                        seg['lang'],
                        seg['text'],
                        f"{seg['start']:.4f}",
                        f"{seg['end']:.4f}",
                    ])
        
        print(f"Exported {total_short_segs} short segments to {args.export}")

if __name__ == '__main__':
    main()
