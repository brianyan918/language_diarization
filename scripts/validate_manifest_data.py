#!/usr/bin/env python3
"""
Validation script to detect data issues that would cause training failures.

This script checks for:
1. Invalid segment durations (too short, negative, zero)
2. Invalid time boundaries (end < start, negative times)
3. Audio file accessibility
4. Missing required fields
5. Language field consistency
6. Text field validation
7. Tokenization issues (cross-lingual segments, empty segments)
8. Segment count and statistics
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import os

def load_manifest(manifest_path):
    """Load JSONL or JSON manifest file."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}")
        sys.exit(1)
    
    data = []
    with open(manifest_path, 'r') as f:
        if manifest_path.suffix == '.jsonl':
            for line_no, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON on line {line_no}: {e}")
                    sys.exit(1)
        else:  # .json
            try:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                elif isinstance(content, dict) and 'entries' in content:
                    data = content['entries']
                else:
                    print(f"ERROR: Unexpected JSON structure in {manifest_path}")
                    sys.exit(1)
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON: {e}")
                sys.exit(1)
    
    return data

def get_segments_from_entry(entry):
    """Extract segments array from manifest entry."""
    if 'segments' in entry:
        return entry['segments']
    return []

def check_segment_validity(segment, entry_idx, seg_idx, manifest_path):
    """Check if a segment is valid and return list of issues."""
    issues = []
    
    # Required fields
    required_fields = ['text', 'lang', 'duration', 'audio_start_sec', 'audio_end_sec']
    for field in required_fields:
        if field not in segment:
            issues.append(f"  MISSING_FIELD: segment[{seg_idx}] missing '{field}'")
    
    if not issues:  # Only check values if fields exist
        # Check duration
        duration = segment.get('duration', 0)
        if duration <= 0:
            issues.append(f"  INVALID_DURATION: segment[{seg_idx}] duration={duration:.4f}s (must be > 0)")
        elif duration < 0.05:
            issues.append(f"  TOO_SHORT: segment[{seg_idx}] duration={duration:.4f}s (< 50ms)")
        
        # Check time boundaries
        start = segment.get('audio_start_sec', 0)
        end = segment.get('audio_end_sec', 0)
        
        if start < 0 or end < 0:
            issues.append(f"  NEGATIVE_TIME: segment[{seg_idx}] start={start:.4f}, end={end:.4f}")
        
        if end <= start:
            issues.append(f"  INVALID_BOUNDS: segment[{seg_idx}] end={end:.4f} <= start={start:.4f}")
        
        # Check text field
        text = segment.get('text', '').strip()
        if not text:
            issues.append(f"  EMPTY_TEXT: segment[{seg_idx}] text is empty or whitespace")
        
        # Check language field
        lang = segment.get('lang', '').strip()
        if not lang:
            issues.append(f"  EMPTY_LANG: segment[{seg_idx}] lang is empty")
        elif len(lang) < 2:
            issues.append(f"  INVALID_LANG: segment[{seg_idx}] lang='{lang}' (too short)")
    
    return issues

def check_cross_lingual_segments(segments):
    """Detect cross-lingual segments (merged segments with mixed languages)."""
    issues = []
    
    for seg_idx, segment in enumerate(segments):
        if 'text' not in segment or 'lang' not in segment:
            continue
        
        text = segment.get('text', '')
        lang = segment.get('lang', '')
        
        # Heuristic: if segment duration > 0.6s and contains multiple language-specific patterns,
        # it might be a merged segment with mixed languages
        duration = segment.get('duration', 0)
        
        # Count spaces - merged segments often have unusual space counts
        space_count = text.count(' ')
        if duration > 0.5 and space_count >= 1:
            # This is a potential cross-lingual segment
            # We can't definitively know without phonetic analysis, but flag it
            issues.append(f"  POTENTIAL_CROSS_LINGUAL: segment[{seg_idx}] duration={duration:.4f}s, lang={lang}, spaces={space_count}, text='{text[:50]}'")
    
    return issues

def check_audio_file(entry):
    """Check if referenced audio file exists."""
    issues = []
    
    if 'audio_path' in entry:
        audio_path = Path(entry['audio_path'])
        if not audio_path.exists():
            issues.append(f"  MISSING_AUDIO: audio file not found: {audio_path}")
    
    return issues

def validate_manifest(manifest_path, verbose=False):
    """Run full validation on manifest file."""
    print(f"\n{'=' * 80}")
    print(f"VALIDATING: {manifest_path}")
    print(f"{'=' * 80}\n")
    
    data = load_manifest(manifest_path)
    print(f"Loaded {len(data)} entries")
    
    # Track issues
    all_issues = defaultdict(list)
    stats = {
        'total_entries': 0,
        'total_segments': 0,
        'min_duration': float('inf'),
        'max_duration': 0,
        'avg_duration': 0,
        'short_segments': 0,  # < 0.5s
        'languages': defaultdict(int),
    }
    
    total_duration = 0
    
    # Validate each entry and segment
    for entry_idx, entry in enumerate(data):
        stats['total_entries'] += 1
        
        # Check audio file
        audio_issues = check_audio_file(entry)
        if audio_issues:
            all_issues['audio'].extend(audio_issues)
        
        # Get segments
        segments = get_segments_from_entry(entry)
        
        for seg_idx, segment in enumerate(segments):
            stats['total_segments'] += 1
            
            # Check segment validity
            seg_issues = check_segment_validity(segment, entry_idx, seg_idx, manifest_path)
            if seg_issues:
                all_issues['segments'].extend(seg_issues)
            
            # Collect stats if valid
            if not seg_issues:
                duration = segment.get('duration', 0)
                stats['min_duration'] = min(stats['min_duration'], duration)
                stats['max_duration'] = max(stats['max_duration'], duration)
                total_duration += duration
                
                if duration < 0.5:
                    stats['short_segments'] += 1
                
                lang = segment.get('lang', 'unknown')
                stats['languages'][lang] += 1
        
        # Check cross-lingual segments
        cross_lingual_issues = check_cross_lingual_segments(segments)
        if cross_lingual_issues:
            all_issues['cross_lingual'].extend(cross_lingual_issues)
    
    # Calculate average
    if stats['total_segments'] > 0:
        stats['avg_duration'] = total_duration / stats['total_segments']
    
    # Print statistics
    print("STATISTICS:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  Duration range: {stats['min_duration']:.4f}s - {stats['max_duration']:.4f}s")
    print(f"  Average duration: {stats['avg_duration']:.4f}s")
    print(f"  Short segments (< 0.5s): {stats['short_segments']} ({100*stats['short_segments']/max(1, stats['total_segments']):.1f}%)")
    print(f"  Language distribution:")
    for lang in sorted(stats['languages'].keys()):
        count = stats['languages'][lang]
        pct = 100 * count / max(1, stats['total_segments'])
        print(f"    {lang}: {count:6d} ({pct:5.1f}%)")
    
    # Print issues
    if all_issues:
        print(f"\n{'=' * 80}")
        print("ISSUES FOUND:")
        print(f"{'=' * 80}\n")
        
        for issue_type in sorted(all_issues.keys()):
            issues = all_issues[issue_type]
            print(f"{issue_type.upper()} ({len(issues)} issues):")
            
            if verbose or issue_type == 'audio':
                # Show all audio issues
                for issue in issues[:20]:  # Limit output
                    print(issue)
                if len(issues) > 20:
                    print(f"  ... and {len(issues) - 20} more")
            else:
                # Show first 5 of each type
                for issue in issues[:5]:
                    print(issue)
                if len(issues) > 5:
                    print(f"  ... and {len(issues) - 5} more")
            print()
        
        return False
    else:
        print("\n✓ No issues found!")
        return True

def compare_manifests(manifest1, manifest2):
    """Compare two manifest files to identify what changed."""
    print(f"\n{'=' * 80}")
    print("COMPARING MANIFESTS")
    print(f"{'=' * 80}\n")
    
    data1 = load_manifest(manifest1)
    data2 = load_manifest(manifest2)
    
    print(f"Original (working):  {len(data1)} entries")
    print(f"New (broken):        {len(data2)} entries")
    
    # Get stats for each
    print("\nORIGINAL MANIFEST:")
    segs1 = sum(len(get_segments_from_entry(e)) for e in data1)
    print(f"  Total segments: {segs1}")
    
    print("\nNEW MANIFEST:")
    segs2 = sum(len(get_segments_from_entry(e)) for e in data2)
    print(f"  Total segments: {segs2}")
    print(f"  Difference: {segs2 - segs1:+d} segments")
    
    # Analyze first few entries to see differences
    print("\nFIRST 3 ENTRIES - SEGMENT COUNT COMPARISON:")
    for i in range(min(3, len(data1), len(data2))):
        segs1_count = len(get_segments_from_entry(data1[i]))
        segs2_count = len(get_segments_from_entry(data2[i]))
        print(f"  Entry {i}: {segs1_count} segments -> {segs2_count} segments ({segs2_count - segs1_count:+d})")
        
        # Show details of what changed
        if segs1_count != segs2_count:
            seg1 = get_segments_from_entry(data1[i])
            seg2 = get_segments_from_entry(data2[i])
            
            print(f"    ORIGINAL (first 2 segments):")
            for j, s in enumerate(seg1[:2]):
                dur = s.get('duration', 0)
                lang = s.get('lang', '?')
                text = s.get('text', '?')[:40]
                print(f"      [{j}] {dur:.4f}s {lang:3} '{text}'")
            
            print(f"    MERGED (first 2 segments):")
            for j, s in enumerate(seg2[:2]):
                dur = s.get('duration', 0)
                lang = s.get('lang', '?')
                text = s.get('text', '?')[:40]
                print(f"      [{j}] {dur:.4f}s {lang:3} '{text}'")

def main():
    parser = argparse.ArgumentParser(
        description='Validate manifest data for training issues'
    )
    parser.add_argument('-m', '--manifest', type=str, required=True,
                        help='Path to manifest file (.json or .jsonl)')
    parser.add_argument('-c', '--compare', type=str,
                        help='Path to second manifest for comparison')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show all issues (not just first 5)')
    
    args = parser.parse_args()
    
    # Validate primary manifest
    is_valid = validate_manifest(args.manifest, verbose=args.verbose)
    
    # Compare if requested
    if args.compare:
        compare_manifests(args.manifest, args.compare)
    
    sys.exit(0 if is_valid else 1)

if __name__ == '__main__':
    main()
