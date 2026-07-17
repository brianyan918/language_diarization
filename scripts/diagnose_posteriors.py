#!/usr/bin/env python3
"""
Diagnostic script to check JSONL file structure for posteriors.

Usage:
    python diagnose_posteriors.py --input_jsonl file.jsonl
    python diagnose_posteriors.py --input_jsonl_glob "exp/runs/*.jsonl"
"""

import argparse
import glob
import json
from collections import defaultdict


def diagnose_jsonl(jsonl_path):
    """Analyze structure of a single JSONL file."""
    
    stats = {
        "total_lines": 0,
        "with_posteriors": 0,
        "without_posteriors": 0,
        "parse_errors": 0,
        "structure_errors": 0,
        "missing_passthrough": 0,
        "first_with_posteriors": None,
        "first_without_posteriors": None,
        "sample_with": None,
        "sample_without": None,
    }
    
    print(f"\n=== Analyzing {jsonl_path} ===\n")
    
    with open(jsonl_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            stats["total_lines"] += 1
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                stats["parse_errors"] += 1
                if stats["parse_errors"] <= 3:
                    print(f"❌ Parse error at line {line_no}: {e}")
                continue
            
            # Check structure
            if not isinstance(obj, dict) or len(obj) != 1:
                stats["structure_errors"] += 1
                if stats["structure_errors"] <= 3:
                    print(f"❌ Structure error at line {line_no}: Expected single-key dict, got {type(obj)}")
                continue
            
            utt_key = next(iter(obj.keys()))
            entry = obj[utt_key]
            
            # Check for posteriors
            has_posteriors = "posteriors" in entry
            has_passthrough = "passthrough" in entry
            
            if has_posteriors:
                stats["with_posteriors"] += 1
                if stats["first_with_posteriors"] is None:
                    stats["first_with_posteriors"] = (line_no, utt_key)
                    stats["sample_with"] = entry
            else:
                stats["without_posteriors"] += 1
                if stats["first_without_posteriors"] is None:
                    stats["first_without_posteriors"] = (line_no, utt_key)
                    stats["sample_without"] = entry
            
            if not has_passthrough:
                stats["missing_passthrough"] += 1
    
    # Print summary
    print(f"Summary:")
    print(f"  Total lines: {stats['total_lines']}")
    print(f"  With posteriors: {stats['with_posteriors']} ({100*stats['with_posteriors']/max(stats['total_lines'],1):.1f}%)")
    print(f"  Without posteriors: {stats['without_posteriors']} ({100*stats['without_posteriors']/max(stats['total_lines'],1):.1f}%)")
    print(f"  Parse errors: {stats['parse_errors']}")
    print(f"  Structure errors: {stats['structure_errors']}")
    print(f"  Missing passthrough: {stats['missing_passthrough']}")
    
    if stats["first_with_posteriors"]:
        line_no, utt_key = stats["first_with_posteriors"]
        print(f"\n✓ First utterance WITH posteriors:")
        print(f"  Line {line_no}, utterance: {utt_key}")
        if stats["sample_with"]:
            entry = stats["sample_with"]
            if "posteriors" in entry:
                post = entry["posteriors"]
                print(f"  Posteriors structure:")
                print(f"    - 'values': shape ({len(post.get('values', []))}, {len(post.get('values', [[]])[0]) if post.get('values') else 0})")
                print(f"    - 'frame_times': length {len(post.get('frame_times', []))}")
                print(f"    - 'frame_duration': {post.get('frame_duration', 'N/A')}")
    
    if stats["first_without_posteriors"]:
        line_no, utt_key = stats["first_without_posteriors"]
        print(f"\n❌ First utterance WITHOUT posteriors:")
        print(f"  Line {line_no}, utterance: {utt_key}")
        if stats["sample_without"]:
            entry = stats["sample_without"]
            print(f"  Keys in entry: {list(entry.keys())}")
            if "pred" in entry:
                print(f"    - 'pred': {len(entry['pred'])} predictions")
            if "passthrough" in entry:
                print(f"    - 'passthrough': keys = {list(entry['passthrough'].keys())}")
    
    return stats


def main():
    ap = argparse.ArgumentParser(description="Diagnose JSONL file structure")
    ap.add_argument("--input_jsonl", help="Single JSONL file")
    ap.add_argument("--input_jsonl_glob", help="Glob pattern for JSONL files")
    ap.add_argument("--sample_lines", type=int, default=10, help="Show sample of lines")
    
    args = ap.parse_args()
    
    if (args.input_jsonl is None) == (args.input_jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")
    
    if args.input_jsonl:
        paths = [args.input_jsonl]
    else:
        paths = sorted(glob.glob(args.input_jsonl_glob or ""))
        if not paths:
            raise RuntimeError(f"No JSONL files matched glob: {args.input_jsonl_glob}")
    
    print(f"Found {len(paths)} file(s)")
    
    total_stats = {
        "total_lines": 0,
        "with_posteriors": 0,
        "without_posteriors": 0,
    }
    
    for path in paths:
        stats = diagnose_jsonl(path)
        total_stats["total_lines"] += stats["total_lines"]
        total_stats["with_posteriors"] += stats["with_posteriors"]
        total_stats["without_posteriors"] += stats["without_posteriors"]
    
    if len(paths) > 1:
        print(f"\n{'='*60}")
        print(f"TOTAL ACROSS {len(paths)} FILES:")
        print(f"  Total lines: {total_stats['total_lines']}")
        print(f"  With posteriors: {total_stats['with_posteriors']} ({100*total_stats['with_posteriors']/max(total_stats['total_lines'],1):.1f}%)")
        print(f"  Without posteriors: {total_stats['without_posteriors']} ({100*total_stats['without_posteriors']/max(total_stats['total_lines'],1):.1f}%)")
        print(f"{'='*60}")
        
        if total_stats["without_posteriors"] > 0:
            print("\n⚠️  RECOMMENDATION:")
            print("   Your inference output is missing posteriors for some utterances.")
            print("   Possible causes:")
            print("   1. Inference didn't use dump_posteriors=true")
            print("   2. Inference was partially run/interrupted")
            print("   3. Old output mixed with new output")
            print("\n   Solution:")
            print("   Re-run inference with:")
            print("   inference.inference_runner.dump_posteriors=true")
        else:
            print("\n✓ All utterances have posteriors! Ready to plot DET curve.")


if __name__ == "__main__":
    main()
