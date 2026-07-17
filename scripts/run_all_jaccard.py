#!/usr/bin/env python3
"""
run_all_jaccard.py

Automatically discover and score all JSONL files in a directory using score_jaccard.py.

Usage:
  python scripts/run_all_jaccard.py \
    --exp_dir /path/to/exp/runs/my_experiment/my_config \
    --vocab data/vocab_102.txt \
    --collar 0.25 \
    --exclude_langs "slk,tel" \
    --pattern "*.jsonl"

The script will:
  1. Find all JSONL files matching the pattern in exp_dir and subdirectories
  2. For each JSONL, run score_jaccard.py
  3. Save results as <jsonl_name>.jaccard.json in the same directory
  4. Summarize results at the end
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def find_jsonl_files(exp_dir: str, pattern: str = "*.jsonl") -> List[str]:
    """
    Find all JSONL files in exp_dir and subdirectories matching pattern.
    """
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        raise ValueError(f"Experiment directory does not exist: {exp_dir}")
    
    # Search recursively
    jsonl_files = sorted(exp_path.rglob(pattern))
    return [str(f) for f in jsonl_files]


def run_jaccard_scoring(
    jsonl_path: str,
    vocab_path: str,
    collar: float,
    exclude_langs: Optional[str] = None,
    output_json: Optional[str] = None,
) -> bool:
    """
    Run score_jaccard.py on a single JSONL file.
    
    Returns True if successful, False otherwise.
    """
    # Build command
    cmd = [
        "python", "scripts/score_jaccard.py",
        "--input_jsonl", jsonl_path,
        "--vocab", vocab_path,
        "--collar", str(collar),
    ]
    
    if exclude_langs:
        cmd.extend(["--exclude_langs", exclude_langs])
    
    if output_json:
        cmd.extend(["--output_json", output_json])
    
    try:
        print(f"\n{'='*80}")
        print(f"Processing: {jsonl_path}")
        print(f"Output: {output_json if output_json else 'stdout'}")
        print(f"{'='*80}")
        
        result = subprocess.run(cmd, check=False, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR running scoring for {jsonl_path}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Run Jaccard scoring on all JSONL files in an experiment directory."
    )
    ap.add_argument(
        "--exp_dir",
        required=True,
        help="Path to experiment directory to search for JSONL files.",
    )
    ap.add_argument(
        "--vocab",
        required=True,
        help="Path to vocabulary file.",
    )
    ap.add_argument(
        "--collar",
        type=float,
        default=0.25,
        help="Boundary collar in seconds (default: 0.25).",
    )
    ap.add_argument(
        "--exclude_langs",
        type=str,
        default="",
        help='Comma-separated list of language codes to exclude (e.g., "slk,tel").',
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern to match JSONL files (default: *.jsonl).",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be run without actually running.",
    )
    args = ap.parse_args()

    # Check that vocab file exists
    if not os.path.exists(args.vocab):
        print(f"ERROR: Vocab file not found: {args.vocab}")
        sys.exit(1)

    # Find JSONL files
    print(f"Searching for JSONL files in: {args.exp_dir}")
    print(f"Pattern: {args.pattern}")
    
    jsonl_files = find_jsonl_files(args.exp_dir, args.pattern)
    
    if not jsonl_files:
        print(f"No JSONL files found matching pattern '{args.pattern}' in {args.exp_dir}")
        sys.exit(0)

    print(f"Found {len(jsonl_files)} JSONL file(s):")
    for f in jsonl_files:
        print(f"  - {f}")

    if args.dry_run:
        print("\n[DRY RUN] Commands that would be run:")
        for jsonl_path in jsonl_files:
            base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
            output_json = os.path.join(os.path.dirname(jsonl_path), f"{base_name}.jaccard.json")
            
            cmd = [
                "python", "scripts/score_jaccard.py",
                "--input_jsonl", jsonl_path,
                "--vocab", args.vocab,
                "--collar", str(args.collar),
            ]
            
            if args.exclude_langs:
                cmd.extend(["--exclude_langs", args.exclude_langs])
            
            cmd.extend(["--output_json", output_json])
            print(f"\n{' '.join(cmd)}")
        return

    # Process each JSONL file
    results: Dict[str, Dict] = {}
    successful = 0
    failed = 0

    for jsonl_path in jsonl_files:
        # Determine output JSON path
        base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
        output_json = os.path.join(os.path.dirname(jsonl_path), f"{base_name}.jaccard.json")

        # Run scoring
        success = run_jaccard_scoring(
            jsonl_path,
            args.vocab,
            args.collar,
            exclude_langs=args.exclude_langs,
            output_json=output_json,
        )

        if success:
            successful += 1
            # Try to load results
            if os.path.exists(output_json):
                try:
                    with open(output_json, "r") as f:
                        results[base_name] = json.load(f)
                except Exception as e:
                    print(f"WARNING: Could not load results from {output_json}: {e}")
        else:
            failed += 1

    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total JSONL files: {len(jsonl_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Print global results
    if results:
        print(f"\n{'='*80}")
        print("GLOBAL JER SCORES")
        print(f"{'='*80}\n")
        
        for name, result in results.items():
            if "global" in result:
                jer = result["global"].get("jer", float("nan"))
                jac = result["global"].get("jaccard_mean", float("nan"))
                utts = result["global"].get("utts", 0)
                ref_speech = result["global"].get("ref_speech", 0)
                print(f"{name:50s} JER={jer:8.6f}  Jaccard={jac:8.6f}  (utts={utts:4d}, ref_speech={ref_speech:10.3f}s)")
            else:
                print(f"{name:50s} (no global results)")


if __name__ == "__main__":
    main()
