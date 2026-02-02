#!/usr/bin/env python3
"""
Filter OUT (remove) the bottom PERCENTAGE of utterances per language pair based on utterance-level score.

Input JSONL format (one object per line):
{
  "id": "...",
  "language": "ara-eng",
  "segments": [...],
  "score": "-0.11938173323869705",
  ...
}

Behavior:
- Groups utterances by obj["language"] (language pair)
- Sorts each group by float(score), ascending
- Removes bottom (percent/100) fraction per group (lowest scores)
- Writes the REMAINING utterances to a new JSONL

Usage:
  python filter_out_bottom_pct_by_langpair.py \
    --input_jsonl data.jsonl \
    --out_jsonl filtered.jsonl \
    --percent 10 \
    --skip_missing

Sharded:
  python filter_out_bottom_pct_by_langpair.py \
    --input_jsonl_glob "/path/*.jsonl" \
    --out_jsonl filtered.jsonl \
    --percent 5 \
    --skip_missing
"""

import argparse
import glob
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def iter_jsonl(jsonl_path: Optional[str], jsonl_glob: Optional[str]):
    if (jsonl_path is None) == (jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")

    paths = [jsonl_path] if jsonl_path else sorted(glob.glob(jsonl_glob))
    if not paths:
        raise RuntimeError("No JSONL files found")

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                yield p, ln, json.loads(line)


def compute_drop_count(k: int, percent: float, rounding: str) -> int:
    """How many to drop from a group of size k."""
    if k <= 0:
        return 0
    frac = percent / 100.0
    raw = k * frac

    if rounding == "ceil":
        n = int(math.ceil(raw))
    elif rounding == "floor":
        n = int(math.floor(raw))
    elif rounding == "round":
        n = int(round(raw))
    else:
        raise ValueError(f"Unknown rounding: {rounding}")

    # If percent > 0, ensure at least 1 gets dropped when there is data
    if percent > 0 and n == 0:
        n = 1
    return max(0, min(k, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", help="Single input JSONL")
    ap.add_argument("--input_jsonl_glob", help="Glob of input JSONLs")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--percent", type=float, required=True, help="BOTTOM percentage to REMOVE per language pair")
    ap.add_argument(
        "--rounding",
        choices=["ceil", "floor", "round"],
        default="ceil",
        help="How to convert percentage to count per group (default: ceil).",
    )
    ap.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip utterances with missing/invalid score (recommended). "
             "These skipped items are not written to output.",
    )
    args = ap.parse_args()

    if not (0.0 <= args.percent <= 100.0):
        raise ValueError("--percent must be between 0 and 100")

    by_pair: Dict[str, List[dict]] = defaultdict(list)
    skipped = 0

    for src, ln, obj in iter_jsonl(args.input_jsonl, args.input_jsonl_glob):
        pair = obj.get("language")
        if not pair:
            skipped += 1
            continue

        score_raw = obj.get("score")
        try:
            score = float(score_raw)
        except Exception:
            if args.skip_missing:
                skipped += 1
                continue
            raise ValueError(f"Invalid score at {src}:{ln}: {score_raw}")

        obj["_parsed_score"] = score  # helper for sorting only
        by_pair[pair].append(obj)

    kept: List[dict] = []
    per_pair_counts: Dict[str, Tuple[int, int, int]] = {}  # total, dropped, kept

    for pair, items in by_pair.items():
        items_sorted = sorted(items, key=lambda x: x["_parsed_score"])  # low -> high
        drop_n = compute_drop_count(len(items_sorted), args.percent, args.rounding)

        # FILTER OUT bottom drop_n
        kept_items = items_sorted[drop_n:]

        per_pair_counts[pair] = (len(items_sorted), drop_n, len(kept_items))
        kept.extend(kept_items)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for obj in kept:
            obj.pop("_parsed_score", None)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("=== FILTER SUMMARY ===")
    print(f"Language pairs: {len(by_pair)}")
    print(f"Removed bottom percent per pair: {args.percent}% (rounding={args.rounding})")
    print(f"Total kept/written: {len(kept)}")
    print(f"Skipped (missing/invalid/no language): {skipped}")
    print(f"Output: {args.out_jsonl}")

    # Show a few pairs
    top_pairs = sorted(per_pair_counts.items(), key=lambda kv: kv[1][0], reverse=True)[:10]
    print("Top 10 pairs by size: (pair: total | dropped | kept)")
    for pair, (tot, dr, kp) in top_pairs:
        print(f"  {pair}: {tot} | {dr} | {kp}")


if __name__ == "__main__":
    main()
