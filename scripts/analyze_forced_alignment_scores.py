#!/usr/bin/env python3
"""
Analyze utterance-level score distributions grouped by language.

Input JSONL format:
{
  "id": "...",
  "language": "ara-eng",
  "segments": [{"audio_start_sec":..., "audio_end_sec":..., "lang":"ara"}, ...],
  "score": "-0.11938173323869705"
}

Reports score distributions:
  1) by language-pair (e.g., ara-eng)
  2) by dominant language (largest duration)
  3) by any-occurring language (multi-count)

Usage:
  python analyze_utterance_score_distribution.py \
    --input_jsonl data.jsonl \
    --out_dir score_analysis
"""

import argparse
import json
import glob
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------
def iter_jsonl(path: Optional[str], glob_path: Optional[str]):
    if (path is None) == (glob_path is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")

    paths = [path] if path else sorted(glob.glob(glob_path))
    if not paths:
        raise RuntimeError("No JSONL files found")

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                yield p, ln, json.loads(line)


def summarize(arr: np.ndarray) -> Dict[str, float]:
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def dominant_language(segments: List[dict]) -> str:
    dur = defaultdict(float)
    for s in segments:
        d = float(s["audio_end_sec"]) - float(s["audio_start_sec"])
        dur[s["lang"]] += max(d, 0.0)
    return max(dur.items(), key=lambda kv: kv[1])[0]


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", help="Single JSONL file")
    ap.add_argument("--input_jsonl_glob", help="Glob of JSONLs")
    ap.add_argument("--out_dir", default="score_analysis")
    ap.add_argument("--save_plots", action="store_true")
    args = ap.parse_args()

    by_pair = defaultdict(list)
    by_dominant = defaultdict(list)
    by_any = defaultdict(list)

    for src, ln, obj in iter_jsonl(args.input_jsonl, args.input_jsonl_glob):
        if "score" not in obj or obj["score"] is None:
            continue

        try:
            score = float(obj["score"])
        except Exception:
            continue

        segments = obj.get("segments", [])
        if not segments:
            continue

        # 1) language pair
        lang_pair = obj.get("language", "UNKNOWN_PAIR")
        by_pair[lang_pair].append(score)

        # 2) dominant language
        dom = dominant_language(segments)
        by_dominant[dom].append(score)

        # 3) any-occurring language
        langs = set(s["lang"] for s in segments)
        for l in langs:
            by_any[l].append(score)

    os.makedirs(args.out_dir, exist_ok=True)

    def report(title: str, data: Dict[str, List[float]]):
        print(f"\n=== {title} ===")
        for k in sorted(data.keys()):
            arr = np.asarray(data[k])
            stats = summarize(arr)
            print(
                f"[{k}] n={stats['count']}  "
                f"mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
                f"p25={stats['p25']:.4f}  p50={stats['p50']:.4f}  p75={stats['p75']:.4f}"
            )

            if args.save_plots:
                plt.figure(figsize=(4, 3))
                plt.hist(arr, bins=40)
                plt.title(f"{title}: {k}")
                plt.xlabel("score")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(os.path.join(args.out_dir, f"{title}_{k}.png"), dpi=150)
                plt.close()

    report("BY_LANGUAGE_PAIR", by_pair)
    report("BY_DOMINANT_LANGUAGE", by_dominant)
    report("BY_ANY_LANGUAGE", by_any)

    print(f"\nSaved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
