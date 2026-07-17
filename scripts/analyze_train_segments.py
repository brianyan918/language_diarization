#!/usr/bin/env python3
"""
analyze_training_switches.py

Analyze code-switched training JSONL like:

{
  "id": "...",
  "segments": [
    {"audio_start_sec":..., "audio_end_sec":..., "duration":..., "lang":"ara"},
    ...
  ],
  ...
}

Reports:
1) Switch count per utt (based on adjacent segments' 'lang' changes; optionally ignore large gaps)
   - avg / min / max over utterances
   - also prints total utts, total ref duration stats (optional)
2) Filtering impact for segment-duration thresholds:
   For each threshold T:
     - how many utts would be removed if ANY segment in the utt has duration < T
     - how many segments would be removed if you drop short segments (diagnostic)
     - distribution of "min segment duration per utt" (via these thresholds)

Notes:
- "switch" is counted when consecutive segments (after sorting by start time and merging adjacent
  same-lang touching segments) change language. Large gaps can be treated as breaks (no switch)
  via --max_gap; set <=0 to disable gap filtering.
- Duration is taken from segment["duration"] if present, else computed as end-start.

Usage:
  python analyze_training_switches.py \
    --input_jsonl train.jsonl \
    --thresholds 0.1,0.25,0.5,0.75,1.0 \
    --max_gap 0.5

  # Or auto thresholds
  python analyze_training_switches.py --input_jsonl train.jsonl --auto_thresholds
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple
from collections import defaultdict

EPS = 1e-12


@dataclass
class Seg:
    start: float
    end: float
    dur: float
    lang: str


def iter_jsonl(path: str) -> Iterator[Tuple[int, dict]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield ln, json.loads(line)


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def load_segments(rec: dict) -> List[Seg]:
    segs_raw = rec.get("segments", []) or []
    out: List[Seg] = []
    for s in segs_raw:
        a = safe_float(s.get("audio_start_sec"), None)
        b = safe_float(s.get("audio_end_sec"), None)
        if a is None or b is None:
            continue
        if b <= a:
            continue
        dur = safe_float(s.get("duration"), None)
        if dur is None:
            dur = b - a
        lang = str(s.get("lang", "UNK"))
        out.append(Seg(start=a, end=b, dur=float(dur), lang=lang))
    out.sort(key=lambda z: (z.start, z.end))
    return out


def merge_adjacent_same_lang(segs: List[Seg]) -> List[Seg]:
    if not segs:
        return []
    out = [Seg(segs[0].start, segs[0].end, segs[0].dur, segs[0].lang)]
    for s in segs[1:]:
        prev = out[-1]
        # touching/overlapping and same lang => merge
        if s.lang == prev.lang and s.start <= prev.end + 1e-9:
            prev.end = max(prev.end, s.end)
            prev.dur = max(0.0, prev.end - prev.start)
        else:
            out.append(Seg(s.start, s.end, s.dur, s.lang))
    return out


def count_switches(segs: List[Seg], max_gap: Optional[float]) -> int:
    """
    Count switches between adjacent segments when language changes.

    If max_gap is not None and > 0:
      - only count potential switch if gap between segments is <= max_gap
      - gaps > max_gap are treated as breaks (no switch counted across them)
    """
    if len(segs) < 2:
        return 0
    sw = 0
    for a, b in zip(segs[:-1], segs[1:]):
        gap = b.start - a.end
        if gap < -1e-6:
            # overlap/disorder: skip switch counting across this boundary
            continue
        if max_gap is not None and max_gap > 0 and gap > max_gap:
            continue
        if a.lang != b.lang:
            sw += 1
    return sw


def summarize_ints(xs: List[int]) -> Dict[str, float]:
    if not xs:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {"avg": sum(xs) / len(xs), "min": float(min(xs)), "max": float(max(xs))}


def summarize_floats(xs: List[float]) -> Dict[str, float]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {"avg": sum(xs) / len(xs), "min": float(min(xs)), "max": float(max(xs))}


def parse_thresholds(s: str) -> List[float]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True, help="Training JSONL (one record per line).")
    ap.add_argument(
        "--thresholds",
        type=str,
        default="0.05,0.1,0.25,0.5,0.75,1.0",
        help="Comma-separated segment-duration thresholds (seconds).",
    )
    ap.add_argument("--auto_thresholds", action="store_true", help="Use a preset threshold list instead of --thresholds.")
    ap.add_argument(
        "--max_gap",
        type=float,
        default=0.5,
        help="Max gap (s) to count switches across; <=0 disables gap filtering.",
    )
    ap.add_argument(
        "--no_merge_adjacent",
        action="store_true",
        help="Do not merge adjacent same-lang segments before counting switches.",
    )
    ap.add_argument(
        "--print_examples",
        type=int,
        default=0,
        help="Print up to K example utt IDs that would be filtered for each threshold.",
    )
    args = ap.parse_args()

    if args.auto_thresholds:
        thresholds = [0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    else:
        thresholds = parse_thresholds(args.thresholds)

    max_gap = None if args.max_gap <= 0 else float(args.max_gap)

    utt_count = 0
    bad_json = 0

    switch_counts: List[int] = []
    min_seg_durs: List[float] = []
    all_seg_durs: List[float] = []

    # For filtering analysis
    # threshold -> count of utts removed
    removed_by_thr: Dict[float, int] = {t: 0 for t in thresholds}
    examples_by_thr: Dict[float, List[str]] = {t: [] for t in thresholds}
    segs_below_thr: Dict[float, int] = {t: 0 for t in thresholds}

    for ln, rec in iter_jsonl(args.input_jsonl):
        utt_count += 1
        try:
            utt_id = str(rec.get("id", f"line{ln}"))
            segs = load_segments(rec)
            if not segs:
                # treat empty as zero switches; min dur undefined -> skip from min stats
                switch_counts.append(0)
                continue

            if not args.no_merge_adjacent:
                segs = merge_adjacent_same_lang(segs)

            # switch stats
            sw = count_switches(segs, max_gap=max_gap)
            switch_counts.append(sw)

            # duration stats
            durs = [max(0.0, s.dur) for s in segs]
            for d in durs:
                all_seg_durs.append(d)

            min_d = min(durs) if durs else float("nan")
            min_seg_durs.append(min_d)

            # threshold impacts
            for t in thresholds:
                below = [d for d in durs if d < t]
                segs_below_thr[t] += len(below)
                if below:
                    removed_by_thr[t] += 1
                    if args.print_examples > 0 and len(examples_by_thr[t]) < args.print_examples:
                        examples_by_thr[t].append(utt_id)

        except Exception:
            bad_json += 1
            continue

    # ---- Switch summary ----
    ssum = summarize_ints(switch_counts)
    print("=== SWITCH STATS (per utterance) ===")
    print(f"utts={utt_count}  bad_json={bad_json}")
    print(f"switches/utt: avg={ssum['avg']:.3f}  min={int(ssum['min'])}  max={int(ssum['max'])}")
    if max_gap is None:
        print("switch counting: max_gap=DISABLED")
    else:
        print(f"switch counting: max_gap={max_gap:.3f}s")
    print(f"merge_adjacent_same_lang: {not args.no_merge_adjacent}")

    # ---- Segment duration summary ----
    dsum = summarize_floats(all_seg_durs)
    mind = summarize_floats(min_seg_durs)
    print("=== SEGMENT DURATION STATS ===")
    print(f"segments: avg={dsum['avg']:.3f}s  min={dsum['min']:.3f}s  max={dsum['max']:.3f}s")
    print(f"min-seg-per-utt: avg={mind['avg']:.3f}s  min={mind['min']:.3f}s  max={mind['max']:.3f}s")

    # ---- Filtering report ----
    print("=== FILTERING IMPACT (remove utt if ANY segment duration < threshold) ===")
    print(f"{'thr(s)':>8}  {'removed_utts':>12}  {'kept_utts':>10}  {'removed_%':>9}  {'segs_below_thr':>14}")
    for t in thresholds:
        removed = removed_by_thr[t]
        kept = utt_count - removed
        pct = 100.0 * removed / max(1, utt_count)
        print(f"{t:8.3f}  {removed:12d}  {kept:10d}  {pct:8.2f}%  {segs_below_thr[t]:14d}")

        if args.print_examples > 0 and examples_by_thr[t]:
            ex = ", ".join(examples_by_thr[t])
            print(f"  examples: {ex}")

    # Optional: quick guidance threshold based on observed min per-utt
    # (how many utts have min_seg_dur < t) is exactly removed_by_thr[t]
    print("Done.")


if __name__ == "__main__":
    main()
