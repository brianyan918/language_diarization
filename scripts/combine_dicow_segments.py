#!/usr/bin/env python3
"""
combine_segments_to_text_allow_overlaps.py

Combine ref/hyp segments into ref_text and hyp_text, while *tracking* overlaps.

Differences vs the strict version:
- Still detects overlaps (within ref and within hyp) and records them
- But DOES NOT fail or skip due to overlaps
- Always produces combined text by sorting segments by (start_time, end_time)
- Gaps are fine
- Zero/negative duration segments are skipped by default (common rounding artifacts)

Input JSONL: one object per line:
{
  "id": "...",
  "language": "...",
  "ref": [ {start_time,end_time,words,...}, ... ],
  "hyp": [ {start_time,end_time,words,...}, ... ]
}

Output JSONL adds:
- ref_text, hyp_text
- overlaps: {
    "ref": [ "...msg...", ... ],
    "hyp": [ "...msg...", ... ]
  }
- overlap_counts: { "ref": int, "hyp": int }
- skipped_segments: { "ref": int, "hyp": int }
Optionally:
- ref_sorted/hyp_sorted if --write_sorted_segments

Usage:
  python combine_segments_to_text_allow_overlaps.py \
    -i in.jsonl -o out.jsonl \
    --min_dur 0.0 \
    --write_overlap_report overlaps.txt
"""

import argparse
import json
from typing import Any, Dict, List, Tuple

EPS = 1e-9


def _as_float(x: Any, field: str, rec_id: str, side: str, idx: int) -> float:
    try:
        return float(x)
    except Exception as e:
        raise RuntimeError(f"Bad {field} for {side}[{idx}] in id={rec_id}: {x}") from e


def _norm_words(w: Any) -> str:
    if w is None:
        return ""
    s = str(w)
    s = " ".join(s.split())
    return s.strip()


def sort_and_find_overlaps(
    segs: List[Dict[str, Any]],
    rec_id: str,
    side: str,
    eps: float = EPS,
    min_dur: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[str], int]:
    """
    Sort segments by (start_time, end_time) and *report* overlaps but do not fail.

    Skips segments with end_time <= start_time (non-positive duration) by default.
    Also skips segments with duration < min_dur if min_dur > 0.

    Returns:
      sorted_segs, overlap_messages, num_skipped
    """
    parsed: List[Tuple[float, float, int, Dict[str, Any]]] = []
    skipped = 0

    for i, s in enumerate(segs):
        st = _as_float(s.get("start_time"), "start_time", rec_id, side, i)
        et = _as_float(s.get("end_time"), "end_time", rec_id, side, i)
        dur = et - st

        if dur <= eps:
            skipped += 1
            continue
        if min_dur > 0 and dur < min_dur:
            skipped += 1
            continue

        parsed.append((st, et, i, s))

    parsed.sort(key=lambda x: (x[0], x[1], x[2]))

    overlap_msgs: List[str] = []
    for (st1, et1, i1, _), (st2, et2, i2, _) in zip(parsed[:-1], parsed[1:]):
        if st2 < et1 - eps:
            overlap_msgs.append(
                f"{side} overlap in id={rec_id}: seg{i1} [{st1:.6f},{et1:.6f}] overlaps seg{i2} [{st2:.6f},{et2:.6f}]"
            )

    sorted_segs = [s for _, __, ___, s in parsed]
    return sorted_segs, overlap_msgs, skipped


def segments_to_text(sorted_segs: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for s in sorted_segs:
        w = _norm_words(s.get("words", ""))
        if w:
            parts.append(w)
    return " ".join(parts).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Input JSONL")
    ap.add_argument("--output", "-o", required=True, help="Output JSONL with ref_text/hyp_text + overlap info")
    ap.add_argument(
        "--min_dur",
        type=float,
        default=0.0,
        help="If >0, also skip segments with duration < min_dur seconds (in addition to non-positive).",
    )
    ap.add_argument(
        "--write_overlap_report",
        default="",
        help="Optional path to write a text report listing overlap failures (both ref and hyp).",
    )
    ap.add_argument(
        "--write_sorted_segments",
        action="store_true",
        help="If set, write back sorted segments as ref_sorted/hyp_sorted in output (larger JSON).",
    )
    args = ap.parse_args()

    overlap_report_lines: List[str] = []
    n_in = 0
    n_out = 0
    total_skipped_ref = 0
    total_skipped_hyp = 0
    total_overlap_ref = 0
    total_overlap_hyp = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for ln, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            n_in += 1

            rec = json.loads(line)
            rec_id = str(rec.get("id", f"LINE{ln}"))

            ref_segs = rec.get("ref", [])
            hyp_segs = rec.get("hyp", [])

            if not isinstance(ref_segs, list) or not isinstance(hyp_segs, list):
                raise RuntimeError(f"Expected 'ref' and 'hyp' to be lists in id={rec_id} (line {ln})")

            ref_sorted, ref_ov, ref_sk = sort_and_find_overlaps(
                ref_segs, rec_id, "ref", min_dur=args.min_dur
            )
            hyp_sorted, hyp_ov, hyp_sk = sort_and_find_overlaps(
                hyp_segs, rec_id, "hyp", min_dur=args.min_dur
            )

            total_skipped_ref += ref_sk
            total_skipped_hyp += hyp_sk
            total_overlap_ref += len(ref_ov)
            total_overlap_hyp += len(hyp_ov)

            if ref_ov or hyp_ov:
                overlap_report_lines.extend([f"line={ln} " + m for m in (ref_ov + hyp_ov)])

            # Always produce combined text (even if overlaps exist)
            rec["ref_text"] = segments_to_text(ref_sorted)
            rec["hyp_text"] = segments_to_text(hyp_sorted)

            rec["skipped_segments"] = {"ref": ref_sk, "hyp": hyp_sk}
            rec["overlaps"] = {"ref": ref_ov, "hyp": hyp_ov}
            rec["overlap_counts"] = {"ref": len(ref_ov), "hyp": len(hyp_ov)}

            if args.write_sorted_segments:
                rec["ref_sorted"] = ref_sorted
                rec["hyp_sorted"] = hyp_sorted

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    if args.write_overlap_report:
        with open(args.write_overlap_report, "w", encoding="utf-8") as f:
            for m in overlap_report_lines:
                f.write(m + "\n")

    print("=== DONE ===")
    print(f"Read:  {n_in}")
    print(f"Wrote: {n_out} -> {args.output}")
    print(f"Skipped segments (non-positive or <min_dur): ref={total_skipped_ref} hyp={total_skipped_hyp}")
    print(f"Overlap pairs found: ref={total_overlap_ref} hyp={total_overlap_hyp}")
    if overlap_report_lines:
        print(f"Records with overlap messages: {len(overlap_report_lines)}")
        if args.write_overlap_report:
            print(f"Overlap report: {args.write_overlap_report}")
    if args.min_dur > 0:
        print(f"min_dur={args.min_dur}")


if __name__ == "__main__":
    main()
