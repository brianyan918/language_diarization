#!/usr/bin/env python3
"""
score_lder_with_threshold.py

Compute full Language Diarization Error Rate (LDER) from JSONL predictions with
frame-level posteriors, applying a threshold to determine embedded vs. matrix language.

Similar to score_lder_detection.py but:
  - Uses frame-level posteriors (not just argmax)
  - Applies threshold T to embedded language posterior
  - Converts threshold-based frames to segments
  - Computes full LDER = (Miss + FA + Conf) / RefSpeech

Metric (DER-style):
  LDER = (Miss + FA + Conf) / RefSpeech

Where:
  - Miss: Reference is embedded but hypothesis is not
  - FA: Reference is not embedded but hypothesis is
  - Conf: Both reference and hypothesis are speech but labels differ

JSONL input (with posteriors): one JSON object per line:
{
  "112": {
    "pred": {
      "alignments": [...],
      "posteriors": {
        "values": [[0.1, 0.8, 0.1], ...],  # Frame-level posteriors (pad, matrix, embed)
        "frame_times": [0.0, 0.04, ...],   # Start time of each frame
        "frame_duration": 0.04              # Duration per frame
      }
    },
    "passthrough": {
      "segment_timestamps": [[...], ...],
      "segment_langs": ["ara","eng",...],
      ...
    }
  }
}
"""

import argparse
import glob
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple
from collections import defaultdict

EPS = 1e-12


@dataclass
class Seg:
    start: float
    end: float
    label: int


@dataclass
class TokSeg:
    start: float
    end: float
    tok: str


# -------------------------
# I/O
# -------------------------
def iter_jsonl_inputs(
    jsonl_path: Optional[str] = None,
    jsonl_glob: Optional[str] = None,
) -> Iterator[Tuple[str, int, dict]]:
    """Yield (source_path, line_number, obj) from single JSONL or globbed JSONLs."""
    if (jsonl_path is None) == (jsonl_glob is None):
        raise ValueError("Provide exactly one of jsonl_path or jsonl_glob")

    if jsonl_path is not None:
        paths = [jsonl_path]
    else:
        paths = sorted(glob.glob(jsonl_glob or ""))
        if not paths:
            raise RuntimeError(f"No JSONL files matched glob: {jsonl_glob}")

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield p, ln, json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"JSON decode error in {p}:{ln}: {e}") from e


# -------------------------
# Segment utilities
# -------------------------
def merge_adjacent(segs: List[Seg]) -> List[Seg]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: (s.start, s.end))
    out = [Seg(segs[0].start, segs[0].end, segs[0].label)]
    for s in segs[1:]:
        prev = out[-1]
        if abs(s.start - prev.end) <= 1e-9 and s.label == prev.label:
            prev.end = max(prev.end, s.end)
        else:
            out.append(Seg(s.start, s.end, s.label))
    return out


def merge_adjacent_tok(segs: List[TokSeg]) -> List[TokSeg]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: (s.start, s.end))
    out = [TokSeg(segs[0].start, segs[0].end, segs[0].tok)]
    for s in segs[1:]:
        prev = out[-1]
        if abs(s.start - prev.end) <= 1e-9 and s.tok == prev.tok:
            prev.end = max(prev.end, s.end)
        else:
            out.append(TokSeg(s.start, s.end, s.tok))
    return out


def build_ref_segments_binary(
    seg_ts: List[List[float]],
    seg_langs: List[str],
    english_token: str,
    non_speech_token: Optional[str],
) -> Tuple[List[Seg], List[TokSeg]]:
    """Build binary ref segments (label 1=matrix, 2=eng)."""
    if len(seg_ts) != len(seg_langs):
        raise ValueError(f"segment_timestamps and segment_langs mismatch: {len(seg_ts)} vs {len(seg_langs)}")

    ref_bin: List[Seg] = []
    ref_tok: List[TokSeg] = []
    for (s, e), tok in zip(seg_ts, seg_langs):
        s = float(s)
        e = float(e)
        if e <= s:
            continue

        ref_tok.append(TokSeg(s, e, str(tok)))

        if non_speech_token is not None and tok == non_speech_token:
            continue
        lab = 2 if tok == english_token else 1
        ref_bin.append(Seg(s, e, lab))

    return merge_adjacent(ref_bin), merge_adjacent_tok(ref_tok)


def build_pred_segments_from_posteriors(
    posteriors_dict: dict,
    threshold: float,
    class_idx: Optional[int] = None,
) -> List[Seg]:
    """
    Convert frame-level posteriors to segments using a threshold.
    
    Args:
        posteriors_dict: Dict with 'values', 'frame_times', 'frame_duration'
        threshold: Confidence threshold for embedded language
        class_idx: Index in posteriors (0-based). If None, uses last index.
    
    Returns:
        List[Seg] with labels 1 (matrix) or 2 (embedded)
    """
    if not posteriors_dict:
        return []
    
    values = posteriors_dict.get("values", [])
    frame_times = posteriors_dict.get("frame_times", [])
    frame_duration = posteriors_dict.get("frame_duration", 0.0)
    
    if not values or not frame_times or frame_duration <= 0:
        return []
    
    values = np.array(values)
    if len(values) != len(frame_times):
        raise ValueError(f"Mismatch: {len(values)} frames vs {len(frame_times)} frame times")
    
    # Use last index if class_idx not specified (embedded language is typically last)
    if class_idx is None:
        class_idx = values.shape[1] - 1
    
    # Get posteriors for the embedded language class
    class_posteriors = values[:, class_idx]
    
    # Label each frame based on threshold
    segments: List[Seg] = []
    for i, (post, t) in enumerate(zip(class_posteriors, frame_times)):
        lab = 2 if post >= threshold else 1
        start = t
        end = t + frame_duration
        
        if segments and segments[-1].label == lab and abs(segments[-1].end - start) <= 1e-9:
            # Merge with previous segment
            segments[-1].end = end
        else:
            segments.append(Seg(start, end, lab))
    
    return segments


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    intervals = [(s, e) for (s, e) in intervals if e > s]
    if not intervals:
        return []
    intervals.sort()
    out = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s <= pe + 1e-9:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def subtract_intervals(base: Tuple[float, float], cuts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    s, e = base
    if e <= s:
        return []
    cuts = merge_intervals([(max(s, cs), min(e, ce)) for cs, ce in cuts])
    if not cuts:
        return [(s, e)]
    out: List[Tuple[float, float]] = []
    cur = s
    for cs, ce in cuts:
        if cs > cur + 1e-9:
            out.append((cur, cs))
        cur = max(cur, ce)
    if cur < e - 1e-9:
        out.append((cur, e))
    return out


def apply_boundary_collar_to_ref(ref: List[Seg], collar: float) -> Tuple[List[Seg], List[Tuple[float, float]]]:
    """Ignore +/- collar seconds around ref label-change boundaries."""
    if collar <= 0:
        return ref, []

    ref = sorted(ref, key=lambda s: (s.start, s.end))
    boundaries: List[float] = []
    for a, b in zip(ref[:-1], ref[1:]):
        if abs(a.end - b.start) <= 1e-9 and a.label != b.label:
            boundaries.append(a.end)

    ignored = merge_intervals([(t - collar, t + collar) for t in boundaries])

    carved: List[Seg] = []
    for seg in ref:
        pieces = subtract_intervals((seg.start, seg.end), ignored)
        for ps, pe in pieces:
            carved.append(Seg(ps, pe, seg.label))

    return merge_adjacent(carved), ignored


def label_at(segs: List[Seg], idx: int, t: float) -> Tuple[Optional[int], int]:
    n = len(segs)
    while idx < n and segs[idx].end <= t + 1e-12:
        idx += 1
    if idx < n and segs[idx].start <= t < segs[idx].end:
        return segs[idx].label, idx
    return None, idx


def tok_at(segs: List[TokSeg], idx: int, t: float) -> Tuple[Optional[str], int]:
    n = len(segs)
    while idx < n and segs[idx].end <= t + 1e-12:
        idx += 1
    if idx < n and segs[idx].start <= t < segs[idx].end:
        return segs[idx].tok, idx
    return None, idx


def in_any_interval(t: float, intervals: List[Tuple[float, float]], idx: int) -> Tuple[bool, int]:
    n = len(intervals)
    while idx < n and intervals[idx][1] <= t + 1e-12:
        idx += 1
    if idx < n:
        s, e = intervals[idx]
        if s <= t < e:
            return True, idx
    return False, idx


def fmt_bin(label: int) -> str:
    if label == 1:
        return "MATRIX(1)"
    if label == 2:
        return "ENG(2)"
    return str(label)


# -------------------------
# Scoring (Language Classification)
# -------------------------
def compute_lder_and_confusions(
    ref: List[Seg],
    hyp: List[Seg],
    collar: float = 0.0,
    include_fa: bool = True,
) -> Tuple[Dict[str, float], Dict[Tuple[int, int], float]]:
    """
    Returns:
      metrics: {LDER, Miss, FA, Conf, RefSpeech}
      conf_pairs: {(ref_label, hyp_label): confused_time_seconds}
    """
    ref = merge_adjacent(ref)
    hyp = merge_adjacent(hyp)

    ref_scored, ignored = apply_boundary_collar_to_ref(ref, collar)

    B = set()
    for s in ref_scored:
        B.add(s.start); B.add(s.end)
    for s in hyp:
        B.add(s.start); B.add(s.end)
    for s, e in ignored:
        B.add(s); B.add(e)

    B = sorted(B)
    if len(B) < 2:
        return (
            {"LDER": float("nan"), "Miss": 0.0, "FA": 0.0, "Conf": 0.0, "RefSpeech": 0.0},
            {},
        )

    Miss = FA = Conf = RefSpeech = 0.0
    conf_pairs: Dict[Tuple[int, int], float] = defaultdict(float)

    i = j = 0
    ig_idx = 0

    for k in range(len(B) - 1):
        t0, t1 = B[k], B[k + 1]
        dt = t1 - t0
        if dt <= EPS:
            continue

        inside_ignored, ig_idx = in_any_interval(t0, ignored, ig_idx)
        if inside_ignored:
            continue

        rlab, i = label_at(ref_scored, i, t0)
        hlab, j = label_at(hyp, j, t0)

        r_is_embedded = (rlab == 2)
        h_is_embedded = (hlab == 2)

        # Count all reference time
        if rlab is not None:
            RefSpeech += dt
        
        # Only score where reference is speech
        if rlab is None:
            continue
        
        if r_is_embedded:
            # Reference is embedded
            if not h_is_embedded:
                # But hypothesis is not -> MISS
                Miss += dt
        else:
            # Reference is matrix (not embedded)
            if h_is_embedded:
                # But hypothesis is embedded -> FA
                FA += dt
            elif hlab is not None and hlab != rlab:
                # Both are speech but different labels -> CONFUSION
                Conf += dt
                conf_pairs[(rlab, hlab)] += dt

    if RefSpeech <= EPS:
        metrics = {"LDER": float("nan"), "Miss": Miss, "FA": FA, "Conf": Conf, "RefSpeech": RefSpeech}
        return metrics, dict(conf_pairs)

    numer = Miss + Conf + (FA if include_fa else 0.0)
    metrics = {"LDER": numer / RefSpeech, "Miss": Miss, "FA": FA, "Conf": Conf, "RefSpeech": RefSpeech}
    return metrics, dict(conf_pairs)


def compute_by_ref_token_breakdown(
    ref_tok: List[TokSeg],
    ref_bin: List[Seg],
    hyp: List[Seg],
    collar: float,
) -> Tuple[
    Dict[str, float],               # ref_speech_by_tok
    Dict[str, float],               # miss_by_tok
    Dict[str, Dict[int, float]],    # conf_by_tok[ref_tok][hyp_label]
]:
    """Breakdown restricted to intervals where REF is speech."""
    ref_bin = merge_adjacent(ref_bin)
    hyp = merge_adjacent(hyp)
    ref_tok = merge_adjacent_tok(ref_tok)

    ref_scored, ignored = apply_boundary_collar_to_ref(ref_bin, collar)

    B = set()
    for s in ref_scored:
        B.add(s.start); B.add(s.end)
    for s in hyp:
        B.add(s.start); B.add(s.end)
    for s in ignored:
        B.add(s[0]); B.add(s[1])
    for s in ref_tok:
        B.add(s.start); B.add(s.end)

    B = sorted(B)
    ref_speech_by_tok: Dict[str, float] = defaultdict(float)
    miss_by_tok: Dict[str, float] = defaultdict(float)
    conf_by_tok: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    if len(B) < 2:
        return {}, {}, {}

    i = j = 0
    ig_idx = 0
    tk_idx = 0

    for k in range(len(B) - 1):
        t0, t1 = B[k], B[k + 1]
        dt = t1 - t0
        if dt <= EPS:
            continue

        inside_ignored, ig_idx = in_any_interval(t0, ignored, ig_idx)
        if inside_ignored:
            continue

        rlab, i = label_at(ref_scored, i, t0)
        hlab, j = label_at(hyp, j, t0)
        rtok, tk_idx = tok_at(ref_tok, tk_idx, t0)

        if rlab is None:
            continue  # ref non-speech

        tok = rtok if rtok is not None else "<UNK>"
        ref_speech_by_tok[tok] += dt

        r_is_embedded = (rlab == 2)
        h_is_embedded = (hlab == 2)

        if r_is_embedded and not h_is_embedded:
            miss_by_tok[tok] += dt
        elif hlab is not None and rlab != hlab:
            conf_by_tok[tok][hlab] += dt

    return dict(ref_speech_by_tok), dict(miss_by_tok), {t: dict(m) for t, m in conf_by_tok.items()}


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", help="Single JSONL file (one JSON object per line).")
    ap.add_argument("--input_jsonl_glob", help='Glob of sharded JSONLs (e.g., "/path/to/*.jsonl").')
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for embedded language posterior (default 0.5).")
    ap.add_argument("--collar", type=float, default=0.0, help="Boundary collar seconds (e.g., 0.25).")
    ap.add_argument("--include_fa", action="store_true", help="Include FA in LDER numerator (classic DER-style).")
    ap.add_argument("--per_utt", action="store_true", help="Print per-utterance metrics.")

    ap.add_argument("--english_token", default="eng", help="REF token treated as embedded language (label=2). Default: eng")
    ap.add_argument("--non_speech_token", default=None, help="Optional REF token to treat as non-speech.")

    ap.add_argument("--conf_topk", type=int, default=25, help="Top-K OVERALL confusion pairs to print (by time).")
    ap.add_argument("--per_ref_topk", type=int, default=10, help="Top-K confusions to print per REF language token.")
    ap.add_argument("--min_ref_time", type=float, default=0.0, help="Skip per-ref report if ref speech < this many seconds.")
    args = ap.parse_args()

    if (args.input_jsonl is None) == (args.input_jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")

    # Global accumulators
    g_miss = g_fa = g_conf = g_ref = 0.0
    g_conf_pairs: Dict[Tuple[int, int], float] = defaultdict(float)
    g_utts = 0
    g_utts_with_posteriors = 0

    # Per-ref-token accumulators
    ref_speech_by_tok: Dict[str, float] = defaultdict(float)
    miss_by_tok: Dict[str, float] = defaultdict(float)
    conf_by_tok: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    conf_total_by_tok: Dict[str, float] = defaultdict(float)

    print(f"Scoring with threshold={args.threshold:.2f}")
    print()

    for src, ln, obj in iter_jsonl_inputs(jsonl_path=args.input_jsonl, jsonl_glob=args.input_jsonl_glob):
        if not isinstance(obj, dict) or len(obj) != 1:
            raise RuntimeError(
                f"Expected each JSONL line to be a dict with exactly 1 key, got {type(obj)} "
                f"len={len(obj) if isinstance(obj, dict) else 'NA'} at {src}:{ln}"
            )

        utt_key = next(iter(obj.keys()))
        entry = obj[utt_key]

        # Handle nested structure
        pred_dict = entry.get("pred", entry)
        
        passthrough = entry.get("passthrough", {})
        seg_ts = passthrough.get("segment_timestamps", [])
        seg_langs = passthrough.get("segment_langs", [])

        ref_bin, ref_tok = build_ref_segments_binary(
            seg_ts, seg_langs, english_token=args.english_token, non_speech_token=args.non_speech_token
        )

        # Try to build hypothesis from posteriors
        posteriors_dict = pred_dict.get("posteriors", None)
        if posteriors_dict is None:
            # Fall back to alignments if available
            pred_list = pred_dict.get("alignments", pred_dict.get("pred", []))
            hyp = []
            for d in pred_list:
                s = float(d["start"])
                e = float(d["end"])
                lab = int(d["label"])
                if e <= s:
                    continue
                hyp.append(Seg(s, e, lab))
            hyp = merge_adjacent(hyp)
        else:
            try:
                hyp = build_pred_segments_from_posteriors(posteriors_dict, args.threshold)
                g_utts_with_posteriors += 1
            except Exception as e:
                print(f"Warning: Error building segments for {utt_key}: {e}; skipping")
                continue

        metrics, conf_pairs = compute_lder_and_confusions(
            ref_bin,
            hyp,
            collar=args.collar,
            include_fa=args.include_fa,
        )

        # Global sums
        g_utts += 1
        g_miss += metrics["Miss"]
        g_fa += metrics["FA"]
        g_conf += metrics["Conf"]
        g_ref += metrics["RefSpeech"]
        for (r, h), t in conf_pairs.items():
            g_conf_pairs[(r, h)] += t

        # Per-ref-token breakdown
        ref_t, miss_t, conf_map = compute_by_ref_token_breakdown(
            ref_tok=ref_tok,
            ref_bin=ref_bin,
            hyp=hyp,
            collar=args.collar,
        )
        for tok, t in ref_t.items():
            ref_speech_by_tok[tok] += t
        for tok, t in miss_t.items():
            miss_by_tok[tok] += t
        for tok, hm in conf_map.items():
            for hlab, t in hm.items():
                conf_by_tok[tok][hlab] += t
                conf_total_by_tok[tok] += t

        if args.per_utt:
            print(
                f"{utt_key}\tLDER={metrics['LDER']:.6f}\t"
                f"Miss={metrics['Miss']:.3f}s\tFA={metrics['FA']:.3f}s\t"
                f"Conf={metrics['Conf']:.3f}s\tRefSpeech={metrics['RefSpeech']:.3f}s"
            )

    # ---- Global report ----
    if g_ref <= EPS:
        print("No reference speech time found; cannot compute LDER.")
        return

    g_numer = g_miss + g_conf + (g_fa if args.include_fa else 0.0)
    g_lder = g_numer / g_ref

    print("=== GLOBAL ===")
    print(f"Utts: {g_utts}")
    print(f"Utts with posteriors: {g_utts_with_posteriors}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"RefSpeech: {g_ref:.3f}s")
    print(f"Miss: {g_miss:.3f}s ({g_miss/g_ref*100:.2f}%)")
    print(f"FA: {g_fa:.3f}s ({g_fa/g_ref*100:.2f}%)  (included: {args.include_fa})")
    print(f"Conf: {g_conf:.3f}s ({g_conf/g_ref*100:.2f}%)")
    print(f"LDER: {g_lder:.6f} ({g_lder*100:.2f}%)")
    print(f"(REF mapping: '{args.english_token}' -> ENG(2), everything else -> MATRIX(1))")

    # ---- LDER by REF language token (speech-only) ----
    print("\n=== LDER BY REF LANGUAGE TOKEN (speech-only; FA not included) ===")
    toks = sorted(ref_speech_by_tok.keys(), key=lambda t: ref_speech_by_tok[t], reverse=True)
    if not toks:
        print("(none)")
    else:
        for tok in toks:
            ref_t = float(ref_speech_by_tok.get(tok, 0.0))
            if ref_t < args.min_ref_time:
                continue
            miss_t = float(miss_by_tok.get(tok, 0.0))
            conf_t = float(conf_total_by_tok.get(tok, 0.0))
            lder_t = (miss_t + conf_t) / ref_t if ref_t > EPS else float("nan")
            print(
                f"[REF {tok}] RefSpeech={ref_t:.3f}s  "
                f"Miss={miss_t:.3f}s ({miss_t/ref_t*100:.2f}%)  "
                f"Conf={conf_t:.3f}s ({conf_t/ref_t*100:.2f}%)  "
                f"LDER={lder_t:.6f} ({lder_t*100:.2f}%)"
            )

    # ---- Overall top confusions ----
    print("\n=== TOP CONFUSIONS OVERALL (ref_bin -> hyp_bin), time-weighted ===")
    if not g_conf_pairs:
        print("(none)")
    else:
        top_items = sorted(g_conf_pairs.items(), key=lambda kv: kv[1], reverse=True)[: max(0, args.conf_topk)]
        for (r, h), t in top_items:
            pct = 100.0 * t / g_ref if g_ref > EPS else 0.0
            print(f"{fmt_bin(r)} -> {fmt_bin(h)}\t{t:.3f}s\t({pct:.2f}% of RefSpeech)")

    # ---- Confusions grouped by REF token ----
    print("\n=== TOP CONFUSIONS BY GROUND-TRUTH TOKEN (REF) ===")
    toks2 = sorted(
        ref_speech_by_tok.keys(),
        key=lambda t: (conf_total_by_tok.get(t, 0.0), ref_speech_by_tok.get(t, 0.0)),
        reverse=True,
    )
    if not toks2:
        print("(none)")
        return

    for tok in toks2:
        ref_t = float(ref_speech_by_tok.get(tok, 0.0))
        if ref_t < args.min_ref_time:
            continue

        conf_t = float(conf_total_by_tok.get(tok, 0.0))
        conf_rate = (conf_t / ref_t) if ref_t > EPS else float("nan")

        print(f"[REF {tok}] RefSpeech={ref_t:.3f}s  Conf={conf_t:.3f}s  ConfRate={conf_rate:.4f}")

        hyp_map = conf_by_tok.get(tok, {})
        if not hyp_map:
            print("  (no confusions)")
            continue

        top_h = sorted(hyp_map.items(), key=lambda kv: kv[1], reverse=True)[: max(0, args.per_ref_topk)]
        for h, t in top_h:
            pct_ref = 100.0 * t / ref_t if ref_t > EPS else 0.0
            print(f"  (expected {'ENG(2)' if tok == args.english_token else 'MATRIX(1)'})  -> {fmt_bin(h)}\t{t:.3f}s\t({pct_ref:.2f}% of REF speech)")


if __name__ == "__main__":
    main()
