#!/usr/bin/env python3
"""
score_jaccard.py

Compute Jaccard Error Rate (JER) from JSONL predictions, plus:
  1) GLOBAL JER
  2) JER BY REF LANGUAGE (Jaccard similarity restricted to time where REF label is that language)
  3) Top confusions overall (time-weighted)
  4) Top confusions grouped by ground-truth language (ref label)

JSONL input: one JSON object per line:
{
  "112": {
    "pred": [...],
    "passthrough": {
      "utt_id": "test-112-ara_1916_BT",
      "segment_timestamps": [[...], ...],
      "segment_langs": ["ara","eng",...],
      ...
    }
  }
}

Vocab file format:
  2 eng
  3 ara
  ...

Jaccard Similarity (IoU):
  For a given label pair, Jaccard = Intersection / Union of time intervals
  JER = 1 - mean(Jaccard_i for all language pairs)

- Exact segment overlap (no frame discretization).
- Optional boundary collar: ignore +/- collar seconds around REF label-change boundaries.
- "Speech" is time covered by reference language segments (segment_timestamps), optionally excluding non_speech_id.
- Optional inferred language filtering: extract language from file_name and exclude specified languages.

Usage:
  python score_jaccard.py --input_jsonl x.jsonl --vocab vocab.txt --collar 0.25 \
    --conf_topk 50 --per_ref_topk 10 --min_ref_time 1.0
  
  With language exclusion (requires file_name in passthrough):
  python score_jaccard.py --input_jsonl x.jsonl --vocab vocab.txt --exclude_langs "eng,ara" \
    --collar 0.25
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple
from collections import defaultdict

EPS = 1e-12


@dataclass
class Seg:
    start: float
    end: float
    label: int


# -------------------------
# I/O
# -------------------------
def load_vocab_id_token(path: str) -> Dict[str, int]:
    """Parse vocab lines like: '2 eng' -> {'eng': 2}"""
    vocab: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Bad vocab line at {path}:{ln} (expected '<id> <token>'): {line}")
            idx = int(parts[0])
            tok = parts[1]
            vocab[tok] = idx
    if not vocab:
        raise ValueError(f"Empty vocab parsed from: {path}")
    return vocab


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """id -> token"""
    inv: Dict[int, str] = {}
    for tok, idx in vocab.items():
        inv[idx] = tok
    return inv


def extract_lang_from_filename(file_name: str) -> Optional[str]:
    """
    Extract language code from file path.
    
    Examples:
      "/path/to/audio/cmn/cmn_1759_CC.wav" -> "cmn"
      "/path/to/audio/eng/eng_123_AB.wav" -> "eng"
      "/path/to/test.wav" -> None
    
    Strategy: Look for 3-letter code in directory path before filename.
    """
    if not file_name:
        return None
    
    # Get the directory part and filename
    dir_path = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    
    # Remove extension from filename
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Try to extract 3-letter code from the start of the filename
    if len(name_without_ext) >= 3:
        potential_lang = name_without_ext[:3]
        # Check if it looks like a language code (all lowercase letters)
        if potential_lang.isalpha() and potential_lang.islower():
            return potential_lang
    
    # Fallback: try to get from parent directory
    parent_dir = os.path.basename(dir_path)
    if len(parent_dir) == 3 and parent_dir.isalpha() and parent_dir.islower():
        return parent_dir
    
    return None


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


def build_ref_segments(seg_ts: List[List[float]], seg_langs: List[str], vocab: Dict[str, int]) -> List[Seg]:
    if len(seg_ts) != len(seg_langs):
        raise ValueError(f"segment_timestamps and segment_langs mismatch: {len(seg_ts)} vs {len(seg_langs)}")
    ref: List[Seg] = []
    for (s, e), lang in zip(seg_ts, seg_langs):
        if lang not in vocab:
            raise KeyError(f"Language '{lang}' not in vocab")
        s = float(s)
        e = float(e)
        if e <= s:
            continue
        ref.append(Seg(s, e, vocab[lang]))
    return merge_adjacent(ref)


def build_pred_segments(pred_list: List[dict]) -> List[Seg]:
    pred: List[Seg] = []
    for d in pred_list:
        s = float(d["start"])
        e = float(d["end"])
        lab = int(d["label"])
        if e <= s:
            continue
        pred.append(Seg(s, e, lab))
    return merge_adjacent(pred)


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
    """
    Ignore +/- collar seconds around ref label-change boundaries where segments touch (end==next.start).
    Returns carved ref and ignored intervals.
    """
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


def in_any_interval(t: float, intervals: List[Tuple[float, float]], idx: int) -> Tuple[bool, int]:
    n = len(intervals)
    while idx < n and intervals[idx][1] <= t + 1e-12:
        idx += 1
    if idx < n:
        s, e = intervals[idx]
        if s <= t < e:
            return True, idx
    return False, idx


def collect_label_intervals(segs: List[Seg], label: int) -> List[Tuple[float, float]]:
    """Collect all time intervals for a given label."""
    intervals: List[Tuple[float, float]] = []
    for seg in segs:
        if seg.label == label:
            intervals.append((seg.start, seg.end))
    return merge_intervals(intervals)


def interval_intersection_time(intervals_a: List[Tuple[float, float]], intervals_b: List[Tuple[float, float]]) -> float:
    """Compute total time where intervals from A and B overlap."""
    total = 0.0
    for sa, ea in intervals_a:
        for sb, eb in intervals_b:
            intersection_start = max(sa, sb)
            intersection_end = min(ea, eb)
            if intersection_end > intersection_start:
                total += intersection_end - intersection_start
    return total


def interval_union_time(intervals_a: List[Tuple[float, float]], intervals_b: List[Tuple[float, float]]) -> float:
    """Compute total time covered by union of intervals from A and B."""
    merged = merge_intervals(intervals_a + intervals_b)
    return sum(e - s for s, e in merged)


# -------------------------
# Scoring
# -------------------------
def compute_jer_and_confusions(
    ref: List[Seg],
    hyp: List[Seg],
    collar: float = 0.0,
    non_speech_id: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[Tuple[int, int], float]]:
    """
    Returns:
      metrics: {JER, Jaccard_mean, RefSpeech}
      conf_pairs: {(ref_label, hyp_label): intersection_over_union} for all label pairs
    
    Note: Collar is NOT applied to Jaccard scoring since Jaccard (IoU) is already
    robust to boundary misalignment. The collar concept (forgiveness zone) is more
    relevant for frame-by-frame metrics like LDER.
    """
    ref = merge_adjacent(ref)
    hyp = merge_adjacent(hyp)

    # Collect all speech labels from ref and hyp (excluding non_speech_id if provided)
    ref_labels = set()
    hyp_labels = set()
    for seg in ref:
        if non_speech_id is None or seg.label != non_speech_id:
            ref_labels.add(seg.label)
    for seg in hyp:
        if non_speech_id is None or seg.label != non_speech_id:
            hyp_labels.add(seg.label)

    all_labels = ref_labels | hyp_labels

    if not all_labels:
        return (
            {"JER": float("nan"), "Jaccard_mean": float("nan"), "RefSpeech": 0.0},
            {},
        )

    # Compute Jaccard for each label
    jaccard_scores: Dict[int, float] = {}
    conf_pairs: Dict[Tuple[int, int], float] = {}

    for label in all_labels:
        ref_intervals = collect_label_intervals(ref, label)
        hyp_intervals = collect_label_intervals(hyp, label)

        intersection = interval_intersection_time(ref_intervals, hyp_intervals)
        union = interval_union_time(ref_intervals, hyp_intervals)

        if union <= EPS:
            jaccard = 1.0  # both are empty, perfect match
        else:
            jaccard = intersection / union

        jaccard_scores[label] = jaccard
        # Store as (label, label) pair for reference
        conf_pairs[(label, label)] = jaccard

    # Compute mean Jaccard
    if jaccard_scores:
        jaccard_mean = sum(jaccard_scores.values()) / len(jaccard_scores)
    else:
        jaccard_mean = float("nan")

    jer = 1.0 - jaccard_mean if not (jaccard_mean != jaccard_mean) else float("nan")  # 1 - nan = nan

    # Compute ref speech time (no collar applied)
    ref_speech = sum(seg.end - seg.start for seg in ref
                     if non_speech_id is None or seg.label != non_speech_id)

    metrics = {
        "JER": jer,
        "Jaccard_mean": jaccard_mean,
        "RefSpeech": ref_speech,
    }

    return metrics, conf_pairs


def compute_by_ref_breakdown(
    ref: List[Seg],
    hyp: List[Seg],
    collar: float,
    non_speech_id: Optional[int],
) -> Tuple[
    Dict[int, float],               # ref_speech_by_label
    Dict[int, float],               # jaccard_by_label
]:
    """
    Breakdown restricted to intervals where REF is speech.

    - ref_speech_by_label: total ref speech time per ref label
    - jaccard_by_label: Jaccard IoU for each ref label
    
    Note: Collar is NOT applied to Jaccard scoring since Jaccard (IoU) is already
    robust to boundary misalignment.
    """
    ref = merge_adjacent(ref)
    hyp = merge_adjacent(hyp)

    # Collect labels that appear in ref as speech
    ref_labels_speech = set()
    for seg in ref:
        if non_speech_id is None or seg.label != non_speech_id:
            ref_labels_speech.add(seg.label)

    if not ref_labels_speech:
        return {}, {}

    ref_speech_by_label: Dict[int, float] = {}
    jaccard_by_label: Dict[int, float] = {}

    for label in ref_labels_speech:
        ref_intervals = collect_label_intervals(ref, label)
        hyp_intervals = collect_label_intervals(hyp, label)

        ref_speech_by_label[label] = sum(e - s for s, e in ref_intervals)

        intersection = interval_intersection_time(ref_intervals, hyp_intervals)
        union = interval_union_time(ref_intervals, hyp_intervals)

        if union <= EPS:
            jaccard = 1.0
        else:
            jaccard = intersection / union

        jaccard_by_label[label] = jaccard

    return ref_speech_by_label, jaccard_by_label


def fmt_label(label: int, inv_vocab: Dict[int, str]) -> str:
    return inv_vocab.get(label, str(label))


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", help="Single JSONL file (one JSON object per line).")
    ap.add_argument("--input_jsonl_glob", help='Glob of sharded JSONLs (e.g., "/path/to/*.jsonl").')
    ap.add_argument("--vocab", required=True, help="Vocab file: '<id> <token>' per line.")
    ap.add_argument("--collar", type=float, default=0.0, help="Boundary collar seconds (e.g., 0.25).")
    ap.add_argument("--non_speech_id", type=int, default=None, help="Optional NONSPEECH label id.")
    ap.add_argument("--per_utt", action="store_true", help="Print per-utterance metrics.")

    ap.add_argument("--conf_topk", type=int, default=25, help="Top-K OVERALL label pairs to print (by Jaccard).")
    ap.add_argument("--per_ref_topk", type=int, default=10, help="Top-K labels to print per REF language.")
    ap.add_argument("--min_ref_time", type=float, default=0.0, help="Skip per-ref report if ref speech < this many seconds.")
    ap.add_argument("--exclude_langs", type=str, default="", help="Comma-separated list of language codes to exclude (e.g., 'eng,ara'). Requires file_name in passthrough.")
    ap.add_argument("--output_json", help="Optional path to save results as JSON.")
    args = ap.parse_args()
    
    # Parse excluded languages
    excluded_langs = set(lang.strip() for lang in args.exclude_langs.split(",") if lang.strip())

    if (args.input_jsonl is None) == (args.input_jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")

    vocab = load_vocab_id_token(args.vocab)
    inv_vocab = invert_vocab(vocab)

    # Global accumulators
    g_jer = 0.0
    g_jaccard_sum = 0.0
    g_jaccard_count = 0
    g_ref_speech = 0.0
    g_label_jaccards: Dict[int, List[float]] = defaultdict(list)  # label -> list of jaccard scores
    g_utts = 0

    # Per-ref accumulators (post-collar, speech only)
    ref_speech_by_label: Dict[int, float] = defaultdict(float)    # ref label -> seconds
    jaccard_by_label: Dict[int, List[float]] = defaultdict(list)  # ref label -> list of Jaccard scores

    skipped_lines = 0
    skipped_no_lang = 0

    for src, ln, obj in iter_jsonl_inputs(jsonl_path=args.input_jsonl, jsonl_glob=args.input_jsonl_glob):
        try:
            if not isinstance(obj, dict) or len(obj) != 1:
                raise RuntimeError(
                    f"Expected each JSONL line to be a dict with exactly 1 key, got {type(obj)} "
                    f"len={len(obj) if isinstance(obj, dict) else 'NA'} at {src}:{ln}"
                )

            utt_key = next(iter(obj.keys()))
            entry = obj[utt_key]

            pred_list = entry.get("pred", [])
            if isinstance(pred_list, dict) and "alignments" in pred_list:
                pred_list = pred_list.get("alignments", [])
            passthrough = entry.get("passthrough", {})
            
            # Extract language from file_name if exclusion list is provided
            if excluded_langs:
                file_name = passthrough.get("file_name", "")
                inferred_lang = extract_lang_from_filename(file_name)
                if not inferred_lang:
                    skipped_no_lang += 1
                    continue
                
                # Skip if language is excluded
                if inferred_lang in excluded_langs:
                    continue

            seg_ts = passthrough.get("segment_timestamps", [])
            seg_langs = passthrough.get("segment_langs", [])

            ref = build_ref_segments(seg_ts, seg_langs, vocab)
            hyp = build_pred_segments(pred_list)

            metrics, conf_pairs = compute_jer_and_confusions(
                ref,
                hyp,
                collar=args.collar,
                non_speech_id=args.non_speech_id,
            )

            # Global sums
            g_utts += 1
            if not (metrics["Jaccard_mean"] != metrics["Jaccard_mean"]):  # not NaN
                g_jaccard_sum += metrics["Jaccard_mean"]
                g_jaccard_count += 1
            g_ref_speech += metrics["RefSpeech"]

            # Accumulate label jaccards globally
            for (label_a, label_b), jaccard in conf_pairs.items():
                if label_a == label_b:  # Only label self-similarity
                    g_label_jaccards[label_a].append(jaccard)

            # Per-ref breakdown (speech-only)
            ref_t, jac_t = compute_by_ref_breakdown(
                ref,
                hyp,
                collar=args.collar,
                non_speech_id=args.non_speech_id,
            )
            for rlab, t in ref_t.items():
                ref_speech_by_label[rlab] += t
            for rlab, jac in jac_t.items():
                jaccard_by_label[rlab].append(jac)

            if args.per_utt:
                print(
                    f"{utt_key}\tJER={metrics['JER']:.6f}\t"
                    f"Jaccard_mean={metrics['Jaccard_mean']:.6f}\tRefSpeech={metrics['RefSpeech']:.3f}s"
                )
        except Exception as e:
            skipped_lines += 1
            continue

    if skipped_lines > 0:
        print(f"Skipped {skipped_lines} lines due to errors.")
    if skipped_no_lang > 0:
        print(f"Skipped {skipped_no_lang} lines due to missing inferred language.")

    # ---- Global report ----
    if g_utts == 0:
        print("No utterances processed.")
        return

    if g_jaccard_count > 0:
        g_jaccard_mean = g_jaccard_sum / g_jaccard_count
        g_jer = 1.0 - g_jaccard_mean
    else:
        g_jaccard_mean = float("nan")
        g_jer = float("nan")

    # Prepare JSON output structure
    results = {
        "global": {
            "utts": g_utts,
            "ref_speech": g_ref_speech,
            "jaccard_mean": g_jaccard_mean,
            "jer": g_jer,
        },
        "by_ref_language": {},
    }

    print("=== GLOBAL ===")
    print(f"Utts: {g_utts}")
    print(f"RefSpeech: {g_ref_speech:.3f}s")
    print(f"Jaccard (mean): {g_jaccard_mean:.6f}")
    print(f"JER: {g_jer:.6f}")

    # ---- JER by REF language ----
    print("=== JER BY REF LANGUAGE ===")
    # Sort by ref speech descending
    ref_labels = sorted(ref_speech_by_label.keys(), key=lambda r: ref_speech_by_label[r], reverse=True)
    if not ref_labels:
        print("(none)")
    else:
        for r in ref_labels:
            ref_t = float(ref_speech_by_label.get(r, 0.0))
            if ref_t < args.min_ref_time:
                continue

            jac_list = jaccard_by_label.get(r, [])
            if jac_list:
                jac_mean = sum(jac_list) / len(jac_list)
                jer_r = 1.0 - jac_mean
            else:
                jac_mean = float("nan")
                jer_r = float("nan")

            label_str = fmt_label(r, inv_vocab)
            results["by_ref_language"][label_str] = {
                "ref_speech": ref_t,
                "jaccard": jac_mean,
                "jer": jer_r
            }

            print(
                f"[REF {label_str}] RefSpeech={ref_t:.3f}s  "
                f"Jaccard={jac_mean:.6f}  JER={jer_r:.6f}"
            )

    # Save JSON output if requested
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n=== Results saved to: {args.output_json} ===")


if __name__ == "__main__":
    main()
