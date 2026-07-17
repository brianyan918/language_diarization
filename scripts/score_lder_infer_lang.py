#!/usr/bin/env python3
"""
score_lder_infer_lang.py

Compute Language Diarization Error Rate (LDER) from JSONL predictions, plus:
  1) GLOBAL LDER
  2) LDER BY INFERRED LANGUAGE (extracted from file_name in passthrough)
  3) Top confusions overall (time-weighted)
  4) Top confusions grouped by ground-truth language

JSONL input: one JSON object per line with file_name in passthrough:
{
  "1455": {
    "pred": {"alignments": [...]},
    "passthrough": {
      "file_name": "/path/to/audio/cmn/cmn_1759_CC.wav",
      "segment_timestamps": [[...], ...],
      "segment_langs": ["cmn","eng",...],
      ...
    }
  }
}

Language extraction:
  From "/data/group_data/swl/old_home/byan/cs_fleurs_large/cs-fleurs/read/test/audio/cmn/cmn_1759_CC.wav"
  Extract "cmn" (language code)

Vocab file format:
  2 eng
  3 ara
  ...

Metric (DER-style):
  LDER = (Miss + FA + Conf) / RefSpeech

Usage:
  python score_lder_infer_lang.py --input_jsonl x.jsonl --vocab vocab.txt --collar 0.25 --include_fa \
    --conf_topk 50 --per_ref_topk 10 --min_ref_time 1.0
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


# -------------------------
# Scoring
# -------------------------
def compute_lder_and_confusions(
    ref: List[Seg],
    hyp: List[Seg],
    collar: float = 0.0,
    non_speech_id: Optional[int] = None,
    include_fa: bool = True,
) -> Tuple[Dict[str, float], Dict[Tuple[int, int], float]]:
    """
    Returns:
      metrics: {LDER, Miss, FA, Conf, RefSpeech}
      conf_pairs: {(ref_label, hyp_label): confused_time_seconds} only when both speech and labels differ
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

        r_is_speech = (rlab is not None) and (non_speech_id is None or rlab != non_speech_id)
        h_is_speech = (hlab is not None) and (non_speech_id is None or hlab != non_speech_id)

        if r_is_speech:
            RefSpeech += dt
            if not h_is_speech:
                Miss += dt
            else:
                if hlab != rlab:
                    Conf += dt
                    conf_pairs[(rlab, hlab)] += dt
        else:
            if h_is_speech:
                FA += dt

    if RefSpeech <= EPS:
        metrics = {"LDER": float("nan"), "Miss": Miss, "FA": FA, "Conf": Conf, "RefSpeech": RefSpeech}
        return metrics, dict(conf_pairs)

    numer = Miss + Conf + (FA if include_fa else 0.0)
    metrics = {"LDER": numer / RefSpeech, "Miss": Miss, "FA": FA, "Conf": Conf, "RefSpeech": RefSpeech}
    return metrics, dict(conf_pairs)


def compute_by_inferred_lang_breakdown(
    ref: List[Seg],
    hyp: List[Seg],
    collar: float,
    non_speech_id: Optional[int],
) -> Tuple[
    Dict[int, float],               # ref_speech_by_label
    Dict[int, float],               # miss_by_label
    Dict[int, Dict[int, float]],    # conf_by_ref[ref][hyp]
]:
    """
    Breakdown restricted to intervals where REF is speech.

    - ref_speech_by_label: total ref speech time per ref label (after collar carving)
    - miss_by_label: time where REF=label but HYP is non-speech
    - conf_by_ref: time where REF=ref_label but HYP=hyp_label (hyp is speech and hyp!=ref)
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
    ref_speech_by_label: Dict[int, float] = defaultdict(float)
    miss_by_label: Dict[int, float] = defaultdict(float)
    conf_by_ref: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    if len(B) < 2:
        return {}, {}, {}

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

        r_is_speech = (rlab is not None) and (non_speech_id is None or rlab != non_speech_id)
        h_is_speech = (hlab is not None) and (non_speech_id is None or hlab != non_speech_id)

        if not r_is_speech:
            continue

        # REF is speech
        assert rlab is not None
        ref_speech_by_label[rlab] += dt

        if not h_is_speech:
            miss_by_label[rlab] += dt
        else:
            if hlab != rlab:
                conf_by_ref[rlab][hlab] += dt

    # convert nested defaultdicts to plain dicts
    return dict(ref_speech_by_label), dict(miss_by_label), {r: dict(m) for r, m in conf_by_ref.items()}


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
    ap.add_argument("--include_fa", action="store_true", help="Include FA in GLOBAL LDER numerator (classic DER-style).")
    ap.add_argument("--per_utt", action="store_true", help="Print per-utterance metrics.")

    ap.add_argument("--conf_topk", type=int, default=25, help="Top-K OVERALL confusion pairs to print (by time).")
    ap.add_argument("--per_lang_topk", type=int, default=10, help="Top-K confusions to print per inferred LANGUAGE.")
    ap.add_argument("--min_ref_time", type=float, default=0.0, help="Skip per-lang report if ref speech < this many seconds.")
    ap.add_argument("--exclude_langs", type=str, default="", help="Comma-separated list of language codes to exclude (e.g., 'eng,ara').")
    ap.add_argument("--output_json", help="Optional path to save results as JSON.")
    args = ap.parse_args()
    
    # Parse excluded languages
    excluded_langs = set(lang.strip() for lang in args.exclude_langs.split(",") if lang.strip())

    if (args.input_jsonl is None) == (args.input_jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")

    vocab = load_vocab_id_token(args.vocab)
    inv_vocab = invert_vocab(vocab)

    # Global accumulators
    g_miss = g_fa = g_conf = g_ref = 0.0
    g_conf_pairs: Dict[Tuple[int, int], float] = defaultdict(float)
    g_utts = 0

    # Per-inferred-lang accumulators (post-collar, speech only)
    ref_speech_by_inferred_lang: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    miss_by_inferred_lang: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    conf_by_inferred_lang: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    conf_total_by_ref: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

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

            # Extract language from file_name
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

            metrics, conf_pairs = compute_lder_and_confusions(
                ref,
                hyp,
                collar=args.collar,
                non_speech_id=args.non_speech_id,
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

            # Per-inferred-lang breakdown (speech-only)
            ref_t, miss_t, conf_map = compute_by_inferred_lang_breakdown(
                ref,
                hyp,
                collar=args.collar,
                non_speech_id=args.non_speech_id,
            )
            for rlab, t in ref_t.items():
                ref_speech_by_inferred_lang[inferred_lang][rlab] += t
            for rlab, t in miss_t.items():
                miss_by_inferred_lang[inferred_lang][rlab] += t
            for rlab, hm in conf_map.items():
                for hlab, t in hm.items():
                    conf_by_inferred_lang[inferred_lang][rlab][hlab] += t
                    conf_total_by_ref[inferred_lang][rlab] += t

            if args.per_utt:
                print(
                    f"{utt_key} [{inferred_lang}]\tLDER={metrics['LDER']:.6f}\t"
                    f"Miss={metrics['Miss']:.3f}s\tFA={metrics['FA']:.3f}s\t"
                    f"Conf={metrics['Conf']:.3f}s\tRefSpeech={metrics['RefSpeech']:.3f}s"
                )
        except Exception as e:
            skipped_lines += 1
            continue

    if skipped_lines > 0:
        print(f"Skipped {skipped_lines} lines due to errors.")
    if skipped_no_lang > 0:
        print(f"Skipped {skipped_no_lang} lines due to missing inferred language.")

    # ---- Global report ----
    if g_ref <= EPS:
        print("No reference speech time found; cannot compute LDER.")
        return

    g_numer = g_miss + g_conf + (g_fa if args.include_fa else 0.0)
    g_lder = g_numer / g_ref

    # Prepare JSON output structure
    results = {
        "global": {
            "utts": g_utts,
            "ref_speech": g_ref,
            "miss": g_miss,
            "fa": g_fa,
            "conf": g_conf,
            "lder": g_lder,
            "fa_included": args.include_fa
        },
        "by_inferred_language": {},
        "top_confusions_overall": [],
        "top_confusions_by_inferred_lang": {}
    }

    print("=== GLOBAL ===")
    print(f"Utts: {g_utts}")
    print(f"RefSpeech: {g_ref:.3f}s")
    print(f"Miss: {g_miss:.3f}s")
    print(f"FA: {g_fa:.3f}s  (included: {args.include_fa})")
    print(f"Conf: {g_conf:.3f}s")
    print(f"LDER: {g_lder:.6f}")

    # ---- LDER by INFERRED LANGUAGE (speech-only) ----
    print("=== LDER BY INFERRED LANGUAGE (from file_name; speech-only; FA not included) ===")
    # Sort by inferred language alphabetically
    inferred_langs = sorted(ref_speech_by_inferred_lang.keys())
    if not inferred_langs:
        print("(none)")
    else:
        for inf_lang in inferred_langs:
            ref_by_label = ref_speech_by_inferred_lang[inf_lang]
            miss_by_label = miss_by_inferred_lang[inf_lang]
            conf_by_ref = conf_by_inferred_lang[inf_lang]
            
            # Calculate totals for this inferred language
            total_ref = sum(ref_by_label.values())
            total_miss = sum(miss_by_label.values())
            total_conf = sum(conf_total_by_ref[inf_lang].values())
            
            if total_ref < args.min_ref_time:
                continue
            
            lder_inf = (total_miss + total_conf) / total_ref if total_ref > EPS else float("nan")
            
            # Add to JSON output
            results["by_inferred_language"][inf_lang] = {
                "ref_speech": total_ref,
                "miss": total_miss,
                "conf": total_conf,
                "lder": lder_inf,
                "by_ref_label": {}
            }
            
            print(
                f"[INFERRED LANG {inf_lang}] RefSpeech={total_ref:.3f}s  "
                f"Miss={total_miss:.3f}s  Conf={total_conf:.3f}s  LDER={lder_inf:.6f}"
            )
            
            # Per-ref-label breakdown within this inferred language
            for rlab in sorted(ref_by_label.keys()):
                ref_t = float(ref_by_label.get(rlab, 0.0))
                miss_t = float(miss_by_label.get(rlab, 0.0))
                conf_t = float(conf_total_by_ref[inf_lang].get(rlab, 0.0))
                lder_r = (miss_t + conf_t) / ref_t if ref_t > EPS else float("nan")
                
                ref_str = fmt_label(rlab, inv_vocab)
                results["by_inferred_language"][inf_lang]["by_ref_label"][ref_str] = {
                    "ref_speech": ref_t,
                    "miss": miss_t,
                    "conf": conf_t,
                    "lder": lder_r
                }
                
                print(
                    f"  [REF {ref_str}] RefSpeech={ref_t:.3f}s  "
                    f"Miss={miss_t:.3f}s  Conf={conf_t:.3f}s  LDER={lder_r:.6f}"
                )

    # ---- Overall top confusions ----
    print("=== TOP CONFUSIONS OVERALL (ref -> hyp), time-weighted ===")
    if not g_conf_pairs:
        print("(none)")
    else:
        top_items = sorted(g_conf_pairs.items(), key=lambda kv: kv[1], reverse=True)[: max(0, args.conf_topk)]
        for (r, h), t in top_items:
            pct = 100.0 * t / g_ref if g_ref > EPS else 0.0
            ref_str = fmt_label(r, inv_vocab)
            hyp_str = fmt_label(h, inv_vocab)
            
            # Add to JSON output
            results["top_confusions_overall"].append({
                "ref": ref_str,
                "hyp": hyp_str,
                "time": t,
                "percent_of_ref_speech": pct
            })
            
            print(f"{ref_str} -> {hyp_str}\t{t:.3f}s\t({pct:.2f}% of RefSpeech)")

    # ---- Confusions grouped by inferred language ----
    print("=== TOP CONFUSIONS BY INFERRED LANGUAGE ===")
    
    if not inferred_langs:
        print("(none)")
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n=== Results saved to: {args.output_json} ===")
        return

    for inf_lang in inferred_langs:
        ref_by_label = ref_speech_by_inferred_lang[inf_lang]
        conf_by_ref = conf_by_inferred_lang[inf_lang]
        
        total_ref = sum(ref_by_label.values())
        if total_ref < args.min_ref_time:
            continue
        
        print(f"[INFERRED LANG {inf_lang}]")
        
        # Sort refs by confusion time (descending)
        ref_labels = sorted(
            ref_by_label.keys(),
            key=lambda r: (conf_total_by_ref[inf_lang].get(r, 0.0), ref_by_label.get(r, 0.0)),
            reverse=True,
        )
        
        lang_confusions = []
        for rlab in ref_labels:
            ref_t = float(ref_by_label.get(rlab, 0.0))
            conf_t = float(conf_total_by_ref[inf_lang].get(rlab, 0.0))
            conf_rate = (conf_t / ref_t) if ref_t > EPS else float("nan")
            
            ref_str = fmt_label(rlab, inv_vocab)
            
            hyp_map = conf_by_ref.get(rlab, {})
            if not hyp_map:
                print(f"  [REF {ref_str}] RefSpeech={ref_t:.3f}s  Conf={conf_t:.3f}s  ConfRate={conf_rate:.4f} (no confusions)")
                continue
            
            print(f"  [REF {ref_str}] RefSpeech={ref_t:.3f}s  Conf={conf_t:.3f}s  ConfRate={conf_rate:.4f}")
            
            top_h = sorted(hyp_map.items(), key=lambda kv: kv[1], reverse=True)[: max(0, args.per_lang_topk)]
            
            confusion_list = []
            for h, t in top_h:
                pct_ref = 100.0 * t / ref_t if ref_t > EPS else 0.0
                hyp_str = fmt_label(h, inv_vocab)
                confusion_list.append({
                    "hyp": hyp_str,
                    "time": t,
                    "percent_of_ref_speech": pct_ref
                })
                print(f"    {ref_str} -> {hyp_str}\t{t:.3f}s\t({pct_ref:.2f}% of REF speech)")
            
            lang_confusions.extend(confusion_list)
        
        results["top_confusions_by_inferred_lang"][inf_lang] = lang_confusions
    
    # Save JSON output if requested
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n=== Results saved to: {args.output_json} ===")
    
    # Print easy copy-paste section
    print("\n" + "="*80)
    print("EASY COPY-PASTE FORMAT (LDER by inferred language, alphabetically sorted)")
    print("="*80 + "\n")
    
    for inf_lang in sorted(inferred_langs):
        ref_by_label = ref_speech_by_inferred_lang[inf_lang]
        total_ref = sum(ref_by_label.values())
        if total_ref < args.min_ref_time:
            continue
        
        total_miss = sum(miss_by_inferred_lang[inf_lang].values())
        total_conf = sum(conf_total_by_ref[inf_lang].values())
        lder_inf = (total_miss + total_conf) / total_ref if total_ref > EPS else float("nan")
        
        print(f"{lder_inf:.6f}")


if __name__ == "__main__":
    main()
