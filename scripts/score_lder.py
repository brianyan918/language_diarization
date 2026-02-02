#!/usr/bin/env python3
"""
score_lder.py

Compute Language Diarization Error Rate (LDER) from JSONL predictions, plus:
  - global confusion summary
  - per-language-group scoring (language group extracted from passthrough["utt_id"])
  - top-2 confusions per language group

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

Language group extraction (as requested):
  group = passthrough["utt_id"].split("-")[-1].split("_")[0]

Vocab file format:
  2 eng
  3 ara
  ...

Metric (DER-style):
  LDER = (Miss + FA + Conf) / RefSpeech

- Exact segment overlap (no frame discretization).
- Optional boundary collar: ignore +/- collar seconds around REF label-change boundaries.
- "Speech" is time covered by reference language segments (segment_timestamps), optionally excluding non_speech_id.
- Hypothesis gaps are treated as NONSPEECH.

Usage:
  Single file:
    python score_lder.py --input_jsonl x.jsonl --vocab vocab.txt --collar 0.25 --include_fa
  Sharded:
    python score_lder.py --input_jsonl_glob "/path/*.jsonl" --vocab vocab.txt --collar 0.25 --include_fa
"""

import argparse
import glob
import json
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


def fmt_label(label: int, inv_vocab: Dict[int, str]) -> str:
    return inv_vocab.get(label, str(label))


def extract_group(passthrough: dict) -> str:
    """
    As requested:
      group = passthrough["utt_id"].split("-")[-1].split("_")[0]
    """
    utt_id = passthrough.get("utt_id", None)
    if not utt_id or not isinstance(utt_id, str):
        return "UNKNOWN_GROUP"
    try:
        return utt_id.split("-")[-1].split("_")[0]
    except Exception:
        return "UNKNOWN_GROUP"


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
    ap.add_argument("--include_fa", action="store_true", help="Include FA in LDER numerator (classic DER-style).")
    ap.add_argument("--per_utt", action="store_true", help="Print per-utterance metrics.")
    ap.add_argument("--conf_topk", type=int, default=0, help="Top-K GLOBAL confusion pairs to print (by time).")
    args = ap.parse_args()

    if (args.input_jsonl is None) == (args.input_jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")

    vocab = load_vocab_id_token(args.vocab)
    inv_vocab = invert_vocab(vocab)

    # Global accumulators
    g_miss = g_fa = g_conf = g_ref = 0.0
    g_conf_pairs: Dict[Tuple[int, int], float] = defaultdict(float)
    g_utts = 0

    # Per-group accumulators
    grp_miss = defaultdict(float)
    grp_fa = defaultdict(float)
    grp_conf = defaultdict(float)
    grp_ref = defaultdict(float)
    grp_utts = defaultdict(int)
    grp_conf_pairs: Dict[str, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))

    for src, ln, obj in iter_jsonl_inputs(jsonl_path=args.input_jsonl, jsonl_glob=args.input_jsonl_glob):
        if not isinstance(obj, dict) or len(obj) != 1:
            raise RuntimeError(
                f"Expected each JSONL line to be a dict with exactly 1 key, got {type(obj)} "
                f"len={len(obj) if isinstance(obj, dict) else 'NA'} at {src}:{ln}"
            )

        utt_key = next(iter(obj.keys()))
        entry = obj[utt_key]

        pred_list = entry.get("pred", [])
        passthrough = entry.get("passthrough", {})

        seg_ts = passthrough.get("segment_timestamps", [])
        seg_langs = passthrough.get("segment_langs", [])

        group = extract_group(passthrough)

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

        # Group sums
        grp_utts[group] += 1
        grp_miss[group] += metrics["Miss"]
        grp_fa[group] += metrics["FA"]
        grp_conf[group] += metrics["Conf"]
        grp_ref[group] += metrics["RefSpeech"]
        for (r, h), t in conf_pairs.items():
            grp_conf_pairs[group][(r, h)] += t

        if args.per_utt:
            print(
                f"{utt_key}\tgroup={group}\tLDER={metrics['LDER']:.6f}\t"
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
    print(f"RefSpeech: {g_ref:.3f}s")
    print(f"Miss: {g_miss:.3f}s")
    print(f"FA: {g_fa:.3f}s  (included: {args.include_fa})")
    print(f"Conf: {g_conf:.3f}s")
    print(f"LDER: {g_lder:.6f}")

    if g_conf_pairs:
        print("=== GLOBAL CONFUSIONS (ref -> hyp), time-weighted ===")
        items = sorted(g_conf_pairs.items(), key=lambda kv: kv[1], reverse=True)[: max(0, args.conf_topk)]
        for (r, h), t in items:
            pct = 100.0 * t / g_ref if g_ref > EPS else 0.0
            print(f"{fmt_label(r, inv_vocab)} -> {fmt_label(h, inv_vocab)}\t{t:.3f}s\t({pct:.2f}% of RefSpeech)")
    else:
        print("=== GLOBAL CONFUSIONS ===")
        print("No confusion time accumulated.")

    # # ---- Per-group report ----
    # print("=== PER-GROUP (by passthrough['utt_id'] suffix lang) ===")
    # # Sort groups by amount of reference speech time (descending)
    # for group in sorted(grp_ref.keys(), key=lambda g: grp_ref[g], reverse=True):
    #     ref_t = grp_ref[group]
    #     if ref_t <= EPS:
    #         lder = float("nan")
    #     else:
    #         numer = grp_miss[group] + grp_conf[group] + (grp_fa[group] if args.include_fa else 0.0)
    #         lder = numer / ref_t

    #     print(
    #         f"[{group}] utts={grp_utts[group]}  RefSpeech={ref_t:.3f}s  "
    #         f"Miss={grp_miss[group]:.3f}s  FA={grp_fa[group]:.3f}s  Conf={grp_conf[group]:.3f}s  "
    #         f"LDER={lder:.6f}"
    #     )

    #     # top-2 confusions for this group
    #     cp = grp_conf_pairs[group]
    #     if cp:
    #         top2 = sorted(cp.items(), key=lambda kv: kv[1], reverse=True)[:2]
    #         for (r, h), t in top2:
    #             pct = 100.0 * t / ref_t if ref_t > EPS else 0.0
    #             print(f"  TOPCONF  {fmt_label(r, inv_vocab)} -> {fmt_label(h, inv_vocab)}\t{t:.3f}s\t({pct:.2f}% of group RefSpeech)")
    #     else:
    #         print("  TOPCONF  (none)")

    # Optional sanity: ensure sum of group refs ~ global ref
    # print(f"DEBUG sum(group RefSpeech) = {sum(grp_ref.values()):.6f}, global RefSpeech = {g_ref:.6f}")


if __name__ == "__main__":
    main()
