#!/usr/bin/env python3
"""
score_switch_f1.py (FIXED, MONOTONIC IN tol)

Key fixes vs your version:
1) Matching is now optimal 1D max-TP matching (two-pointer).
   => TP is non-decreasing as tol increases; recall cannot drop with larger tol.
2) FP_within / FP_far are computed AFTER matching (diagnostics only),
   so they don't interfere with TP/FN.

Everything else (formats, switch extraction, bucket reporting) kept the same.
"""

import argparse
import glob
import json
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional
from collections import defaultdict

EPS = 1e-12


@dataclass
class Seg:
    start: float
    end: float
    label: str  # ref: language code; hyp: token (id->tok) when possible


@dataclass
class RefSwitch:
    t: float
    bucket: str  # short/medium/long


# -------------------------
# I/O
# -------------------------
def load_vocab_id_token(path: str) -> Dict[int, str]:
    """Vocab lines: '2 eng' => {2:'eng'}"""
    id2tok: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Bad vocab line at {path}:{ln}: {line}")
            idx = int(parts[0])
            tok = parts[1]
            id2tok[idx] = tok
    return id2tok


def iter_jsonl_inputs(jsonl_path: Optional[str], jsonl_glob: Optional[str]) -> Iterator[dict]:
    if (jsonl_path is None) == (jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")
    paths = [jsonl_path] if jsonl_path else sorted(glob.glob(jsonl_glob))
    if not paths:
        raise RuntimeError("No JSONL files found")
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


# -------------------------
# Helpers
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


def bucket_by_next_seg_duration(dur: float) -> str:
    if dur < 0.5:
        return "short"
    if dur <= 2.0:
        return "medium"
    return "long"


def extract_ref_segments_flat(obj: dict) -> List[Seg]:
    segs = []
    for s in obj.get("segments", []):
        a = float(s["audio_start_sec"])
        b = float(s["audio_end_sec"])
        if b > a:
            segs.append(Seg(a, b, str(s["lang"])))
    return merge_adjacent(segs)


def extract_ref_segments_nested(entry: dict) -> List[Seg]:
    pt = entry.get("passthrough", {})
    ts = pt.get("segment_timestamps", [])
    langs = pt.get("segment_langs", [])
    segs = []
    for (a, b), lang in zip(ts, langs):
        a = float(a)
        b = float(b)
        if b > a:
            segs.append(Seg(a, b, str(lang)))
    return merge_adjacent(segs)


def extract_hyp_segments_nested(entry: dict, id2tok: Dict[int, str]) -> List[Seg]:
    segs = []
    for d in entry.get("pred", []):
        a = float(d["start"])
        b = float(d["end"])
        if b <= a:
            continue
        lab_id = int(d["label"])
        segs.append(Seg(a, b, id2tok.get(lab_id, str(lab_id))))
    return merge_adjacent(segs)


def extract_hyp_segments_flat(obj: dict, id2tok: Dict[int, str]) -> List[Seg]:
    segs = []
    for d in obj.get("pred", []):
        a = float(d["start"])
        b = float(d["end"])
        if b <= a:
            continue
        lab_id = int(d["label"])
        segs.append(Seg(a, b, id2tok.get(lab_id, str(lab_id))))
    return merge_adjacent(segs)


def extract_ref_switches(ref: List[Seg], max_gap: float) -> List[RefSwitch]:
    out = []
    for a, b in zip(ref[:-1], ref[1:]):
        gap = b.start - a.end
        if gap < -1e-6:
            continue
        if gap > max_gap:
            continue
        if a.label != b.label:
            dur_next = max(0.0, b.end - b.start)
            out.append(RefSwitch(t=b.start, bucket=bucket_by_next_seg_duration(dur_next)))
    return out


def extract_pred_switch_times(hyp: List[Seg], max_gap: float) -> List[float]:
    out = []
    for a, b in zip(hyp[:-1], hyp[1:]):
        gap = b.start - a.end
        if gap < -1e-6:
            continue
        if gap > max_gap:
            continue
        if a.label != b.label:
            out.append(b.start)
    return out


# -------------------------
# Correct matching (monotonic in tol)
# -------------------------
def match_max_tp(ref_times: List[float], pred_times: List[float], tol: float):
    """
    Optimal max-cardinality matching in 1D with tolerance using two pointers.

    Returns:
      TP, FN, matched_pred_indices (indices in sorted pred_times)
    """
    ref_times = sorted(ref_times)
    pred_times = sorted(pred_times)

    i = j = 0
    TP = 0
    matched_pred = set()

    while i < len(ref_times) and j < len(pred_times):
        r = ref_times[i]
        p = pred_times[j]

        if p < r - tol - 1e-9:
            # pred too early to match this or any later ref
            j += 1
        elif p > r + tol + 1e-9:
            # no remaining pred can match this ref
            i += 1
        else:
            # match
            TP += 1
            matched_pred.add(j)
            i += 1
            j += 1

    FN = len(ref_times) - TP
    return TP, FN, matched_pred


def fp_within_far(ref_times: List[float], pred_times: List[float], tol: float, matched_pred_indices: set):
    """
    Diagnostic split of unmatched preds:
      FP_within: unmatched pred within tol of ANY ref time
      FP_far   : unmatched pred not within tol of any ref time
    """
    ref_times = sorted(ref_times)
    pred_times = sorted(pred_times)

    fp_within = 0
    fp_far = 0

    # pointer over ref_times for neighborhood checks
    i = 0
    for j, p in enumerate(pred_times):
        if j in matched_pred_indices:
            continue

        while i < len(ref_times) and ref_times[i] < p - tol - 1e-9:
            i += 1

        within = False
        if i < len(ref_times) and abs(ref_times[i] - p) <= tol + 1e-9:
            within = True
        elif i > 0 and abs(ref_times[i - 1] - p) <= tol + 1e-9:
            within = True

        if within:
            fp_within += 1
        else:
            fp_far += 1

    return fp_within, fp_far


def match_one_utt(ref_sw: List[RefSwitch], pred_t: List[float], tol: float):
    """
    Monotonic, correct per-utt matching:
    - Maximize TP with two-pointer matching.
    - Bucket TP/FN by ref switch bucket.
    - Compute FP_within/FP_far diagnostics after matching.
    """
    ref_sw = sorted(ref_sw, key=lambda x: x.t)
    pred_t = sorted(pred_t)

    ref_times = [rs.t for rs in ref_sw]
    TP, FN, matched_pred = match_max_tp(ref_times, pred_t, tol)

    # Bucket accounting for TP/FN:
    # We need to know which ref switches were matched. With two-pointer matching,
    # matches are in time order, so we can reconstruct which refs matched by replaying.
    bucket_ref = defaultdict(int)
    bucket_tp = defaultdict(int)
    bucket_fn = defaultdict(int)
    for rs in ref_sw:
        bucket_ref[rs.bucket] += 1

    # Re-play matching to bucket TP/FN per ref
    i = j = 0
    while i < len(ref_sw) and j < len(pred_t):
        r = ref_sw[i].t
        p = pred_t[j]
        if p < r - tol - 1e-9:
            j += 1
        elif p > r + tol + 1e-9:
            bucket_fn[ref_sw[i].bucket] += 1
            i += 1
        else:
            bucket_tp[ref_sw[i].bucket] += 1
            i += 1
            j += 1
    while i < len(ref_sw):
        bucket_fn[ref_sw[i].bucket] += 1
        i += 1

    FP_within, FP_far = fp_within_far(ref_times, pred_t, tol, matched_pred)

    return {
        "TP": TP,
        "FN": FN,
        "FP_within": FP_within,
        "FP_far": FP_far,
        "bucket_ref": bucket_ref,
        "bucket_tp": bucket_tp,
        "bucket_fn": bucket_fn,
        "bucket_fp_within": defaultdict(int, {k: 0 for k in ["short", "medium", "long"]}),  # kept for compatibility
        "pred_switches": len(pred_t),
        "ref_switches": len(ref_sw),
    }


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", help="Single JSONL file")
    ap.add_argument("--input_jsonl_glob", help="Glob of JSONLs")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--tol", type=float, default=0.25)
    ap.add_argument("--max_gap", type=float, default=0.5)
    ap.add_argument("--by_langpair", action="store_true", help="Also report per obj['language'] (e.g., ara-eng)")
    args = ap.parse_args()

    id2tok = load_vocab_id_token(args.vocab)

    g = {"TP": 0, "FN": 0, "FP_within": 0, "FP_far": 0, "RefSwitches": 0, "PredSwitches": 0}
    gb_ref = defaultdict(int)
    gb_tp = defaultdict(int)
    gb_fn = defaultdict(int)

    per = defaultdict(
        lambda: {
            "TP": 0, "FN": 0, "FP_within": 0, "FP_far": 0, "RefSwitches": 0, "PredSwitches": 0,
            "b_ref": defaultdict(int), "b_tp": defaultdict(int), "b_fn": defaultdict(int)
        }
    )

    utts = 0

    for obj in iter_jsonl_inputs(args.input_jsonl, args.input_jsonl_glob):
        utts += 1

        if isinstance(obj, dict) and len(obj) == 1 and "pred" not in obj and "segments" not in obj:
            entry = next(iter(obj.values()))
            ref = extract_ref_segments_nested(entry)
            hyp = extract_hyp_segments_nested(entry, id2tok)
            langpair = entry.get("passthrough", {}).get("language", "UNKNOWN_PAIR")
        else:
            ref = extract_ref_segments_flat(obj)
            hyp = extract_hyp_segments_flat(obj, id2tok)
            langpair = obj.get("language", "UNKNOWN_PAIR")

        ref_sw = extract_ref_switches(ref, args.max_gap)
        pred_sw = extract_pred_switch_times(hyp, args.max_gap)

        res = match_one_utt(ref_sw, pred_sw, args.tol)

        g["TP"] += res["TP"]
        g["FN"] += res["FN"]
        g["FP_within"] += res["FP_within"]
        g["FP_far"] += res["FP_far"]
        g["RefSwitches"] += res["ref_switches"]
        g["PredSwitches"] += res["pred_switches"]

        for b in ["short", "medium", "long"]:
            gb_ref[b] += res["bucket_ref"][b]
            gb_tp[b] += res["bucket_tp"][b]
            gb_fn[b] += res["bucket_fn"][b]

        if args.by_langpair:
            d = per[langpair]
            d["TP"] += res["TP"]
            d["FN"] += res["FN"]
            d["FP_within"] += res["FP_within"]
            d["FP_far"] += res["FP_far"]
            d["RefSwitches"] += res["ref_switches"]
            d["PredSwitches"] += res["pred_switches"]
            for b in ["short", "medium", "long"]:
                d["b_ref"][b] += res["bucket_ref"][b]
                d["b_tp"][b] += res["bucket_tp"][b]
                d["b_fn"][b] += res["bucket_fn"][b]

    P, R, F1 = prf(g["TP"], g["FP_within"] + g["FP_far"], g["FN"])
    print("=== GLOBAL SWITCH-POINT METRICS (PER-UTT, MAX-TP MATCHING) ===")
    print(f"utts={utts}  tol={args.tol:.3f}s  max_gap={args.max_gap:.3f}s")
    print(
        f"RefSwitches={g['RefSwitches']}  PredSwitches={g['PredSwitches']}  "
        f"TP={g['TP']}  FN={g['FN']}  FP_within={g['FP_within']}  FP_far={g['FP_far']}"
    )
    print(f"P={P:.6f}  R={R:.6f}  F1={F1:.6f}")

    print("=== BY SWITCH DURATION (bucketed by next REF segment duration) ===")
    for b in ["short", "medium", "long"]:
        p, r, f1 = prf(gb_tp[b], 0, gb_fn[b])  # bucket FP is not well-defined; use global FP for global P
        # If you want bucket precision, you need bucket FP attribution (non-trivial). We report recall/F1_proxy here.
        print(
            f"{b:6s}  ref={gb_ref[b]:7d}  tp={gb_tp[b]:7d}  fn={gb_fn[b]:7d}  "
            f"R={r:.6f}"
        )

    if args.by_langpair:
        print("=== PER LANGUAGE-PAIR ===")
        for lp in sorted(per.keys(), key=lambda k: per[k]["RefSwitches"], reverse=True):
            d = per[lp]
            p, r, f1 = prf(d["TP"], d["FP_within"] + d["FP_far"], d["FN"])
            print(f"[{lp}] RefSwitches={d['RefSwitches']} PredSwitches={d['PredSwitches']} P={p:.6f} R={r:.6f} F1={f1:.6f}")
            for b in ["short", "medium", "long"]:
                _, rb, _ = prf(d["b_tp"][b], 0, d["b_fn"][b])
                print(f"  {b:6s}  ref={d['b_ref'][b]:7d} tp={d['b_tp'][b]:7d} fn={d['b_fn'][b]:7d}  R={rb:.6f}")


if __name__ == "__main__":
    main()
