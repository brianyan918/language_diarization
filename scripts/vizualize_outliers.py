#!/usr/bin/env python3
"""
rank_and_viz_lder_switch.py

Compute **LDER** and **switch-point F1** per utterance, then visualize:
  - the N worst LDER utterances (highest LDER)
  - the N worst switch-F1 utterances (lowest F1)

Outputs:
  <out_dir>/per_utt_metrics.jsonl
  <out_dir>/worst_lder/*.png
  <out_dir>/worst_switchf1/*.png

Assumptions / Supported input formats
-------------------------------------
(1) Nested (your common format): one JSON object per line, dict with exactly 1 key:
{
  "112": {
    "pred": [{"start":..,"end":..,"label":..}, ...],
    "passthrough": {
      "utt_id": "...",
      "segment_timestamps": [[s,e], ...],
      "segment_langs": ["ara","eng", ...],
      "file_name": "...",    # optional
      "language": "ara-eng"  # optional
    }
  }
}

(2) Flat (also supported):
{
  "segments": [{"audio_start_sec":..,"audio_end_sec":..,"lang":"ara"}, ...],
  "pred": [{"start":..,"end":..,"label":..}, ...],
  "utt_id": "...",      # optional
  "file_name": "...",   # optional
  "language": "ara-eng" # optional
}

Vocab file:
  <id> <token>   e.g. "2 eng"

Notes on metrics
----------------
LDER:
  LDER = (Miss + Conf + (FA if --include_fa)) / RefSpeech
  - Exact interval scoring (no frames)
  - Optional collar around REF boundaries
  - Hyp gaps treated as NONSPEECH (hlab=None)

Switch F1:
  - Extract REF switch times at boundaries where label changes and gap <= max_gap
  - Extract HYP switch times similarly (on merged hyp segments)
  - Optimal max-TP 1D matching within +/- tol (two-pointer; monotonic in tol)
  - If ref_switches==0 and pred_switches==0 => F1=1
    If ref_switches==0 and pred_switches>0 => F1=0

Visualization
-------------
For each selected utterance, we render REF vs HYP bars and overlay:
  - green vertical lines: REF switch times
  - red vertical lines: HYP switch times
"""

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl

EPS = 1e-12


# -------------------------
# Data structures
# -------------------------
@dataclass
class Seg:
    start: float
    end: float
    label: int  # for LDER + plotting


@dataclass
class SegStr:
    start: float
    end: float
    label: str  # for switch extraction


@dataclass
class RefSwitch:
    t: float
    bucket: str


# -------------------------
# Vocab
# -------------------------
def load_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Parse vocab lines like: '2 eng' -> tok2id and id2tok."""
    tok2id: Dict[str, int] = {}
    id2tok: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Bad vocab line at {path}:{ln}: {line}")
            idx = int(parts[0])
            tok = parts[1]
            tok2id[tok] = idx
            id2tok[idx] = tok
    if not tok2id:
        raise ValueError(f"Empty vocab: {path}")
    return tok2id, id2tok


# -------------------------
# Input iterator
# -------------------------
def iter_jsonl_inputs(jsonl_path: Optional[str], jsonl_glob: Optional[str]) -> Iterator[Tuple[str, int, dict]]:
    if (jsonl_path is None) == (jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")
    paths = [jsonl_path] if jsonl_path else sorted(glob.glob(jsonl_glob or ""))
    if not paths:
        raise RuntimeError("No JSONL files found")
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                yield p, ln, json.loads(line)


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


def merge_adjacent_str(segs: List[SegStr]) -> List[SegStr]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: (s.start, s.end))
    out = [SegStr(segs[0].start, segs[0].end, segs[0].label)]
    for s in segs[1:]:
        prev = out[-1]
        if abs(s.start - prev.end) <= 1e-9 and s.label == prev.label:
            prev.end = max(prev.end, s.end)
        else:
            out.append(SegStr(s.start, s.end, s.label))
    return out


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
# Extractors (ref/hyp)
# -------------------------
def parse_record(obj: dict) -> Tuple[str, dict, dict]:
    """
    Returns (utt_key, entry, passthrough_like)
    - Nested: utt_key is dict key; entry is its value; pt=entry["passthrough"]
    - Flat: utt_key is obj.get("utt_id") or fallback; entry is obj; pt=obj (as passthrough-like)
    """
    if isinstance(obj, dict) and len(obj) == 1 and "pred" not in obj and "segments" not in obj:
        utt_key = next(iter(obj.keys()))
        entry = obj[utt_key]
        pt = entry.get("passthrough", {}) or {}
        return utt_key, entry, pt
    # flat
    entry = obj
    utt_key = str(obj.get("utt_id") or obj.get("id") or "UNKNOWN_UTT")
    pt = obj  # passthrough-like
    return utt_key, entry, pt


def build_ref_segments(entry: dict, pt: dict, tok2id: Dict[str, int]) -> List[Seg]:
    # nested style preferred
    if "segment_timestamps" in pt and "segment_langs" in pt:
        ts = pt.get("segment_timestamps", [])
        langs = pt.get("segment_langs", [])
        if len(ts) != len(langs):
            raise ValueError(f"segment_timestamps and segment_langs mismatch: {len(ts)} vs {len(langs)}")
        ref: List[Seg] = []
        for (s, e), lang in zip(ts, langs):
            s = float(s)
            e = float(e)
            if e <= s:
                continue
            if lang not in tok2id:
                raise KeyError(f"Language '{lang}' not in vocab")
            ref.append(Seg(s, e, tok2id[str(lang)]))
        return merge_adjacent(ref)

    # flat style
    segs = []
    for s in entry.get("segments", []):
        a = float(s["audio_start_sec"])
        b = float(s["audio_end_sec"])
        if b <= a:
            continue
        lang = str(s["lang"])
        if lang not in tok2id:
            raise KeyError(f"Language '{lang}' not in vocab")
        segs.append(Seg(a, b, tok2id[lang]))
    return merge_adjacent(segs)


def build_hyp_segments(entry: dict) -> List[Seg]:
    hyp: List[Seg] = []
    for d in entry.get("pred", []):
        s = float(d["start"])
        e = float(d["end"])
        lab = int(d["label"])
        if e > s:
            hyp.append(Seg(s, e, lab))
    return merge_adjacent(hyp)


def build_ref_segments_str(ref: List[Seg], id2tok: Dict[int, str]) -> List[SegStr]:
    return merge_adjacent_str([SegStr(s.start, s.end, id2tok.get(s.label, str(s.label))) for s in ref])


def build_hyp_segments_str(hyp: List[Seg], id2tok: Dict[int, str]) -> List[SegStr]:
    return merge_adjacent_str([SegStr(s.start, s.end, id2tok.get(s.label, str(s.label))) for s in hyp])


# -------------------------
# LDER per utterance
# -------------------------
def compute_lder(
    ref: List[Seg],
    hyp: List[Seg],
    collar: float = 0.0,
    non_speech_id: Optional[int] = None,
    include_fa: bool = True,
) -> Dict[str, float]:
    """
    Returns dict with: LDER, Miss, FA, Conf, RefSpeech
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
        return {"LDER": float("nan"), "Miss": 0.0, "FA": 0.0, "Conf": 0.0, "RefSpeech": 0.0}

    Miss = FA = Conf = RefSpeech = 0.0
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
        else:
            if h_is_speech:
                FA += dt

    if RefSpeech <= EPS:
        return {"LDER": float("nan"), "Miss": Miss, "FA": FA, "Conf": Conf, "RefSpeech": RefSpeech}

    numer = Miss + Conf + (FA if include_fa else 0.0)
    return {"LDER": numer / RefSpeech, "Miss": Miss, "FA": FA, "Conf": Conf, "RefSpeech": RefSpeech}


# -------------------------
# Switch F1 per utterance
# -------------------------
def bucket_by_next_seg_duration(dur: float) -> str:
    if dur < 0.5:
        return "short"
    if dur <= 2.0:
        return "medium"
    return "long"


def extract_ref_switches(ref: List[SegStr], max_gap: float) -> List[RefSwitch]:
    out: List[RefSwitch] = []
    for a, b in zip(ref[:-1], ref[1:]):
        gap = b.start - a.end
        if gap < -1e-6:
            continue
        if gap > max_gap:
            continue
        if a.label != b.label:
            out.append(RefSwitch(t=b.start, bucket=bucket_by_next_seg_duration(max(0.0, b.end - b.start))))
    return out


def extract_pred_switch_times(hyp: List[SegStr], max_gap: float) -> List[float]:
    out: List[float] = []
    for a, b in zip(hyp[:-1], hyp[1:]):
        gap = b.start - a.end
        if gap < -1e-6:
            continue
        if gap > max_gap:
            continue
        if a.label != b.label:
            out.append(b.start)
    return out


def match_max_tp(ref_times: List[float], pred_times: List[float], tol: float) -> Tuple[int, int, int]:
    """
    Returns (TP, FP, FN) for 1D max-TP matching within +/- tol.
    """
    ref_times = sorted(ref_times)
    pred_times = sorted(pred_times)

    i = j = 0
    tp = 0

    while i < len(ref_times) and j < len(pred_times):
        r = ref_times[i]
        p = pred_times[j]
        if p < r - tol - 1e-9:
            j += 1
        elif p > r + tol + 1e-9:
            i += 1
        else:
            tp += 1
            i += 1
            j += 1

    fn = len(ref_times) - tp
    fp = len(pred_times) - tp
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_switch_f1_per_utt(
    ref_str: List[SegStr],
    hyp_str: List[SegStr],
    tol: float,
    max_gap: float,
) -> Dict[str, float]:
    ref_sw = extract_ref_switches(ref_str, max_gap=max_gap)
    pred_t = extract_pred_switch_times(hyp_str, max_gap=max_gap)

    ref_times = [x.t for x in ref_sw]

    # edge cases to make ranking intuitive
    if len(ref_times) == 0 and len(pred_t) == 0:
        return {"P": 1.0, "R": 1.0, "F1": 1.0, "TP": 0, "FP": 0, "FN": 0, "RefSwitches": 0, "PredSwitches": 0}
    if len(ref_times) == 0 and len(pred_t) > 0:
        return {"P": 0.0, "R": 1.0, "F1": 0.0, "TP": 0, "FP": len(pred_t), "FN": 0,
                "RefSwitches": 0, "PredSwitches": len(pred_t)}

    tp, fp, fn = match_max_tp(ref_times, pred_t, tol=tol)
    p, r, f1 = prf(tp, fp, fn)
    return {"P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn,
            "RefSwitches": len(ref_times), "PredSwitches": len(pred_t)}


# -------------------------
# Visualization
# -------------------------
def label_name(label: int, id2tok: Dict[int, str]) -> str:
    return id2tok.get(label, str(label))


def color_for_label(label: int, cmap) -> Tuple[float, float, float, float]:
    # deterministic pseudo-random color by label id
    x = (abs(int(label)) * 2654435761) % 2**32
    v = (x / 2**32)
    return cmap(v)


def clamp_segments(segs: List[Seg], tmin: float, tmax: float) -> List[Seg]:
    out = []
    for s in segs:
        a = max(tmin, s.start)
        b = min(tmax, s.end)
        if b > a:
            out.append(Seg(a, b, s.label))
    return out


def get_utt_title(utt_key: str, pt: dict) -> str:
    utt_id = pt.get("utt_id", "") if isinstance(pt, dict) else ""
    fn = pt.get("file_name", "") if isinstance(pt, dict) else ""
    if utt_id:
        return f"{utt_key} | {utt_id}"
    if fn:
        return f"{utt_key} | {os.path.basename(fn)}"
    return utt_key


def plot_utt_with_switches(
    ref: List[Seg],
    hyp: List[Seg],
    ref_sw_t: List[float],
    hyp_sw_t: List[float],
    title: str,
    subtitle: str,
    id2tok: Dict[int, str],
    out_path: str,
    max_duration: Optional[float],
):
    t0 = 0.0
    t1 = 0.0
    for s in ref + hyp:
        t1 = max(t1, s.end)

    if max_duration is not None and max_duration > 0:
        t1 = min(t1, max_duration)
        ref = clamp_segments(ref, t0, t1)
        hyp = clamp_segments(hyp, t0, t1)
        ref_sw_t = [t for t in ref_sw_t if t0 <= t <= t1]
        hyp_sw_t = [t for t in hyp_sw_t if t0 <= t <= t1]

    # Slightly taller to accommodate fig-level text cleanly
    fig, ax = plt.subplots(figsize=(14, 2.9), dpi=150)
    cmap = mpl.cm.get_cmap("tab20")

    bar_h = 0.35
    y_ref = 0.65
    y_hyp = 0.15

    def draw_row(segs: List[Seg], y: float):
        for s in segs:
            ax.broken_barh(
                [(s.start, s.end - s.start)],
                (y, bar_h),
                facecolors=color_for_label(s.label, cmap),
                edgecolors="none",
            )

    draw_row(ref, y_ref)
    draw_row(hyp, y_hyp)

    # overlay switches
    for t in ref_sw_t:
        ax.axvline(t, ymin=0.0, ymax=1.0, linewidth=1.0, alpha=0.8, color="green")
    for t in hyp_sw_t:
        ax.axvline(t, ymin=0.0, ymax=1.0, linewidth=1.0, alpha=0.8, color="red")

    ax.set_xlim(t0, max(t1, 0.01))
    ax.set_ylim(0, 1.25)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")

    # ---- legend (same idea, but a bit lower) ----
    present = []
    seen = set()
    for s in ref + hyp:
        if s.label not in seen:
            seen.add(s.label)
            present.append(s.label)
    present = present[:10]

    handles = [mpl.patches.Patch(color=color_for_label(lab, cmap), label=label_name(lab, id2tok)) for lab in present]
    handles += [
        mpl.lines.Line2D([0], [0], color="green", lw=2, label="REF switch"),
        mpl.lines.Line2D([0], [0], color="red", lw=2, label="HYP switch"),
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.42),
            ncol=min(len(handles), 6),
            frameon=False,
            fontsize=8,
        )

    # ---- put title + subtitle at FIGURE level to avoid overlap ----
    fig.suptitle(title, fontsize=10, y=0.98)
    fig.text(0.01, 0.915, subtitle, fontsize=8, va="top", ha="left")

    # Reserve room at top for suptitle/subtitle, and bottom for legend
    fig.subplots_adjust(top=0.78, bottom=0.28)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", help="Single JSONL file")
    ap.add_argument("--input_jsonl_glob", help='Glob of JSONLs (e.g., "/path/*.jsonl")')
    ap.add_argument("--vocab", required=True)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=20, help="N worst utterances to visualize for each metric")

    # LDER knobs
    ap.add_argument("--collar", type=float, default=0.0)
    ap.add_argument("--non_speech_id", type=int, default=None)
    ap.add_argument("--include_fa", action="store_true", help="Include FA in LDER numerator")

    # Switch-F1 knobs
    ap.add_argument("--tol", type=float, default=0.25)
    ap.add_argument("--max_gap", type=float, default=0.5)

    # viz
    ap.add_argument("--max_duration", type=float, default=0.0)
    ap.add_argument("--prefix", type=str, default="utt")

    args = ap.parse_args()
    tok2id, id2tok = load_vocab(args.vocab)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    per_utt = []
    max_dur = args.max_duration if args.max_duration and args.max_duration > 0 else None

    # compute per-utt metrics
    for src, ln, obj in iter_jsonl_inputs(args.input_jsonl, args.input_jsonl_glob):
        try:
            utt_key, entry, pt = parse_record(obj)
            ref = build_ref_segments(entry, pt, tok2id)
            hyp = build_hyp_segments(entry)

            lder = compute_lder(
                ref, hyp,
                collar=args.collar,
                non_speech_id=args.non_speech_id,
                include_fa=args.include_fa,
            )

            ref_str = build_ref_segments_str(ref, id2tok)
            hyp_str = build_hyp_segments_str(hyp, id2tok)
            sw = compute_switch_f1_per_utt(ref_str, hyp_str, tol=args.tol, max_gap=args.max_gap)

            # switch times for visualization
            ref_sw = extract_ref_switches(ref_str, max_gap=args.max_gap)
            hyp_sw_t = extract_pred_switch_times(hyp_str, max_gap=args.max_gap)

            per_utt.append({
                "utt_key": utt_key,
                "src": src,
                "line": ln,
                "utt_id": pt.get("utt_id", "") if isinstance(pt, dict) else "",
                "language": pt.get("language", entry.get("language", "UNKNOWN_PAIR")) if isinstance(pt, dict) else entry.get("language", "UNKNOWN_PAIR"),
                "file_name": pt.get("file_name", "") if isinstance(pt, dict) else entry.get("file_name", ""),
                "lder": lder,
                "switch": sw,
                "_viz": {  # not written to jsonl
                    "ref": ref,
                    "hyp": hyp,
                    "ref_sw_t": [x.t for x in ref_sw],
                    "hyp_sw_t": hyp_sw_t,
                    "title": get_utt_title(utt_key, pt),
                }
            })
        except Exception as e:
            # keep going, but record error
            per_utt.append({
                "utt_key": f"ERROR@{src}:{ln}",
                "src": src,
                "line": ln,
                "error": str(e),
            })

    # write per-utt metrics jsonl (without heavy viz payload)
    metrics_path = os.path.join(out_dir, "per_utt_metrics.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for r in per_utt:
            rr = dict(r)
            rr.pop("_viz", None)
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    print(f"Wrote per-utt metrics: {metrics_path}")

    # select worst N by LDER (higher is worse)
    lder_candidates = [
        r for r in per_utt
        if "_viz" in r and isinstance(r.get("lder", {}).get("LDER", None), (int, float)) and not math.isnan(r["lder"]["LDER"])
    ]
    lder_candidates.sort(key=lambda r: r["lder"]["LDER"], reverse=True)
    worst_lder = lder_candidates[: max(0, args.n)]

    # select worst N by switch F1 (lower is worse)
    sw_candidates = [
        r for r in per_utt
        if "_viz" in r and isinstance(r.get("switch", {}).get("F1", None), (int, float)) and not math.isnan(r["switch"]["F1"])
    ]
    sw_candidates.sort(key=lambda r: r["switch"]["F1"])
    worst_sw = sw_candidates[: max(0, args.n)]

    # visualize worst LDER
    out_lder_dir = os.path.join(out_dir, "worst_lder")
    os.makedirs(out_lder_dir, exist_ok=True)
    for i, r in enumerate(worst_lder):
        vz = r["_viz"]
        l = r["lder"]
        s = r["switch"]
        subtitle = (
            f"LDER={l['LDER']:.4f}  Miss={l['Miss']:.2f}s  Conf={l['Conf']:.2f}s  "
            f"FA={l['FA']:.2f}s (incl={bool(args.include_fa)})  RefSpeech={l['RefSpeech']:.2f}s | "
            f"SwitchF1={s['F1']:.4f} (P={s['P']:.3f}, R={s['R']:.3f})  "
            f"RefSw={int(s['RefSwitches'])}  HypSw={int(s['PredSwitches'])}"
        )
        out_path = os.path.join(out_lder_dir, f"{args.prefix}.worst_lder.{i:04d}.{r['utt_key']}.png")
        plot_utt_with_switches(
            ref=vz["ref"],
            hyp=vz["hyp"],
            ref_sw_t=vz["ref_sw_t"],
            hyp_sw_t=vz["hyp_sw_t"],
            title=vz["title"],
            subtitle=subtitle,
            id2tok=id2tok,
            out_path=out_path,
            max_duration=max_dur,
        )
    print(f"Wrote {len(worst_lder)} worst-LDER figures to: {out_lder_dir}")

    # visualize worst switch F1
    out_sw_dir = os.path.join(out_dir, "worst_switchf1")
    os.makedirs(out_sw_dir, exist_ok=True)
    for i, r in enumerate(worst_sw):
        vz = r["_viz"]
        l = r["lder"]
        s = r["switch"]

        # LDER could be nan for some, handle gracefully
        lder_str = f"{l['LDER']:.4f}" if isinstance(l.get("LDER", None), (int, float)) and not math.isnan(l["LDER"]) else "nan"
        subtitle = (
            f"SwitchF1={s['F1']:.4f}  P={s['P']:.4f}  R={s['R']:.4f}  "
            f"TP={int(s['TP'])} FP={int(s['FP'])} FN={int(s['FN'])} | "
            f"RefSw={int(s['RefSwitches'])} HypSw={int(s['PredSwitches'])}  "
            f"(tol={args.tol:.2f}s max_gap={args.max_gap:.2f}s) | "
            f"LDER={lder_str}"
        )
        out_path = os.path.join(out_sw_dir, f"{args.prefix}.worst_switchf1.{i:04d}.{r['utt_key']}.png")
        plot_utt_with_switches(
            ref=vz["ref"],
            hyp=vz["hyp"],
            ref_sw_t=vz["ref_sw_t"],
            hyp_sw_t=vz["hyp_sw_t"],
            title=vz["title"],
            subtitle=subtitle,
            id2tok=id2tok,
            out_path=out_path,
            max_duration=max_dur,
        )
    print(f"Wrote {len(worst_sw)} worst-switchF1 figures to: {out_sw_dir}")


if __name__ == "__main__":
    main()
