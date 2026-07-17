#!/usr/bin/env python3
"""
plot_det_curves.py

Plot DET (Detection Error Tradeoff) curves by varying the threshold for accepting
embedded language (label=2) vs matrix language (label=1) based on frame-level posteriors.

This script:
1. Loads predictions with frame-level posteriors and timing info
2. For each threshold T in [0, 1]:
   - Convert frame posteriors to segments using threshold T
   - Compute FAR (False Alarm Rate) and Miss Rate (FNR)
3. Plots DET curve with FAR on x-axis and Miss Rate on y-axis

Metrics (all time-weighted, in reference time):
  - Miss Rate = Miss / RefSpeech
  - FA Rate = FA / NonRefSpeech  (or FA / RefSpeech for classic DER-style)
  - Ref time uses the same boundary collar approach as LDER script

JSONL input (with posteriors): one JSON object per line:
{
  "112": {
    "pred": [{"start":0.0,"end":1.2,"label":1}, ...],  # Original argmax predictions
    "posteriors": {
      "values": [[0.1, 0.9], [0.2, 0.8], ...],  # Frame-level posteriors
      "frame_times": [0.0, 0.04, 0.08, ...],    # Start time of each frame
      "frame_duration": 0.04                     # Duration per frame
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
import matplotlib.pyplot as plt
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
# Segment utilities (from score_lder_detection.py)
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


def build_pred_segments(pred_list: List[dict]) -> List[Seg]:
    """Build predicted segments from list of dicts."""
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
    """Apply boundary collar to ignore transitions."""
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
# Posteriors -> Segments
# -------------------------
def build_pred_segments_from_posteriors(
    posteriors_dict: dict,
    threshold: float,
    class_idx: Optional[int] = None,  # Index of embedded language in posteriors. If None, use last index.
) -> List[Seg]:
    """
    Convert frame-level posteriors to segments using a threshold.
    
    Args:
        posteriors_dict: Dict with 'values', 'frame_times', 'frame_duration'
        threshold: Confidence threshold for class_idx
        class_idx: Index in posteriors (0-based). If None, uses the last index (embedded language).
    
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
    
    values = np.array(values)  # (num_frames, vocab_size)
    if len(values) != len(frame_times):
        raise ValueError(f"Mismatch: {len(values)} frames vs {len(frame_times)} frame times")
    
    # Use last index if class_idx not specified (embedded language is typically last)
    if class_idx is None:
        class_idx = values.shape[1] - 1
    
    # Get posteriors for the target class
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


# -------------------------
# DET Scoring (Language Classification, not speech detection)
# -------------------------
def compute_det_metrics_language(
    ref: List[Seg],
    hyp: List[Seg],
    embedded_label: int = 2,
    collar: float = 0.0,
    compute_fa_normalized: str = "ref_speech",  # "ref_speech" or "non_ref_speech"
) -> Dict[str, float]:
    """
    Compute DET metrics for language classification (embedded vs. matrix).
    
    Args:
        ref: Reference segments with labels (1=matrix, 2=embedded)
        hyp: Hypothesis segments with labels (1=matrix, 2=embedded)
        embedded_label: Which label represents embedded language (default 2)
        collar: Boundary collar in seconds
        compute_fa_normalized: How to normalize FA
          - "ref_speech": FA / RefSpeech (classic DER-style)
          - "non_ref_speech": FA / NonRefSpeech
    
    Returns:
        {"Miss": miss_rate, "FA": fa_rate, "Miss_sec": miss_seconds, "FA_sec": fa_seconds, 
         "RefSpeech": ref_speech, "NonRefSpeech": non_ref_speech}
    
    Metrics:
        - Miss: Reference is EMBEDDED but hypothesis is NOT embedded
        - FA: Reference is NOT embedded but hypothesis IS embedded
        - RefSpeech: Total reference time (both matrix and embedded)
        - NonRefSpeech: Time when reference is NOT embedded (matrix language)
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
        return {
            "Miss": float("nan"),
            "FA": float("nan"),
            "Miss_sec": 0.0,
            "FA_sec": 0.0,
            "RefSpeech": 0.0,
            "NonRefSpeech": 0.0,
        }

    Miss = FA = RefSpeech = NonRefSpeech = 0.0
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

        # Count all reference time (both matrix and embedded)
        if rlab is not None:
            RefSpeech += dt
        
        # Track when reference is NOT embedded (matrix language)
        if rlab is not None and rlab != embedded_label:
            NonRefSpeech += dt
        
        # Only score where reference is speech
        if rlab is None:
            continue
        
        r_is_embedded = (rlab == embedded_label)
        h_is_embedded = (hlab == embedded_label)

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

    if compute_fa_normalized == "ref_speech":
        fa_norm = RefSpeech
    else:
        fa_norm = NonRefSpeech

    miss_rate = Miss / RefSpeech if RefSpeech > EPS else float("nan")
    fa_rate = FA / fa_norm if fa_norm > EPS else float("nan")

    return {
        "Miss": miss_rate,
        "FA": fa_rate,
        "Miss_sec": Miss,
        "FA_sec": FA,
        "RefSpeech": RefSpeech,
        "NonRefSpeech": NonRefSpeech,
    }


def compute_det_metrics(
    ref: List[Seg],
    hyp: List[Seg],
    collar: float = 0.0,
    compute_fa_normalized: str = "ref_speech",  # "ref_speech" or "non_ref_speech"
) -> Dict[str, float]:
    """Backward compatibility wrapper - now calls compute_det_metrics_language."""
    return compute_det_metrics_language(ref, hyp, embedded_label=2, collar=collar, 
                                       compute_fa_normalized=compute_fa_normalized)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot DET curves by varying embedded language threshold")
    ap.add_argument("--input_jsonl", help="Single JSONL file with posteriors.")
    ap.add_argument("--input_jsonl_glob", help='Glob of sharded JSONLs (e.g., "/path/to/*.jsonl").')
    ap.add_argument("--collar", type=float, default=0.0, help="Boundary collar seconds.")
    ap.add_argument("--english_token", default="eng", help="Embedded language token.")
    ap.add_argument("--non_speech_token", default=None, help="Optional non-speech token.")
    ap.add_argument("--output_plot", default="det_curve.png", help="Output plot file.")
    ap.add_argument("--thresholds", default=None, help="Comma-separated thresholds (e.g., '0.3,0.5,0.7'). If None, use [0, 0.1, ..., 1.0]")
    ap.add_argument("--fa_normalized", choices=["ref_speech", "non_ref_speech"], default="ref_speech", 
                    help="How to normalize FA: by ref speech (DER-style) or non-ref speech")
    ap.add_argument("--title", default="DET Curve: Embedded Language Detection", help="Plot title")
    ap.add_argument("--per_utt", action="store_true", help="Print per-utterance results.")
    args = ap.parse_args()

    if (args.input_jsonl is None) == (args.input_jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")

    # Parse thresholds
    if args.thresholds:
        thresholds = [float(x.strip()) for x in args.thresholds.split(",")]
    else:
        thresholds = np.linspace(0, 1, 11).tolist()  # [0, 0.1, ..., 1.0]

    thresholds = sorted(set(thresholds))
    print(f"Testing thresholds: {thresholds}")

    # Accumulate metrics across all utterances
    metrics_by_threshold: Dict[float, Dict[str, float]] = {t: {"Miss": 0.0, "FA": 0.0, "Miss_sec": 0.0, "FA_sec": 0.0, "RefSpeech": 0.0, "NonRefSpeech": 0.0} for t in thresholds}
    n_utts = 0

    for src, ln, obj in iter_jsonl_inputs(jsonl_path=args.input_jsonl, jsonl_glob=args.input_jsonl_glob):
        if not isinstance(obj, dict) or len(obj) != 1:
            raise RuntimeError(
                f"Expected each JSONL line to be a dict with exactly 1 key, got {type(obj)} "
                f"len={len(obj) if isinstance(obj, dict) else 'NA'} at {src}:{ln}"
            )

        utt_key = next(iter(obj.keys()))
        entry = obj[utt_key]

        # Handle nested structure: entry may have {"pred": {...}} or be flat
        pred_dict = entry.get("pred", entry)  # Fall back to entry itself if no "pred" key
        
        passthrough = entry.get("passthrough", {})
        seg_ts = passthrough.get("segment_timestamps", [])
        seg_langs = passthrough.get("segment_langs", [])

        ref_bin, _ = build_ref_segments_binary(
            seg_ts, seg_langs, english_token=args.english_token, non_speech_token=args.non_speech_token
        )

        posteriors_dict = pred_dict.get("posteriors", None)
        if posteriors_dict is None:
            print(f"Warning: No posteriors found for {utt_key} (line {ln}); skipping")
            continue

        n_utts += 1

        # Compute metrics for each threshold
        for threshold in thresholds:
            try:
                hyp = build_pred_segments_from_posteriors(posteriors_dict, threshold)  # Uses last index by default
            except Exception as e:
                print(f"Error building segments for {utt_key} threshold {threshold}: {e}")
                continue

            metrics = compute_det_metrics(
                ref_bin,
                hyp,
                collar=args.collar,
                compute_fa_normalized=args.fa_normalized,
            )

            # Accumulate
            metrics_by_threshold[threshold]["Miss_sec"] += metrics["Miss_sec"]
            metrics_by_threshold[threshold]["FA_sec"] += metrics["FA_sec"]
            metrics_by_threshold[threshold]["RefSpeech"] += metrics["RefSpeech"]
            metrics_by_threshold[threshold]["NonRefSpeech"] += metrics["NonRefSpeech"]

            if args.per_utt:
                print(
                    f"{utt_key}\tThreshold={threshold:.2f}\t"
                    f"Miss={metrics['Miss']:.4f}\tFA={metrics['FA']:.4f}"
                )

    # Compute aggregated rates
    print(f"\nProcessed {n_utts} utterances")
    print(f"\n=== DET Metrics by Threshold ===")

    miss_rates = []
    fa_rates = []
    det_data = []

    for threshold in thresholds:
        m = metrics_by_threshold[threshold]
        ref_speech = m["RefSpeech"]

        if args.fa_normalized == "ref_speech":
            fa_norm = ref_speech
        else:
            fa_norm = m["NonRefSpeech"]

        miss_rate = m["Miss_sec"] / ref_speech if ref_speech > EPS else float("nan")
        fa_rate = m["FA_sec"] / fa_norm if fa_norm > EPS else float("nan")

        miss_rates.append(miss_rate)
        fa_rates.append(fa_rate)
        det_data.append({
            "threshold": threshold,
            "miss_rate": miss_rate,
            "fa_rate": fa_rate,
            "miss_sec": m["Miss_sec"],
            "fa_sec": m["FA_sec"],
            "ref_speech": ref_speech,
            "non_ref_speech": m["NonRefSpeech"]
        })

        print(
            f"Threshold={threshold:.2f}\tMiss={miss_rate:.6f}\tFA={fa_rate:.6f}\t"
            f"(Miss_sec={m['Miss_sec']:.3f}s, FA_sec={m['FA_sec']:.3f}s, RefSpeech={ref_speech:.3f}s)"
        )

    # Estimate EER (Equal Error Rate)
    eer = None
    min_diff = float("inf")
    for mr, fa, t in zip(miss_rates, fa_rates, thresholds):
        diff = abs(mr - fa)
        if diff < min_diff:
            min_diff = diff
            eer = (mr + fa) / 2
            eer_threshold = t
    print(f"\nEstimated EER: {eer:.4f} at threshold {eer_threshold:.2f}")

    # Save DET curve data to file
    det_data_file = args.output_plot.replace(".png", "_det_data.json")
    import json
    with open(det_data_file, "w", encoding="utf-8") as f:
        json.dump(det_data, f, indent=2)
    print(f"DET curve data saved to {det_data_file}")

    # Plot DET curve
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter out NaN values for plotting
    valid_indices = [i for i in range(len(thresholds)) if not (np.isnan(miss_rates[i]) or np.isnan(fa_rates[i]))]
    valid_miss = [miss_rates[i] for i in valid_indices]
    valid_fa = [fa_rates[i] for i in valid_indices]
    valid_thresholds = [thresholds[i] for i in valid_indices]

    ax.plot(valid_fa, valid_miss, "b-o", linewidth=2, markersize=6, label="DET Curve")

    # Annotate points with thresholds
    for i, threshold in enumerate(valid_thresholds):
        ax.annotate(f"{threshold:.2f}", (valid_fa[i], valid_miss[i]), 
                   textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)

    # Mark EER point
    if eer is not None:
        ax.plot([fa_rates[thresholds.index(eer_threshold)]], [miss_rates[thresholds.index(eer_threshold)]], 'ro', markersize=10, label=f'EER={eer:.3f}')

    ax.set_xlabel("False Alarm Rate (FA)", fontsize=12)
    ax.set_ylabel("Miss Rate (FN)", fontsize=12)
    ax.set_title(args.title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Set limits to [0, 1]
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    print(f"\nPlot saved to {args.output_plot}")
    plt.show()


if __name__ == "__main__":
    main()
