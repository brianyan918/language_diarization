#!/usr/bin/env python3
"""
plot_det_curves_multilang.py

Plot DET (Detection Error Tradeoff) curves for multi-class frame-level LID,
treating English as embedded language and all others as matrix language.

This script:
1. Loads frame-level LID posteriors (multi-class, not binary)
2. For each threshold T in [0, 1]:
   - Aggregate posteriors by language class
   - Map English (eng) to label 2, all others to label 1
   - Convert frame posteriors to segments using threshold T
   - Compute FAR (False Alarm Rate) and Miss Rate (FNR)
3. Plots DET curve with FAR on x-axis and Miss Rate on y-axis

Vocab structure (from vocab file):
  0 - pad
  1 - eng (embedded language)
  2 - ara
  3 - fra
  ... (other languages)

JSONL input (with multi-class posteriors): one JSON object per line:
{
  "112": {
    "pred": {
      "alignments": [...],
      "posteriors": {
        "values": [[0.1, 0.4, 0.3, 0.2, ...], ...],  # Frame-level posteriors (multi-class)
        "frame_times": [0.0, 0.04, ...],             # Start time of each frame
        "frame_duration": 0.04                        # Duration per frame
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
import matplotlib.pyplot as plt
from scipy.special import ndtri
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


def load_vocab(vocab_file: str) -> Dict[str, int]:
    """Load vocab file with '<idx> <token>' per line and return {token: idx}."""
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                idx, token = parts
                try:
                    idx = int(idx)
                except ValueError:
                    continue
                vocab[token] = idx
            elif len(parts) == 1:
                # fallback for old format
                token = parts[0]
                vocab[token] = len(vocab)
    return vocab


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
    vocab: Dict[str, int],
    english_token: str,
    non_speech_token: Optional[str],
) -> Tuple[List[Seg], List[TokSeg]]:
    """
    Build binary ref segments (label 1=matrix, 2=eng) using vocab mapping.
    """
    if len(seg_ts) != len(seg_langs):
        raise ValueError(f"segment_timestamps and segment_langs mismatch: {len(seg_ts)} vs {len(seg_langs)}")

    ref_bin: List[Seg] = []
    ref_tok: List[TokSeg] = []
    eng_idx = vocab.get(english_token)
    for (s, e), lang in zip(seg_ts, seg_langs):
        s = float(s)
        e = float(e)
        if e <= s:
            continue

        # Map lang to vocab index if it's an int or str
        if lang in vocab:
            lang_idx = vocab[lang]
        else:
            try:
                lang_idx = int(lang)
            except Exception:
                lang_idx = None

        ref_tok.append(TokSeg(s, e, str(lang)))

        if non_speech_token is not None and lang == non_speech_token:
            continue
        lab = 2 if lang_idx == eng_idx else 1
        ref_bin.append(Seg(s, e, lab))

    return merge_adjacent(ref_bin), merge_adjacent_tok(ref_tok)


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
# Posteriors -> Segments (Multi-class)
# -------------------------
def build_pred_segments_from_multilang_posteriors(
    posteriors_dict: dict,
    threshold: float,
    vocab: Dict[str, int],
    english_token: str = "eng",
) -> List[Seg]:
    """
    Convert frame-level multi-class LID posteriors to binary segments using a threshold.
    
    Args:
        posteriors_dict: Dict with 'values', 'frame_times', 'frame_duration'
        threshold: Confidence threshold for English (embedded language)
        vocab: {token: idx} mapping from vocabulary file
        english_token: Token representing English (default "eng")
    
    Returns:
        List[Seg] with labels 1 (matrix/non-English) or 2 (English/embedded)
    """
    if not posteriors_dict:
        return []
    
    values = posteriors_dict.get("values", [])
    frame_times = posteriors_dict.get("frame_times", [])
    frame_duration = posteriors_dict.get("frame_duration", 0.0)
    
    if not values or not frame_times or frame_duration <= 0:
        return []
    
    values = np.array(values)  # (num_frames, num_languages)
    if len(values) != len(frame_times):
        raise ValueError(f"Mismatch: {len(values)} frames vs {len(frame_times)} frame times")
    
    # Get index of English in vocabulary
    if english_token not in vocab:
        raise ValueError(f"English token '{english_token}' not found in vocabulary")
    eng_idx = vocab[english_token]
    
    if eng_idx >= values.shape[1]:
        raise ValueError(f"English index {eng_idx} out of range for posteriors with {values.shape[1]} classes")
    
    # Get posteriors for English language
    eng_posteriors = values[:, eng_idx]
    
    # Label each frame based on threshold
    segments: List[Seg] = []
    for i, (post, t) in enumerate(zip(eng_posteriors, frame_times)):
        lab = 2 if post >= threshold else 1  # 2 = English/embedded, 1 = matrix
        start = t
        end = t + frame_duration
        
        if segments and segments[-1].label == lab and abs(segments[-1].end - start) <= 1e-9:
            # Merge with previous segment
            segments[-1].end = end
        else:
            segments.append(Seg(start, end, lab))
    
    return segments


# -------------------------
# Probit (Normal Deviate) Scale
# -------------------------
def rate_to_probit(rate: float) -> float:
    """Convert error rate to probit (normal deviate) scale.
    
    Args:
        rate: Error rate in [0, 1]
    
    Returns:
        Probit value (normal deviate scale)
    """
    # Clip to avoid numerical issues at extremes
    rate = np.clip(rate, 1e-6, 1 - 1e-6)
    return ndtri(rate)


def probit_to_rate(probit: float) -> float:
    """Convert probit (normal deviate) back to error rate.
    
    Args:
        probit: Normal deviate value
    
    Returns:
        Error rate in (0, 1)
    """
    from scipy.special import ndtr
    return ndtr(probit)


# -------------------------
# DET Scoring (Language Classification)
# -------------------------
def compute_det_metrics_language(
    ref: List[Seg],
    hyp: List[Seg],
    embedded_label: int = 2,
    collar: float = 0.0,
    compute_fa_normalized: str = "ref_speech",
) -> Dict[str, float]:
    """Compute DET metrics for language classification (embedded vs. matrix)."""
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

        # Count all reference time
        if rlab is not None:
            RefSpeech += dt
        
        # Track when reference is NOT embedded
        if rlab is not None and rlab != embedded_label:
            NonRefSpeech += dt
        
        # Only score where reference is speech
        if rlab is None:
            continue
        
        r_is_embedded = (rlab == embedded_label)
        h_is_embedded = (hlab == embedded_label)

        if r_is_embedded:
            if not h_is_embedded:
                Miss += dt
        else:
            if h_is_embedded:
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
    compute_fa_normalized: str = "ref_speech",
) -> Dict[str, float]:
    """Wrapper for language classification DET metrics."""
    return compute_det_metrics_language(ref, hyp, embedded_label=2, collar=collar, 
                                       compute_fa_normalized=compute_fa_normalized)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot DET curves for multi-class frame-level LID")
    ap.add_argument("--input_jsonl", help="Single JSONL file with posteriors.")
    ap.add_argument("--input_jsonl_glob", help='Glob of sharded JSONLs (e.g., "/path/to/*.jsonl").')
    ap.add_argument("--vocab_file", required=True, help="Vocabulary file (one token per line).")
    ap.add_argument("--collar", type=float, default=0.0, help="Boundary collar seconds.")
    ap.add_argument("--english_token", default="eng", help="English token name in vocab.")
    ap.add_argument("--non_speech_token", default=None, help="Optional non-speech token.")
    ap.add_argument("--output_plot", default="det_curve_multilang.png", help="Output plot file.")
    ap.add_argument("--thresholds", default=None, help="Comma-separated thresholds (e.g., '0.3,0.5,0.7'). If None, use [0, 0.1, ..., 1.0]")
    ap.add_argument("--fa_normalized", choices=["ref_speech", "non_ref_speech"], default="ref_speech", 
                    help="How to normalize FA: by ref speech (DER-style) or non-ref speech")
    ap.add_argument("--title", default="DET Curve: English Detection (Multi-class LID)", help="Plot title")
    ap.add_argument("--per_utt", action="store_true", help="Print per-utterance results.")
    args = ap.parse_args()

    if (args.input_jsonl is None) == (args.input_jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")

    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab_file}...")
    vocab = load_vocab(args.vocab_file)
    print(f"  Loaded {len(vocab)} tokens")
    if args.english_token in vocab:
        print(f"  English token '{args.english_token}' at index {vocab[args.english_token]}")
    else:
        print(f"  WARNING: English token '{args.english_token}' not found in vocabulary!")

    # Parse thresholds
    if args.thresholds:
        thresholds = [float(x.strip()) for x in args.thresholds.split(",")]
    else:
        thresholds = np.linspace(0, 1, 11).tolist()  # [0, 0.1, ..., 1.0]

    thresholds = sorted(set(thresholds))
    print(f"Testing thresholds: {thresholds}\n")

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

        # Handle nested structure
        pred_dict = entry.get("pred", entry)
        
        passthrough = entry.get("passthrough", {})
        seg_ts = passthrough.get("segment_timestamps", [])
        seg_langs = passthrough.get("segment_langs", [])

        ref_bin, _ = build_ref_segments_binary(
            seg_ts, seg_langs, vocab, english_token=args.english_token, non_speech_token=args.non_speech_token
        )

        posteriors_dict = pred_dict.get("posteriors", None)
        if posteriors_dict is None:
            print(f"Warning: No posteriors found for {utt_key} (line {ln}); skipping")
            continue

        n_utts += 1

        # Compute metrics for each threshold
        for threshold in thresholds:
            try:
                hyp = build_pred_segments_from_multilang_posteriors(
                    posteriors_dict, threshold, vocab, english_token=args.english_token
                )
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

    # Convert to probit scale
    valid_miss_probit = [rate_to_probit(m) for m in valid_miss]
    valid_fa_probit = [rate_to_probit(f) for f in valid_fa]

    ax.plot(valid_fa_probit, valid_miss_probit, "b-o", linewidth=2, markersize=6, label="DET Curve")

    # Annotate points with thresholds
    for i, threshold in enumerate(valid_thresholds):
        ax.annotate(f"{threshold:.2f}", (valid_fa_probit[i], valid_miss_probit[i]), 
                   textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)

    # Mark EER point
    if eer is not None:
        eer_idx = thresholds.index(eer_threshold)
        eer_fa_probit = rate_to_probit(fa_rates[eer_idx])
        eer_miss_probit = rate_to_probit(miss_rates[eer_idx])
        ax.plot([eer_fa_probit], [eer_miss_probit], 'ro', markersize=10, label=f'EER={eer:.3f}')

    ax.set_xlabel("False Alarm Rate (FA)", fontsize=12)
    ax.set_ylabel("Miss Rate (FN)", fontsize=12)
    ax.set_title(args.title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Set probit scale limits
    # Probit scale typically ranges from about -3 to 3 (for error rates 0.1% to 99.9%)
    probit_min, probit_max = -3.5, 3.5
    ax.set_xlim([probit_min, probit_max])
    ax.set_ylim([probit_min, probit_max])

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    print(f"\nPlot saved to {args.output_plot}")
    plt.show()


if __name__ == "__main__":
    main()
