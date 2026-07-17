#!/usr/bin/env python3
"""
generate_error_simulation.py

Simulate a range of language diarization error rates by progressively correcting
predicted diarizations from their current state toward ground truth (silver).

Core workflow:
1. Compare silver (ref) vs predicted (hyp) diarizations
2. Detect error segments (contiguous ref segments mislabeled in predictions)
3. Distribute corrections across N buckets deterministically
4. Generate N intermediate JSONL files with progressively corrected predictions
5. Output format matches input (preserves passthrough, updates pred)

Input JSONL format (one line per utterance):
{
  "utt_id": {
    "pred": [{"start": float, "end": float, "label": int, "score": null}, ...],
    "passthrough": {
      "segment_timestamps": [[start, end], ...],
      "segment_langs": ["lang1", "lang2", ...],
      ... other fields ...
    }
  }
}

Output: num_buckets JSONL files representing spectrum from silver to prediction
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterator
from collections import defaultdict
import random


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Seg:
    """Time segment with a label."""
    start: float
    end: float
    label: int


@dataclass
class ErrorSegment:
    """Represents a contiguous error in prediction."""
    start: float
    end: float
    ref_label: int          # Correct reference label
    pred_label: int         # Incorrect predicted label
    utt_id: str            # Utterance identifier
    segment_idx: int       # Index in original ref


# ============================================================================
# Segment utilities (adapted from score_jaccard.py)
# ============================================================================

def merge_adjacent(segs: List[Seg]) -> List[Seg]:
    """Merge adjacent segments with the same label."""
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
    """Build reference segments from passthrough data."""
    if len(seg_ts) != len(seg_langs):
        raise ValueError(f"Mismatch: {len(seg_ts)} timestamps vs {len(seg_langs)} langs")
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
    """Build prediction segments from pred list."""
    pred: List[Seg] = []
    for d in pred_list:
        s = float(d["start"])
        e = float(d["end"])
        lab = int(d["label"])
        if e <= s:
            continue
        pred.append(Seg(s, e, lab))
    return merge_adjacent(pred)


# ============================================================================
# Error Detection
# ============================================================================

def get_dominant_label_in_interval(
    hyp: List[Seg],
    start: float,
    end: float,
) -> int:
    """
    Get the most time-represented label in the interval [start, end).
    If multiple labels have equal time, pick the one with smallest index.
    """
    label_time: Dict[int, float] = defaultdict(float)
    
    for seg in hyp:
        if seg.end <= start or seg.start >= end:
            continue
        overlap_start = max(seg.start, start)
        overlap_end = min(seg.end, end)
        if overlap_end > overlap_start:
            label_time[seg.label] += overlap_end - overlap_start
    
    if not label_time:
        return 0  # Fallback (shouldn't happen with well-formed data)
    
    return max(label_time.keys(), key=lambda k: (label_time[k], -k))


def detect_error_segments(
    ref: List[Seg],
    hyp: List[Seg],
    utt_id: str,
) -> List[ErrorSegment]:
    """
    Detect error segments: ref segments where hyp has incorrect labels.
    
    An error is detected if the ref segment is not entirely covered by
    the correct ref_label in hyp.
    """
    errors = []
    
    for ref_idx, ref_seg in enumerate(ref):
        # Get hyp segments overlapping this ref segment
        overlapping_hyp = [h for h in hyp 
                          if h.end > ref_seg.start and h.start < ref_seg.end]
        
        if not overlapping_hyp:
            # No prediction for this ref segment = error
            errors.append(ErrorSegment(
                start=ref_seg.start,
                end=ref_seg.end,
                ref_label=ref_seg.label,
                pred_label=-1,  # No prediction
                utt_id=utt_id,
                segment_idx=ref_idx,
            ))
            continue
        
        # Get hyp labels in this interval
        hyp_labels = set(h.label for h in overlapping_hyp)
        
        # Check if ref_label is present and is the only label
        if ref_seg.label not in hyp_labels or len(hyp_labels) > 1:
            # Error: ref_label not present or mixed with other labels
            pred_label = get_dominant_label_in_interval(overlapping_hyp, ref_seg.start, ref_seg.end)
            errors.append(ErrorSegment(
                start=ref_seg.start,
                end=ref_seg.end,
                ref_label=ref_seg.label,
                pred_label=pred_label,
                utt_id=utt_id,
                segment_idx=ref_idx,
            ))
    
    return errors


# ============================================================================
# Error Correction Scheduling
# ============================================================================

def distribute_corrections_across_buckets(
    all_utterance_errors: Dict[str, List[ErrorSegment]],
    num_buckets: int,
) -> List[Dict[str, List[ErrorSegment]]]:
    """
    Distribute error corrections deterministically across iterations.
    
    For each utterance with N errors:
    - Each iteration corrects ceil(N / num_buckets) errors
    - Errors are sorted by start time for reproducibility
    - Returns (num_buckets - 1) iterations (bucket 0 is silver, bucket N is pred)
    
    Returns:
        List of dicts, each mapping utt_id -> errors_to_correct_in_this_iteration
    """
    corrections_per_iter = []
    
    # Make mutable copy
    remaining_errors = {
        utt_id: sorted(list(errors), key=lambda e: e.start)
        for utt_id, errors in all_utterance_errors.items()
    }
    
    for bucket_idx in range(num_buckets - 1):
        this_iter_corrections = {}
        
        for utt_id, errors in remaining_errors.items():
            if not errors:
                continue
            
            # How many total errors does this utterance have?
            total_errors = len(all_utterance_errors[utt_id])
            errors_to_correct_per_iter = math.ceil(total_errors / num_buckets)
            
            # Take first N from remaining (sorted by start time)
            to_correct = errors[:errors_to_correct_per_iter]
            if to_correct:
                this_iter_corrections[utt_id] = to_correct
            
            # Update remaining
            remaining_errors[utt_id] = errors[errors_to_correct_per_iter:]
        
        corrections_per_iter.append(this_iter_corrections)
    
    return corrections_per_iter


# ============================================================================
# Error Correction Application
# ============================================================================

def apply_error_correction(
    original_pred_segs: List[Seg],
    error: ErrorSegment,
) -> List[Seg]:
    """
    Apply a single error correction to pred segments.
    
    Replace the prediction label in [error.start, error.end) with error.ref_label.
    """
    corrected = []
    
    for seg in original_pred_segs:
        # Segment entirely before error interval
        if seg.end <= error.start + 1e-9:
            corrected.append(seg)
        # Segment entirely after error interval
        elif seg.start >= error.end - 1e-9:
            corrected.append(seg)
        # Segment overlaps error interval
        else:
            # Add part before error interval
            if seg.start < error.start - 1e-9:
                corrected.append(Seg(seg.start, error.start, seg.label))
            
            # Add corrected part (ref_label)
            overlap_start = max(seg.start, error.start)
            overlap_end = min(seg.end, error.end)
            if overlap_end > overlap_start + 1e-9:
                corrected.append(Seg(overlap_start, overlap_end, error.ref_label))
            
            # Add part after error interval
            if seg.end > error.end + 1e-9:
                corrected.append(Seg(error.end, seg.end, seg.label))
    
    return merge_adjacent(sorted(corrected, key=lambda s: s.start))


def apply_multiple_corrections(
    original_pred_segs: List[Seg],
    errors: List[ErrorSegment],
) -> List[Seg]:
    """Apply multiple error corrections sequentially."""
    corrected = list(original_pred_segs)
    for error in sorted(errors, key=lambda e: e.start):
        corrected = apply_error_correction(corrected, error)
    return corrected


# ============================================================================
# I/O
# ============================================================================

def load_vocab_id_token(path: str) -> Dict[str, int]:
    """Load vocab: 'id token' per line -> {token: id}."""
    vocab: Dict[str, int] = {}
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
            vocab[tok] = idx
    return vocab


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """Invert vocab: {id: token}."""
    return {idx: tok for tok, idx in vocab.items()}


def iter_jsonl_inputs(jsonl_path: str) -> Iterator[Tuple[str, dict]]:
    """Yield (utt_id, full_entry) from JSONL."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict) or len(obj) != 1:
                    raise ValueError(f"Expected single-key dict, got {type(obj)}")
                utt_id = next(iter(obj.keys()))
                entry = obj[utt_id]
                yield utt_id, entry
            except Exception as e:
                print(f"ERROR at {jsonl_path}:{ln}: {e}")
                raise


def seg_list_to_dict(segs: List[Seg]) -> List[dict]:
    """Convert Seg objects to dict format for JSON output."""
    return [
        {"start": s.start, "end": s.end, "label": s.label, "score": None}
        for s in segs
    ]


# ============================================================================
# Main Pipeline
# ============================================================================

def generate_error_simulation_points(
    input_jsonl: str,
    vocab_path: str,
    num_buckets: int,
    output_dir: str,
    seed: Optional[int] = 42,
) -> None:
    """
    Main pipeline: generate intermediate diarization files from prediction to silver.
    
    Args:
        input_jsonl: Path to input JSONL with pred and passthrough
        vocab_path: Path to vocab file (id token per line)
        num_buckets: Number of intermediate points to generate
        output_dir: Directory to save output JSONL files
        seed: Random seed for reproducibility (optional)
    """
    if seed is not None:
        random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load vocab
    vocab = load_vocab_id_token(vocab_path)
    inv_vocab = invert_vocab(vocab)
    
    print(f"[*] Loaded vocab with {len(vocab)} languages")
    print(f"[*] Loading input JSONL: {input_jsonl}")
    
    # ====== PHASE 1: Detect all errors ======
    all_utterance_errors: Dict[str, List[ErrorSegment]] = {}
    utterance_data: Dict[str, Tuple[dict, List[Seg], List[Seg]]] = {}  # utt_id -> (entry, ref, pred)
    total_errors = 0
    utts_with_errors = 0
    
    for utt_id, entry in iter_jsonl_inputs(input_jsonl):
        try:
            pred_list = entry.get("pred", [])
            passthrough = entry.get("passthrough", {})
            
            seg_ts = passthrough.get("segment_timestamps", [])
            seg_langs = passthrough.get("segment_langs", [])
            
            # Build segments
            ref = build_ref_segments(seg_ts, seg_langs, vocab)
            hyp = build_pred_segments(pred_list)
            
            # Detect errors
            errors = detect_error_segments(ref, hyp, utt_id)
            
            if errors:
                all_utterance_errors[utt_id] = errors
                utterance_data[utt_id] = (entry, ref, hyp)
                utts_with_errors += 1
                total_errors += len(errors)
            else:
                # No errors, but still store for reference
                utterance_data[utt_id] = (entry, ref, hyp)
        
        except Exception as e:
            print(f"ERROR processing {utt_id}: {e}")
            raise
    
    print(f"[*] Detected errors in {utts_with_errors} utterances, {total_errors} total error segments")
    
    # ====== PHASE 2: Distribute corrections ======
    corrections_per_iter = distribute_corrections_across_buckets(
        all_utterance_errors,
        num_buckets,
    )
    print(f"[*] Distributed into {len(corrections_per_iter)} iterations")
    
    # ====== PHASE 3: Generate buckets ======
    
    # Bucket 0: Silver (all errors corrected)
    print(f"[*] Generating bucket 0 (silver)...")
    bucket_0_path = os.path.join(output_dir, "bucket_000_silver.jsonl")
    with open(bucket_0_path, "w", encoding="utf-8") as f:
        for utt_id, (entry, ref, _pred) in utterance_data.items():
            # Use ref as the "corrected" prediction
            corrected_pred = seg_list_to_dict(ref)
            
            output_entry = {
                utt_id: {
                    "pred": corrected_pred,
                    "passthrough": entry.get("passthrough", {}),
                }
            }
            f.write(json.dumps(output_entry) + "\n")
    print(f"  -> {bucket_0_path}")
    
    # Buckets 1 to N-1: Intermediate states
    cumulative_corrections: Dict[str, List[ErrorSegment]] = defaultdict(list)
    
    for iter_idx, corrections_this_iter in enumerate(corrections_per_iter, start=1):
        bucket_idx = iter_idx
        
        # Accumulate corrections
        for utt_id, errors in corrections_this_iter.items():
            cumulative_corrections[utt_id].extend(errors)
        
        print(f"[*] Generating bucket {bucket_idx} (iteration {iter_idx}/{len(corrections_per_iter)})...")
        
        bucket_path = os.path.join(output_dir, f"bucket_{bucket_idx:03d}.jsonl")
        with open(bucket_path, "w", encoding="utf-8") as f:
            for utt_id, (entry, ref, pred) in utterance_data.items():
                # Apply corrections to pred
                pred_copy = list(pred)
                if utt_id in cumulative_corrections:
                    pred_copy = apply_multiple_corrections(pred_copy, cumulative_corrections[utt_id])
                
                corrected_pred = seg_list_to_dict(pred_copy)
                
                output_entry = {
                    utt_id: {
                        "pred": corrected_pred,
                        "passthrough": entry.get("passthrough", {}),
                    }
                }
                f.write(json.dumps(output_entry) + "\n")
        
        print(f"  -> {bucket_path}")
    
    # Final bucket: Original prediction (no corrections)
    final_bucket_idx = num_buckets
    print(f"[*] Generating bucket {final_bucket_idx} (original prediction)...")
    bucket_final_path = os.path.join(output_dir, f"bucket_{final_bucket_idx:03d}_prediction.jsonl")
    with open(bucket_final_path, "w", encoding="utf-8") as f:
        for utt_id, (entry, _ref, _pred) in utterance_data.items():
            output_entry = {
                utt_id: entry
            }
            f.write(json.dumps(output_entry) + "\n")
    print(f"  -> {bucket_final_path}")
    
    # ====== PHASE 4: Save metadata ======
    metadata = {
        "num_buckets": num_buckets,
        "seed": seed,
        "input_jsonl": input_jsonl,
        "vocab_path": vocab_path,
        "total_utterances": len(utterance_data),
        "utterances_with_errors": utts_with_errors,
        "total_error_segments": total_errors,
        "errors_per_bucket": [
            sum(len(errors) for errors in corrections_this_iter.values())
            for corrections_this_iter in corrections_per_iter
        ],
        "output_files": [
            "bucket_000_silver.jsonl",
        ] + [
            f"bucket_{i:03d}.jsonl" for i in range(1, num_buckets)
        ] + [
            f"bucket_{num_buckets:03d}_prediction.jsonl",
        ],
    }
    
    metadata_path = os.path.join(output_dir, "simulation_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[*] Saved metadata to: {metadata_path}")
    
    print(f"\n[✓] Simulation complete!")
    print(f"    Output directory: {output_dir}")
    print(f"    Bucket files: {num_buckets + 1}")
    print(f"    Silver (0 errors) -> Intermediate -> Prediction (~19% JER)")


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate error simulation points between silver and predicted diarization."
    )
    ap.add_argument("--input_jsonl", required=True, help="Input JSONL with pred and passthrough.")
    ap.add_argument("--vocab", required=True, help="Vocab file (id token per line).")
    ap.add_argument("--num_buckets", type=int, required=True, help="Number of intermediate points (N+1 total files).")
    ap.add_argument("--output_dir", required=True, help="Directory to save bucket JSONL files.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = ap.parse_args()
    
    generate_error_simulation_points(
        input_jsonl=args.input_jsonl,
        vocab_path=args.vocab,
        num_buckets=args.num_buckets,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
