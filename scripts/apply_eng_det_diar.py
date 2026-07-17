#!/usr/bin/env python3
"""
Apply a fixed language-detection diarization rule to an input JSON (dict) or JSONL,
producing segment-level predictions.

You specified the best method:
- score method C:  score[t] = log p(target) - max_{l != target} log p(l)
- smoothing: median filter with window = 0.25 seconds  (per utterance)
- threshold: -0.96870195
- no priors

Interpretation for diarization output:
- We produce a binary per-frame decision: "target_lang" vs "not_target_lang"
  using score_C_smoothed >= threshold.
- Then we convert to segments by grouping consecutive frames with the same decision.
- For frames predicted as target_lang => label = vocab_id(target_lang)
- For frames predicted as not_target_lang => label = vocab_id(other_label)

IMPORTANT: Your request says each segment must have a single "label" which is
"the index of the language". With only an English detection threshold, we must
choose what label to emit for "not English" segments.

This script supports two modes for "non-target" frames:
  1) --non_target_mode best_alt
     label = argmax_{l != target} posterior[l] per frame (then segmented)
     This yields full multi-class diarization (but the boundary decision is still driven by English detection).
  2) --non_target_mode fixed --non_target_lang <token>
     label all non-target frames as a fixed language token (less useful).

Default is best_alt, which is almost always what you want.

Input format (your new one):
{
  "utt_id": {
    "pred": {
      "posteriors": {
        "values": [[...], ...],       # frames x vocab_size softmax probs
        "frame_times": [0.0, 0.04, ...],
        "frame_duration": 0.04
      },
      "alignments": {...}   # ignored
    },
    "passthrough": {...}    # passed through to output
  },
  ...
}

Output format:
{
  "0": {
    "pred": [{"start": 0.0, "end": 1.0, "label": 3}, ...],
    "passthrough": { ... }   # whatever passthrough info you want to keep
  },
  "1": {...}
}

We index output keys as strings "0", "1", ... in the order processed.

Usage:
  python apply_eng_det_diar.py \
    --input in.json \
    --vocab vocab.txt \
    --target_lang eng \
    --smooth_sec 0.25 \
    --threshold -0.96870195 \
    --out out.json

Optional:
  --non_target_mode best_alt
  --min_seg_dur 0.0
  --merge_gap 0.0
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, Any, Iterable, Tuple, List, Optional
from tqdm import tqdm

import numpy as np

try:
    from scipy.signal import medfilt
except Exception:
    medfilt = None


# -----------------------------
# Vocab
# -----------------------------

def load_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    tok2id: Dict[str, int] = {}
    id2tok: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            tok = parts[1]
            tok2id[tok] = idx
            id2tok[idx] = tok
    return tok2id, id2tok


# -----------------------------
# Input iterator
# -----------------------------

def iter_utts_any(input_path: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Supports:
      - JSON dict mapping utt_id -> obj
      - JSONL where each line is {"utt_id": {...}} or {"utt_id": "...", ...}
    """
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Try JSON dict first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            # mapping case
            for uid, u in obj.items():
                if isinstance(u, dict):
                    yield str(uid), u
            return
    except Exception:
        pass

    # Fallback: JSONL
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if isinstance(o, dict) and len(o) == 1:
                uid, u = next(iter(o.items()))
                yield str(uid), u
            elif isinstance(o, dict) and "utt_id" in o:
                yield str(o["utt_id"]), o
            else:
                raise ValueError(f"Unrecognized JSONL line keys={list(o.keys())[:10]}")


# -----------------------------
# Score C + smoothing + segmentation
# -----------------------------

def score_c_logp_minus_max_other(post: np.ndarray, target_id: int, eps_prob: float) -> np.ndarray:
    """
    post: (T,V) softmax probabilities.
    score C: log p(target) - max_{l != target} log p(l)
    """
    post = post.astype(np.float64)
    post = np.clip(post, eps_prob, 1.0)
    logpost = np.log(post)
    lt = logpost[:, target_id]

    # compute max other
    # do it without huge overhead; easiest safe method:
    other = logpost.copy()
    other[:, target_id] = -np.inf
    max_other = np.max(other, axis=1)
    return lt - max_other


def smooth_median(x: np.ndarray, win_frames: int) -> np.ndarray:
    if win_frames <= 1:
        return x
    if medfilt is None:
        raise RuntimeError("scipy is required for median smoothing (scipy.signal.medfilt).")
    # medfilt requires odd kernel
    if win_frames % 2 == 0:
        win_frames += 1
    return medfilt(x, kernel_size=win_frames)


def frames_to_segments(
    frame_times: np.ndarray,
    frame_duration: float,
    labels: np.ndarray,  # int labels per frame
    *,
    min_seg_dur: float = 0.0,
    merge_gap: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Convert per-frame labels to contiguous segments with start/end in seconds.

    - Each frame i covers [t_i, t_i+dt)
    - Consecutive frames with same label are merged.
    - Optionally drop segments shorter than min_seg_dur (seconds).
    - Optionally merge segments of the same label separated by <= merge_gap seconds.
    """
    T = labels.shape[0]
    if T == 0:
        return []

    # initial segmentation by consecutive labels
    segs: List[Tuple[float, float, int]] = []
    cur_lab = int(labels[0])
    cur_start = float(frame_times[0])
    cur_end = float(frame_times[0] + frame_duration)

    for i in range(1, T):
        t = float(frame_times[i])
        e = float(t + frame_duration)
        lab = int(labels[i])
        if lab == cur_lab and abs(t - cur_end) <= 1e-6:
            cur_end = e
        else:
            segs.append((cur_start, cur_end, cur_lab))
            cur_lab = lab
            cur_start = t
            cur_end = e
    segs.append((cur_start, cur_end, cur_lab))

    # merge small gaps between same-label segments
    if merge_gap > 0.0 and len(segs) >= 2:
        merged: List[Tuple[float, float, int]] = [segs[0]]
        for (s, e, lab) in segs[1:]:
            ps, pe, plab = merged[-1]
            if lab == plab and (s - pe) <= merge_gap + 1e-9:
                merged[-1] = (ps, max(pe, e), plab)
            else:
                merged.append((s, e, lab))
        segs = merged

    # apply min duration
    out: List[Dict[str, Any]] = []
    for (s, e, lab) in segs:
        if (e - s) + 1e-12 < min_seg_dur:
            continue
        out.append({"start": float(s), "end": float(e), "label": int(lab)})
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON or JSONL (your new format).")
    ap.add_argument("--vocab", required=True, help="Vocab mapping '<id> <token>'")
    ap.add_argument("--out", required=True, help="Output JSON path.")

    ap.add_argument("--target_lang", default="eng")
    ap.add_argument("--smooth_sec", type=float, default=0.25, help="Median smoothing window in seconds.")
    ap.add_argument("--threshold", type=float, required=True, help="Threshold on smoothed score C.")
    ap.add_argument("--eps_prob", type=float, default=1e-6, help="Clamp epsilon for posteriors before log.")

    ap.add_argument("--non_target_mode", choices=["best_alt", "fixed"], default="best_alt",
                    help="How to label non-target frames.")
    ap.add_argument("--non_target_lang", default=None,
                    help="Required if --non_target_mode fixed: token to use for non-target.")

    ap.add_argument("--min_seg_dur", type=float, default=0.0, help="Drop segments shorter than this (sec).")
    ap.add_argument("--merge_gap", type=float, default=0.0, help="Merge same-label segments separated by <= gap (sec).")

    ap.add_argument("--keep_passthrough_keys", default=None,
                    help="Comma-separated passthrough keys to keep (default: keep full passthrough).")
    return ap.parse_args()


def main():
    args = parse_args()
    tok2id, id2tok = load_vocab(args.vocab)

    if args.target_lang not in tok2id:
        raise ValueError(f"target_lang '{args.target_lang}' not found in vocab.")
    target_id = tok2id[args.target_lang]

    if args.non_target_mode == "fixed":
        if not args.non_target_lang:
            raise ValueError("--non_target_lang is required when --non_target_mode fixed")
        if args.non_target_lang not in tok2id:
            raise ValueError(f"non_target_lang '{args.non_target_lang}' not found in vocab.")
        fixed_non_target_id = tok2id[args.non_target_lang]
    else:
        fixed_non_target_id = None

    keep_keys = None
    if args.keep_passthrough_keys:
        keep_keys = [k.strip() for k in args.keep_passthrough_keys.split(",") if k.strip()]

    out_idx = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for utt_id, utt in tqdm(iter_utts_any(args.input)):
            # Locate posteriors under utt["pred"]["posteriors"]
            pred = utt.get("pred", {})
            post_info = pred.get("posteriors", None)
            if post_info is None:
                raise ValueError(f"utt '{utt_id}' missing pred.posteriors")

            values = np.asarray(post_info["values"], dtype=np.float64)  # (T,V)
            frame_times = np.asarray(post_info["frame_times"], dtype=np.float64)
            dt = float(post_info["frame_duration"])

            if values.ndim != 2:
                raise ValueError(f"utt '{utt_id}': posteriors.values must be 2D (T,V)")
            if frame_times.ndim != 1 or frame_times.shape[0] != values.shape[0]:
                raise ValueError(f"utt '{utt_id}': frame_times length mismatch with values")

            # Compute score C and smooth
            s = score_c_logp_minus_max_other(values, target_id=target_id, eps_prob=args.eps_prob)

            win_frames = max(1, int(round(args.smooth_sec / dt))) if args.smooth_sec > 0 else 1
            s_sm = smooth_median(s, win_frames)

            # Decide English vs not-English
            is_target = (s_sm >= float(args.threshold))

            # Produce per-frame language label IDs
            if args.non_target_mode == "fixed":
                labels = np.where(is_target, target_id, fixed_non_target_id).astype(np.int32)
            else:
                # best_alt: for non-target frames, label with argmax over non-target languages
                # (for target frames, label = target_id)
                # Note: This uses raw posteriors argmax; you can also use argmax over logpost if you prefer.
                best = np.argmax(values, axis=1).astype(np.int32)

                # Ensure best_alt excludes target when is_target==False:
                # If best==target but is_target==False (can happen if threshold is strict),
                # choose second-best.
                need_second = (~is_target) & (best == target_id)
                if np.any(need_second):
                    # find second best for those frames
                    v = values[need_second].copy()
                    v[:, target_id] = -np.inf
                    best2 = np.argmax(v, axis=1).astype(np.int32)
                    best[need_second] = best2

                labels = np.where(is_target, target_id, best).astype(np.int32)

            # Convert frame labels to segments
            pred_segments = frames_to_segments(
                frame_times=frame_times,
                frame_duration=dt,
                labels=labels,
                min_seg_dur=float(args.min_seg_dur),
                merge_gap=float(args.merge_gap),
            )

            # passthrough: keep entire passthrough (or subset)
            passthrough = utt.get("passthrough", {})
            if keep_keys is not None:
                passthrough = {k: passthrough.get(k) for k in keep_keys}

            out_obj = {
                utt_id: {"pred": pred_segments, "passthrough": {**passthrough}}
            }
            out_idx += 1

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


    print(f"Wrote: {args.out}  (utts={out_idx})")


if __name__ == "__main__":
    main()