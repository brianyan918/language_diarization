#!/usr/bin/env python3
"""
Plot DET curves for English (or any target lang) detection from frame-level softmax posteriors,
with reference language segments given as time intervals.

Input format (per-utt object), e.g.
{
  "utt_id": {
    "posteriors": {
      "values": [[...], [...], ...],       # frames x vocab_size (softmax probs)
      "frame_times": [0.0, 0.04, 0.08, ...],
      "frame_duration": 0.04
    },
    "passthrough": {
      "segment_timestamps": [[0, 2.5], [2.5, 5.0]],
      "segment_langs": ["ara", "eng"]
    }
  },
  ...
}

Also supports JSONL where each line is {"utt_id": {...}} (one utt per line) or
{"utt_id": "...", "posteriors": {...}, "passthrough": {...}}.

Outputs:
- A PNG (or PDF) plot of DET curves for multiple scoring methods
- A JSON file with DET points (thresholds, Pfa, Pmiss, probit-x/y, EER, etc.)

Usage:
  python det_from_posteriors.py \
    --input in.json \
    --vocab vocab.txt \
    --target_lang eng \
    --out_plot det.png \
    --out_json det_data.json

Notes:
- DET is computed over frames (time-weighted automatically if constant frame_duration).
- Reference segments are converted to frame-level labels using the frame midpoint rule.
- Frames not covered by any reference segment can be excluded (default) or treated as non-target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Tuple, List, Optional
from tqdm import tqdm

import numpy as np

try:
    from scipy.stats import norm
except Exception as e:
    norm = None


# -----------------------------
# Utilities: loading
# -----------------------------

def load_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Vocab file lines like: "2 eng" (id token).
    Returns token->id and id->token dicts.
    """
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


def iter_utts(input_path: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yield (utt_id, utt_obj) where utt_obj contains "posteriors" and "passthrough".
    Supports:
      - JSON dict mapping utt_id -> {...}
      - JSONL where each line is {"utt_id": {...}} OR {"utt_id": "...", "posteriors":..., "passthrough":...}
    """
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Try JSON dict first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            # If it's the main mapping (utt_id -> utt_obj)
            # If it looks like a single utt with "utt_id" field, handle below
            if "posteriors" in obj and "passthrough" in obj and "utt_id" in obj:
                uid = str(obj["utt_id"])
                yield uid, obj
                return

            # mapping case
            for uid, u in obj.items():
                if not isinstance(u, dict):
                    continue
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
            if isinstance(o, dict) and "utt_id" in o and "posteriors" in o and "passthrough" in o:
                yield str(o["utt_id"]), o
            elif isinstance(o, dict) and len(o) == 1:
                uid, u = next(iter(o.items()))
                yield str(uid), u
            else:
                raise ValueError(f"Unrecognized JSONL line structure: keys={list(o.keys())[:10]}")


# -----------------------------
# Reference -> frame labels
# -----------------------------

def label_frames_midpoint(
    frame_times: np.ndarray,
    frame_duration: float,
    seg_times: np.ndarray,
    seg_langs: List[str],
    *,
    uncovered_label: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert segment-level reference to frame-level labels via midpoint rule.

    Returns:
      labels_str: (T,) dtype=object, language label per frame or uncovered_label
      covered: (T,) bool, whether the midpoint falls inside any segment
    """
    T = frame_times.shape[0]
    mids = frame_times + 0.5 * frame_duration

    labels = np.empty(T, dtype=object)
    covered = np.zeros(T, dtype=bool)

    # segments assumed sorted, non-overlapping
    j = 0
    nseg = seg_times.shape[0]
    for i in range(T):
        t = float(mids[i])
        # advance segment pointer while t beyond current segment end
        while j < nseg and t >= float(seg_times[j, 1]):
            j += 1
        if j < nseg and float(seg_times[j, 0]) <= t < float(seg_times[j, 1]):
            labels[i] = seg_langs[j]
            covered[i] = True
        else:
            labels[i] = uncovered_label
            covered[i] = False

    return labels, covered


# -----------------------------
# DET computation
# -----------------------------

@dataclass
class DetCurve:
    thresholds: List[float]
    pfa: List[float]
    pmiss: List[float]
    x_probit: List[float]
    y_probit: List[float]
    eer: float
    eer_threshold: float
    n_target: int
    n_nontarget: int


def _safe_ppf(p: np.ndarray, eps: float) -> np.ndarray:
    if norm is None:
        raise RuntimeError("scipy is required for DET probit transform (scipy.stats.norm.ppf).")
    p = np.clip(p, eps, 1.0 - eps)
    return norm.ppf(p)


def compute_det_curve(scores: np.ndarray, labels01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Efficient DET point generation from scores and binary labels (1=target, 0=nontarget).
    Returns thresholds, Pfa, Pmiss arrays including endpoints.
    """
    assert scores.ndim == 1 and labels01.ndim == 1 and scores.shape[0] == labels01.shape[0]
    y = labels01.astype(np.int32)

    N_tar = int(y.sum())
    N_non = int((1 - y).sum())
    if N_tar == 0 or N_non == 0:
        raise ValueError(f"Need both target and non-target frames, got N_tar={N_tar}, N_non={N_non}")

    # Sort by score descending; as threshold decreases, more positives predicted
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = y[order]

    # Cumulative counts when predicting positive for top-k frames
    tp_cum = np.cumsum(y)                      # TP at each k
    fp_cum = np.cumsum(1 - y)                  # FP at each k

    # Record points only at score-change boundaries (unique thresholds)
    # threshold = s[idx] means predict positive for scores >= threshold
    change = np.r_[True, s[1:] != s[:-1]]
    idxs = np.nonzero(change)[0]

    thresholds = []
    pfa = []
    pmiss = []

    # Endpoint: threshold = +inf => predict none => Pfa=0, Pmiss=1
    thresholds.append(float("inf"))
    pfa.append(0.0)
    pmiss.append(1.0)

    # For each unique score threshold, include all items up to the last occurrence of that score.
    # We use the last index of each run:
    run_ends = np.r_[idxs[1:] - 1, len(s) - 1]
    for end in run_ends:
        thr = float(s[end])
        TP = int(tp_cum[end])
        FP = int(fp_cum[end])
        Pfa = FP / N_non
        Pmiss = 1.0 - (TP / N_tar)
        thresholds.append(thr)
        pfa.append(Pfa)
        pmiss.append(Pmiss)

    # Endpoint: threshold = -inf => predict all => Pfa=1, Pmiss=0
    thresholds.append(float("-inf"))
    pfa.append(1.0)
    pmiss.append(0.0)

    return np.asarray(thresholds, dtype=float), np.asarray(pfa, dtype=float), np.asarray(pmiss, dtype=float)


def estimate_eer(thresholds: np.ndarray, pfa: np.ndarray, pmiss: np.ndarray) -> Tuple[float, float]:
    """
    Estimate EER by finding crossing of pfa and pmiss and linearly interpolating.
    Returns (eer, eer_threshold).
    """
    d = pfa - pmiss
    # Find a sign change closest to zero; if none, pick minimal abs difference.
    sign = np.sign(d)
    # indices where sign changes between i and i+1
    changes = np.where(sign[:-1] * sign[1:] < 0)[0]
    if len(changes) == 0:
        k = int(np.argmin(np.abs(d)))
        eer = float((pfa[k] + pmiss[k]) / 2.0)
        return eer, float(thresholds[k])

    i = int(changes[0])
    # linear interpolation in d between i and i+1
    d0, d1 = float(d[i]), float(d[i + 1])
    t0, t1 = float(thresholds[i]), float(thresholds[i + 1])
    # Avoid division by zero
    if d1 == d0:
        alpha = 0.5
    else:
        alpha = (0.0 - d0) / (d1 - d0)
        alpha = min(1.0, max(0.0, alpha))
    # interpolate EER between the two points
    pfa_e = float(pfa[i] + alpha * (pfa[i + 1] - pfa[i]))
    pm_e = float(pmiss[i] + alpha * (pmiss[i + 1] - pmiss[i]))
    eer = (pfa_e + pm_e) / 2.0
    thr = t0 + alpha * (t1 - t0) if (math.isfinite(t0) and math.isfinite(t1)) else float(thresholds[i])
    return float(eer), float(thr)


# -----------------------------
# Scoring methods
# -----------------------------

def compute_scores_from_posteriors(
    post: np.ndarray,
    target_id: int,
    *,
    eps_prob: float,
) -> Dict[str, np.ndarray]:
    """
    post: (T, V) probabilities summing to 1.
    Returns dict of method_name -> scores (T,)
    """
    if post.ndim != 2:
        raise ValueError(f"posteriors must be 2D (T,V), got shape {post.shape}")
    T, V = post.shape
    if not (0 <= target_id < V):
        raise ValueError(f"target_id={target_id} out of range for V={V}")

    p = post[:, target_id].astype(np.float64)
    p = np.clip(p, eps_prob, 1.0 - eps_prob)
    logp = np.log(p)

    # A) log p(eng)
    score_a = logp

    # B) logit: log(p) - log(1-p)
    score_b = logp - np.log1p(-p)  # log(1-p) stable

    # C) log p(eng) - max_{l != eng} log p(l)
    # clamp whole posterior for stability before log
    post_clamped = np.clip(post.astype(np.float64), eps_prob, 1.0)
    logpost = np.log(post_clamped)
    # max over non-target
    # (avoid copying huge arrays by temporarily setting target col to -inf)
    other = logpost.copy()
    other[:, target_id] = -np.inf
    max_other = np.max(other, axis=1)
    score_c = logp - max_other

    return {
        "A_log_p": score_a,
        "B_logit": score_b,
        "C_log_p_minus_max_other": score_c,
    }


# -----------------------------
# Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON or JSONL with posteriors + passthrough reference.")
    ap.add_argument("--vocab", required=True, help="Vocab mapping file: '<id> <token>' per line.")
    ap.add_argument("--target_lang", default="eng", help="Target language token in vocab (default: eng).")

    ap.add_argument("--out_plot", required=True, help="Output plot path (e.g., det.png).")
    ap.add_argument("--out_json", required=True, help="Output DET data JSON path.")

    ap.add_argument("--exclude_uncovered", action="store_true",
                    help="Exclude frames not covered by any reference segment (recommended).")
    ap.add_argument("--treat_uncovered_as_nontarget", action="store_true",
                    help="If set, uncovered frames become non-target (ignored unless --exclude_uncovered is false).")

    ap.add_argument("--smooth_sec", type=float, default=0.0,
                    help="Optional moving-average smoothing window in seconds (0 disables).")
    ap.add_argument("--eps_prob", type=float, default=1e-6,
                    help="Clamp epsilon for probabilities before log/logit.")
    ap.add_argument("--eps_rate", type=float, default=1e-6,
                    help="Clamp epsilon for Pfa/Pmiss before probit.")

    ap.add_argument("--title", default=None, help="Optional plot title.")
    return ap.parse_args()


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    # simple centered moving average
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, kernel, mode="same")


def main():
    args = parse_args()

    if norm is None:
        raise RuntimeError("scipy is required. Please install scipy to use this script.")

    tok2id, _ = load_vocab(args.vocab)
    if args.target_lang not in tok2id:
        raise ValueError(f"target_lang '{args.target_lang}' not found in vocab file.")
    target_id = tok2id[args.target_lang]

    all_scores: Dict[str, List[np.ndarray]] = {
        "A_log_p": [],
        "B_logit": [],
        "C_log_p_minus_max_other": [],
    }
    all_labels: List[np.ndarray] = []

    n_utts = 0
    n_frames_total = 0
    n_frames_used = 0
    n_uncovered = 0

    for utt_id, utt in tqdm(iter_utts(args.input)):
        if "pred" not in utt or "passthrough" not in utt:
            raise ValueError(f"utt '{utt_id}' missing 'posteriors' or 'passthrough' keys")

        post_info = utt["pred"]["posteriors"]
        ref_info = utt["passthrough"]

        values = np.asarray(post_info["values"], dtype=np.float64)          # (T,V)
        frame_times = np.asarray(post_info["frame_times"], dtype=np.float64)  # (T,)
        dt = float(post_info["frame_duration"])

        seg_ts = np.asarray(ref_info["segment_timestamps"], dtype=np.float64)  # (S,2)
        seg_langs = list(ref_info["segment_langs"])

        if seg_ts.ndim != 2 or seg_ts.shape[1] != 2:
            raise ValueError(f"utt '{utt_id}': segment_timestamps must be shape (S,2), got {seg_ts.shape}")
        if len(seg_langs) != seg_ts.shape[0]:
            raise ValueError(f"utt '{utt_id}': segment_langs length != segment_timestamps rows")

        labels_str, covered = label_frames_midpoint(
            frame_times=frame_times,
            frame_duration=dt,
            seg_times=seg_ts,
            seg_langs=seg_langs,
            uncovered_label=None,
        )

        # build y (binary), and mask
        y = (labels_str == args.target_lang).astype(np.int32)

        T = values.shape[0]
        n_utts += 1
        n_frames_total += T
        n_uncovered += int((~covered).sum())

        if args.exclude_uncovered:
            mask = covered
        else:
            mask = np.ones(T, dtype=bool)
            if (not args.treat_uncovered_as_nontarget) and (not args.exclude_uncovered):
                # If not excluding and not treating as non-target, then we must drop uncovered anyway.
                # This is the safest default to avoid injecting unknown regions.
                mask = covered

        y_m = y[mask]
        post_m = values[mask, :]

        if y_m.size == 0:
            continue

        # optional smoothing on p_eng (before score), per-utt
        # (smoothing across utt boundary would be wrong)
        if args.smooth_sec and args.smooth_sec > 0:
            win = max(1, int(round(args.smooth_sec / dt)))
            p_eng = post_m[:, target_id].astype(np.float64)
            p_eng = np.clip(p_eng, args.eps_prob, 1.0 - args.eps_prob)
            p_eng_s = moving_average(p_eng, win)
            # re-normalize p_eng into (0,1) safely
            p_eng_s = np.clip(p_eng_s, args.eps_prob, 1.0 - args.eps_prob)
            # replace just target column for scoring A/B; C needs full posteriors, so we smooth score later for C.
            post_for_ab = post_m.copy()
            post_for_ab[:, target_id] = p_eng_s

            # compute A/B on smoothed target posterior; compute C on original post (or you can smooth score_c below)
            scores_ab = compute_scores_from_posteriors(post_for_ab, target_id, eps_prob=args.eps_prob)
            scores_c = compute_scores_from_posteriors(post_m, target_id, eps_prob=args.eps_prob)

            scores = {
                "A_log_p": scores_ab["A_log_p"],
                "B_logit": scores_ab["B_logit"],
                "C_log_p_minus_max_other": scores_c["C_log_p_minus_max_other"],
            }
        else:
            scores = compute_scores_from_posteriors(post_m, target_id, eps_prob=args.eps_prob)

        for k in all_scores.keys():
            all_scores[k].append(scores[k])
        all_labels.append(y_m)

        n_frames_used += int(y_m.shape[0])

    if n_frames_used == 0:
        raise RuntimeError("No frames selected for evaluation. Check masking / uncovered handling.")

    labels_all = np.concatenate(all_labels, axis=0).astype(np.int32)

    out: Dict[str, Any] = {
        "input": os.path.abspath(args.input),
        "vocab": os.path.abspath(args.vocab),
        "target_lang": args.target_lang,
        "target_id": target_id,
        "num_utts": n_utts,
        "num_frames_total": n_frames_total,
        "num_frames_used": n_frames_used,
        "num_frames_uncovered": n_uncovered,
        "exclude_uncovered": bool(args.exclude_uncovered),
        "treat_uncovered_as_nontarget": bool(args.treat_uncovered_as_nontarget),
        "smooth_sec": float(args.smooth_sec),
        "eps_prob": float(args.eps_prob),
        "eps_rate": float(args.eps_rate),
        "methods": {},
    }

    # Plot
    import matplotlib.pyplot as plt

    plt.figure()
    title = args.title or f"DET curves: target={args.target_lang} (frames, dt={None})"
    plt.title(title)
    plt.xlabel("False Alarm probability (probit scale)")
    plt.ylabel("Miss probability (probit scale)")

    # Common legend order
    method_order = [
        ("A_log_p", "A: log p(target)"),
        ("B_logit", "B: logit(p(target))"),
        ("C_log_p_minus_max_other", "C: log p(target) - max log p(other)"),
    ]

    for key, label in method_order:
        scores_all = np.concatenate(all_scores[key], axis=0).astype(np.float64)

        thr, pfa, pmiss = compute_det_curve(scores_all, labels_all)
        eer, eer_thr = estimate_eer(thr, pfa, pmiss)

        # probit transform
        # clip rates; using args.eps_rate (or you can choose 0.5/N style)
        x = _safe_ppf(pfa, args.eps_rate)
        y = _safe_ppf(pmiss, args.eps_rate)

        plt.plot(x, y, label=f"{label} (EER={eer:.4f})")

        out["methods"][key] = {
            "label": label,
            "thresholds": thr.tolist(),
            "pfa": pfa.tolist(),
            "pmiss": pmiss.tolist(),
            "x_probit": x.tolist(),
            "y_probit": y.tolist(),
            "eer": float(eer),
            "eer_threshold": float(eer_thr),
            "n_target": int(labels_all.sum()),
            "n_nontarget": int((1 - labels_all).sum()),
        }

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=200)
    plt.close()

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote plot: {args.out_plot}")
    print(f"Wrote DET data: {args.out_json}")
    print(f"Frames used: {n_frames_used}  (target={int(labels_all.sum())}, non-target={int((1-labels_all).sum())})")


if __name__ == "__main__":
    main()