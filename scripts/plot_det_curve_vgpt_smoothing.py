#!/usr/bin/env python3
"""
DET curves for English detection using Score C (log p(target) - max log p(other)),
with a sweep over smoothing methods + window sizes.

Input supports JSON dict mapping utt_id -> {...} or JSONL.
Vocab file lines: "<id> <token>"

Example:
  python det_sweep_smoothing.py \
    --input in.json \
    --vocab vocab.txt \
    --target_lang eng \
    --smooth_methods none,mean,median,ema \
    --smooth_windows_sec 0,0.25,0.5,1.0 \
    --exclude_uncovered \
    --out_plot det_smoothing.png \
    --out_json det_smoothing.json
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
    from scipy.signal import medfilt
except Exception:
    norm = None
    medfilt = None


# -----------------------------
# Loading helpers
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


def iter_utts(input_path: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Try JSON dict
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if "utt_id" in obj and "posteriors" in obj and "passthrough" in obj:
                yield str(obj["utt_id"]), obj
                return
            for uid, u in obj.items():
                if isinstance(u, dict):
                    yield str(uid), u
            return
    except Exception:
        pass

    # JSONL
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
                raise ValueError(f"Unrecognized JSONL line keys={list(o.keys())[:10]}")


# -----------------------------
# Ref segments -> frame labels
# -----------------------------

def label_frames_midpoint(
    frame_times: np.ndarray,
    frame_duration: float,
    seg_times: np.ndarray,
    seg_langs: List[str],
    *,
    uncovered_label: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    T = frame_times.shape[0]
    mids = frame_times + 0.5 * frame_duration

    labels = np.empty(T, dtype=object)
    covered = np.zeros(T, dtype=bool)

    j = 0
    nseg = seg_times.shape[0]
    for i in range(T):
        t = float(mids[i])
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
# Score C + smoothing
# -----------------------------

def score_c_logp_minus_max_other(post: np.ndarray, target_id: int, eps_prob: float) -> np.ndarray:
    """
    post: (T,V) softmax probabilities.
    Returns score C: log p(target) - max_{l!=target} log p(l)
    """
    post = post.astype(np.float64)
    post = np.clip(post, eps_prob, 1.0)
    logpost = np.log(post)

    logp_t = logpost[:, target_id]

    # compute max other without copying the whole matrix:
    # take row-wise max, but with target excluded via a temporary view-like trick:
    # simplest reliable approach: compute max over all then handle cases where target is max
    max_all = np.max(logpost, axis=1)
    argmax_all = np.argmax(logpost, axis=1)

    # if target isn't max, max_other = max_all
    max_other = max_all.copy()

    # if target IS max, we need second max
    idx = np.where(argmax_all == target_id)[0]
    if idx.size > 0:
        # compute second max for those rows
        # mask target to -inf for those rows only
        tmp = logpost[idx, :].copy()
        tmp[:, target_id] = -np.inf
        max_other[idx] = np.max(tmp, axis=1)

    return logp_t - max_other


def smooth_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, kernel, mode="same")


def smooth_median(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    if medfilt is None:
        raise RuntimeError("scipy.signal.medfilt is required for median smoothing (install scipy).")
    # medfilt requires odd kernel size
    if win % 2 == 0:
        win += 1
    return medfilt(x, kernel_size=win)


def smooth_ema(x: np.ndarray, win: int) -> np.ndarray:
    """
    Exponential moving average with alpha chosen so that the effective window ~ win.
    A common heuristic: alpha = 2/(win+1)
    """
    if win <= 1:
        return x
    alpha = 2.0 / (win + 1.0)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, x.shape[0]):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def apply_smoothing(x: np.ndarray, method: str, win_frames: int) -> np.ndarray:
    method = method.lower()
    if method == "none":
        return x
    if method == "mean":
        return smooth_mean(x, win_frames)
    if method == "median":
        return smooth_median(x, win_frames)
    if method == "ema":
        return smooth_ema(x, win_frames)
    raise ValueError(f"Unknown smoothing method: {method}")


# -----------------------------
# DET computation
# -----------------------------

def compute_det_curve(scores: np.ndarray, labels01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    assert scores.ndim == 1 and labels01.ndim == 1 and scores.shape[0] == labels01.shape[0]
    y = labels01.astype(np.int32)

    N_tar = int(y.sum())
    N_non = int((1 - y).sum())
    if N_tar == 0 or N_non == 0:
        raise ValueError(f"Need both target and non-target frames, got N_tar={N_tar}, N_non={N_non}")

    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = y[order]

    tp_cum = np.cumsum(y)
    fp_cum = np.cumsum(1 - y)

    change = np.r_[True, s[1:] != s[:-1]]
    idxs = np.nonzero(change)[0]
    run_ends = np.r_[idxs[1:] - 1, len(s) - 1]

    thresholds = [float("inf")]
    pfa = [0.0]
    pmiss = [1.0]

    for end in run_ends:
        thr = float(s[end])
        TP = int(tp_cum[end])
        FP = int(fp_cum[end])
        thresholds.append(thr)
        pfa.append(FP / N_non)
        pmiss.append(1.0 - (TP / N_tar))

    thresholds.append(float("-inf"))
    pfa.append(1.0)
    pmiss.append(0.0)

    return np.asarray(thresholds), np.asarray(pfa), np.asarray(pmiss), N_tar, N_non


def estimate_eer(thresholds: np.ndarray, pfa: np.ndarray, pmiss: np.ndarray) -> Tuple[float, float]:
    d = pfa - pmiss
    sign = np.sign(d)
    changes = np.where(sign[:-1] * sign[1:] < 0)[0]
    if len(changes) == 0:
        k = int(np.argmin(np.abs(d)))
        return float((pfa[k] + pmiss[k]) / 2.0), float(thresholds[k])

    i = int(changes[0])
    d0, d1 = float(d[i]), float(d[i + 1])
    t0, t1 = float(thresholds[i]), float(thresholds[i + 1])
    if d1 == d0:
        alpha = 0.5
    else:
        alpha = (0.0 - d0) / (d1 - d0)
        alpha = min(1.0, max(0.0, alpha))

    pfa_e = float(pfa[i] + alpha * (pfa[i + 1] - pfa[i]))
    pm_e = float(pmiss[i] + alpha * (pmiss[i + 1] - pmiss[i]))
    eer = (pfa_e + pm_e) / 2.0

    if math.isfinite(t0) and math.isfinite(t1):
        thr = t0 + alpha * (t1 - t0)
    else:
        thr = float(thresholds[i])
    return float(eer), float(thr)


def probit_transform(p: np.ndarray, eps: float) -> np.ndarray:
    if norm is None:
        raise RuntimeError("scipy is required: scipy.stats.norm.ppf")
    p = np.clip(p, eps, 1.0 - eps)
    return norm.ppf(p)


# -----------------------------
# Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--target_lang", default="eng")

    ap.add_argument("--smooth_methods", default="none,mean,median,ema",
                    help="Comma-separated: none,mean,median,ema")
    ap.add_argument("--smooth_windows_sec", default="0,0.25,0.5,1.0",
                    help="Comma-separated window sizes in seconds (include 0 for no smoothing).")

    ap.add_argument("--exclude_uncovered", action="store_true",
                    help="Exclude frames not covered by any reference segment (recommended).")
    ap.add_argument("--treat_uncovered_as_nontarget", action="store_true",
                    help="If not excluding uncovered, treat uncovered as non-target.")

    ap.add_argument("--eps_prob", type=float, default=1e-6)
    ap.add_argument("--eps_rate", type=float, default=1e-6)

    ap.add_argument("--out_plot", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--title", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if norm is None:
        raise RuntimeError("This script requires scipy (scipy.stats.norm.ppf). Please install scipy.")

    tok2id, _ = load_vocab(args.vocab)
    if args.target_lang not in tok2id:
        raise ValueError(f"target_lang '{args.target_lang}' not found in vocab.")
    target_id = tok2id[args.target_lang]

    methods = [m.strip().lower() for m in args.smooth_methods.split(",") if m.strip()]
    windows_sec = [float(x.strip()) for x in args.smooth_windows_sec.split(",") if x.strip()]

    # Store per-setting arrays to concatenate later
    setting_scores: Dict[str, List[np.ndarray]] = {f"{m}:{w:g}": [] for m in methods for w in windows_sec}
    all_labels: List[np.ndarray] = []

    n_utts = 0
    n_frames_total = 0
    n_frames_used = 0
    n_uncovered = 0
    dt_global: Optional[float] = None

    for utt_id, utt in tqdm(iter_utts(args.input)):
        post_info = utt["pred"]["posteriors"]
        ref_info = utt["passthrough"]

        values = np.asarray(post_info["values"], dtype=np.float64)          # (T,V)
        frame_times = np.asarray(post_info["frame_times"], dtype=np.float64)
        dt = float(post_info["frame_duration"])
        dt_global = dt if dt_global is None else dt_global

        seg_ts = np.asarray(ref_info["segment_timestamps"], dtype=np.float64)
        seg_langs = list(ref_info["segment_langs"])

        labels_str, covered = label_frames_midpoint(frame_times, dt, seg_ts, seg_langs, uncovered_label=None)
        y = (labels_str == args.target_lang).astype(np.int32)

        T = values.shape[0]
        n_utts += 1
        n_frames_total += T
        n_uncovered += int((~covered).sum())

        if args.exclude_uncovered:
            mask = covered
        else:
            if args.treat_uncovered_as_nontarget:
                mask = np.ones(T, dtype=bool)
                # uncovered labels already non-target because labels_str == target_lang will be False
            else:
                mask = covered  # safest default

        y_m = y[mask]
        post_m = values[mask, :]
        if y_m.size == 0:
            continue

        # base score C (no smoothing yet)
        s = score_c_logp_minus_max_other(post_m, target_id=target_id, eps_prob=args.eps_prob)

        # apply smoothing per setting, per utterance (never across utterance boundaries)
        for m in methods:
            for wsec in windows_sec:
                key = f"{m}:{wsec:g}"
                if wsec <= 0 or m == "none":
                    s_sm = s
                else:
                    win_frames = max(1, int(round(wsec / dt)))
                    s_sm = apply_smoothing(s, m, win_frames)
                setting_scores[key].append(s_sm)

        all_labels.append(y_m)
        n_frames_used += int(y_m.shape[0])

    if n_frames_used == 0:
        raise RuntimeError("No frames selected for evaluation (masking removed everything).")

    labels_all = np.concatenate(all_labels).astype(np.int32)

    # Plot
    import matplotlib.pyplot as plt
    plt.figure()
    title = args.title or f"DET sweep (Score C), target={args.target_lang}, dt={dt_global}"
    plt.title(title)
    plt.xlabel("False Alarm probability (probit)")
    plt.ylabel("Miss probability (probit)")

    out: Dict[str, Any] = {
        "input": os.path.abspath(args.input),
        "vocab": os.path.abspath(args.vocab),
        "target_lang": args.target_lang,
        "target_id": target_id,
        "score": "C_log_p_minus_max_other",
        "num_utts": n_utts,
        "num_frames_total": n_frames_total,
        "num_frames_used": n_frames_used,
        "num_frames_uncovered": n_uncovered,
        "exclude_uncovered": bool(args.exclude_uncovered),
        "treat_uncovered_as_nontarget": bool(args.treat_uncovered_as_nontarget),
        "eps_prob": float(args.eps_prob),
        "eps_rate": float(args.eps_rate),
        "dt": float(dt_global) if dt_global is not None else None,
        "settings": {},
    }

    # Sort legend in a pleasant order: by method then window
    def setting_sort_key(k: str):
        m, w = k.split(":")
        return (methods.index(m), float(w))

    for key in sorted(setting_scores.keys(), key=setting_sort_key):
        scores_all = np.concatenate(setting_scores[key]).astype(np.float64)

        thr, pfa, pmiss, N_tar, N_non = compute_det_curve(scores_all, labels_all)
        eer, eer_thr = estimate_eer(thr, pfa, pmiss)

        x = probit_transform(pfa, args.eps_rate)
        y = probit_transform(pmiss, args.eps_rate)

        plt.plot(x, y, label=f"{key} (EER={eer:.4f})")

        out["settings"][key] = {
            "method": key.split(":")[0],
            "window_sec": float(key.split(":")[1]),
            "thresholds": thr.tolist(),
            "pfa": pfa.tolist(),
            "pmiss": pmiss.tolist(),
            "x_probit": x.tolist(),
            "y_probit": y.tolist(),
            "eer": float(eer),
            "eer_threshold": float(eer_thr),
            "n_target": int(N_tar),
            "n_nontarget": int(N_non),
        }

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=200)
    plt.close()

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote plot: {args.out_plot}")
    print(f"Wrote DET JSON: {args.out_json}")
    print(f"Frames used: {n_frames_used}  (target={int(labels_all.sum())}, non-target={int((1-labels_all).sum())})")


if __name__ == "__main__":
    main()