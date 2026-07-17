#!/usr/bin/env python3
"""
Plot DET curves for a target language (default: eng) from frame-level softmax posteriors,
comparing:
  - 3 scoring methods (A/B/C)
  - 3 prior modes (none / subtract_train / subtract_train_add_test)
=> total 9 curves

Input (eval) format (JSON dict of utt_id -> {...} OR JSONL):
{
  "utt_id": {
    "posteriors": {
      "values": [[...], ...],       # frames x vocab_size, softmax probs sum to 1
      "frame_times": [0.0, 0.04, ...],
      "frame_duration": 0.04
    },
    "passthrough": {
      "segment_timestamps": [[0,2.5],[2.5,5.0]],
      "segment_langs": ["ara","eng"]
    }
  },
  ...
}

Vocab file:
  <id> <token>
Example:
  2 eng
  3 ara

Priors inputs:
- JSON mapping token->prior probability (recommended), e.g.
  {"eng": 0.12, "ara": 0.05, ...}
(IDs ok too: {"2": 0.12, "3": 0.05, ...}, but tokens preferred)

This script assumes you will provide train_prior and test_prior JSONs.
(Next step: we can write a script to compute these priors from segment timestamps.)

Outputs:
- Plot of 9 DET curves
- JSON containing DET points and EER for each (score_method, prior_mode) pair

Usage:
  python det_9modes_priors.py \
    --eval eval.json \
    --vocab vocab.txt \
    --train_prior train_prior.json \
    --test_prior test_prior.json \
    --target_lang eng \
    --exclude_uncovered \
    --out_plot det_9modes.png \
    --out_json det_9modes.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Any, Iterable, Tuple, List, Optional
from tqdm import tqdm

import numpy as np

try:
    from scipy.stats import norm
except Exception:
    norm = None


# -----------------------------
# IO helpers
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
            if "utt_id" in obj and "pred" in obj and "passthrough" in obj:
                yield str(obj["utt_id"]), obj
                return
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
            if isinstance(o, dict) and "utt_id" in o and "pred" in o and "passthrough" in o:
                yield str(o["utt_id"]), o
            elif isinstance(o, dict) and len(o) == 1:
                uid, u = next(iter(o.items()))
                yield str(uid), u
            else:
                raise ValueError(f"Unrecognized JSONL line structure: keys={list(o.keys())[:10]}")


def load_priors_json(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Priors file must be a JSON object/dict, got {type(obj)}")
    pri: Dict[str, float] = {}
    for k, v in obj["priors_token"].items():
        pri[str(k)] = float(v)
    return pri


def priors_to_vector(
    priors_map: Dict[str, float],
    tok2id: Dict[str, int],
    id2tok: Dict[int, str],
    *,
    eps_prior: float,
) -> np.ndarray:
    """
    Convert priors map into a dense vector indexed by vocab id (max_id+1 length).
    Accepts keys as tokens ("eng") or ids ("2").
    Missing entries get eps_prior before renormalization.
    """
    max_id = max(id2tok.keys())
    p = np.full(max_id + 1, eps_prior, dtype=np.float64)

    for k, v in priors_map.items():
        if k in tok2id:
            idx = tok2id[k]
            p[idx] = float(v)
        else:
            # maybe id string
            try:
                idx = int(k)
                if idx in id2tok:
                    p[idx] = float(v)
            except Exception:
                pass

    # Renormalize over all ids we know
    known_ids = sorted(id2tok.keys())
    s = float(np.sum(p[known_ids]))
    if s <= 0:
        raise ValueError("Sum of prior probabilities is <= 0 after filling; check priors input.")
    p[known_ids] /= s
    return p


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
    Midpoint rule: label each frame by the GT segment containing (t + dt/2).
    Returns (labels_str[T], covered[T]).
    """
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
# Prior correction on logits
# -----------------------------

def apply_prior_mode_to_logpost(
    logpost: np.ndarray,              # (T,V) log q(l|x)
    train_prior_vec: np.ndarray,      # (V,)
    test_prior_vec: np.ndarray,       # (V,)
    prior_mode: str,
    *,
    eps_prior: float,
) -> np.ndarray:
    """
    Adjust log-posteriors by adding a per-class bias b_l:

      none:                       b_l = 0
      subtract_train:             b_l = -log P_train(l)
      subtract_train_add_test:    b_l = -log P_train(l) + log P_test(l)

    Returns adjusted log-scores z_l(x) = log q(l|x) + b_l
    (We do NOT renormalize with logsumexp; for A/B/C, only relative values matter.)
    """
    prior_mode = prior_mode.lower()
    # clamp priors
    Pt = np.clip(train_prior_vec, eps_prior, 1.0)
    Pq = np.clip(test_prior_vec, eps_prior, 1.0)

    if prior_mode == "none":
        return logpost
    elif prior_mode == "subtract_train":
        b = -np.log(Pt)
        return logpost + b[None, :]
    elif prior_mode == "subtract_train_add_test":
        b = -np.log(Pt) + np.log(Pq)
        return logpost + b[None, :]
    else:
        raise ValueError(f"Unknown prior_mode: {prior_mode}")


# -----------------------------
# Scoring methods from adjusted log-scores
# -----------------------------

def compute_scores_abc_from_adjusted_log(
    adj_log: np.ndarray,     # (T,V) = log q + bias
    target_id: int,
) -> Dict[str, np.ndarray]:
    """
    Given adjusted per-class log-scores, compute:
      A: score = adj_log[target]
      B: score = adj_log[target] - logsumexp(adj_log[others])  (a logit-like LLR)
      C: score = adj_log[target] - max(adj_log[others])

    Note: For B, this is a more principled "one-vs-rest" LLR-style score in log-space
          and is compatible with prior biases without requiring re-normalizing probabilities.
    """
    T, V = adj_log.shape
    lt = adj_log[:, target_id]

    # C: target - max other
    other_max = np.max(np.where(
        np.arange(V)[None, :] == target_id,
        -np.inf,
        adj_log
    ), axis=1)
    score_c = lt - other_max

    # A: just target score
    score_a = lt

    # B: target - logsumexp(other)
    # stable logsumexp over others:
    other = adj_log.copy()
    other[:, target_id] = -np.inf
    m = np.max(other, axis=1)
    lse_other = m + np.log(np.sum(np.exp(other - m[:, None]), axis=1))
    score_b = lt - lse_other

    return {
        "A_log_p": score_a,
        "B_one_vs_rest_lse": score_b,
        "C_minus_max_other": score_c,
    }


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
    ap.add_argument("--eval", required=True, help="Eval JSON/JSONL with posteriors + passthrough reference.")
    ap.add_argument("--vocab", required=True, help="Vocab mapping file '<id> <token>'")
    ap.add_argument("--train_prior", required=True, help="Train prior JSON (token->prob or id->prob).")
    ap.add_argument("--test_prior", required=True, help="Test prior JSON (token->prob or id->prob).")
    ap.add_argument("--target_lang", default="eng")

    ap.add_argument("--exclude_uncovered", action="store_true",
                    help="Exclude frames not covered by any reference segment (recommended).")
    ap.add_argument("--treat_uncovered_as_nontarget", action="store_true",
                    help="If not excluding uncovered, treat uncovered as non-target.")

    ap.add_argument("--eps_prob", type=float, default=1e-6,
                    help="Clamp epsilon for softmax probabilities before log.")
    ap.add_argument("--eps_prior", type=float, default=1e-9,
                    help="Clamp epsilon for priors before log.")
    ap.add_argument("--eps_rate", type=float, default=1e-6,
                    help="Clamp epsilon for Pfa/Pmiss before probit.")

    ap.add_argument("--out_plot", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--title", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if norm is None:
        raise RuntimeError("This script requires scipy (scipy.stats.norm.ppf). Please install scipy.")

    tok2id, id2tok = load_vocab(args.vocab)
    if args.target_lang not in tok2id:
        raise ValueError(f"target_lang '{args.target_lang}' not found in vocab.")
    target_id = tok2id[args.target_lang]

    train_prior_map = load_priors_json(args.train_prior)
    test_prior_map = load_priors_json(args.test_prior)

    train_prior_vec = priors_to_vector(train_prior_map, tok2id, id2tok, eps_prior=args.eps_prior)
    test_prior_vec = priors_to_vector(test_prior_map, tok2id, id2tok, eps_prior=args.eps_prior)

    prior_modes = ["none", "subtract_train", "subtract_train_add_test"]
    score_methods = ["A_log_p", "B_one_vs_rest_lse", "C_minus_max_other"]

    # collect per (score_method, prior_mode)
    buckets: Dict[Tuple[str, str], List[np.ndarray]] = {
        (sm, pm): [] for sm in score_methods for pm in prior_modes
    }
    all_labels: List[np.ndarray] = []

    n_utts = 0
    n_frames_total = 0
    n_frames_used = 0
    n_uncovered = 0
    dt_global: Optional[float] = None

    for utt_id, utt in tqdm(iter_utts(args.eval)):
        if "pred" not in utt or "passthrough" not in utt:
            raise ValueError(f"utt '{utt_id}' missing 'pred' or 'passthrough' keys")

        post_info = utt["pred"]["posteriors"]
        ref_info = utt["passthrough"]

        values = np.asarray(post_info["values"], dtype=np.float64)  # (T,V) probs
        frame_times = np.asarray(post_info["frame_times"], dtype=np.float64)
        dt = float(post_info["frame_duration"])
        dt_global = dt if dt_global is None else dt_global

        seg_ts = np.asarray(ref_info["segment_timestamps"], dtype=np.float64)  # (S,2)
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
            else:
                mask = covered  # safest default

        y_m = y[mask]
        post_m = values[mask, :]
        if y_m.size == 0:
            continue

        # log-softmax probs -> logpost
        post_m = np.clip(post_m, args.eps_prob, 1.0)
        logpost = np.log(post_m)

        # compute all prior modes, then scores A/B/C for each
        for pm in prior_modes:
            adj = apply_prior_mode_to_logpost(
                logpost, train_prior_vec, test_prior_vec, pm, eps_prior=args.eps_prior
            )
            scores = compute_scores_abc_from_adjusted_log(adj, target_id)

            for sm in score_methods:
                buckets[(sm, pm)].append(scores[sm])

        all_labels.append(y_m)
        n_frames_used += int(y_m.shape[0])

    if n_frames_used == 0:
        raise RuntimeError("No frames selected for evaluation. Check masking / uncovered handling.")

    labels_all = np.concatenate(all_labels).astype(np.int32)

    # Plot
    import matplotlib.pyplot as plt
    plt.figure()
    title = args.title or f"DET (target={args.target_lang}), 3 scores × 3 prior modes, dt={dt_global}"
    plt.title(title)
    plt.xlabel("False Alarm probability (probit)")
    plt.ylabel("Miss probability (probit)")

    out: Dict[str, Any] = {
        "eval": os.path.abspath(args.eval),
        "vocab": os.path.abspath(args.vocab),
        "train_prior": os.path.abspath(args.train_prior),
        "test_prior": os.path.abspath(args.test_prior),
        "target_lang": args.target_lang,
        "target_id": target_id,
        "prior_modes": prior_modes,
        "score_methods": score_methods,
        "num_utts": n_utts,
        "num_frames_total": n_frames_total,
        "num_frames_used": n_frames_used,
        "num_frames_uncovered": n_uncovered,
        "exclude_uncovered": bool(args.exclude_uncovered),
        "treat_uncovered_as_nontarget": bool(args.treat_uncovered_as_nontarget),
        "dt": float(dt_global) if dt_global is not None else None,
        "eps_prob": float(args.eps_prob),
        "eps_prior": float(args.eps_prior),
        "eps_rate": float(args.eps_rate),
        "curves": {},  # key = "<score_method>|<prior_mode>"
    }

    # nicer ordering in legend: group by score method, then prior mode
    def curve_key(sm: str, pm: str) -> str:
        return f"{sm}|{pm}"

    for sm in score_methods:
        for pm in prior_modes:
            scores_all = np.concatenate(buckets[(sm, pm)]).astype(np.float64)

            thr, pfa, pmiss, N_tar, N_non = compute_det_curve(scores_all, labels_all)
            eer, eer_thr = estimate_eer(thr, pfa, pmiss)

            x = probit_transform(pfa, args.eps_rate)
            y = probit_transform(pmiss, args.eps_rate)

            label = f"{sm} + {pm} (EER={eer:.4f})"
            plt.plot(x, y, label=label)

            out["curves"][curve_key(sm, pm)] = {
                "score_method": sm,
                "prior_mode": pm,
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
    plt.legend(fontsize=8, ncols=1)
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