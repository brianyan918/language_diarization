#!/usr/bin/env python3
"""
Compute language priors (duration-weighted) from JSONL manifests containing segment language
and segment timing.

You said:
- Each JSONL line has a "segments" list.
- Each segment has: "audio_start_sec", "audio_end_sec", "duration", "lang".
- Total utterance duration should be determined by max(end) - min(start).
- Priors should be computed as total duration per language / total duration across languages.
- For train: aggregate across multiple input jsonl files.
- For test: one input jsonl at a time.

We also accept a vocab mapping file ("<id> <token>") so we can:
- report priors by token, and
- optionally report priors by id too.

Important choices:
- By default, we compute priors from *segments* durations (sum of segment durations).
  This matches "time labeled as language" and is usually what you want.
- We also compute a diagnostic "coverage" ratio comparing sum(segment durations) vs utt_span
  (max_end - min_start) to catch gaps/overlaps.

Outputs:
- JSON with:
  - priors_token: {lang_token: prior_prob}
  - priors_id:    {id_str: prior_prob}
  - total_duration_sec, per_lang_duration_sec
  - diagnostics: coverage, gaps, overlaps indicators

Usage examples:

# Train priors (aggregate multiple files)
python compute_lang_priors.py \
  --vocab vocab.txt \
  --inputs train1.jsonl train2.jsonl train3.jsonl \
  --out train_prior.json

# Test priors (one file)
python compute_lang_priors.py \
  --vocab vocab.txt \
  --inputs test.jsonl \
  --out test_prior.json

Notes:
- If a segment's lang is not in vocab, it's counted under "__OOV__" (and reported).
- If you want to drop OOV segments entirely, pass --drop_oov.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

import math


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
# Core computation
# -----------------------------

@dataclass
class UttDiagnostics:
    utt_id: str
    utt_span_sec: float
    seg_sum_sec: float
    coverage_ratio: float
    num_segments: int
    has_gaps: bool
    has_overlaps: bool


def _segments_gap_overlap(seg_times: List[Tuple[float, float]], eps: float = 1e-6) -> Tuple[bool, bool]:
    """
    Determine whether segments contain gaps or overlaps when ordered by start time.
    """
    if not seg_times:
        return False, False
    seg_times_sorted = sorted(seg_times, key=lambda x: (x[0], x[1]))
    has_gaps = False
    has_overlaps = False
    prev_end = seg_times_sorted[0][1]
    for (s, e) in seg_times_sorted[1:]:
        if s > prev_end + eps:
            has_gaps = True
        if s < prev_end - eps:
            has_overlaps = True
        prev_end = max(prev_end, e)
    return has_gaps, has_overlaps


def update_counts_from_utt(
    obj: Dict[str, Any],
    tok2id: Dict[str, int],
    lang_dur: Dict[str, float],
    *,
    drop_oov: bool,
    oov_key: str = "__OOV__",
) -> Optional[UttDiagnostics]:
    """
    Update duration totals from one JSONL record. Returns diagnostics for this utt (or None if skipped).
    """
    utt_id = str(obj.get("id", obj.get("utt_id", "")))

    segments = obj.get("segments", None)
    if segments is None:
        # also allow nested passthrough-style formats (just in case)
        passthrough = obj.get("passthrough", {})
        segments = passthrough.get("segments", None)
    if not segments:
        return None

    starts = []
    ends = []
    seg_sum = 0.0
    seg_times: List[Tuple[float, float]] = []

    for seg in segments:
        try:
            s = float(seg["audio_start_sec"])
            e = float(seg["audio_end_sec"])
        except Exception:
            # fallback: infer from duration if needed
            s = float(seg.get("audio_start_sec", 0.0))
            d = float(seg.get("duration", 0.0))
            e = s + d

        d = float(seg.get("duration", max(0.0, e - s)))
        if d < 0:
            continue

        lang = str(seg.get("lang", "")).strip()
        if not lang:
            continue

        # track for diagnostics
        starts.append(s)
        ends.append(e)
        seg_sum += d
        seg_times.append((s, e))

        # vocab handling
        if lang in tok2id:
            lang_dur[lang] += d
        else:
            if drop_oov:
                continue
            lang_dur[oov_key] += d

    if not starts or not ends:
        return None

    utt_span = max(ends) - min(starts)
    # avoid divide by zero
    coverage = seg_sum / utt_span if utt_span > 1e-9 else 0.0

    has_gaps, has_overlaps = _segments_gap_overlap(seg_times)

    return UttDiagnostics(
        utt_id=utt_id,
        utt_span_sec=float(utt_span),
        seg_sum_sec=float(seg_sum),
        coverage_ratio=float(coverage),
        num_segments=len(seg_times),
        has_gaps=bool(has_gaps),
        has_overlaps=bool(has_overlaps),
    )


def compute_priors_from_jsonl(
    input_paths: List[str],
    tok2id: Dict[str, int],
    *,
    drop_oov: bool,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    Returns:
      priors_token, dur_token, diagnostics_summary
    """
    lang_dur: Dict[str, float] = defaultdict(float)
    diags: List[UttDiagnostics] = []

    total_lines = 0
    total_utts_used = 0

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                obj = json.loads(line)

                diag = update_counts_from_utt(obj, tok2id, lang_dur, drop_oov=drop_oov)
                if diag is not None:
                    total_utts_used += 1
                    diags.append(diag)

    total_dur = float(sum(lang_dur.values()))
    if total_dur <= 0:
        raise RuntimeError("Total labeled duration is 0. Check your inputs / segments fields.")

    priors_token: Dict[str, float] = {k: float(v / total_dur) for k, v in lang_dur.items()}
    dur_token: Dict[str, float] = dict(lang_dur)

    # Diagnostics summary
    coverages = [d.coverage_ratio for d in diags if d.utt_span_sec > 1e-9]
    gaps = sum(1 for d in diags if d.has_gaps)
    overlaps = sum(1 for d in diags if d.has_overlaps)

    diagnostics_summary: Dict[str, Any] = {
        "total_input_lines": total_lines,
        "total_utts_used": total_utts_used,
        "total_labeled_duration_sec": total_dur,
        "coverage_ratio_mean": float(sum(coverages) / len(coverages)) if coverages else None,
        "coverage_ratio_p50": float(sorted(coverages)[len(coverages)//2]) if coverages else None,
        "coverage_ratio_p10": float(sorted(coverages)[max(0, int(0.10*len(coverages))-1)]) if coverages else None,
        "coverage_ratio_p90": float(sorted(coverages)[min(len(coverages)-1, int(0.90*len(coverages)))]) if coverages else None,
        "num_utts_with_gaps": int(gaps),
        "num_utts_with_overlaps": int(overlaps),
    }

    return priors_token, dur_token, diagnostics_summary


def priors_token_to_id(
    priors_token: Dict[str, float],
    tok2id: Dict[str, int],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for tok, p in priors_token.items():
        if tok in tok2id:
            out[str(tok2id[tok])] = float(p)
    return out


# -----------------------------
# Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True, help="Vocab file '<id> <token>'")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more JSONL files to aggregate.")
    ap.add_argument("--out", required=True, help="Output JSON path for priors.")
    ap.add_argument("--drop_oov", action="store_true", help="Drop segments whose lang isn't in vocab.")
    ap.add_argument("--topk", type=int, default=50, help="Include top-k languages by duration in summary.")
    return ap.parse_args()


def main():
    args = parse_args()
    tok2id, id2tok = load_vocab(args.vocab)

    priors_token, dur_token, diag = compute_priors_from_jsonl(
        input_paths=args.inputs,
        tok2id=tok2id,
        drop_oov=args.drop_oov,
    )

    # create a stable sorted view for summary
    items = sorted(dur_token.items(), key=lambda kv: kv[1], reverse=True)
    topk = items[: max(0, args.topk)]

    priors_id = priors_token_to_id(priors_token, tok2id)

    out = {
        "vocab": os.path.abspath(args.vocab) if "os" in globals() else args.vocab,  # safe if os missing
        "inputs": args.inputs,
        "drop_oov": bool(args.drop_oov),
        "diagnostics": diag,
        "total_labeled_duration_sec": float(sum(dur_token.values())),
        "per_lang_duration_sec": dur_token,      # full map (can be big)
        "priors_token": priors_token,            # full map
        "priors_id": priors_id,                  # subset map for vocab entries
        "summary_topk_by_duration": [
            {"lang": k, "duration_sec": float(v), "prior": float(priors_token[k])}
            for k, v in topk
        ],
    }

    # tiny fix: os was used above in one place; make sure it's imported
    # (kept defensive because some folks delete imports)
    import os as _os
    out["vocab"] = _os.path.abspath(args.vocab)
    out["inputs"] = [ _os.path.abspath(p) for p in args.inputs ]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # print a concise summary
    print(f"Wrote priors: {args.out}")
    print(f"Total labeled duration (sec): {out['total_labeled_duration_sec']:.2f}")
    if "__OOV__" in dur_token:
        print(f"OOV duration (sec): {dur_token['__OOV__']:.2f}  (drop_oov={args.drop_oov})")
    print("Top languages:")
    for row in out["summary_topk_by_duration"][: min(10, len(out["summary_topk_by_duration"]))]:
        print(f"  {row['lang']:>8s}  dur={row['duration_sec']:.2f}s  prior={row['prior']:.6f}")


if __name__ == "__main__":
    main()