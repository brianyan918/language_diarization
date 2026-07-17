#!/usr/bin/env python3
"""
Analyze LDER components for language diarization:
  (A) out-of-utterance confusions: predicted lang not present in utterance
  (B) in-utterance confusions: predicted lang present in utterance but wrong segment

Input:
- A glob of prediction files. Each file may be:
  1) JSON dict: {example_key: example_obj, ...}
  2) JSONL: each line is either:
        a) {example_key: example_obj}   (your shown format)
        b) example_obj                  (less common; supported)

Each example object looks like:
{
  "pred": [{"start":..., "end":..., "label": int, "score": ...}, ...],
  "passthrough": {
      "utt_id": ...,
      "segment_timestamps": [[s,e], ...],
      "segment_langs": ["ara","eng",...],
      ...
  }
}

- A vocab file mapping label IDs to language codes. Supported formats:
  1) One language code per line => line index is label id (0-based)
  2) JSON dict: {"0": "ara", "1":"eng", ...} or {0:"ara", ...}
  3) TSV/space: "<id>\t<lang>" or "<id> <lang>" per line

Output:
- Aggregate LDER and component rates
- By language pair/set (sorted by hours desc): same metrics + top error directions

Alignment:
- Errors are computed by time overlap between reference segments and predictions.
- Any ref-covered time with no prediction coverage counts as out-of-utterance under "<NO_PRED>".
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


# -----------------------------
# Vocab loading
# -----------------------------

def load_vocab(path: str) -> Dict[int, str]:
    # Try JSON dict
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("JSON vocab must be a dict mapping id->lang")
        return {int(k): str(v) for k, v in obj.items()}

    vocab: Dict[int, str] = {}
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # If every line contains whitespace, assume "<id> <lang>"
    has_explicit_ids = len(lines) > 0 and all(any(ch.isspace() for ch in ln) for ln in lines)

    if has_explicit_ids:
        for ln in lines:
            parts = ln.split()
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            lang = parts[1]
            vocab[idx] = lang
    else:
        for idx, ln in enumerate(lines):
            vocab[idx] = ln

    if not vocab:
        raise ValueError(f"Failed to parse vocab from {path}")
    return vocab


# -----------------------------
# Read prediction files: JSON or JSONL
# -----------------------------

def iter_examples_from_file(fp: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Supports:
      - JSON dict file: {key: example_obj, ...}
      - JSONL:
          a) each line: {key: example_obj}
          b) each line: example_obj
    """
    with open(fp, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        # Try JSON dict first if it looks like JSON
        if first_char == "{":
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    for ex_key, ex_obj in data.items():
                        if isinstance(ex_obj, dict):
                            yield str(ex_key), ex_obj
                        else:
                            # If ex_obj isn't dict, still yield and let analyzer error explicitly
                            yield str(ex_key), ex_obj  # type: ignore[misc]
                    return
            except json.JSONDecodeError:
                # fall back to JSONL
                f.seek(0)

        # JSONL fallback
        f.seek(0)
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Common JSONL pattern: {"112": {...}}
            if isinstance(obj, dict) and len(obj) == 1 and isinstance(next(iter(obj.values())), dict):
                ex_key, ex_obj = next(iter(obj.items()))
                yield str(ex_key), ex_obj
            else:
                # Less common: line is already example object
                yield f"{os.path.basename(fp)}:{i}", obj


# -----------------------------
# Interval overlap alignment
# -----------------------------

@dataclass
class Seg:
    start: float
    end: float
    label: Optional[str]  # language code or None

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


def to_ref_segments(segment_timestamps: List[List[float]], segment_langs: List[str]) -> List[Seg]:
    if len(segment_timestamps) != len(segment_langs):
        raise ValueError("segment_timestamps and segment_langs length mismatch")
    out: List[Seg] = []
    for (s, e), lang in zip(segment_timestamps, segment_langs):
        out.append(Seg(float(s), float(e), str(lang)))
    out.sort(key=lambda x: (x.start, x.end))
    return out


def to_pred_segments(pred_list: List[Dict[str, Any]], vocab: Dict[int, str]) -> List[Seg]:
    out: List[Seg] = []
    for p in pred_list:
        s = float(p["start"])
        e = float(p["end"])
        lab_id = p.get("label", None)
        if lab_id is None:
            lab = None
        else:
            lab = vocab.get(int(lab_id), f"<UNK_{lab_id}>")
        out.append(Seg(s, e, lab))
    out.sort(key=lambda x: (x.start, x.end))
    return out


def fill_pred_gaps_over_ref(ref: List[Seg], pred: List[Seg]) -> List[Seg]:
    """
    Ensure we have a predicted label covering all time that is covered by reference segments.
    Gaps are labeled as "<NO_PRED>".
    """
    if not ref:
        return pred

    filled: List[Seg] = []
    pred_i = 0

    for r in ref:
        t = r.start
        while t < r.end:
            while pred_i < len(pred) and pred[pred_i].end <= t:
                pred_i += 1

            if pred_i >= len(pred) or pred[pred_i].start >= r.end:
                filled.append(Seg(t, r.end, "<NO_PRED>"))
                t = r.end
                break

            p = pred[pred_i]
            if p.start > t:
                gap_end = min(p.start, r.end)
                filled.append(Seg(t, gap_end, "<NO_PRED>"))
                t = gap_end
                continue

            chunk_end = min(p.end, r.end)
            filled.append(Seg(t, chunk_end, p.label))
            t = chunk_end

    # merge adjacent same-label segments
    merged: List[Seg] = []
    for s in filled:
        if s.dur <= 0:
            continue
        if merged and merged[-1].label == s.label and abs(merged[-1].end - s.start) < 1e-6:
            merged[-1] = Seg(merged[-1].start, s.end, s.label)
        else:
            merged.append(s)
    return merged


def overlap_duration(a: Seg, b: Seg) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


# -----------------------------
# Metrics aggregation
# -----------------------------

@dataclass
class Stats:
    total_ref_dur: float = 0.0
    err_in_utt: float = 0.0
    err_out_utt: float = 0.0

    in_utt_confusions: Counter = None  # (ref, pred) -> seconds
    out_utt_preds: Counter = None      # pred -> seconds

    def __post_init__(self):
        if self.in_utt_confusions is None:
            self.in_utt_confusions = Counter()
        if self.out_utt_preds is None:
            self.out_utt_preds = Counter()

    def add(self, other: "Stats"):
        self.total_ref_dur += other.total_ref_dur
        self.err_in_utt += other.err_in_utt
        self.err_out_utt += other.err_out_utt
        self.in_utt_confusions.update(other.in_utt_confusions)
        self.out_utt_preds.update(other.out_utt_preds)

    @property
    def lder(self) -> float:
        return 0.0 if self.total_ref_dur <= 0 else (self.err_in_utt + self.err_out_utt) / self.total_ref_dur

    @property
    def rate_in_utt(self) -> float:
        return 0.0 if self.total_ref_dur <= 0 else self.err_in_utt / self.total_ref_dur

    @property
    def rate_out_utt(self) -> float:
        return 0.0 if self.total_ref_dur <= 0 else self.err_out_utt / self.total_ref_dur


def langset_key(ref_langs: List[str]) -> str:
    uniq = sorted(set(ref_langs))
    return "-".join(uniq) if uniq else "<NONE>"


def analyze_one_example(ex_obj: Dict[str, Any], vocab: Dict[int, str]) -> Tuple[str, Stats]:
    pred_list = ex_obj.get("pred", [])
    passthrough = ex_obj.get("passthrough", {})

    seg_ts = passthrough["segment_timestamps"]
    seg_langs = passthrough["segment_langs"]

    ref = to_ref_segments(seg_ts, seg_langs)
    pred = to_pred_segments(pred_list, vocab)
    pred_cov = fill_pred_gaps_over_ref(ref, pred)

    utt_langset = set(seg_langs)
    key = langset_key(seg_langs)

    st = Stats()

    i = 0
    j = 0
    while i < len(ref) and j < len(pred_cov):
        r = ref[i]
        p = pred_cov[j]

        ov = overlap_duration(r, p)
        if ov > 0:
            st.total_ref_dur += ov
            ref_lang = r.label
            pred_lang = p.label

            if pred_lang != ref_lang:
                if pred_lang in utt_langset:
                    st.err_in_utt += ov
                    st.in_utt_confusions[(ref_lang, pred_lang)] += ov
                else:
                    st.err_out_utt += ov
                    st.out_utt_preds[pred_lang] += ov

        if r.end <= p.end:
            i += 1
        else:
            j += 1

    return key, st


# -----------------------------
# Printing
# -----------------------------

def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def print_stats(title: str, st: Stats, topk: int = 10):
    print(f"\n== {title} ==")
    print(f"Total ref duration: {st.total_ref_dur/3600.0:.2f} hours")
    print(f"LDER:              {fmt_pct(st.lder)}")
    print(f"  in-utt:          {fmt_pct(st.rate_in_utt)}")
    print(f"  out-of-utt:      {fmt_pct(st.rate_out_utt)}")

    if st.in_utt_confusions and len(st.in_utt_confusions) > 0:
        print(f"\nTop in-utt confusions (ref -> pred) by seconds (top {topk}):")
        for (r, p), d in st.in_utt_confusions.most_common(topk):
            print(f"  {r:>6} -> {p:<10} : {d:.2f}s")

    if st.out_utt_preds and len(st.out_utt_preds) > 0:
        print(f"\nTop out-of-utt predicted labels by seconds (top {topk}):")
        for p, d in st.out_utt_preds.most_common(topk):
            print(f"  {str(p):<12} : {d:.2f}s")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-glob", required=True, help='Glob for prediction files, e.g. "runs/preds/*"')
    ap.add_argument("--vocab", required=True, help="Vocab file mapping label id -> language code")
    ap.add_argument("--topk", type=int, default=10, help="Top-K confusions to print")

    ap.add_argument("--min-hours", type=float, default=0.0,
                    help="Only print per-pair rows with >= this many hours of ref duration")
    ap.add_argument("--max-pairs", type=int, default=2000,
                    help="Max language sets to print (sorted by hours desc)")
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)

    files = sorted(glob.glob(args.pred_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.pred_glob}")

    agg = Stats()
    by_pair: Dict[str, Stats] = defaultdict(Stats)

    total_examples = 0
    bad_examples = 0

    for fp in tqdm(files, desc="Reading files"):
        for ex_key, ex_obj in iter_examples_from_file(fp):
            total_examples += 1
            try:
                pair_key, st = analyze_one_example(ex_obj, vocab)
                agg.add(st)
                by_pair[pair_key].add(st)
            except Exception as e:
                bad_examples += 1
                print(f"[WARN] Failed example {ex_key} in {fp}: {e}")

    print("\n============================")
    print("LDER component breakdown")
    print("============================")
    print(f"Files:           {len(files)}")
    print(f"Examples:        {total_examples}")
    print(f"Failed examples: {bad_examples}")

    print_stats("Aggregate", agg, topk=args.topk)

    rows = sorted(by_pair.items(), key=lambda kv: kv[1].total_ref_dur, reverse=True)

    print("\n============================")
    print("By language pair/set (sorted by hours desc)")
    print("============================")

    printed = 0
    for k, st in rows:
        hours = st.total_ref_dur / 3600.0
        if hours < args.min_hours:
            continue

        print(f"\n-- {k} --  ({hours:.2f} hours)")
        print(f"  LDER {fmt_pct(st.lder)} | in-utt {fmt_pct(st.rate_in_utt)} | out-of-utt {fmt_pct(st.rate_out_utt)}")

        if st.in_utt_confusions and len(st.in_utt_confusions) > 0:
            (r, p), d = st.in_utt_confusions.most_common(1)[0]
            print(f"  top in-utt: {r}->{p} ({d:.1f}s)")
        if st.out_utt_preds and len(st.out_utt_preds) > 0:
            p0, d0 = st.out_utt_preds.most_common(1)[0]
            print(f"  top out-of-utt: {p0} ({d0:.1f}s)")

        printed += 1
        if printed >= args.max_pairs:
            break


if __name__ == "__main__":
    main()
