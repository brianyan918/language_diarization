#!/usr/bin/env python3
"""
score_segments_cer.py

Score Character Error Rate (CER) for segments from segment_transcriptions.

Input JSONL format (from transcribe_segments.py):
{
  "id": "...",
  "language": "ara-eng",
  "segment_transcriptions": [
    {
      "index": 0,
      "audio_start_sec": 0.0,
      "audio_end_sec": 2.8,
      "duration": 2.8,
      "lang": "ara",
      "ref_text": "reference text",
      "whisper_text": "hypothesis text",
      "whisper_segments": [...]  # optional
    },
    ...
  ],
  ...
}

Output:
- Per-segment CER scores
- Per-language CER aggregation
- Top confusions (SUB/INS/DEL)
- Worst examples per language
- JSON summary with confusion matrix

Usage:
  python score_segments_cer.py -i input.jsonl \
    --ignore_ws --remove_punct --lower \
    --top_confusions 50 --per_lang_top_confusions 15 \
    --out_json summary.json
"""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Tuple, List, Optional

WS_RE = re.compile(r"\s+")
EPS_SYM = "<eps>"

# 2-letter to 3-letter language code mapping
LANG_2_TO_3 = {
    'ar': 'ara', 'en': 'eng', 'fr': 'fra', 'es': 'spa', 'de': 'deu', 'it': 'ita', 'pt': 'por', 'ru': 'rus',
    'ja': 'jpn', 'zh': 'zho', 'hi': 'hin', 'bn': 'ben', 'ta': 'tam', 'te': 'tel', 'tr': 'tur', 'pl': 'pol',
    'nl': 'nld', 'sv': 'swe', 'no': 'nor', 'da': 'dan', 'fi': 'fin', 'he': 'heb', 'th': 'tha', 'ko': 'kor',
    'vi': 'vie', 'id': 'ind', 'fil': 'fil', 'my': 'mya', 'km': 'khm', 'lo': 'lao', 'ca': 'cat', 'eu': 'eus',
    'ga': 'gle', 'gl': 'glg', 'el': 'ell', 'hu': 'hun', 'ro': 'ron', 'sk': 'slk', 'sl': 'slv', 'cs': 'ces',
    'hr': 'hrv', 'sr': 'srp', 'mk': 'mkd', 'bg': 'bul', 'uk': 'ukr', 'ka': 'kat', 'fa': 'fas', 'ur': 'urd',
    'ps': 'pus', 'ku': 'kur', 'am': 'amh', 'om': 'orm', 'sw': 'swa', 'so': 'som',
}

LANG_3_TO_2 = {v: k for k, v in LANG_2_TO_3.items()}


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def strip_unicode_punct(s: str) -> str:
    """Remove Unicode punctuation characters."""
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def normalize_text(text: str, ignore_ws: bool, remove_punct: bool, lower: bool) -> str:
    """Normalize text for scoring."""
    if remove_punct:
        text = strip_unicode_punct(text)
    if ignore_ws:
        text = WS_RE.sub("", text)
    if lower:
        text = text.lower()
    return text


def levenshtein_counts_and_confusions(
    ref: str, hyp: str
) -> Tuple[int, int, int, int, Dict[Tuple[str, str], int]]:
    """
    Full DP Levenshtein with backtrace and confusion tracking.
    
    Returns:
      edits, ins, del, sub, confusions
    
    confusions includes:
      - substitution: (ref_char, hyp_char)
      - deletion:     (ref_char, <eps>)
      - insertion:    (<eps>, hyp_char)
    """
    n = len(ref)
    m = len(hyp)
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    
    for i in range(1, n + 1):
        rc = ref[i - 1]
        for j in range(1, m + 1):
            hc = hyp[j - 1]
            cost_sub = 0 if rc == hc else 1
            v_del = dp[i - 1][j] + 1
            v_ins = dp[i][j - 1] + 1
            v_sub = dp[i - 1][j - 1] + cost_sub
            
            best = v_sub
            best_op = "M" if cost_sub == 0 else "S"
            if v_del < best:
                best = v_del
                best_op = "D"
            if v_ins < best:
                best = v_ins
                best_op = "I"
            
            dp[i][j] = best
            op[i][j] = best_op
    
    i, j = n, m
    ins = dele = sub = 0
    conf = defaultdict(int)
    
    while i > 0 or j > 0:
        cur_op = op[i][j]
        if cur_op == "M":
            i -= 1
            j -= 1
        elif cur_op == "S":
            sub += 1
            conf[(ref[i - 1], hyp[j - 1])] += 1
            i -= 1
            j -= 1
        elif cur_op == "D":
            dele += 1
            conf[(ref[i - 1], EPS_SYM)] += 1
            i -= 1
        elif cur_op == "I":
            ins += 1
            conf[(EPS_SYM, hyp[j - 1])] += 1
            j -= 1
        else:
            if i > 0 and j > 0:
                if ref[i - 1] == hyp[j - 1]:
                    i -= 1
                    j -= 1
                else:
                    sub += 1
                    conf[(ref[i - 1], hyp[j - 1])] += 1
                    i -= 1
                    j -= 1
            elif i > 0:
                dele += 1
                conf[(ref[i - 1], EPS_SYM)] += 1
                i -= 1
            else:
                ins += 1
                conf[(EPS_SYM, hyp[j - 1])] += 1
                j -= 1
    
    edits = dp[n][m]
    return edits, ins, dele, sub, dict(conf)


def format_char(c: str) -> str:
    """Format character for display in confusion matrix."""
    if c == EPS_SYM:
        return EPS_SYM
    if c == " ":
        return "<space>"
    if c == "\t":
        return "<tab>"
    if c == "\n":
        return "<nl>"
    if c == "\r":
        return "<cr>"
    return c


def confusion_type(a: str, b: str) -> str:
    """Classify confusion type: SUB, DEL, INS, or M (match)."""
    if a == EPS_SYM and b != EPS_SYM:
        return "INS"
    if a != EPS_SYM and b == EPS_SYM:
        return "DEL"
    if a != EPS_SYM and b != EPS_SYM:
        return "SUB" if a != b else "M"
    return "?"


def merge_confusions(dst: Dict[Tuple[str, str], int], src: Dict[Tuple[str, str], int]) -> None:
    """Merge confusion counts from src into dst."""
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + int(v)


def top_k_confusions(conf: Dict[Tuple[str, str], int], k: int, only_type: Optional[str] = None):
    """Get top-K confusions, optionally filtered by type (SUB/INS/DEL)."""
    items = []
    for (a, b), cnt in conf.items():
        typ = confusion_type(a, b)
        if only_type is not None and typ != only_type:
            continue
        items.append((cnt, a, b, typ))
    items.sort(reverse=True, key=lambda x: (x[0], x[3], x[1], x[2]))
    return items[:k]


def get_duration_bucket(duration: float, bucket_size: float = 1.0) -> str:
    """Assign duration to bucket like '<0.25', '0.25-0.50', '0.50-0.75', etc."""
    if duration < bucket_size:
        return f"<{bucket_size:.2f}".rstrip('0').rstrip('.')
    # Calculate bucket boundaries with proper rounding
    lower = round(int(duration / bucket_size) * bucket_size, 2)
    upper = round(lower + bucket_size, 2)
    # Format with enough precision but strip trailing zeros
    lower_str = f"{lower:.2f}".rstrip('0').rstrip('.')
    upper_str = f"{upper:.2f}".rstrip('0').rstrip('.')
    return f"{lower_str}-{upper_str}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input JSONL with segment_transcriptions")
    ap.add_argument("--ignore_ws", action="store_true", help="Remove all whitespace")
    ap.add_argument("--remove_punct", action="store_true", help="Remove Unicode punctuation")
    ap.add_argument("--lower", action="store_true", help="Lowercase")
    ap.add_argument("--out_json", default=None, help="Optional path to write summary JSON")
    ap.add_argument("--top_confusions", type=int, default=30, help="Top-K confusions globally (default: 30)")
    ap.add_argument("--per_lang_top_confusions", type=int, default=10, help="Top-K confusions per language (default: 10)")
    ap.add_argument("--top_by_type", type=int, default=0, help="If >0, also print top-K confusions separately for SUB/INS/DEL")
    ap.add_argument("--bucket_size", type=float, default=1.0, help="Duration bucket size in seconds (default: 1.0)")
    args = ap.parse_args()
    
    overall_stats = {
        "edits": 0, "ref_len": 0, "count": 0,
        "ins": 0, "del": 0, "sub": 0
    }
    overall_conf: Dict[Tuple[str, str], int] = {}
    
    per_lang = defaultdict(lambda: {
        "edits": 0, "ref_len": 0, "count": 0,
        "ins": 0, "del": 0, "sub": 0,
        "conf_map": {},
        "examples": [],
        "worst_examples": []
    })
    
    per_duration = defaultdict(lambda: {
        "edits": 0, "ref_len": 0, "count": 0,
        "ins": 0, "del": 0, "sub": 0,
        "conf_map": {},
        "examples": [],
        "worst_examples": []
    })
    
    per_lang_duration = defaultdict(lambda: defaultdict(lambda: {
        "edits": 0, "ref_len": 0, "count": 0,
        "ins": 0, "del": 0, "sub": 0,
        "conf_map": {},
        "examples": [],
        "worst_examples": []
    }))
    
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            
            segments = rec.get("segment_transcriptions", [])
            if not isinstance(segments, list) or not segments:
                continue
            
            rec_id = rec.get("id", "unknown")
            
            # Score each segment
            for seg in segments:
                ref_text_raw = safe_str(seg.get("ref_text", ""))
                hyp_text_raw = safe_str(seg.get("whisper_text", ""))
                lang_code = safe_str(seg.get("lang", "unknown")).strip()
                seg_idx = seg.get("index", -1)
                duration = seg.get("duration", 0.0)
                duration_bucket = get_duration_bucket(duration, args.bucket_size)
                
                # Normalize 2-letter code to 3-letter if needed
                if len(lang_code) == 2 and lang_code.isalpha():
                    lang_code = LANG_2_TO_3.get(lang_code, lang_code)
                
                if not ref_text_raw or not hyp_text_raw:
                    continue
                
                ref = normalize_text(ref_text_raw, args.ignore_ws, args.remove_punct, args.lower)
                hyp = normalize_text(hyp_text_raw, args.ignore_ws, args.remove_punct, args.lower)
                
                if not ref:
                    continue
                
                edits, ins, dele, sub, conf = levenshtein_counts_and_confusions(ref, hyp)
                rlen = len(ref)
                cer = edits / max(1, rlen)
                
                # Update overall
                overall_stats["edits"] += edits
                overall_stats["ref_len"] += rlen
                overall_stats["count"] += 1
                overall_stats["ins"] += ins
                overall_stats["del"] += dele
                overall_stats["sub"] += sub
                merge_confusions(overall_conf, conf)
                
                # Update per-language
                d = per_lang[lang_code]
                d["edits"] += edits
                d["ref_len"] += rlen
                d["count"] += 1
                d["ins"] += ins
                d["del"] += dele
                d["sub"] += sub
                merge_confusions(d["conf_map"], conf)
                d["examples"].append({
                    "id": rec_id,
                    "seg_idx": seg_idx,
                    "cer": cer,
                    "edits": edits,
                    "ref_len": rlen,
                    "duration": duration,
                })
                
                # Update per-duration
                d_dur = per_duration[duration_bucket]
                d_dur["edits"] += edits
                d_dur["ref_len"] += rlen
                d_dur["count"] += 1
                d_dur["ins"] += ins
                d_dur["del"] += dele
                d_dur["sub"] += sub
                merge_confusions(d_dur["conf_map"], conf)
                d_dur["examples"].append({
                    "id": rec_id,
                    "seg_idx": seg_idx,
                    "cer": cer,
                    "edits": edits,
                    "ref_len": rlen,
                    "duration": duration,
                })
                
                # Update per-language-duration
                d_ld = per_lang_duration[lang_code][duration_bucket]
                d_ld["edits"] += edits
                d_ld["ref_len"] += rlen
                d_ld["count"] += 1
                d_ld["ins"] += ins
                d_ld["del"] += dele
                d_ld["sub"] += sub
                merge_confusions(d_ld["conf_map"], conf)
                d_ld["examples"].append({
                    "id": rec_id,
                    "seg_idx": seg_idx,
                    "cer": cer,
                    "edits": edits,
                    "ref_len": rlen,
                    "duration": duration,
                })
                
                # Track worst examples (keep top 5 worst per bucket)
                example = {
                    "id": rec_id,
                    "seg_idx": seg_idx,
                    "cer": cer,
                    "edits": edits,
                    "ref_len": rlen,
                    "duration": duration,
                    "ref": ref[:100],
                    "hyp": hyp[:100],
                }
                d["worst_examples"].append(example)
                d["worst_examples"].sort(key=lambda x: x["cer"], reverse=True)
                if len(d["worst_examples"]) > 5:
                    d["worst_examples"] = d["worst_examples"][:5]
                
                d_dur["worst_examples"].append(example)
                d_dur["worst_examples"].sort(key=lambda x: x["cer"], reverse=True)
                if len(d_dur["worst_examples"]) > 5:
                    d_dur["worst_examples"] = d_dur["worst_examples"][:5]
                
                d_ld["worst_examples"].append(example)
                d_ld["worst_examples"].sort(key=lambda x: x["cer"], reverse=True)
                if len(d_ld["worst_examples"]) > 5:
                    d_ld["worst_examples"] = d_ld["worst_examples"][:5]
    
    # Compute overall CER
    overall_cer = overall_stats["edits"] / max(1, overall_stats["ref_len"])
    
    print("\n=== Segments CER Summary ===")
    print(f"Scored segments: {overall_stats['count']}")
    print(f"Overall CER: {overall_cer:.6f}")
    print(f"  edits={overall_stats['edits']}, ref_chars={overall_stats['ref_len']}")
    print(f"  ins={overall_stats['ins']}, del={overall_stats['del']}, sub={overall_stats['sub']}")
    print(f"Normalization: ignore_ws={args.ignore_ws} remove_punct={args.remove_punct} lower={args.lower}")
    
    # Print global top confusions
    if args.top_confusions and args.top_confusions > 0:
        print(f"\n=== Top {args.top_confusions} confusions (global; SUB/INS/DEL) ===")
        print("count\ttype\tref\t->\thyp")
        for cnt, a, b, typ in top_k_confusions(overall_conf, args.top_confusions):
            print(f"{cnt}\t{typ}\t{format_char(a)}\t->\t{format_char(b)}")
    
    # Print confusions by type
    if args.top_by_type and args.top_by_type > 0:
        for typ in ["SUB", "INS", "DEL"]:
            print(f"\n=== Top {args.top_by_type} {typ} (global) ===")
            print("count\ttype\tref\t->\thyp")
            for cnt, a, b, t in top_k_confusions(overall_conf, args.top_by_type, only_type=typ):
                print(f"{cnt}\t{t}\t{format_char(a)}\t->\t{format_char(b)}")
    
    print("\n=== CER per language ===")
    print("language\tcount\tcer\tedits\tins\tdel\tsub\tref_chars")
    
    rows = []
    for lang, d in per_lang.items():
        if d["count"] > 0:
            cer = d["edits"] / max(1, d["ref_len"])
            rows.append((cer, lang, d))
    rows.sort(reverse=True)
    
    for cer, lang, d in rows:
        print(f"{lang}\t{d['count']}\t{cer:.6f}\t{d['edits']}\t{d['ins']}\t{d['del']}\t{d['sub']}\t{d['ref_len']}")
        
        # Print per-language top confusions
        if args.per_lang_top_confusions and args.per_lang_top_confusions > 0:
            topc = top_k_confusions(d["conf_map"], args.per_lang_top_confusions)
            if topc:
                print("  top_confusions:")
                for cnt, a, b, typ in topc:
                    print(f"    {cnt}\t{typ}\t{format_char(a)}\t->\t{format_char(b)}")
    
    # Print duration breakdown
    print("\n=== CER per duration bucket ===")
    print("duration_bucket\tcount\tcer\tedits\tins\tdel\tsub\tref_chars")
    
    dur_rows = []
    for dur_bucket, d in per_duration.items():
        if d["count"] > 0:
            cer = d["edits"] / max(1, d["ref_len"])
            dur_rows.append((dur_bucket, cer, d))
    # Sort by bucket order: <0.5, 0.5-1.0, 1.0-1.5, etc.
    def parse_bucket_key(bucket_str):
        if '<' in bucket_str:
            return float(bucket_str.replace('<', ''))
        else:
            return float(bucket_str.split('-')[0])
    
    dur_rows.sort(key=lambda x: parse_bucket_key(x[0]))
    
    for dur_bucket, cer, d in dur_rows:
        print(f"{dur_bucket}\t{d['count']}\t{cer:.6f}\t{d['edits']}\t{d['ins']}\t{d['del']}\t{d['sub']}\t{d['ref_len']}")
    
    # Print per-language per-duration breakdown (summary)
    print("\n=== CER per language per duration bucket ===")
    print("language\tduration_bucket\tcount\tcer\tedits\tins\tdel\tsub\tref_chars")
    
    for lang in sorted(per_lang_duration.keys()):
        lang_dur_data = per_lang_duration[lang]
        for dur_bucket in sorted(lang_dur_data.keys(), key=lambda x: parse_bucket_key(x)):
            d = lang_dur_data[dur_bucket]
            if d["count"] > 0:
                cer = d["edits"] / max(1, d["ref_len"])
                print(f"{lang}\t{dur_bucket}\t{d['count']}\t{cer:.6f}\t{d['edits']}\t{d['ins']}\t{d['del']}\t{d['sub']}\t{d['ref_len']}")
    
    # Print worst examples per language
    print("\n=== Worst Examples (Top 5 per Language) ===")
    for cer, lang, d in rows[:10]:  # Show for top 10 worst languages
        print(f"\n{lang.upper()} (CER={cer:.4f}, {d['count']} segments)")
        print("-" * 100)
        for i, ex in enumerate(d.get("worst_examples", [])[:3], 1):
            print(f"  {i}. Record ID: {ex['id']}, Segment: {ex['seg_idx']}")
            print(f"     CER: {ex['cer']:.4f} ({ex['edits']} edits / {ex['ref_len']} chars)")
            print(f"     Ref: {repr(ex['ref'])}")
            print(f"     Hyp: {repr(ex['hyp'])}")
            print()
    
    # Output JSON
    if args.out_json:
        out = {
            "normalization": {
                "ignore_ws": args.ignore_ws,
                "remove_punct": args.remove_punct,
                "lower": args.lower,
            },
            "overall": {
                "cer": overall_cer,
                "edits": overall_stats["edits"],
                "ins": overall_stats["ins"],
                "del": overall_stats["del"],
                "sub": overall_stats["sub"],
                "ref_chars": overall_stats["ref_len"],
                "scored_segments": overall_stats["count"],
                "top_confusions_global": [
                    {"count": cnt, "type": typ, "ref": a, "hyp": b}
                    for cnt, a, b, typ in top_k_confusions(overall_conf, args.top_confusions)
                ]
                if args.top_confusions and args.top_confusions > 0
                else [],
            },
            "per_language": {
                lang: {
                    "cer": d["edits"] / max(1, d["ref_len"]) if d["count"] > 0 else None,
                    "edits": d["edits"],
                    "ins": d["ins"],
                    "del": d["del"],
                    "sub": d["sub"],
                    "ref_chars": d["ref_len"],
                    "count": d["count"],
                    "top_confusions": [
                        {"count": cnt, "type": typ, "ref": a, "hyp": b}
                        for cnt, a, b, typ in top_k_confusions(d["conf_map"], args.per_lang_top_confusions)
                    ]
                    if args.per_lang_top_confusions and args.per_lang_top_confusions > 0
                    else [],
                }
                for _, lang, d in rows
            },
            "per_duration": {
                dur_bucket: {
                    "cer": d["edits"] / max(1, d["ref_len"]) if d["count"] > 0 else None,
                    "edits": d["edits"],
                    "ins": d["ins"],
                    "del": d["del"],
                    "sub": d["sub"],
                    "ref_chars": d["ref_len"],
                    "count": d["count"],
                }
                for dur_bucket, _, d in dur_rows
            },
            "per_language_per_duration": {
                lang: {
                    dur_bucket: {
                        "cer": per_lang_duration[lang][dur_bucket]["edits"] / max(1, per_lang_duration[lang][dur_bucket]["ref_len"]) if per_lang_duration[lang][dur_bucket]["count"] > 0 else None,
                        "edits": per_lang_duration[lang][dur_bucket]["edits"],
                        "ins": per_lang_duration[lang][dur_bucket]["ins"],
                        "del": per_lang_duration[lang][dur_bucket]["del"],
                        "sub": per_lang_duration[lang][dur_bucket]["sub"],
                        "ref_chars": per_lang_duration[lang][dur_bucket]["ref_len"],
                        "count": per_lang_duration[lang][dur_bucket]["count"],
                    }
                    for dur_bucket in per_lang_duration[lang].keys()
                    if per_lang_duration[lang][dur_bucket]["count"] > 0
                }
                for lang in per_lang_duration.keys()
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as wf:
            json.dump(out, wf, ensure_ascii=False, indent=2)
        print(f"\nWrote summary JSON to: {args.out_json}")


if __name__ == "__main__":
    main()
