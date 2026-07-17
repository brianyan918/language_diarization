#!/usr/bin/env python3
"""
score_target_lang_cer.py

Score Character Error Rate (CER) for target languages separately.

Supports three input formats:
1. whisper_outputs format: {"whisper_outputs": {"por": "...", "eng": "..."}}
2. segment format: segments array with "speaker" field (e.g., "ru_lbl16" -> "ru" -> "rus")
3. segment format: segments array with "lang" field (e.g., "ara", "eng")

Reference extraction:
- If segments with "lang" field exist: extract per-language text from segments
- Else if "**...**" markers present in text: text wrapped in ** is one language, rest is another
- Map 2-letter codes to 3-letter (e.g., "eng" from marker, "ara" from segments)

Usage:
  python score_target_lang_cer.py -i input.jsonl \
    --lang_field language --ignore_ws --remove_punct --lower \
    --out_json summary.json
"""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Tuple, List, Optional

BOLD_MARK_RE = re.compile(r"\*\*")
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


def levenshtein_counts(ref: str, hyp: str) -> Tuple[int, int, int, int]:
    """Return edits, ins, del, sub."""
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
    
    while i > 0 or j > 0:
        cur_op = op[i][j]
        if cur_op == "M":
            i -= 1
            j -= 1
        elif cur_op == "S":
            sub += 1
            i -= 1
            j -= 1
        elif cur_op == "D":
            dele += 1
            i -= 1
        elif cur_op == "I":
            ins += 1
            j -= 1
        else:
            if i > 0 and j > 0:
                if ref[i - 1] == hyp[j - 1]:
                    i -= 1
                    j -= 1
                else:
                    sub += 1
                    i -= 1
                    j -= 1
            elif i > 0:
                dele += 1
                i -= 1
            else:
                ins += 1
                j -= 1
    
    edits = dp[n][m]
    return edits, ins, dele, sub


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


def extract_by_markers(text: str) -> Tuple[str, str]:
    """
    Extract text by ** markers.
    Returns (eng_text, non_eng_text).
    """
    eng_parts = []
    other_parts = []
    
    # Find all ** ... ** blocks
    pos = 0
    for match in re.finditer(r'\*\*([^*]*)\*\*', text):
        # Add non-marked text before this block
        if match.start() > pos:
            other_parts.append(text[pos:match.start()])
        # Add marked text (English)
        eng_parts.append(match.group(1))
        pos = match.end()
    
    # Add remaining text
    if pos < len(text):
        other_parts.append(text[pos:])
    
    eng_text = ' '.join(eng_parts).strip()
    other_text = ' '.join(other_parts).strip()
    
    return eng_text, other_text


def extract_format1(rec: Dict) -> Dict[str, str]:
    """Extract from whisper_outputs format.
    Supports both:
    - {"whisper_outputs": {"ara": "text", "eng": "text"}}
    - {"whisper_outputs": {"ara": {"text": "..."}, "eng": {"text": "..."}}}
    """
    whisper_outputs = rec.get("whisper_outputs", {})
    result = {}
    
    for lang, value in whisper_outputs.items():
        if isinstance(value, dict):
            # New format: {"text": "..."}
            text = safe_str(value.get("text", ""))
        else:
            # Old format: direct string
            text = safe_str(value)
        
        if text:
            result[lang] = text
    
    return result


def extract_format2_hyp_only(rec: Dict) -> Dict[str, str]:
    """Extract from hyp array format with speaker field."""
    hyp_array = rec.get("hyp", [])
    lang_to_text = defaultdict(list)
    
    for seg in hyp_array:
        speaker = safe_str(seg.get("speaker", ""))
        words = safe_str(seg.get("words", ""))
        
        # Extract 2-letter code from speaker (e.g., "ru_lbl16" -> "ru", "AH_ar" -> "ar")
        # Try suffix pattern first (AH_ar, AH_en)
        match = re.search(r'_(ar|en|es|fr|de|it|pt|ru|ja|zh|hi|bn|ta|te|tr|pl|nl|sv|no|da|fi|he|th|ko|vi|id|my|km|lo|ca|eu|ga|gl|el|hu|ro|sk|sl|cs|hr|sr|mk|bg|uk|ka|fa|ur|ps|ku|am|om|sw|so)$', speaker)
        if match:
            lang_2 = match.group(1)
        else:
            # Try prefix pattern (ru_lbl16 -> ru)
            match = re.match(r'^([a-z]{2})', speaker)
            if match:
                lang_2 = match.group(1)
            else:
                continue
        
        lang_3 = LANG_2_TO_3.get(lang_2, lang_2)
        if words:
            lang_to_text[lang_3].append(words)
    
    # Join segments for each language
    result = {}
    for lang, parts in lang_to_text.items():
        result[lang] = ' '.join(parts).strip()
    
    return result


def extract_ref_from_segments_lang_field(rec: Dict) -> Dict[str, str]:
    """
    Extract reference from segments array with "lang" field.
    Each segment has: {"text": "...", "lang": "ara"/"eng"/etc}
    Returns: {"ara": "text1 text2", "eng": "text3 text4"}
    """
    segments = rec.get("segments", [])
    if not isinstance(segments, list):
        return {}
    
    lang_to_text = defaultdict(list)
    
    for seg in segments:
        lang = safe_str(seg.get("lang", "")).strip()
        text = safe_str(seg.get("text", "")).strip()
        
        # Normalize 2-letter code to 3-letter if needed
        if len(lang) == 2 and lang.isalpha():
            lang = LANG_2_TO_3.get(lang, lang)
        
        if lang and text:
            lang_to_text[lang].append(text)
    
    # Join segments for each language with space separator
    result = {}
    for lang, parts in lang_to_text.items():
        result[lang] = ' '.join(parts).strip()
    
    return result


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input JSONL")
    ap.add_argument("--lang_field", default="language", help="Language field (default: language)")
    ap.add_argument("--ignore_ws", action="store_true", help="Remove all whitespace")
    ap.add_argument("--remove_punct", action="store_true", help="Remove Unicode punctuation")
    ap.add_argument("--lower", action="store_true", help="Lowercase")
    ap.add_argument("--out_json", default=None, help="Optional path to write summary JSON")
    ap.add_argument("--top_confusions", type=int, default=30, help="Top-K confusions globally (default: 30)")
    ap.add_argument("--per_lang_top_confusions", type=int, default=10, help="Top-K confusions per language (default: 10)")
    ap.add_argument("--top_by_type", type=int, default=0, help="If >0, also print top-K confusions separately for SUB/INS/DEL")
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
        "worst_examples": []  # Track worst scoring utterances
    })
    
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            
            # Always extract reference from "text" field
            ref_text = safe_str(rec.get("text", ""))
            if not ref_text:
                continue
            
            # Determine format and extract hypotheses
            if "whisper_outputs" in rec:
                # Format 1: whisper_outputs dict
                hyp_by_lang = extract_format1(rec)
            elif "hyp" in rec and isinstance(rec.get("hyp"), list):
                # Format 2 & 3: hyp is an array with speaker field
                hyp_by_lang = extract_format2_hyp_only(rec)
            else:
                continue
            
            if not hyp_by_lang:
                continue
            
            # Try to extract reference from segments with "lang" field first (Format 3)
            ref_by_lang = extract_ref_from_segments_lang_field(rec)
            
            if ref_by_lang:
                # Format 3: segments with "lang" field
                pass  # Use ref_by_lang directly
            else:
                # Format 1 & 2: Use markers from text field
                eng_ref, other_ref = extract_by_markers(ref_text)
                
                # Get matrix language to determine which ref goes with which lang
                lang_str = safe_str(rec.get(args.lang_field, ""))
                if lang_str:
                    # Split by "-" and filter for valid 3-letter codes or convert 2-letter codes
                    parts = lang_str.split("-")
                    matrix_lang = None
                    for part in parts:
                        if len(part) == 3 and part.isalpha():
                            matrix_lang = part
                            break
                        elif len(part) == 2 and part.isalpha() and part in LANG_2_TO_3:
                            matrix_lang = LANG_2_TO_3[part]
                            break
                    if not matrix_lang:
                        matrix_lang = "unknown"
                else:
                    matrix_lang = "unknown"
                
                # Map references: eng_ref for "eng", other_ref for matrix_lang or non-eng languages
                ref_by_lang = {}
                if eng_ref:
                    ref_by_lang["eng"] = eng_ref
                if other_ref:
                    # If we know the matrix language, use it; otherwise guess from hyp_by_lang
                    if matrix_lang != "unknown" and matrix_lang != "eng":
                        ref_by_lang[matrix_lang] = other_ref
                    else:
                        # Assign to first non-eng language in hyp_by_lang
                        for lang in hyp_by_lang:
                            if lang != "eng" and lang not in ref_by_lang:
                                ref_by_lang[lang] = other_ref
                                break
            
            # Score each language
            for lang, hyp_text in hyp_by_lang.items():
                ref_text_for_lang = ref_by_lang.get(lang, "")
                
                if not ref_text_for_lang:
                    continue
                
                ref = normalize_text(ref_text_for_lang, args.ignore_ws, args.remove_punct, args.lower)
                hyp = normalize_text(hyp_text, args.ignore_ws, args.remove_punct, args.lower)
                
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
                d = per_lang[lang]
                d["edits"] += edits
                d["ref_len"] += rlen
                d["count"] += 1
                d["ins"] += ins
                d["del"] += dele
                d["sub"] += sub
                merge_confusions(d["conf_map"], conf)
                d["examples"].append({
                    "id": rec.get("id"),
                    "cer": cer,
                    "edits": edits,
                    "ref_len": rlen,
                })
                
                # Track worst examples (keep top 5 worst)
                example = {
                    "id": rec.get("id"),
                    "cer": cer,
                    "edits": edits,
                    "ref_len": rlen,
                    "ref": ref[:100],  # First 100 chars
                    "hyp": hyp[:100],  # First 100 chars
                }
                d["worst_examples"].append(example)
                d["worst_examples"].sort(key=lambda x: x["cer"], reverse=True)
                if len(d["worst_examples"]) > 5:
                    d["worst_examples"] = d["worst_examples"][:5]
    
    # Compute overall CER
    overall_cer = overall_stats["edits"] / max(1, overall_stats["ref_len"])
    
    print("\n=== Target Language CER Summary ===")
    print(f"Scored utterances: {overall_stats['count']}")
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
    
    # Print worst examples per language
    print("\n=== Worst Examples (Top 5 per Language) ===")
    for cer, lang, d in rows[:10]:  # Show for top 10 worst languages
        print(f"\n{lang.upper()} (CER={cer:.4f}, {d['count']} utterances)")
        print("-" * 100)
        for i, ex in enumerate(d.get("worst_examples", [])[:3], 1):
            print(f"  {i}. ID: {ex['id']}")
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
                "scored_utterances": overall_stats["count"],
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
        }
        with open(args.out_json, "w", encoding="utf-8") as wf:
            json.dump(out, wf, ensure_ascii=False, indent=2)
        print(f"\nWrote summary JSON to: {args.out_json}")


if __name__ == "__main__":
    main()
