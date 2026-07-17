#!/usr/bin/env python3
"""
score_cer_parts_jsonl.py

Alternative CER scorer that:
1. Combines hyp parts in order (by time) *within each speaker*
2. Extracts ref parts:
    - ENG part: text inside **...**
    - MATRIX part: rest of the text
3. Scores each speaker against BOTH ref parts, then assigns one speaker to ENG
    and one speaker to MATRIX to minimize total CER
4. Reports CER separately for ENG and MATRIX parts (selected assignment)

Input JSONL: one object per line with 'ref' (list of segments), 'hyp' (list of segments).
Each segment has: 'speaker', 'words', 'start_time', 'end_time'

Usage:
  python score_cer_parts_jsonl.py -i in.jsonl \
    --ignore_ws --remove_punct --lower \
    --report_top 5 --out_json summary.json
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


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def strip_unicode_punct(s: str) -> str:
    """Remove Unicode punctuation characters."""
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def normalize_text(text: str, *, ignore_ws: bool, remove_punct: bool, lower: bool) -> str:
    """Normalize text for scoring."""
    if remove_punct:
        text = strip_unicode_punct(text)

    if ignore_ws:
        text = WS_RE.sub("", text)

    if lower:
        text = text.lower()

    return text


def extract_eng_and_matrix(ref_text: str) -> Tuple[str, str]:
    """
    Extract ENG part (inside **...**) and MATRIX part (rest) from ref_text.
    Returns (eng_part, matrix_part)
    
    IMPORTANT:
    - eng_part: text inside ** markers (concatenated with spaces)
    - matrix_part: text with ** markers AND their contents removed (not just the markers)
    """
    # Find text inside ** markers
    eng_matches = re.findall(r"\*\*([^*]*)\*\*", ref_text)
    eng_part = " ".join(eng_matches)

    # Remove ** markers AND their contents to get matrix part
    # Replace **...** with nothing (not just the markers)
    matrix_part = re.sub(r"\*\*[^*]*\*\*", "", ref_text)
    # Clean up extra spaces
    matrix_part = WS_RE.sub(" ", matrix_part).strip()

    return eng_part, matrix_part


def levenshtein_distance(ref: str, hyp: str) -> int:
    """Simple Levenshtein distance."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[n][m]


def levenshtein_with_confusions(ref: str, hyp: str) -> Tuple[int, Dict[Tuple[str, str], int], int, int, int]:
    """Levenshtein distance with character confusions and separate INS/DEL/SUB counts."""
    n, m = len(ref), len(hyp)
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

    # Backtrace
    i, j = n, m
    conf = defaultdict(int)
    ins_cnt = sub_cnt = del_cnt = 0

    while i > 0 or j > 0:
        cur_op = op[i][j]
        if cur_op == "M":
            i -= 1
            j -= 1
        elif cur_op == "S":
            conf[(ref[i - 1], hyp[j - 1])] += 1
            sub_cnt += 1
            i -= 1
            j -= 1
        elif cur_op == "D":
            conf[(ref[i - 1], EPS_SYM)] += 1
            del_cnt += 1
            i -= 1
        elif cur_op == "I":
            conf[(EPS_SYM, hyp[j - 1])] += 1
            ins_cnt += 1
            j -= 1
        else:
            if i > 0 and j > 0:
                if ref[i - 1] == hyp[j - 1]:
                    i -= 1
                    j -= 1
                else:
                    conf[(ref[i - 1], hyp[j - 1])] += 1
                    sub_cnt += 1
                    i -= 1
                    j -= 1
            elif i > 0:
                conf[(ref[i - 1], EPS_SYM)] += 1
                del_cnt += 1
                i -= 1
            else:
                conf[(EPS_SYM, hyp[j - 1])] += 1
                ins_cnt += 1
                j -= 1

    return dp[n][m], dict(conf), ins_cnt, del_cnt, sub_cnt


def cer_from_counts(edits: int, ref_len: int) -> float:
    return edits / max(1, ref_len)


def combine_hyp_by_speaker(hyp_segments: List[Dict]) -> Dict[str, str]:
    """
    Combine hyp segments per speaker (temporal order within each speaker).
    Returns dict: speaker -> combined_text
    
    Note: hyp segments represent different language speakers. Each speaker's 
    segments are combined in temporal order separately.
    """
    by_speaker = defaultdict(list)
    for seg in hyp_segments:
        speaker = seg.get("speaker", "unknown")
        words = safe_str(seg.get("words", ""))
        start_time = seg.get("start_time", float("inf"))
        by_speaker[speaker].append((start_time, words))

    # For each speaker, combine their segments in temporal order
    result = {}
    for speaker in sorted(by_speaker.keys()):
        items = sorted(by_speaker[speaker], key=lambda x: x[0])
        combined_words = [words for _, words in items]
        result[speaker] = " ".join(combined_words)

    return result


def extract_ref_parts(ref_segments: List[Dict]) -> Tuple[str, str]:
    """
    Extract ENG and MATRIX parts from ref segments.
    Returns (eng_part, matrix_part)
    """
    all_words = []
    for seg in ref_segments:
        words = safe_str(seg.get("words", ""))
        all_words.append(words)

    ref_text = " ".join(all_words)
    return extract_eng_and_matrix(ref_text)


def strip_id_prefix(s: str) -> str:
    """Strip prefix like 'test-0-' from id/language."""
    return re.sub(r'^[a-z]+-\d+-', '', s)


def format_char(c: str) -> str:
    """Format a character for display."""
    if c == EPS_SYM:
        return EPS_SYM
    if c == " ":
        return "<space>"
    if c == "\t":
        return "<tab>"
    if c == "\n":
        return "<nl>"
    return c


def top_k_confusions(conf: Dict[Tuple[str, str], int], k: int) -> List[Tuple[int, str, str]]:
    """Return top-k confusions (ref, hyp) sorted by count."""
    items = sorted(conf.items(), key=lambda x: x[1], reverse=True)
    return [(cnt, a, b) for (a, b), cnt in items[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input JSONL")
    ap.add_argument("--ref_field", default="text", help="Reference text field (default: ref_text)")
    ap.add_argument("--lang_field", default="language", help="Language field (default: language)")
    ap.add_argument("--exclude_langs", nargs="*", default=[], help="Languages to exclude")
    ap.add_argument("--include_langs", nargs="*", default=None, help="If set, ONLY score these langs")

    ap.add_argument("--ignore_ws", action="store_true", help="Remove ALL whitespace chars before scoring")
    ap.add_argument("--remove_punct", action="store_true", help="Remove Unicode punctuation chars")
    ap.add_argument("--lower", action="store_true", help="Lowercase before scoring")

    ap.add_argument("--report_top", type=int, default=0, help="Print worst-N examples per lang (0 = off)")
    ap.add_argument("--top_confusions", type=int, default=0, help="Print top-K confusions per part globally (0 = off)")
    ap.add_argument("--per_lang_confusions", type=int, default=0, help="Print top-K confusions per part per language (0 = off)")
    ap.add_argument("--out_json", default=None, help="Optional path to write summary JSON")
    ap.add_argument("--ignore_id_prefix", action="store_true", help="Strip prefix from language field")

    args = ap.parse_args()

    exclude = set(args.exclude_langs)
    include = set(args.include_langs) if args.include_langs is not None else None

    # Global accumulators
    overall_eng_edits = 0
    overall_eng_ref_len = 0
    overall_eng_ins = 0
    overall_eng_del = 0
    overall_eng_sub = 0
    overall_matrix_edits = 0
    overall_matrix_ref_len = 0
    overall_matrix_ins = 0
    overall_matrix_del = 0
    overall_matrix_sub = 0
    overall_count = 0
    skipped_missing = 0
    skipped_excluded = 0
    overall_eng_conf = defaultdict(int)
    overall_matrix_conf = defaultdict(int)

    per_lang = defaultdict(
        lambda: {
            "eng_edits": 0,
            "eng_ref_len": 0,
            "eng_ins": 0,
            "eng_del": 0,
            "eng_sub": 0,
            "matrix_edits": 0,
            "matrix_ref_len": 0,
            "matrix_ins": 0,
            "matrix_del": 0,
            "matrix_sub": 0,
            "count": 0,
            "skipped_missing": 0,
            "eng_conf": defaultdict(int),
            "matrix_conf": defaultdict(int),
            "examples": [],
        }
    )

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            lang = safe_str(rec.get(args.lang_field, "UNKNOWN"))
            if args.ignore_id_prefix:
                lang = strip_id_prefix(lang)

            if lang in exclude or (include is not None and lang not in include):
                skipped_excluded += 1
                continue

            ref_segs = rec.get("ref", [])
            hyp_segs = rec.get("hyp", [])

            if not hyp_segs:
                skipped_missing += 1
                per_lang[lang]["skipped_missing"] += 1
                continue

            # Extract ref parts (eng vs matrix)
            # Try to get ref_text from the specified field, or fall back to ref segments
            ref_text = safe_str(rec.get(args.ref_field, ""))
            if ref_text:
                eng_ref, matrix_ref = extract_eng_and_matrix(ref_text)
            else:
                if not ref_segs:
                    skipped_missing += 1
                    per_lang[lang]["skipped_missing"] += 1
                    continue
                eng_ref, matrix_ref = extract_ref_parts(ref_segs)

            # Combine hyp by speaker
            hyp_by_speaker = combine_hyp_by_speaker(hyp_segs)

            if not eng_ref and not matrix_ref:
                skipped_missing += 1
                per_lang[lang]["skipped_missing"] += 1
                continue

            # Normalize ref parts
            eng_ref_norm = normalize_text(eng_ref, ignore_ws=args.ignore_ws, remove_punct=args.remove_punct, lower=args.lower)
            matrix_ref_norm = normalize_text(matrix_ref, ignore_ws=args.ignore_ws, remove_punct=args.remove_punct, lower=args.lower)

            # Score each speaker against both refs, then assign one speaker to ENG
            # and one speaker to MATRIX to minimize total CER.
            per_speaker = {}
            for speaker, hyp_combined in hyp_by_speaker.items():
                hyp_norm = normalize_text(hyp_combined, ignore_ws=args.ignore_ws, remove_punct=args.remove_punct, lower=args.lower)

                # Score against ENG
                eng_edits, eng_conf, eng_ins, eng_del, eng_sub = levenshtein_with_confusions(eng_ref_norm, hyp_norm) if eng_ref_norm else (0, {}, 0, 0, 0)
                eng_ref_len = len(eng_ref_norm)
                eng_cer = cer_from_counts(eng_edits, eng_ref_len)

                # Score against MATRIX
                matrix_edits, matrix_conf, matrix_ins, matrix_del, matrix_sub = levenshtein_with_confusions(matrix_ref_norm, hyp_norm) if matrix_ref_norm else (0, {}, 0, 0, 0)
                matrix_ref_len = len(matrix_ref_norm)
                matrix_cer = cer_from_counts(matrix_edits, matrix_ref_len)

                per_speaker[speaker] = {
                    "hyp": hyp_norm,
                    "eng_edits": eng_edits,
                    "eng_ref_len": eng_ref_len,
                    "eng_cer": eng_cer,
                    "eng_ins": eng_ins,
                    "eng_del": eng_del,
                    "eng_sub": eng_sub,
                    "eng_conf": eng_conf,
                    "matrix_edits": matrix_edits,
                    "matrix_ref_len": matrix_ref_len,
                    "matrix_cer": matrix_cer,
                    "matrix_ins": matrix_ins,
                    "matrix_del": matrix_del,
                    "matrix_sub": matrix_sub,
                    "matrix_conf": matrix_conf,
                }

            if not per_speaker:
                skipped_missing += 1
                per_lang[lang]["skipped_missing"] += 1
                continue

            # Ensure we have two speakers to assign (ENG and MATRIX)
            if len(per_speaker) == 1:
                missing_speaker = "__MISSING__"
                missing_hyp = ""

                eng_edits, eng_conf, eng_ins, eng_del, eng_sub = levenshtein_with_confusions(eng_ref_norm, missing_hyp) if eng_ref_norm else (0, {}, 0, 0, 0)
                eng_ref_len = len(eng_ref_norm)
                eng_cer = cer_from_counts(eng_edits, eng_ref_len)

                matrix_edits, matrix_conf, matrix_ins, matrix_del, matrix_sub = levenshtein_with_confusions(matrix_ref_norm, missing_hyp) if matrix_ref_norm else (0, {}, 0, 0, 0)
                matrix_ref_len = len(matrix_ref_norm)
                matrix_cer = cer_from_counts(matrix_edits, matrix_ref_len)

                per_speaker[missing_speaker] = {
                    "hyp": missing_hyp,
                    "eng_edits": eng_edits,
                    "eng_ref_len": eng_ref_len,
                    "eng_cer": eng_cer,
                    "eng_ins": eng_ins,
                    "eng_del": eng_del,
                    "eng_sub": eng_sub,
                    "eng_conf": eng_conf,
                    "matrix_edits": matrix_edits,
                    "matrix_ref_len": matrix_ref_len,
                    "matrix_cer": matrix_cer,
                    "matrix_ins": matrix_ins,
                    "matrix_del": matrix_del,
                    "matrix_sub": matrix_sub,
                    "matrix_conf": matrix_conf,
                }

            speakers = list(per_speaker.keys())
            best_pair = None
            best_score = None

            for eng_speaker in speakers:
                for matrix_speaker in speakers:
                    if eng_speaker == matrix_speaker:
                        continue
                    total_cer = per_speaker[eng_speaker]["eng_cer"] + per_speaker[matrix_speaker]["matrix_cer"]
                    if best_score is None or total_cer < best_score:
                        best_score = total_cer
                        best_pair = (eng_speaker, matrix_speaker)

            if best_pair is None:
                skipped_missing += 1
                per_lang[lang]["skipped_missing"] += 1
                continue

            eng_speaker, matrix_speaker = best_pair
            eng_sel = per_speaker[eng_speaker]
            matrix_sel = per_speaker[matrix_speaker]

            # Accumulate global stats (selected assignment only)
            overall_eng_edits += eng_sel["eng_edits"]
            overall_eng_ref_len += eng_sel["eng_ref_len"]
            overall_eng_ins += eng_sel["eng_ins"]
            overall_eng_del += eng_sel["eng_del"]
            overall_eng_sub += eng_sel["eng_sub"]
            overall_matrix_edits += matrix_sel["matrix_edits"]
            overall_matrix_ref_len += matrix_sel["matrix_ref_len"]
            overall_matrix_ins += matrix_sel["matrix_ins"]
            overall_matrix_del += matrix_sel["matrix_del"]
            overall_matrix_sub += matrix_sel["matrix_sub"]
            overall_count += 1

            for (a, b), cnt in eng_sel["eng_conf"].items():
                overall_eng_conf[(a, b)] += cnt
            for (a, b), cnt in matrix_sel["matrix_conf"].items():
                overall_matrix_conf[(a, b)] += cnt

            d = per_lang[lang]
            d["eng_edits"] += eng_sel["eng_edits"]
            d["eng_ref_len"] += eng_sel["eng_ref_len"]
            d["eng_ins"] += eng_sel["eng_ins"]
            d["eng_del"] += eng_sel["eng_del"]
            d["eng_sub"] += eng_sel["eng_sub"]
            d["matrix_edits"] += matrix_sel["matrix_edits"]
            d["matrix_ref_len"] += matrix_sel["matrix_ref_len"]
            d["matrix_ins"] += matrix_sel["matrix_ins"]
            d["matrix_del"] += matrix_sel["matrix_del"]
            d["matrix_sub"] += matrix_sel["matrix_sub"]
            d["count"] += 1

            for (a, b), cnt in eng_sel["eng_conf"].items():
                d["eng_conf"][(a, b)] += cnt
            for (a, b), cnt in matrix_sel["matrix_conf"].items():
                d["matrix_conf"][(a, b)] += cnt

            if args.report_top and args.report_top > 0:
                d["examples"].append(
                    {
                        "id": rec.get("id"),
                        "eng_speaker": eng_speaker,
                        "matrix_speaker": matrix_speaker,
                        "selected_total_cer": best_score,
                        "per_speaker": {
                            spk: {
                                "eng_cer": info["eng_cer"],
                                "matrix_cer": info["matrix_cer"],
                                "hyp": info["hyp"],
                            }
                            for spk, info in per_speaker.items()
                        },
                        "eng_ref": eng_ref_norm,
                        "matrix_ref": matrix_ref_norm,
                    }
                )

    print("\n=== CER Summary (Parts-based scoring) ===")
    print(f"Scored utterances: {overall_count}")
    print(f"Skipped (missing ref/hyp): {skipped_missing}")
    print(f"Skipped (excluded/include-filtered): {skipped_excluded}")

    overall_eng_cer = cer_from_counts(overall_eng_edits, overall_eng_ref_len)
    overall_matrix_cer = cer_from_counts(overall_matrix_edits, overall_matrix_ref_len)

    print(f"\nENG part:")
    print(f"  CER: {overall_eng_cer:.6f}  (edits={overall_eng_edits}, ref_chars={overall_eng_ref_len})")
    print(f"  INS: {overall_eng_ins}, DEL: {overall_eng_del}, SUB: {overall_eng_sub}")
    print(f"  INS rate: {100.0*overall_eng_ins/max(1,overall_eng_ref_len):.2f}%, DEL rate: {100.0*overall_eng_del/max(1,overall_eng_ref_len):.2f}%, SUB rate: {100.0*overall_eng_sub/max(1,overall_eng_ref_len):.2f}%")
    print(f"\nMATRIX part:")
    print(f"  CER: {overall_matrix_cer:.6f}  (edits={overall_matrix_edits}, ref_chars={overall_matrix_ref_len})")
    print(f"  INS: {overall_matrix_ins}, DEL: {overall_matrix_del}, SUB: {overall_matrix_sub}")
    print(f"  INS rate: {100.0*overall_matrix_ins/max(1,overall_matrix_ref_len):.2f}%, DEL rate: {100.0*overall_matrix_del/max(1,overall_matrix_ref_len):.2f}%, SUB rate: {100.0*overall_matrix_sub/max(1,overall_matrix_ref_len):.2f}%")

    print(f"\nNormalization: ignore_ws={args.ignore_ws} remove_punct={args.remove_punct} lower={args.lower}")

    # Print global confusions if requested
    if args.top_confusions and args.top_confusions > 0:
        print(f"\n=== Top {args.top_confusions} confusions (ENG part) ===")
        print("count\tref\t->\thyp")
        for cnt, a, b in top_k_confusions(overall_eng_conf, args.top_confusions):
            print(f"{cnt}\t{format_char(a)}\t->\t{format_char(b)}")
        
        print(f"\n=== Top {args.top_confusions} confusions (MATRIX part) ===")
        print("count\tref\t->\thyp")
        for cnt, a, b in top_k_confusions(overall_matrix_conf, args.top_confusions):
            print(f"{cnt}\t{format_char(a)}\t->\t{format_char(b)}")

    print("\n=== CER per language (Parts-based) ===")
    header = ["language", "n", "eng_cer", "eng_ins", "eng_del", "eng_sub", "matrix_cer", "matrix_ins", "matrix_del", "matrix_sub", "skipped"]
    print("\t".join(header))

    rows = []
    for lang, d in per_lang.items():
        if d["count"] <= 0:
            continue
        eng_cer = cer_from_counts(d["eng_edits"], d["eng_ref_len"])
        matrix_cer = cer_from_counts(d["matrix_edits"], d["matrix_ref_len"])
        rows.append((eng_cer, matrix_cer, lang, d))
    rows.sort(reverse=True)

    for eng_cer, matrix_cer, lang, d in rows:
        eng_ins_rate = 100.0 * d["eng_ins"] / max(1, d["eng_ref_len"])
        eng_del_rate = 100.0 * d["eng_del"] / max(1, d["eng_ref_len"])
        eng_sub_rate = 100.0 * d["eng_sub"] / max(1, d["eng_ref_len"])
        matrix_ins_rate = 100.0 * d["matrix_ins"] / max(1, d["matrix_ref_len"])
        matrix_del_rate = 100.0 * d["matrix_del"] / max(1, d["matrix_ref_len"])
        matrix_sub_rate = 100.0 * d["matrix_sub"] / max(1, d["matrix_ref_len"])
        
        print(
            "\t".join(
                [
                    lang,
                    str(d["count"]),
                    f"{eng_cer:.6f}",
                    f"{eng_ins_rate:.2f}%",
                    f"{eng_del_rate:.2f}%",
                    f"{eng_sub_rate:.2f}%",
                    f"{matrix_cer:.6f}",
                    f"{matrix_ins_rate:.2f}%",
                    f"{matrix_del_rate:.2f}%",
                    f"{matrix_sub_rate:.2f}%",
                    str(d["skipped_missing"]),
                ]
            )
        )
        
        # Print per-language confusions if requested
        if args.per_lang_confusions and args.per_lang_confusions > 0:
            eng_conf = d.get("eng_conf", {})
            matrix_conf = d.get("matrix_conf", {})
            if eng_conf:
                print(f"  ENG confusions (top {args.per_lang_confusions}):")
                for cnt, a, b in top_k_confusions(eng_conf, args.per_lang_confusions):
                    print(f"    {cnt}\t{format_char(a)}\t->\t{format_char(b)}")
            if matrix_conf:
                print(f"  MATRIX confusions (top {args.per_lang_confusions}):")
                for cnt, a, b in top_k_confusions(matrix_conf, args.per_lang_confusions):
                    print(f"    {cnt}\t{format_char(a)}\t->\t{format_char(b)}")

    if args.report_top and args.report_top > 0:
        print(f"\n=== Worst {args.report_top} examples per language (by selected total CER) ===")
        for eng_cer, matrix_cer, lang, d in rows:
            exs = d["examples"]
            if not exs:
                continue
            exs.sort(key=lambda x: x["selected_total_cer"], reverse=True)
            print(f"\n-- {lang} --")
            for ex in exs[: args.report_top]:
                print(
                    f"id={ex['id']}  ENG={ex['eng_speaker']}  MATRIX={ex['matrix_speaker']}  total_cer={ex['selected_total_cer']:.6f}"
                )
                print("PER-SPEAKER CERs:")
                for spk, info in ex["per_speaker"].items():
                    print(f"  {spk}: eng_cer={info['eng_cer']:.6f}  matrix_cer={info['matrix_cer']:.6f}")
                print(f"ENG_REF:    {ex['eng_ref']}")
                print(f"MATRIX_REF: {ex['matrix_ref']}")
                print()

    if args.out_json:
        out = {
            "lang_field": args.lang_field,
            "excluded_langs": sorted(list(exclude)),
            "included_langs": sorted(list(include)) if include is not None else None,
            "normalization": {
                "ignore_ws": args.ignore_ws,
                "remove_punct": args.remove_punct,
                "lower": args.lower,
            },
            "overall": {
                "eng_cer": overall_eng_cer,
                "eng_edits": overall_eng_edits,
                "eng_ref_chars": overall_eng_ref_len,
                "matrix_cer": overall_matrix_cer,
                "matrix_edits": overall_matrix_edits,
                "matrix_ref_chars": overall_matrix_ref_len,
                "scored_utterances": overall_count,
                "skipped_missing": skipped_missing,
                "skipped_excluded": skipped_excluded,
            },
            "per_language": {
                lang: {
                    "eng_cer": cer_from_counts(d["eng_edits"], d["eng_ref_len"]) if d["count"] > 0 else None,
                    "eng_edits": d["eng_edits"],
                    "eng_ref_chars": d["eng_ref_len"],
                    "matrix_cer": cer_from_counts(d["matrix_edits"], d["matrix_ref_len"]) if d["count"] > 0 else None,
                    "matrix_edits": d["matrix_edits"],
                    "matrix_ref_chars": d["matrix_ref_len"],
                    "count": d["count"],
                    "skipped_missing": d["skipped_missing"],
                }
                for _, _, lang, d in rows
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as wf:
            json.dump(out, wf, ensure_ascii=False, indent=2)
        print(f"\nWrote summary JSON to: {args.out_json}")


if __name__ == "__main__":
    main()
