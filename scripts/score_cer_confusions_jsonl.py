#!/usr/bin/env python3
"""
score_cer_confusions.py

Character Error Rate (CER) scorer with:
- Total edits + INS/DEL/SUB (overall + per-language)
- Top confusions INCLUDING:
    SUB: (ref_char -> hyp_char)
    DEL: (ref_char -> <eps>)
    INS: (<eps> -> hyp_char)
- Optional: top confusions broken down by type (SUB only, INS only, DEL only)
- Optional worst-N examples per language
- OPTIONAL punctuation removal
- OPTIONAL whitespace ignoring (remove ALL whitespace before scoring)

Key normalization options:
  --ignore_ws         : remove all whitespace chars from ref/hyp (space/tab/newline/etc)
  --remove_punct      : remove Unicode punctuation from ref/hyp

Input JSONL: one object per line.

Usage:
  python score_cer_confusions.py -i in.jsonl \
    --ref_field text --hyp_field whisper_pred_text --lang_field language \
    --ignore_ws --remove_punct --lower \
    --top_confusions 50 --per_lang_top_confusions 10 \
    --top_by_type 20 \
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

EPS_SYM = "<eps>"  # used for insertions/deletions in confusion table


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def strip_unicode_punct(s: str) -> str:
    """
    Remove Unicode punctuation characters.
    Uses Unicode category starting with 'P' (Pc, Pd, Ps, Pe, Pi, Pf, Po).
    """
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def normalize_text(text: str, *, is_ref: bool, ignore_ws: bool, remove_punct: bool, lower: bool) -> str:
    """
    Shared normalization for ref/hyp.
    - For ref: remove ** markers
    - Optionally remove punctuation
    - Optionally remove ALL whitespace (ignore_ws)
    - Optionally lowercase
    """
    if is_ref:
        text = BOLD_MARK_RE.sub("", text)

    if remove_punct:
        text = strip_unicode_punct(text)

    if ignore_ws:
        # remove ALL whitespace (spaces, newlines, tabs, etc.)
        text = WS_RE.sub("", text)
    else:
        # keep as-is (no collapsing)
        text = text

    if lower:
        text = text.lower()

    return text


def levenshtein_counts_and_confusions(
    ref: str, hyp: str
) -> Tuple[int, int, int, int, Dict[Tuple[str, str], int]]:
    """
    Full DP Levenshtein with backtrace.

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
    op = [[None] * (m + 1) for _ in range(n + 1)]  # 'I','D','S','M'

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

            # Tie-break preference: M/S > D > I
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
            # Fallback (should be rare)
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


def cer_from_counts(edits: int, ref_len: int) -> float:
    return edits / max(1, ref_len)


def merge_confusions(dst: Dict[Tuple[str, str], int], src: Dict[Tuple[str, str], int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + int(v)


def format_char(c: str) -> str:
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
    if a == EPS_SYM and b != EPS_SYM:
        return "INS"
    if a != EPS_SYM and b == EPS_SYM:
        return "DEL"
    if a != EPS_SYM and b != EPS_SYM:
        return "SUB" if a != b else "M"
    return "?"


def top_k_confusions(conf: Dict[Tuple[str, str], int], k: int, only_type: Optional[str] = None):
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
    ap.add_argument("--ref_field", default="text", help="Reference field (default: text)")
    ap.add_argument("--hyp_field", default="whisper_pred_text", help="Hyp field (default: whisper_pred_text)")
    ap.add_argument("--lang_field", default="language", help="Language field (default: language)")
    ap.add_argument("--exclude_langs", nargs="*", default=[], help="Languages to exclude entirely")
    ap.add_argument("--include_langs", nargs="*", default=None, help="If set, ONLY score these langs")

    ap.add_argument(
        "--ignore_ws",
        action="store_true",
        help="Ignore whitespace by removing ALL whitespace chars before scoring.",
    )
    ap.add_argument(
        "--remove_punct",
        action="store_true",
        help="Remove Unicode punctuation chars before scoring (applies to both ref/hyp).",
    )
    ap.add_argument("--lower", action="store_true", help="Lowercase both ref/hyp before scoring (useful for Latin)")

    ap.add_argument("--report_top", type=int, default=0, help="Print worst-N examples per lang (0 = off)")
    ap.add_argument("--top_confusions", type=int, default=30, help="Top-K confusions (SUB/INS/DEL) globally")
    ap.add_argument("--per_lang_top_confusions", type=int, default=10, help="Top-K confusions per language")
    ap.add_argument(
        "--top_by_type",
        type=int,
        default=0,
        help="If >0, also print top-K confusions separately for SUB, INS, DEL (global).",
    )
    ap.add_argument("--out_json", default=None, help="Optional path to write summary JSON")
    args = ap.parse_args()

    exclude = set(args.exclude_langs)
    include = set(args.include_langs) if args.include_langs is not None else None

    overall_edits = 0
    overall_ref_len = 0
    overall_count = 0
    overall_ins = 0
    overall_del = 0
    overall_sub = 0
    overall_conf: Dict[Tuple[str, str], int] = {}

    skipped_missing = 0
    skipped_excluded = 0

    per_lang = defaultdict(
        lambda: {
            "edits": 0,
            "ref_len": 0,
            "count": 0,
            "ins": 0,
            "del": 0,
            "sub": 0,
            "conf_map": {},
            "skipped_missing": 0,
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

            if lang in exclude or (include is not None and lang not in include):
                skipped_excluded += 1
                continue

            ref_raw = safe_str(rec.get(args.ref_field, ""))
            hyp_raw = safe_str(rec.get(args.hyp_field, ""))

            if not ref_raw or not hyp_raw:
                skipped_missing += 1
                per_lang[lang]["skipped_missing"] += 1
                continue

            ref = normalize_text(
                ref_raw,
                is_ref=True,
                ignore_ws=args.ignore_ws,
                remove_punct=args.remove_punct,
                lower=args.lower,
            )
            hyp = normalize_text(
                hyp_raw,
                is_ref=False,
                ignore_ws=args.ignore_ws,
                remove_punct=args.remove_punct,
                lower=args.lower,
            )

            edits, ins, dele, sub, conf = levenshtein_counts_and_confusions(ref, hyp)
            rlen = len(ref)
            cer = cer_from_counts(edits, rlen)

            overall_edits += edits
            overall_ref_len += rlen
            overall_count += 1
            overall_ins += ins
            overall_del += dele
            overall_sub += sub
            merge_confusions(overall_conf, conf)

            d = per_lang[lang]
            d["edits"] += edits
            d["ref_len"] += rlen
            d["count"] += 1
            d["ins"] += ins
            d["del"] += dele
            d["sub"] += sub
            merge_confusions(d["conf_map"], conf)

            if args.report_top and args.report_top > 0:
                d["examples"].append(
                    {
                        "id": rec.get("id"),
                        "cer": cer,
                        "edits": edits,
                        "ins": ins,
                        "del": dele,
                        "sub": sub,
                        "ref": ref,
                        "hyp": hyp,
                    }
                )

    overall_cer = cer_from_counts(overall_edits, overall_ref_len)

    print("\n=== CER Summary ===")
    print(f"Scored utterances: {overall_count}")
    print(f"Skipped (missing ref/hyp): {skipped_missing}")
    print(f"Skipped (excluded/include-filtered): {skipped_excluded}")
    print(
        f"Overall CER: {overall_cer:.6f}  "
        f"(edits={overall_edits}, ref_chars={overall_ref_len}, ins={overall_ins}, del={overall_del}, sub={overall_sub})"
    )
    print(
        f"Normalization: ignore_ws={args.ignore_ws} remove_punct={args.remove_punct} lower={args.lower}"
    )

    if args.top_confusions and args.top_confusions > 0:
        print(f"\n=== Top {args.top_confusions} confusions (global; SUB/INS/DEL) ===")
        print("count\ttype\tref\t->\thyp")
        for cnt, a, b, typ in top_k_confusions(overall_conf, args.top_confusions):
            print(f"{cnt}\t{typ}\t{format_char(a)}\t->\t{format_char(b)}")

    if args.top_by_type and args.top_by_type > 0:
        for typ in ["SUB", "INS", "DEL"]:
            print(f"\n=== Top {args.top_by_type} {typ} (global) ===")
            print("count\ttype\tref\t->\thyp")
            for cnt, a, b, t in top_k_confusions(overall_conf, args.top_by_type, only_type=typ):
                print(f"{cnt}\t{t}\t{format_char(a)}\t->\t{format_char(b)}")

    print("\n=== CER per language ===")
    header = ["language", "n", "cer", "edits", "ins", "del", "sub", "ref_chars", "skipped_missing"]
    print("\t".join(header))

    rows = []
    for lang, d in per_lang.items():
        if d["count"] <= 0:
            continue
        cer = cer_from_counts(d["edits"], d["ref_len"])
        rows.append((cer, lang, d))
    rows.sort(reverse=True)

    for cer, lang, d in rows:
        print(
            "\t".join(
                [
                    lang,
                    str(d["count"]),
                    f"{cer:.6f}",
                    str(d["edits"]),
                    str(d["ins"]),
                    str(d["del"]),
                    str(d["sub"]),
                    str(d["ref_len"]),
                    str(d["skipped_missing"]),
                ]
            )
        )
        if args.per_lang_top_confusions and args.per_lang_top_confusions > 0:
            topc = top_k_confusions(d["conf_map"], args.per_lang_top_confusions)
            if topc:
                print("  top_confusions:")
                for cnt, a, b, typ in topc:
                    print(f"    {cnt}\t{typ}\t{format_char(a)}\t->\t{format_char(b)}")

    if args.report_top and args.report_top > 0:
        print(f"\n=== Worst {args.report_top} examples per language (by CER) ===")
        for cer, lang, d in rows:
            exs = d["examples"]
            if not exs:
                continue
            exs.sort(key=lambda x: x["cer"], reverse=True)
            print(f"\n-- {lang} --")
            for ex in exs[: args.report_top]:
                print(
                    f"id={ex['id']}  cer={ex['cer']:.6f}  "
                    f"edits={ex['edits']} ins={ex['ins']} del={ex['del']} sub={ex['sub']}"
                )
                print(f"REF: {ex['ref']}")
                print(f"HYP: {ex['hyp']}")
                print()

    if args.out_json:
        out = {
            "ref_field": args.ref_field,
            "hyp_field": args.hyp_field,
            "lang_field": args.lang_field,
            "excluded_langs": sorted(list(exclude)),
            "included_langs": sorted(list(include)) if include is not None else None,
            "normalization": {
                "ignore_ws": args.ignore_ws,
                "remove_punct": args.remove_punct,
                "lower": args.lower,
            },
            "overall": {
                "cer": overall_cer,
                "edits": overall_edits,
                "ins": overall_ins,
                "del": overall_del,
                "sub": overall_sub,
                "ref_chars": overall_ref_len,
                "scored_utterances": overall_count,
                "skipped_missing": skipped_missing,
                "skipped_excluded": skipped_excluded,
            },
            "top_confusions_global": [
                {"count": cnt, "type": typ, "ref": a, "hyp": b}
                for cnt, a, b, typ in top_k_confusions(overall_conf, args.top_confusions)
            ]
            if args.top_confusions and args.top_confusions > 0
            else [],
            "per_language": {
                lang: {
                    "cer": cer_from_counts(d["edits"], d["ref_len"]) if d["count"] > 0 else None,
                    "edits": d["edits"],
                    "ins": d["ins"],
                    "del": d["del"],
                    "sub": d["sub"],
                    "ref_chars": d["ref_len"],
                    "count": d["count"],
                    "skipped_missing": d["skipped_missing"],
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
