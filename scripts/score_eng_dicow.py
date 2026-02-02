#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# Ref English spans: **...**
REF_ENG_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)

# Hyp timestamps: <|0.00|> etc.
TS_RE = re.compile(r"<\|.*?\|>")

WS_RE = re.compile(r"\s+")


def collapse_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def extract_ref_english(text: str) -> str:
    """
    Extract only English portions from ref, which are inside **...**.
    Returns a single string (spans joined by a space).
    """
    spans = [collapse_ws(m.group(1)) for m in REF_ENG_RE.finditer(text)]
    spans = [s for s in spans if s]
    return collapse_ws(" ".join(spans))


def extract_hyp_english_from_parts(parts: Any, part_name: str) -> Optional[str]:
    """
    Find a part entry with name matching part_name (exact or endswith),
    return cleaned text with timestamps removed.
    """
    if not isinstance(parts, list):
        return None

    chosen = None
    for p in parts:
        if not isinstance(p, dict):
            continue
        pname = p.get("part")
        if not isinstance(pname, str):
            continue
        if pname == part_name or pname.endswith("/" + part_name) or pname.endswith(part_name):
            chosen = p
            break

    if chosen is None:
        return None

    txt = chosen.get("text")
    if txt is None:
        return None
    txt = str(txt)
    txt = TS_RE.sub(" ", txt)          # remove <|...|>
    txt = collapse_ws(txt)
    return txt if txt else ""


def levenshtein(a: str, b: str) -> int:
    """
    Character-level Levenshtein distance (O(min(n,m)) memory).
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # ensure b is shorter
    if len(b) > len(a):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input JSONL")
    ap.add_argument("--ref_field", default="text", help="Ref field containing ** spans (default: text)")
    ap.add_argument("--parts_field", default="parts", help="Field containing list of parts (default: parts)")
    ap.add_argument("--part_name", default="en.txt", help="Part filename to use as hyp (default: en.txt)")
    ap.add_argument("--lang_field", default="language", help="Language field for per-lang report (default: language)")
    ap.add_argument("--exclude_langs", nargs="*", default=[], help="Languages to exclude entirely")
    ap.add_argument("--include_langs", nargs="*", default=None, help="If set, ONLY score these langs")
    ap.add_argument("--lower", action="store_true", help="Lowercase ref/hyp before scoring (useful for Latin)")
    ap.add_argument("--report_top", type=int, default=0, help="Show worst-N examples per language (0=off)")
    args = ap.parse_args()

    exclude = set(args.exclude_langs)
    include = set(args.include_langs) if args.include_langs is not None else None

    overall_edits = 0
    overall_ref_chars = 0
    scored = 0
    skipped_excluded = 0
    skipped_missing = 0

    per_lang = defaultdict(lambda: {"edits": 0, "ref_chars": 0, "count": 0, "skipped_missing": 0, "examples": []})

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            lang = str(rec.get(args.lang_field, "UNKNOWN"))

            if lang in exclude or (include is not None and lang not in include):
                skipped_excluded += 1
                continue

            ref_raw = rec.get(args.ref_field, "")
            ref_raw = "" if ref_raw is None else str(ref_raw)

            ref_eng = extract_ref_english(ref_raw)
            hyp_eng = extract_hyp_english_from_parts(rec.get(args.parts_field), args.part_name)

            # Score only when both exist and ref has something (otherwise CER denominator weird)
            if hyp_eng is None or ref_eng == "":
                skipped_missing += 1
                per_lang[lang]["skipped_missing"] += 1
                continue

            if args.lower:
                ref_eng = ref_eng.lower()
                hyp_eng = hyp_eng.lower()

            edits = levenshtein(ref_eng, hyp_eng)
            rlen = len(ref_eng)
            cer = edits / max(1, rlen)

            overall_edits += edits
            overall_ref_chars += rlen
            scored += 1

            per_lang[lang]["edits"] += edits
            per_lang[lang]["ref_chars"] += rlen
            per_lang[lang]["count"] += 1

            if args.report_top and args.report_top > 0:
                per_lang[lang]["examples"].append({
                    "id": rec.get("id"),
                    "cer": cer,
                    "ref_eng": ref_eng,
                    "hyp_eng": hyp_eng,
                })

    overall_cer = overall_edits / max(1, overall_ref_chars)

    print("\n=== English-only CER (ref=**...**, hyp=parts/en.txt) ===")
    print(f"Scored utterances: {scored}")
    print(f"Skipped (excluded/include-filtered): {skipped_excluded}")
    print(f"Skipped (missing hyp or no ref English spans): {skipped_missing}")
    print(f"Overall CER: {overall_cer:.6f}  (edits={overall_edits}, ref_chars={overall_ref_chars})")

    print("\n=== CER per language ===")
    print("\t".join(["language", "n", "cer", "edits", "ref_chars", "skipped_missing"]))

    rows = []
    for lang, d in per_lang.items():
        if d["count"] == 0:
            continue
        cer = d["edits"] / max(1, d["ref_chars"])
        rows.append((cer, lang, d))
    rows.sort(reverse=True)

    for cer, lang, d in rows:
        print("\t".join([
            lang,
            str(d["count"]),
            f"{cer:.6f}",
            str(d["edits"]),
            str(d["ref_chars"]),
            str(d["skipped_missing"]),
        ]))

    if args.report_top and args.report_top > 0:
        print(f"\n=== Worst {args.report_top} examples per language (by CER) ===")
        for cer, lang, d in rows:
            exs = d["examples"]
            if not exs:
                continue
            exs.sort(key=lambda x: x["cer"], reverse=True)
            print(f"\n-- {lang} --")
            for ex in exs[: args.report_top]:
                print(f"id={ex['id']}  cer={ex['cer']:.6f}")
                print(f"REF_EN: {ex['ref_eng']}")
                print(f"HYP_EN: {ex['hyp_eng']}")
                print()

if __name__ == "__main__":
    main()
