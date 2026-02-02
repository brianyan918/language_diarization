#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, List


BOLD_MARK_RE = re.compile(r"\*\*")          # remove ** markers
WHITESPACE_RE = re.compile(r"\s+")          # optional normalization


def normalize_ref(text: str, collapse_ws: bool) -> str:
    # remove ** markers
    text = BOLD_MARK_RE.sub("", text)
    if collapse_ws:
        text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_hyp(text: str, collapse_ws: bool) -> str:
    if collapse_ws:
        text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def levenshtein_distance(a: str, b: str) -> int:
    """
    Classic DP edit distance (insert/delete/substitute) at character level.
    Memory O(min(len(a), len(b))).
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Ensure b is the shorter for lower memory
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


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def compute_cer(ref: str, hyp: str) -> Tuple[int, int, float]:
    """
    Returns (edit_distance, ref_len, cer).
    CER = edits / max(1, ref_len)
    """
    edits = levenshtein_distance(ref, hyp)
    ref_len = len(ref)
    cer = edits / max(1, ref_len)
    return edits, ref_len, cer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input JSONL")
    ap.add_argument("--ref_field", default="text", help="Reference field (default: text)")
    ap.add_argument("--hyp_field", default="whisper_pred_text", help="Hyp field (default: whisper_pred_text)")
    ap.add_argument("--lang_field", default="language", help="Language field (default: language)")
    ap.add_argument("--exclude_langs", nargs="*", default=[], help="Languages to exclude entirely")
    ap.add_argument("--include_langs", nargs="*", default=None, help="If set, ONLY score these langs")
    ap.add_argument("--collapse_ws", action="store_true", help="Collapse whitespace before scoring")
    ap.add_argument("--lower", action="store_true", help="Lowercase both ref/hyp before scoring (useful for Latin)")
    ap.add_argument("--report_top", type=int, default=0, help="Print worst-N examples per lang (0 = off)")
    ap.add_argument("--out_json", default=None, help="Optional path to write summary JSON")
    args = ap.parse_args()

    exclude = set(args.exclude_langs)
    include = set(args.include_langs) if args.include_langs is not None else None

    # Accumulators
    overall_edits = 0
    overall_ref_len = 0
    overall_count = 0
    skipped_missing = 0
    skipped_excluded = 0

    per_lang = defaultdict(lambda: {"edits": 0, "ref_len": 0, "count": 0, "skipped_missing": 0, "examples": []})

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            lang = safe_str(rec.get(args.lang_field, "UNKNOWN"))

            # include/exclude logic
            if lang in exclude or (include is not None and lang not in include):
                skipped_excluded += 1
                continue

            ref_raw = safe_str(rec.get(args.ref_field, ""))
            hyp_raw = safe_str(rec.get(args.hyp_field, ""))

            if not ref_raw or not hyp_raw:
                skipped_missing += 1
                per_lang[lang]["skipped_missing"] += 1
                continue

            ref = normalize_ref(ref_raw, args.collapse_ws)
            hyp = normalize_hyp(hyp_raw, args.collapse_ws)

            if args.lower:
                ref = ref.lower()
                hyp = hyp.lower()

            edits, rlen, cer = compute_cer(ref, hyp)

            overall_edits += edits
            overall_ref_len += rlen
            overall_count += 1

            per_lang[lang]["edits"] += edits
            per_lang[lang]["ref_len"] += rlen
            per_lang[lang]["count"] += 1

            if args.report_top and args.report_top > 0:
                # store some examples for later sorting (worst CER)
                ex = {
                    "id": rec.get("id"),
                    "cer": cer,
                    "ref": ref,
                    "hyp": hyp,
                }
                per_lang[lang]["examples"].append(ex)

    overall_cer = overall_edits / max(1, overall_ref_len)

    # Print summary
    print("\n=== CER Summary ===")
    print(f"Scored utterances: {overall_count}")
    print(f"Skipped (missing ref/hyp): {skipped_missing}")
    print(f"Skipped (excluded/include-filtered): {skipped_excluded}")
    print(f"Overall CER: {overall_cer:.6f}  (edits={overall_edits}, ref_chars={overall_ref_len})")

    print("\n=== CER per language ===")
    header = ["language", "n", "cer", "edits", "ref_chars", "skipped_missing"]
    print("\t".join(header))

    # stable ordering: by CER descending then count
    rows = []
    for lang, d in per_lang.items():
        n = d["count"]
        if n == 0:
            continue
        cer = d["edits"] / max(1, d["ref_len"])
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
                    str(d["ref_len"]),
                    str(d["skipped_missing"]),
                ]
            )
        )

    # Worst examples
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
                print(f"REF: {ex['ref']}")
                print(f"HYP: {ex['hyp']}")
                print()

    # Optional JSON output
    if args.out_json:
        out = {
            "ref_field": args.ref_field,
            "hyp_field": args.hyp_field,
            "lang_field": args.lang_field,
            "excluded_langs": sorted(list(exclude)),
            "included_langs": sorted(list(include)) if include is not None else None,
            "collapse_ws": args.collapse_ws,
            "lower": args.lower,
            "overall": {
                "cer": overall_cer,
                "edits": overall_edits,
                "ref_chars": overall_ref_len,
                "scored_utterances": overall_count,
                "skipped_missing": skipped_missing,
                "skipped_excluded": skipped_excluded,
            },
            "per_language": {
                lang: {
                    "cer": (d["edits"] / max(1, d["ref_len"])) if d["count"] > 0 else None,
                    "edits": d["edits"],
                    "ref_chars": d["ref_len"],
                    "count": d["count"],
                    "skipped_missing": d["skipped_missing"],
                }
                for _, lang, d in rows
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as wf:
            json.dump(out, wf, ensure_ascii=False, indent=2)
        print(f"\nWrote summary JSON to: {args.out_json}")


if __name__ == "__main__":
    main()
