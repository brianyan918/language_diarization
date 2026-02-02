#!/usr/bin/env python3
import argparse
import json
import re
from typing import Any, Dict, List, Tuple

# **...** spans mark English in REF
REF_ENG_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)

# English-like tokens in HYP: any ASCII letter
HAS_ASCII_LETTER = re.compile(r"[A-Za-z]")

WS_RE = re.compile(r"\s+")


def collapse_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def extract_ref_eng(text: str) -> str:
    spans = [collapse_ws(m.group(1)) for m in REF_ENG_RE.finditer(text)]
    spans = [s for s in spans if s]
    return collapse_ws(" ".join(spans))


def extract_ref_noneng(text: str) -> str:
    # remove English spans entirely (including markers), keep the rest
    non = REF_ENG_RE.sub(" ", text)
    # also remove stray ** if any
    non = non.replace("**", " ")
    return collapse_ws(non)


def split_hyp_tokens(hyp: str) -> Tuple[List[str], List[str]]:
    """
    Returns (eng_tokens, noneng_tokens) based on ASCII-letter heuristic.
    """
    toks = [t for t in WS_RE.split(hyp.strip()) if t]
    eng = [t for t in toks if HAS_ASCII_LETTER.search(t)]
    non = [t for t in toks if not HAS_ASCII_LETTER.search(t)]
    return eng, non


def edit_counts(ref: str, hyp: str) -> Tuple[int, int, int, int]:
    """
    Character-level Levenshtein with operation counts.
    Returns (ins, del, sub, dist).
    Tie-break: SUB/MATCH > DEL > INS (stable).
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]  # 'M', 'S', 'D', 'I'

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"
    bt[0][0] = "M"

    for i in range(1, n + 1):
        r = ref[i - 1]
        for j in range(1, m + 1):
            h = hyp[j - 1]
            cost_sub = dp[i - 1][j - 1] + (0 if r == h else 1)
            cost_del = dp[i - 1][j] + 1
            cost_ins = dp[i][j - 1] + 1
            best = min(cost_sub, cost_del, cost_ins)
            dp[i][j] = best
            if best == cost_sub:
                bt[i][j] = "M" if r == h else "S"
            elif best == cost_del:
                bt[i][j] = "D"
            else:
                bt[i][j] = "I"

    i, j = n, m
    ins = dele = sub = 0
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            sub += 1
            i -= 1
            j -= 1
        elif op == "D":
            dele += 1
            i -= 1
        elif op == "I":
            ins += 1
            j -= 1
        else:
            break

    dist = ins + dele + sub
    return ins, dele, sub, dist


def fmt_rate(num: int, denom: int) -> str:
    d = max(1, denom)
    return f"{num/d:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Baseline whisper JSONL")
    ap.add_argument("--target_lang", default="cmn-eng", help="Only score this language (default: cmn-eng)")
    ap.add_argument("--lang_field", default="language")
    ap.add_argument("--ref_field", default="text")
    ap.add_argument("--hyp_field", default="whisper_pred_text")
    ap.add_argument("--lower", action="store_true", help="Lowercase both sides before scoring")
    ap.add_argument("--report_top", type=int, default=0, help="Worst-N examples for each split (0=off)")
    args = ap.parse_args()

    # Totals for ENG and NON-ENG splits
    totals = {
        "eng": {"ins": 0, "del": 0, "sub": 0, "ref_chars": 0, "count": 0, "skipped": 0, "worst": []},
        "non": {"ins": 0, "del": 0, "sub": 0, "ref_chars": 0, "count": 0, "skipped": 0, "worst": []},
    }
    skipped_other_lang = 0
    skipped_no_hyp = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec: Dict[str, Any] = json.loads(line)
            lang = str(rec.get(args.lang_field, ""))
            if lang != args.target_lang:
                skipped_other_lang += 1
                continue

            hyp_raw = rec.get(args.hyp_field)
            hyp_raw = None if hyp_raw is None else str(hyp_raw)
            if hyp_raw is None or hyp_raw.strip() == "":
                skipped_no_hyp += 1
                continue

            ref_raw = "" if rec.get(args.ref_field) is None else str(rec.get(args.ref_field))

            ref_eng = extract_ref_eng(ref_raw)
            ref_non = extract_ref_noneng(ref_raw)

            hyp_eng_tokens, hyp_non_tokens = split_hyp_tokens(hyp_raw)
            hyp_eng = collapse_ws(" ".join(hyp_eng_tokens))
            hyp_non = collapse_ws(" ".join(hyp_non_tokens))

            if args.lower:
                ref_eng, hyp_eng = ref_eng.lower(), hyp_eng.lower()
                ref_non, hyp_non = ref_non.lower(), hyp_non.lower()

            rid = str(rec.get("id", ""))

            # Score ENG split only if ref_eng exists
            if ref_eng:
                ins, dele, sub, dist = edit_counts(ref_eng, hyp_eng)
                totals["eng"]["ins"] += ins
                totals["eng"]["del"] += dele
                totals["eng"]["sub"] += sub
                totals["eng"]["ref_chars"] += len(ref_eng)
                totals["eng"]["count"] += 1
                if args.report_top and args.report_top > 0:
                    cer = dist / max(1, len(ref_eng))
                    totals["eng"]["worst"].append((cer, rid, ref_eng, hyp_eng, len(ref_eng), ins, dele, sub, dist))
            else:
                totals["eng"]["skipped"] += 1

            # Score NON-ENG split only if ref_non exists
            if ref_non:
                ins, dele, sub, dist = edit_counts(ref_non, hyp_non)
                totals["non"]["ins"] += ins
                totals["non"]["del"] += dele
                totals["non"]["sub"] += sub
                totals["non"]["ref_chars"] += len(ref_non)
                totals["non"]["count"] += 1
                if args.report_top and args.report_top > 0:
                    cer = dist / max(1, len(ref_non))
                    totals["non"]["worst"].append((cer, rid, ref_non, hyp_non, len(ref_non), ins, dele, sub, dist))
            else:
                totals["non"]["skipped"] += 1

    def print_block(name: str, t: Dict[str, Any]):
        denom = max(1, t["ref_chars"])
        cer = (t["ins"] + t["del"] + t["sub"]) / denom
        print(f"\n--- {name.upper()} split ---")
        print(f"Scored utterances: {t['count']}")
        print(f"Skipped (no ref chars in this split): {t['skipped']}")
        print(f"Ref chars: {t['ref_chars']}")
        print(f"CER:      {cer:.6f}")
        print(f"INS rate: {t['ins']/denom:.6f}   (count={t['ins']})")
        print(f"DEL rate: {t['del']/denom:.6f}   (count={t['del']})")
        print(f"SUB rate: {t['sub']/denom:.6f}   (count={t['sub']})")
        print(f"Edits:    {t['ins'] + t['del'] + t['sub']} total")

        if args.report_top and args.report_top > 0 and t["worst"]:
            t["worst"].sort(key=lambda x: x[0], reverse=True)
            print(f"\nWorst {args.report_top} examples for {name} (by CER):")
            for cer, rid, ref_s, hyp_s, rlen, ins, dele, sub, dist in t["worst"][: args.report_top]:
                d = max(1, rlen)
                print(
                    f"id={rid}  cer={cer:.6f}  "
                    f"ins={ins}/{d}({ins/d:.4f}) del={dele}/{d}({dele/d:.4f}) sub={sub}/{d}({sub/d:.4f})"
                )
                print(f"REF_{name.upper()}: {ref_s}")
                print(f"HYP_{name.upper()}: {hyp_s}")
                print()

    print(f"\n=== CER for {args.target_lang} (baseline hyp={args.hyp_field}) ===")
    print(f"Skipped other languages: {skipped_other_lang}")
    print(f"Skipped (no hyp): {skipped_no_hyp}")

    print_block("eng", totals["eng"])
    print_block("non", totals["non"])


if __name__ == "__main__":
    main()
