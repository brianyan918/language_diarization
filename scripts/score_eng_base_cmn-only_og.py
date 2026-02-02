#!/usr/bin/env python3
import argparse
import json
import re
from typing import Any, Dict, List, Tuple

REF_ENG_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
HAS_ASCII_LETTER = re.compile(r"[A-Za-z]")
WS_RE = re.compile(r"\s+")


def collapse_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def extract_ref_english(text: str) -> str:
    spans = [collapse_ws(m.group(1)) for m in REF_ENG_RE.finditer(text)]
    spans = [s for s in spans if s]
    return collapse_ws(" ".join(spans))


def extract_hyp_english_tokens(hyp: str) -> str:
    toks = [t for t in WS_RE.split(hyp.strip()) if t]
    eng = [t for t in toks if HAS_ASCII_LETTER.search(t)]
    return collapse_ws(" ".join(eng))


def edit_counts(ref: str, hyp: str) -> Tuple[int, int, int, int]:
    """
    Character-level Levenshtein with op counts.
    Returns (ins, del, sub, dist).
    Tie-break: SUB/MATCH > DEL > INS.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Baseline whisper JSONL")
    ap.add_argument("--lang_field", default="language")
    ap.add_argument("--target_lang", default="cmn-eng")
    ap.add_argument("--ref_field", default="text")
    ap.add_argument("--hyp_field", default="whisper_pred_text")
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--report_top", type=int, default=0)
    args = ap.parse_args()

    tot_ins = tot_del = tot_sub = 0
    tot_ref_chars = 0
    scored = 0
    skipped_no_ref_eng = 0
    skipped_no_hyp = 0
    skipped_other_lang = 0

    worst: List[Tuple[float, str, str, str, int, int, int, int, int]] = []
    # (cer, id, ref_eng, hyp_eng, ref_len, ins, del, sub, dist)

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
            ref_eng = extract_ref_english(ref_raw)
            if ref_eng == "":
                skipped_no_ref_eng += 1
                continue

            hyp_eng = extract_hyp_english_tokens(hyp_raw)

            if args.lower:
                ref_eng = ref_eng.lower()
                hyp_eng = hyp_eng.lower()

            ins, dele, sub, dist = edit_counts(ref_eng, hyp_eng)
            rlen = len(ref_eng)
            cer = dist / max(1, rlen)

            tot_ins += ins
            tot_del += dele
            tot_sub += sub
            tot_ref_chars += rlen
            scored += 1

            if args.report_top and args.report_top > 0:
                rid = str(rec.get("id", ""))
                worst.append((cer, rid, ref_eng, hyp_eng, rlen, ins, dele, sub, dist))

    denom = max(1, tot_ref_chars)
    overall_cer = (tot_ins + tot_del + tot_sub) / denom
    ins_rate = tot_ins / denom
    del_rate = tot_del / denom
    sub_rate = tot_sub / denom

    print(f"\n=== English-only CER for {args.target_lang} (baseline hyp={args.hyp_field}) ===")
    print(f"Scored utterances: {scored}")
    print(f"Skipped other languages: {skipped_other_lang}")
    print(f"Skipped (no hyp): {skipped_no_hyp}")
    print(f"Skipped (no ref English spans): {skipped_no_ref_eng}")

    print("\n--- Overall (normalized by ref English chars) ---")
    print(f"Ref chars: {tot_ref_chars}")
    print(f"CER:      {overall_cer:.6f}")
    print(f"INS rate: {ins_rate:.6f}   (count={tot_ins})")
    print(f"DEL rate: {del_rate:.6f}   (count={tot_del})")
    print(f"SUB rate: {sub_rate:.6f}   (count={tot_sub})")
    print(f"Edits:    {tot_ins + tot_del + tot_sub} total")

    if args.report_top and args.report_top > 0 and worst:
        worst.sort(key=lambda x: x[0], reverse=True)
        print(f"\n=== Worst {args.report_top} examples (by CER) ===")
        for cer, rid, ref_eng, hyp_eng, rlen, ins, dele, sub, dist in worst[: args.report_top]:
            denom_u = max(1, rlen)
            print(
                f"id={rid}  cer={cer:.6f}  "
                f"ins={ins}/{denom_u}({ins/denom_u:.4f}) "
                f"del={dele}/{denom_u}({dele/denom_u:.4f}) "
                f"sub={sub}/{denom_u}({sub/denom_u:.4f})"
            )
            print(f"REF_EN: {ref_eng}")
            print(f"HYP_EN: {hyp_eng}")
            print()

if __name__ == "__main__":
    main()
