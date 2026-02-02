#!/usr/bin/env python3
import json
import re
import sys

REF_ENG_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
ASCII_LETTER = re.compile(r"[A-Za-z]")
CJK_CHAR = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")
WS_RE = re.compile(r"\s+")


def collapse_ws(s):
    return WS_RE.sub(" ", s).strip()


def extract_ref_eng(text):
    return collapse_ws(" ".join(m.group(1) for m in REF_ENG_RE.finditer(text)))


def extract_ref_non(text):
    text = REF_ENG_RE.sub(" ", text)
    return collapse_ws(text.replace("**", " "))


def split_hyp_charwise(hyp):
    eng = []
    non = []
    cur_eng = []

    def flush():
        if cur_eng:
            eng.append("".join(cur_eng))
            cur_eng.clear()

    for ch in hyp:
        if ASCII_LETTER.match(ch):
            cur_eng.append(ch)
        elif cur_eng and (ch.isdigit() or ch in "'-_."):
            cur_eng.append(ch)
        else:
            flush()
            if CJK_CHAR.match(ch):
                non.append(ch)
            else:
                non.append(" ")

    flush()
    return collapse_ws(" ".join(eng)), collapse_ws("".join(non))


def levenshtein_alignment(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m+1):
        dp[0][j] = j
        bt[0][j] = "I"

    for i in range(1, n+1):
        for j in range(1, m+1):
            sub = dp[i-1][j-1] + (ref[i-1] != hyp[j-1])
            dele = dp[i-1][j] + 1
            ins = dp[i][j-1] + 1
            best = min(sub, dele, ins)
            dp[i][j] = best
            bt[i][j] = (
                "M" if sub == best and ref[i-1] == hyp[j-1] else
                "S" if sub == best else
                "D" if dele == best else
                "I"
            )

    i, j = n, m
    aligned = []
    ins = dele = sub = 0

    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "M":
            aligned.append(("M", ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif op == "S":
            aligned.append(("S", ref[i-1], hyp[j-1]))
            sub += 1
            i -= 1; j -= 1
        elif op == "D":
            aligned.append(("D", ref[i-1], "∅"))
            dele += 1
            i -= 1
        else:
            aligned.append(("I", "∅", hyp[j-1]))
            ins += 1
            j -= 1

    return aligned[::-1], ins, dele, sub


def main(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("language") != "cmn-eng":
                continue

            ref = ex["text"]
            hyp = ex["whisper_pred_text"]

            ref_eng = extract_ref_eng(ref)
            ref_non = extract_ref_non(ref)
            hyp_eng, hyp_non = split_hyp_charwise(hyp)

            print("\n===== DEBUG EXAMPLE =====")
            print("ID:", ex.get("id"))
            print("REF:", ref)
            print("HYP:", hyp)

            print("\n--- REF ENG ---")
            print(ref_eng)
            print("\n--- HYP ENG ---")
            print(hyp_eng)

            print("\n--- REF NON ---")
            print(ref_non)
            print("\n--- HYP NON ---")
            print(hyp_non)

            print("\n--- NON-ENG ALIGNMENT ---")
            align, ins, dele, sub = levenshtein_alignment(ref_non, hyp_non)
            for op, r, h in align:
                print(f"{op}  {r}  {h}")

            print("\nCOUNTS:")
            print("INS:", ins, "DEL:", dele, "SUB:", sub)
            print("REF LEN:", len(ref_non))
            print("CER:", (ins + dele + sub) / max(1, len(ref_non)))

            return  # only print ONE example


if __name__ == "__main__":
    main(sys.argv[1])
