#!/usr/bin/env python
import argparse
import json
import re
import sys
from typing import List, Dict, Any

# --- Step 1: merge adjacent **english** segments -----------------------------

# Pattern: **foo**   **bar**
# We capture:   [**foo**][spaces][**bar**]
MERGE_PATTERN = re.compile(r"\*\*([^*]+?)\*\*\s+\*\*([^*]+?)\*\*")

def merge_adjacent_english(text: str) -> str:
    """
    Merge adjacent **...** segments separated only by whitespace.
    E.g. '**prison** **Abu** **Ghraib**' -> '**prison Abu Ghraib**'
    """
    while True:
        # Replace one level of adjacency; repeat until no more matches.
        new_text, n_subs = MERGE_PATTERN.subn(
            lambda m: f"**{m.group(1).strip()} {m.group(2).strip()}**",
            text,
        )
        if n_subs == 0:
            break
        text = new_text
    return text

# --- Step 2: segment by language using **...** markers -----------------------

EN_SPAN_PATTERN = re.compile(r"\*\*(.+?)\*\*")

def segment_by_language(text: str, language) -> List[Dict[str, str]]:
    """
    Given text where English segments are marked as **...** (possibly multi-word
    after merging), split into segments:

    - Non-English chunks (outside **...**) -> lang='other'
    - Each **...** chunk -> lang='eng' (without the ** markers)
    """
    segments: List[Dict[str, str]] = []
    pos = 0

    for m in EN_SPAN_PATTERN.finditer(text):
        before = text[pos:m.start()]
        eng_inner = m.group(1)  # contents inside **...**

        if before.strip():
            segments.append({"lang": language[0], "text": before.strip()})

        if eng_inner.strip():
            segments.append({"lang": language[1], "text": eng_inner.strip()})

        pos = m.end()

    tail = text[pos:]
    if tail.strip():
        segments.append({"lang": language[0], "text": tail.strip()})

    return segments

# --- Main I/O ----------------------------------------------------------------

def process_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    text = rec.get("text", "")
    merged = merge_adjacent_english(text)
    rec["text"] = merged  # modify in place
    rec["segments"] = segment_by_language(merged, rec.get("language").split("-"))
    return rec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="-",
        help="Input JSONL file (default: stdin)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="-",
        help="Output JSONL file (default: stdout)",
    )
    args = parser.parse_args()

    in_f = sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8")
    out_f = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")

    with in_f, out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = process_record(rec)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
