#!/usr/bin/env python3
import argparse
import json
import re
from typing import Dict, Any, List

# Merge adjacent **...** blocks separated by only whitespace
MERGE_PATTERN = re.compile(r"\*\*([^*]+?)\*\*\s+\*\*([^*]+?)\*\*")

def merge_adjacent_marked(text: str) -> str:
    """
    '**a** **b** **c**' -> '**a b c**'
    """
    while True:
        new_text, n = MERGE_PATTERN.subn(
            lambda m: f"**{m.group(1).strip()} {m.group(2).strip()}**",
            text
        )
        if n == 0:
            return text
        text = new_text

def process_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    lang_field = rec.get("language", "")
    parts = [p.strip() for p in lang_field.split("-") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected exactly 2 langs in rec['language'], got: {lang_field}")

    matrix_lang, embedded_lang = parts[0], parts[1]

    ft = rec.get("fasttext_word_lid")
    if not ft or "words" not in ft:
        raise ValueError("Missing rec['fasttext_word_lid']['words']")

    words_info: List[Dict[str, Any]] = ft["words"]

    # Rebuild text from the token stream, wrapping embedded-language tokens
    rebuilt_tokens: List[str] = []
    for wi in words_info:
        w = wi.get("word", "")
        pred = wi.get("pred_lang")  # expected ISO-3 like 'eng', 'deu', etc.

        if pred == embedded_lang:
            rebuilt_tokens.append(f"**{w}**")
        else:
            rebuilt_tokens.append(w)

    new_text = " ".join(rebuilt_tokens)
    new_text = merge_adjacent_marked(new_text)

    rec["text"] = new_text
    # Optional: store what embedded/matrix were used for wrapping
    rec["postprocess"] = {
        "matrix_lang": matrix_lang,
        "embedded_lang": embedded_lang,
        "wrapped_field": "fasttext_word_lid.words[*].pred_lang",
    }
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input JSONL (with fasttext_word_lid)")
    ap.add_argument("-o", "--output", required=True, help="Output JSONL (text rewritten)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = process_record(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
