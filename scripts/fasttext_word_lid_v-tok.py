#!/usr/bin/env python3
"""
fasttext_word_lid.py

Token-level language ID using fastText with Unicode-aware tokenization and optional merging.

Fixes:
- Script-mixed glued tokens like "impressএর", "tutorialএ" by tokenizing into script runs.
- SEAME-style Han tokenization: optional --collapse_han_spaces removes spaces between Han chars.
- NEW: optional --merge_same_script_runs merges adjacent *content* tokens of the same script class
  (joined by a single space) to give more context to fastText.

Example (Arabic):
  whitespace: "يبلغ ارتفاع"
  script tokens: ["يبلغ", "ارتفاع"]
  merged: ["يبلغ ارتفاع"]   <-- better context for fastText

Output:
rec["fasttext_word_lid"] = {
  "candidate_langs": [...],          # ISO-639-3
  "tokenization": "...",
  "collapse_han_spaces": bool,
  "merge_same_script_runs": bool,
  "words": [
    {"word": "...", "pred_lang": "ara", "scores": {... optional ...}},
    ...
  ]
}

Dependencies:
  pip install fasttext
  pip install regex   # recommended (better Unicode script detection)

Usage:
  python fasttext_word_lid.py \
    -i in.jsonl -o out.jsonl -m lid.176.bin \
    --collapse_han_spaces \
    --merge_same_script_runs \
    --store_probs
"""

import argparse
import json
import unicodedata
from typing import Dict, Any, List, Optional, Tuple

import fasttext

try:
    import regex as regx  # pip install regex
except Exception:
    regx = None


# ISO-639-3 -> fastText ISO-2-ish labels
ISO3_TO_ISO2 = {
    "eng": "en",
    "deu": "de",
    "fra": "fr",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "nld": "nl",
    "rus": "ru",
    "ara": "ar",
    "hin": "hi",
    "ben": "bn",
    "tur": "tr",
    "vie": "vi",
    "ukr": "uk",
    "pol": "pl",
    "swe": "sv",
    "nor": "no",
    "dan": "da",
    "fin": "fi",
    "ces": "cs",
    "ron": "ro",
    "hun": "hu",
    "ell": "el",
    "heb": "he",
    "tha": "th",
    "ind": "id",
    "mal": "ms",
    "tgl": "tl",
    "cmn": "zh",
    "zho": "zh",
    "jpn": "ja",
    "kor": "ko",
}


# ---------------------------
# Candidate mapping utilities
# ---------------------------
def iso3_candidates(lang_field: str, add_eng: bool) -> List[str]:
    if not lang_field:
        return []
    
    rv = [x.strip() for x in str(lang_field).split("-") if x.strip()]
    if "eng" not in rv:
        return rv + ["eng"]
    else:
        return rv


def iso3_to_fasttext_label(iso3: str) -> Optional[str]:
    iso2 = ISO3_TO_ISO2.get(iso3)
    if not iso2:
        return None
    return f"__label__{iso2}"


# ---------------------------
# fastText scoring
# ---------------------------
def score_token_for_label(model, token: str, label: str, topk: int) -> float:
    labels, probs = model.predict(token, k=topk)
    lab2prob = {l: float(p) for l, p in zip(labels, probs)}
    return lab2prob.get(label, 0.0)


def predict_restricted(model, token: str, candidate_iso3: List[str], topk: int) -> Dict[str, Any]:
    candidates = []
    for iso3 in candidate_iso3:
        ft_label = iso3_to_fasttext_label(iso3)
        if ft_label:
            candidates.append((iso3, ft_label))

    if not candidates:
        labels, probs = model.predict(token, k=1)
        if labels:
            return {
                "pred_iso3": None,
                "pred_label": labels[0],
                "scores": {labels[0]: float(probs[0])},
                "note": "no ISO3->ISO2 mapping for candidates; used unrestricted fastText top-1",
            }
        return {"pred_iso3": None, "pred_label": None, "scores": {}, "note": "no prediction"}

    scores: Dict[str, float] = {}
    best_iso3 = None
    best_label = None
    best_score = -1.0

    for iso3, ft_label in candidates:
        s = score_token_for_label(model, token, ft_label, topk=topk)
        scores[iso3] = s
        if s > best_score:
            best_score = s
            best_iso3 = iso3
            best_label = ft_label

    return {"pred_iso3": best_iso3, "pred_label": best_label, "scores": scores}


# ---------------------------
# Tokenization helpers
# ---------------------------
def is_content_token(tok: str) -> bool:
    for ch in tok:
        cat = unicodedata.category(ch)
        if cat and cat[0] in ("L", "M", "N"):
            return True
    return False


def collapse_spaces_between_han(text: str) -> str:
    if not text:
        return text
    if regx is not None:
        return regx.sub(r"(?<=\p{Script=Han})\s+(?=\p{Script=Han})", "", text)

    def is_han(ch: str) -> bool:
        o = ord(ch)
        return 0x4E00 <= o <= 0x9FFF

    out = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch.isspace():
            j = i - 1
            while j >= 0 and text[j].isspace():
                j -= 1
            k = i + 1
            while k < n and text[k].isspace():
                k += 1
            if j >= 0 and k < n and is_han(text[j]) and is_han(text[k]):
                i = k
                continue
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def script_class_regex(ch: str) -> str:
    """
    Return a coarse script class label for a character using regex Script properties.
    """
    assert regx is not None
    if regx.match(r"\p{Script=Latin}", ch):
        return "LATIN"
    if regx.match(r"\p{Script=Arabic}", ch):
        return "ARABIC"
    if regx.match(r"\p{Script=Devanagari}", ch):
        return "DEVANAGARI"
    if regx.match(r"\p{Script=Bengali}", ch):
        return "BENGALI"
    if regx.match(r"\p{Script=Han}", ch):
        return "HAN"
    if regx.match(r"\p{Script=Hiragana}", ch):
        return "HIRAGANA"
    if regx.match(r"\p{Script=Katakana}", ch):
        return "KATAKANA"
    if regx.match(r"\p{Script=Hangul}", ch):
        return "HANGUL"
    # Generic letter/mark/number
    cat0 = unicodedata.category(ch)[0]
    if cat0 in ("L", "M", "N"):
        return "OTHER_LETTER"
    return "PUNC"


def script_class_fallback(ch: str) -> str:
    o = ord(ch)
    if ch.isspace():
        return "WS"
    if o < 128 and (ch.isalnum() or ch in ("_",)):
        return "LATIN"
    if (0x0600 <= o <= 0x06FF) or (0x0750 <= o <= 0x077F) or (0x08A0 <= o <= 0x08FF):
        return "ARABIC"
    if 0x0900 <= o <= 0x097F:
        return "DEVANAGARI"
    if 0x0980 <= o <= 0x09FF:
        return "BENGALI"
    if 0x4E00 <= o <= 0x9FFF:
        return "HAN"
    if 0x3040 <= o <= 0x309F:
        return "HIRAGANA"
    if 0x30A0 <= o <= 0x30FF:
        return "KATAKANA"
    if 0xAC00 <= o <= 0xD7AF:
        return "HANGUL"
    cat0 = unicodedata.category(ch)[0]
    if cat0 in ("L", "M", "N"):
        return "OTHER_LETTER"
    return "PUNC"


def tokenize_script_runs(text: str) -> List[str]:
    """
    Tokenize into script runs + numbers + single punctuation tokens.
    """
    text = (text or "").strip()
    if not text:
        return []

    if regx is not None:
        pattern = regx.compile(
            r"(?:\p{Script=Latin}+"
            r"|\p{Script=Arabic}+"
            r"|\p{Script=Devanagari}+"
            r"|\p{Script=Bengali}+"
            r"|\p{Script=Han}+"
            r"|\p{Script=Hiragana}+"
            r"|\p{Script=Katakana}+"
            r"|\p{Script=Hangul}+"
            r"|\p{Letter}+"
            r"|\p{Number}+"
            r"|[^\s])",
            regx.UNICODE,
        )
        return pattern.findall(text)

    # fallback: group by coarse class, punctuation is separate
    toks: List[str] = []
    cur: List[str] = []
    cur_c: Optional[str] = None

    for ch in text:
        c = script_class_fallback(ch)
        if c == "WS":
            if cur:
                toks.append("".join(cur))
                cur = []
                cur_c = None
            continue
        if c == "PUNC":
            if cur:
                toks.append("".join(cur))
                cur = []
                cur_c = None
            toks.append(ch)
            continue
        if cur_c is None:
            cur = [ch]
            cur_c = c
        elif c == cur_c:
            cur.append(ch)
        else:
            toks.append("".join(cur))
            cur = [ch]
            cur_c = c

    if cur:
        toks.append("".join(cur))
    return toks


def token_script_class(tok: str) -> str:
    """
    Determine a token's script class by inspecting its first content character.
    If no content char, returns PUNC.
    """
    for ch in tok:
        cat0 = unicodedata.category(ch)[0]
        if cat0 in ("L", "M", "N"):
            if regx is not None:
                return script_class_regex(ch)
            return script_class_fallback(ch)
    return "PUNC"


def merge_adjacent_same_script(tokens: List[str], merge_with_space: bool = True) -> List[str]:
    """
    Merge adjacent content tokens with the same script class.
    Punctuation breaks groups.
    """
    out: List[str] = []
    cur_parts: List[str] = []
    cur_cls: Optional[str] = None

    def flush():
        nonlocal cur_parts, cur_cls
        if cur_parts:
            out.append((" ".join(cur_parts)) if merge_with_space else ("".join(cur_parts)))
        cur_parts = []
        cur_cls = None

    for tok in tokens:
        if not is_content_token(tok):
            # punctuation / symbols: flush and optionally keep (we keep them as separate tokens)
            flush()
            out.append(tok)
            continue

        cls = token_script_class(tok)
        if cur_cls is None:
            cur_cls = cls
            cur_parts = [tok]
            continue

        if cls == cur_cls:
            cur_parts.append(tok)
        else:
            flush()
            cur_cls = cls
            cur_parts = [tok]

    flush()
    return out


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Input JSONL")
    ap.add_argument("--output", "-o", required=True, help="Output JSONL")
    ap.add_argument("--fasttext_model", "-m", required=True, help="Path to lid.176.bin or similar")
    ap.add_argument("--store_probs", action="store_true", help="Store per-candidate probs for each token (bigger output)")
    ap.add_argument("--topk", type=int, default=20, help="fastText top-k to query per token")
    ap.add_argument("--emit_punct", action="store_true", help="Also emit punctuation tokens (default: skip)")
    ap.add_argument(
        "--collapse_han_spaces",
        action="store_true",
        help="Collapse spaces ONLY between adjacent Han characters (helps SEAME-style tokenization).",
    )
    ap.add_argument(
        "--merge_same_script_runs",
        action="store_true",
        help="Merge adjacent content tokens with the same script class (joined by spaces) before fastText.",
    )
    ap.add_argument("--text_field", type=str, default="text")
    ap.add_argument("--add_eng", action="store_true")
    args = ap.parse_args()

    model = fasttext.load_model(args.fasttext_model)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            text = rec.get(args.text_field, "") or ""
            if args.collapse_han_spaces:
                text = collapse_spaces_between_han(text)

            lang_field = rec.get("language", "") or ""
            cand_iso3 = iso3_candidates(lang_field, args.add_eng)

            toks = tokenize_script_runs(text)

            if args.merge_same_script_runs:
                toks = merge_adjacent_same_script(toks, merge_with_space=True)

            out_words = []
            for tok in toks:
                if not args.emit_punct and not is_content_token(tok):
                    continue

                pred = predict_restricted(model, tok, cand_iso3, topk=args.topk)

                entry = {"word": tok, "pred_lang": pred.get("pred_iso3")}
                if args.store_probs:
                    entry["scores"] = pred.get("scores", {})
                if pred.get("pred_iso3") is None:
                    entry["pred_label"] = pred.get("pred_label")
                    if "note" in pred:
                        entry["note"] = pred["note"]

                out_words.append(entry)

            rec["fasttext_word_lid"] = {
                "candidate_langs": cand_iso3,
                "tokenization": "script_runs_regex" if regx is not None else "script_runs_fallback",
                "collapse_han_spaces": bool(args.collapse_han_spaces),
                "merge_same_script_runs": bool(args.merge_same_script_runs),
                "words": out_words,
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
