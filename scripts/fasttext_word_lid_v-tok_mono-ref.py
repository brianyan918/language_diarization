#!/usr/bin/env python3
"""
fasttext_word_lid.py

Token-level language ID with optional monolingual reference override.

Modes:
1) With --mono_dir:
   - Parse rec["id"] as <lang>_<sentenceID>_<spk>
   - Load mono_dir/<lang>.json
   - Tokenize monolingual reference into WORD TOKENS
   - If a token exactly matches a reference token → force pred_lang=<lang>
   - Otherwise fallback to fastText

2) Without --mono_dir:
   - Pure fastText-based token-level LID (original behavior)

Fixes:
- Word-level mono matching (no substring bugs like "to" in "touto")
- Never merge LATIN script runs
- iso3_candidates respects --add_eng

Dependencies:
  pip install fasttext
  pip install regex   # strongly recommended
"""

import argparse
import json
import os
import unicodedata
from typing import Dict, Any, List, Optional, Tuple, Set

import fasttext

try:
    import regex as regx
except Exception:
    regx = None


# ---------------------------
# ISO mapping
# ---------------------------
ISO3_TO_ISO2 = {
    # Core European
    "eng": "en",
    "deu": "de",
    "fra": "fr",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "nld": "nl",
    "pol": "pl",
    "ces": "cs",
    "slk": "sk",
    "ron": "ro",
    "hun": "hu",
    "bul": "bg",
    "ukr": "uk",
    "rus": "ru",
    "ell": "el",
    "swe": "sv",
    "dan": "da",
    "nor": "no",
    "fin": "fi",
    "cat": "ca",
    "cym": "cy",
    "gle": "ga",
    "isl": "is",
    "lav": "lv",
    "lit": "lt",
    "est": "et",
    "slv": "sl",

    # Semitic / Middle East
    "ara": "ar",
    "heb": "he",
    "tur": "tr",
    "fas": "fa",
    "urd": "ur",

    # South & Southeast Asia
    "hin": "hi",
    "ben": "bn",
    "tel": "te",
    "tam": "ta",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "pan": "pa",
    "guj": "gu",
    "sin": "si",
    "tha": "th",
    "vie": "vi",
    "ind": "id",
    "jav": "jv",
    "msa": "ms",
    "tgl": "tl",

    # East Asia
    "zho": "zh",
    "cmn": "zh",   # Mandarin → zh
    "yue": "zh",   # Cantonese → zh (necessary for FLEURS)
    "jpn": "ja",
    "kor": "ko",

    # Caucasus / Central Asia
    "aze": "az",
    "kat": "ka",
    "kaz": "kk",
    "uzb": "uz",

    # Africa
    "swh": "sw",
    "amh": "am",
    "hau": "ha",
    "ibo": "ig",
    "yor": "yo",
    "zul": "zu",
    "xho": "xh",
}

# ---------------------------
# Candidate utilities
# ---------------------------
def iso3_candidates(lang_field: str, add_eng: bool) -> List[str]:
    if not lang_field:
        rv: List[str] = []
    else:
        rv = [x.strip() for x in str(lang_field).split("-") if x.strip()]
    if add_eng and "eng" not in rv:
        rv.append("eng")
    return rv


def iso3_to_fasttext_label(iso3: str) -> Optional[str]:
    iso2 = ISO3_TO_ISO2.get(iso3)
    if iso2 is None and iso3 not in ["ceb", "zlm", "tgk", "mya", "kir", "lug"]:
        print(iso3)
    return f"__label__{iso2}" if iso2 else None


# ---------------------------
# fastText fallback
# ---------------------------
def score_token_for_label(model, token: str, label: str, topk: int) -> float:
    labels, probs = model.predict(token, k=topk)
    return {l: float(p) for l, p in zip(labels, probs)}.get(label, 0.0)


def predict_restricted(model, token: str, candidate_iso3: List[str], topk: int) -> Dict[str, Any]:
    candidates = [(iso3, iso3_to_fasttext_label(iso3)) for iso3 in candidate_iso3]
    candidates = [(i, l) for i, l in candidates if l]

    if not candidates:
        labels, probs = model.predict(token, k=1)
        return {
            "pred_iso3": None,
            "pred_label": labels[0] if labels else None,
            "scores": {labels[0]: float(probs[0])} if labels else {},
        }

    scores = {}
    best_iso3, best_label, best_score = None, None, -1.0

    for iso3, lbl in candidates:
        s = score_token_for_label(model, token, lbl, topk)
        scores[iso3] = s
        if s > best_score:
            best_score, best_iso3, best_label = s, iso3, lbl

    if best_score <= 0.0:
        labels, probs = model.predict(token, k=1)
        return {
            "pred_iso3": best_iso3,
            "pred_label": labels[0] if labels else None,
            "scores": scores,
            "note": "all candidate scores were 0.0",
        }

    return {"pred_iso3": best_iso3, "pred_label": best_label, "scores": scores}


# ---------------------------
# Tokenization
# ---------------------------
def is_content_token(tok: str) -> bool:
    return any(unicodedata.category(ch)[0] in ("L", "M", "N") for ch in tok)


def collapse_spaces_between_han(text: str) -> str:
    if regx:
        return regx.sub(r"(?<=\p{Script=Han})\s+(?=\p{Script=Han})", "", text)
    return text


def script_class(ch: str) -> str:
    if regx:
        for s in ["Latin", "Arabic", "Devanagari", "Bengali", "Han", "Hiragana", "Katakana", "Hangul"]:
            if regx.match(fr"\p{{Script={s}}}", ch):
                return s.upper()
    o = ord(ch)
    if ch.isspace():
        return "WS"
    if o < 128 and ch.isalnum():
        return "LATIN"
    if 0x4E00 <= o <= 0x9FFF:
        return "HAN"
    return "OTHER"


def tokenize_script_runs(text: str) -> List[str]:
    if not text:
        return []
    if regx:
        return regx.findall(
            r"(?:\p{Script=Latin}+|\p{Script=Arabic}+|\p{Script=Devanagari}+|"
            r"\p{Script=Bengali}+|\p{Script=Han}+|\p{Script=Hiragana}+|"
            r"\p{Script=Katakana}+|\p{Script=Hangul}+|\p{Letter}+|\p{Number}+|[^\s])",
            text,
        )

    toks, cur, cur_cls = [], [], None
    for ch in text:
        c = script_class(ch)
        if c == "WS":
            if cur:
                toks.append("".join(cur))
                cur, cur_cls = [], None
            continue
        if not cur or c == cur_cls:
            cur.append(ch)
            cur_cls = c
        else:
            toks.append("".join(cur))
            cur, cur_cls = [ch], c
    if cur:
        toks.append("".join(cur))
    return toks


def token_script_class(tok: str) -> str:
    for ch in tok:
        if unicodedata.category(ch)[0] in ("L", "M", "N"):
            return script_class(ch)
    return "PUNC"


def merge_adjacent_same_script(
    tokens: List[str], merge_with_space=True, no_merge_classes: Set[str] = {"LATIN"}
) -> List[str]:
    out, cur, cur_cls = [], [], None

    def flush():
        nonlocal cur, cur_cls
        if cur:
            out.append(" ".join(cur) if merge_with_space else "".join(cur))
        cur, cur_cls = [], None

    for tok in tokens:
        if not is_content_token(tok):
            flush()
            out.append(tok)
            continue

        cls = token_script_class(tok)
        if cls in no_merge_classes:
            flush()
            out.append(tok)
            continue

        if cur_cls == cls:
            cur.append(tok)
        else:
            flush()
            cur, cur_cls = [tok], cls

    flush()
    return out


# ---------------------------
# Monolingual reference helpers
# ---------------------------
def parse_lang_and_sentence_id(rec_id: str) -> Tuple[Optional[str], Optional[str]]:
    parts = str(rec_id).split("_")
    return (parts[0], parts[1]) if len(parts) >= 2 else (None, None)


def load_mono_lang_file(mono_dir: str, lang: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(mono_dir, f"{lang}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {str(x["sentenceId"]): x for x in json.load(f) if "sentenceId" in x}


def normalize_for_match(s: str, lower: bool) -> str:
    s = unicodedata.normalize("NFC", s or "")
    return s.lower() if lower else s


def build_mono_ref_token_set(
    ref_text: str, collapse_han: bool, merge_runs: bool, lower: bool
) -> Set[str]:
    if collapse_han:
        ref_text = collapse_spaces_between_han(ref_text)
    toks = tokenize_script_runs(ref_text)
    if merge_runs:
        toks = merge_adjacent_same_script(toks)
    return {
        normalize_for_match(t, lower)
        for t in toks
        if is_content_token(t)
    }


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-m", "--fasttext_model", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--add_eng", action="store_true")
    ap.add_argument("--emit_punct", action="store_true")
    ap.add_argument("--collapse_han_spaces", action="store_true")
    ap.add_argument("--merge_same_script_runs", action="store_true")

    # Optional mono
    ap.add_argument("--mono_dir", type=str, default=None)
    ap.add_argument("--mono_use_field", default="reference", choices=["reference", "codeSwitchedSentence"])
    ap.add_argument("--mono_match_lower", action="store_true")

    args = ap.parse_args()

    model = fasttext.load_model(args.fasttext_model)
    mono_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def get_mono_item(lang: str, sid: str) -> Optional[Dict[str, Any]]:
        if not args.mono_dir:
            return None
        if lang not in mono_cache:
            mono_cache[lang] = load_mono_lang_file(args.mono_dir, lang)
        return mono_cache[lang].get(sid)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)

            text = rec.get(args.text_field, "") or ""
            if args.collapse_han_spaces:
                text = collapse_spaces_between_han(text)

            toks = tokenize_script_runs(text)
            if args.merge_same_script_runs:
                toks = merge_adjacent_same_script(toks)

            cand_iso3 = iso3_candidates(rec.get("language", ""), args.add_eng)

            mono_tokens = None
            mono_lang = None
            if args.mono_dir:
                mono_lang, mono_sid = parse_lang_and_sentence_id(rec.get("id", ""))
                item = get_mono_item(mono_lang, mono_sid) if mono_lang and mono_sid else None
                if item:
                    mono_tokens = build_mono_ref_token_set(
                        item.get(args.mono_use_field, ""),
                        args.collapse_han_spaces,
                        args.merge_same_script_runs,
                        args.mono_match_lower,
                    )

            out_words = []
            for tok in toks:
                if not args.emit_punct and not is_content_token(tok):
                    continue

                if mono_tokens is not None and mono_lang:
                    if normalize_for_match(tok, args.mono_match_lower) in mono_tokens:
                        out_words.append({"word": tok, "pred_lang": mono_lang, "source": "mono_ref"})
                        continue

                pred = predict_restricted(model, tok, cand_iso3, args.topk)
                out_words.append({
                    "word": tok,
                    "pred_lang": pred.get("pred_iso3"),
                    "source": "fasttext",
                })

            rec["fasttext_word_lid"] = {
                "mono_enabled": bool(args.mono_dir),
                "words": out_words,
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
