#!/usr/bin/env python3
import argparse
import json
import re
from typing import Dict, Any, List, Tuple, Optional

import fasttext

WS_SPLIT = re.compile(r"\s+")

# Minimal ISO-3 -> ISO-2 mapping for common cases.
# Extend as needed.
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
    "cmn": "zh",  # sometimes 'zho' also used
    "zho": "zh",
    "jpn": "ja",
    "kor": "ko",
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
    # add more if your dataset needs them
}

def iso3_candidates(lang_field: str) -> List[str]:
    return [x.strip() for x in lang_field.split("-") if x.strip()]

def iso3_to_fasttext_label(iso3: str) -> Optional[str]:
    """
    Map ISO-3 (e.g., 'deu') to fastText label (e.g., '__label__de').
    Returns None if unknown.
    """
    iso2 = ISO3_TO_ISO2.get(iso3)
    if not iso2:
        return None
    return f"__label__{iso2}"

def score_word_for_label(model, word: str, label: str) -> float:
    """
    fastText supervised model provides raw scores via predict(k).
    We can ask for all labels (k=-1) but that's expensive.
    Instead, ask for top K once and use its probability if present.
    If label isn't present in top K, approximate with a very small prob.
    """
    # For very short tokens or punctuation, fastText can be noisy; still score.
    labels, probs = model.predict(word, k=20)  # top-20 is usually enough
    lab2prob = {l: float(p) for l, p in zip(labels, probs)}
    return lab2prob.get(label, 0.0)

def predict_restricted(model, word: str, candidate_iso3: List[str]) -> Dict[str, Any]:
    """
    Return best candidate among the provided ISO-3 codes by scoring their
    corresponding fastText labels.
    """
    candidates = []
    for iso3 in candidate_iso3:
        ft_label = iso3_to_fasttext_label(iso3)
        if ft_label:
            candidates.append((iso3, ft_label))

    # If we can't map any candidates, fall back to unrestricted prediction
    if not candidates:
        labels, probs = model.predict(word, k=1)
        if labels:
            return {
                "pred_iso3": None,
                "pred_label": labels[0],
                "scores": {labels[0]: float(probs[0])},
                "note": "no ISO3->ISO2 mapping available for candidates; used unrestricted fastText top-1",
            }
        return {"pred_iso3": None, "pred_label": None, "scores": {}, "note": "no prediction"}

    scores: Dict[str, float] = {}
    best_iso3 = None
    best_label = None
    best_score = -1.0

    for iso3, ft_label in candidates:
        s = score_word_for_label(model, word, ft_label)
        scores[iso3] = s
        if s > best_score:
            best_score = s
            best_iso3 = iso3
            best_label = ft_label

    return {
        "pred_iso3": best_iso3,
        "pred_label": best_label,
        "scores": scores,  # probs per ISO-3 candidate
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Input JSONL")
    ap.add_argument("--output", "-o", required=True, help="Output JSONL")
    ap.add_argument("--fasttext_model", "-m", required=True, help="Path to lid.176.bin or similar")
    ap.add_argument("--store_probs", action="store_true", help="Store per-candidate probs for each word (bigger output)")
    args = ap.parse_args()

    model = fasttext.load_model(args.fasttext_model)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            text = rec.get("text", "")
            lang_field = rec.get("language", "")
            cand_iso3 = iso3_candidates(lang_field)

            words = [w for w in WS_SPLIT.split(text.strip()) if w]
            out_words = []

            for w in words:
                pred = predict_restricted(model, w, cand_iso3)
                entry = {
                    "word": w,
                    "pred_lang": pred.get("pred_iso3"),
                }
                if args.store_probs:
                    entry["scores"] = pred.get("scores", {})
                # If mapping failed, keep label/note for debugging
                if pred.get("pred_iso3") is None:
                    entry["pred_label"] = pred.get("pred_label")
                    if "note" in pred:
                        entry["note"] = pred["note"]

                out_words.append(entry)

            rec["fasttext_word_lid"] = {
                "candidate_langs": cand_iso3,
                "words": out_words,
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
