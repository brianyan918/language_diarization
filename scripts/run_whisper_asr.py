#!/usr/bin/env python3
import argparse
import json
from typing import Dict, Any, Optional

import torch
from transformers import pipeline
from tqdm import tqdm
import os

# ISO-3 → Whisper language codes
ISO3_TO_WHISPER = {
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
    "zho": "zh",
    "cmn": "zh",
    "jpn": "ja",
    "kor": "ko",
    "tur": "tr",
    "vie": "vi",
    "ukr": "uk",
    "pol": "pl",
    "ben": "bn",
}

def primary_iso3(lang_field: str) -> Optional[str]:
    """
    "deu-eng" → "deu"
    "ara"     → "ara"
    """
    if not lang_field:
        return None
    parts = [p.strip() for p in lang_field.split("-") if p.strip()]
    return parts[0] if parts else None

def iso3_to_whisper(lang_field: str) -> Optional[str]:
    iso3 = primary_iso3(lang_field)
    if iso3 is None:
        return None
    return ISO3_TO_WHISPER.get(iso3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--audio_field", default="file_name")
    ap.add_argument("--text_out_field", default="whisper_pred_text")
    ap.add_argument("--language_field", default="language")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])

    # NEW (minimal): force vs detect
    ap.add_argument(
        "--language_mode",
        default="force",
        choices=["force", "detect"],
        help="force: use language_field to force Whisper language; detect: let Whisper auto-detect language",
    )

    args = ap.parse_args()

    # Device selection for pipeline
    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else -1
    elif args.device == "cuda":
        device = 0
    else:
        device = -1

    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        torch_dtype=torch.float16 if device != -1 else torch.float32,
    )

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as fin, \
        open(args.output, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue

            rec: Dict[str, Any] = json.loads(line)
            audio_path = rec.get(args.audio_field)

            whisper_lang = iso3_to_whisper(rec.get(args.language_field, ""))

            generate_kwargs = {"task": args.task}

            # CHANGED: only force language in "force" mode
            if args.language_mode == "force" and whisper_lang:
                generate_kwargs["language"] = whisper_lang

            try:
                out = asr(audio_path, generate_kwargs=generate_kwargs)
                rec[args.text_out_field] = out["text"]
                rec["whisper_task"] = args.task
                rec["whisper_language_mode"] = args.language_mode
                rec["whisper_forced_language"] = whisper_lang if args.language_mode == "force" else None
            except Exception as e:
                rec[args.text_out_field] = None
                rec["whisper_language_mode"] = args.language_mode
                rec["whisper_error"] = f"{type(e).__name__}: {e}"

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
