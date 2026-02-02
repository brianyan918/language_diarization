#!/usr/bin/env python3
"""
fleurs_jsonl_to_segments.py (PARALLEL)

Parallelizes audio-duration probing (sf.info) + segment construction.

Design:
- Producer: reads JSONL lines and submits work items (line index, src, ln, obj) to a ProcessPool.
- Workers: compute ISO3 + resolve path + read duration from WAV header + return (idx, out_line_json, stats_flags).
- Writer: writes results in ORIGINAL INPUT ORDER using an in-memory reorder buffer.

Why process pool?
- sf.info hits the filesystem a lot; parallelism helps on network filesystems / many small files.

Usage:
  python fleurs_jsonl_to_segments.py \
    --input_jsonl fleurs_formatted/train/metadata.jsonl \
    --out_jsonl fleurs_formatted/train/metadata.segments.jsonl \
    --num_workers 16

Or:
  python fleurs_jsonl_to_segments.py \
    --input_jsonl_glob "/path/to/*.jsonl" \
    --out_jsonl out.segments.jsonl \
    --num_workers 16

Notes:
- Keeps file_name absolute in output (same as your script).
- Fails fast if an audio path is missing.
"""

import argparse
import glob
import json
import os
from typing import Dict, Iterator, Optional, Tuple, List

import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed

# Optional dependency (recommended)
try:
    import pycountry  # type: ignore
except Exception:
    pycountry = None


ISO1_TO_ISO3_FALLBACK = {
    "am": "amh",
    "ar": "ara",
    "cs": "ces",
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "hi": "hin",
    "hu": "hun",
    "it": "ita",
    "ja": "jpn",
    "ko": "kor",
    "nl": "nld",
    "pl": "pol",
    "pt": "por",
    "ru": "rus",
    "sk": "slk",
    "te": "tel",
    "tr": "tur",
    # NOTE: zh ISO-639-3 is "zho". If you want Mandarin specifically, use "cmn".
    "zh": "zho",
}


def iter_jsonl_inputs(jsonl_path: Optional[str], jsonl_glob: Optional[str]) -> Iterator[Tuple[str, int, dict]]:
    if (jsonl_path is None) == (jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")

    paths = [jsonl_path] if jsonl_path else sorted(glob.glob(jsonl_glob or ""))
    if not paths:
        raise RuntimeError("No JSONL files found")

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield p, ln, json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"JSON decode error in {p}:{ln}: {e}") from e


def fleurs_code_to_iso1(lang_code: str) -> str:
    if not lang_code:
        return ""
    return lang_code.split("_", 1)[0].lower()


def iso1_to_iso3(iso1: str) -> str:
    iso1 = (iso1 or "").lower().strip()
    if not iso1:
        return ""

    if pycountry is not None:
        lang = pycountry.languages.get(alpha_2=iso1)
        if lang is not None and getattr(lang, "alpha_3", None):
            return str(lang.alpha_3)

    if iso1 in ISO1_TO_ISO3_FALLBACK:
        return ISO1_TO_ISO3_FALLBACK[iso1]

    # Fail-soft (keeps behavior similar to your original), but you can hard-fail if you prefer:
    return iso1


def audio_duration_seconds(path: str) -> float:
    info = sf.info(path)
    if info.samplerate <= 0:
        raise RuntimeError(f"Bad samplerate in audio header: {path}")
    return float(info.frames) / float(info.samplerate)


def _process_one(idx: int, src: str, ln: int, obj: dict, root: str, warn_unknown_lang: bool):
    """
    Worker function (must be top-level for multiprocessing).
    Returns:
      idx, out_line(str), missing_audio(bool), unknown_lang(bool), warn_msg(Optional[str])
    """
    lang_code = str(obj.get("language", "")).strip()
    iso1 = fleurs_code_to_iso1(lang_code)
    iso3 = iso1_to_iso3(iso1)

    unknown_lang = False
    warn_msg = None
    if iso3 == iso1 and iso1 not in ISO1_TO_ISO3_FALLBACK and pycountry is None:
        unknown_lang = True
        if warn_unknown_lang:
            warn_msg = f"[WARN] Unknown ISO1->ISO3 mapping (pycountry missing). language={lang_code} at {src}:{ln}"

    wav_path = str(obj.get("file_name", "")).strip()
    if not wav_path:
        raise RuntimeError(f"Missing file_name at {src}:{ln}")

    if not os.path.isabs(wav_path):
        if not root:
            raise RuntimeError(f"Relative file_name but --root_fl_dir not provided: {wav_path} at {src}:{ln}")
        wav_path = os.path.join(root, wav_path)

    wav_path = os.path.abspath(wav_path)
    if not os.path.isfile(wav_path):
        raise RuntimeError(f"Audio file not found: {wav_path} (from {src}:{ln})")

    dur = audio_duration_seconds(wav_path)

    text = str(obj.get("text", ""))
    norm = str(obj.get("text_norm", text))

    seg = {
        "audio_start_sec": 0.0,
        "audio_end_sec": dur,
        "duration": dur,
        "text": text,
        "normalized_text": norm,
        "lang": iso3,
    }

    out_obj = dict(obj)
    out_obj["file_name"] = wav_path
    out_obj["duration"] = round(dur, 8)
    out_obj["language_iso3"] = iso3
    out_obj["segments"] = [seg]

    out_line = json.dumps(out_obj, ensure_ascii=False)
    return idx, out_line, unknown_lang, warn_msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", help="Single input JSONL")
    ap.add_argument("--input_jsonl_glob", help="Glob of input JSONLs")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL with single full-span segment")
    ap.add_argument("--root_fl_dir", default="", help="Optional: resolve relative file_name under this root")
    ap.add_argument("--warn_unknown_lang", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8, help="Process count for parallel duration probing")
    ap.add_argument("--max_in_flight", type=int, default=2000, help="Bound memory: max submitted tasks pending results")
    args = ap.parse_args()

    root = os.path.abspath(args.root_fl_dir) if args.root_fl_dir else ""

    n_read = 0
    unknown_lang_count = 0

    # We'll preserve input order by writing in increasing idx.
    next_to_write = 0
    buffer: Dict[int, str] = {}

    # To keep memory bounded, we submit up to max_in_flight tasks, then drain.
    futures = []
    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as ex:

            def drain_some(block: bool = True):
                nonlocal next_to_write, unknown_lang_count
                if not futures:
                    return
                if block:
                    done_iter = as_completed(list(futures))
                else:
                    # non-blocking-ish: collect those already done
                    done_iter = [fu for fu in list(futures) if fu.done()]

                for fu in done_iter:
                    if not fu.done():
                        continue
                    futures.remove(fu)
                    idx, out_line, unknown_lang, warn_msg = fu.result()
                    if unknown_lang:
                        unknown_lang_count += 1
                        if warn_msg:
                            print(warn_msg)

                    buffer[idx] = out_line

                    # flush contiguous ready lines
                    while next_to_write in buffer:
                        out_f.write(buffer.pop(next_to_write) + "\n")
                        next_to_write += 1

                    # if we were non-blocking, don't loop too long
                    if not block:
                        break

            for src, ln, obj in iter_jsonl_inputs(args.input_jsonl, args.input_jsonl_glob):
                idx = n_read
                n_read += 1

                futures.append(ex.submit(_process_one, idx, src, ln, obj, root, args.warn_unknown_lang))

                # Keep at most max_in_flight tasks pending to bound RAM
                if len(futures) >= args.max_in_flight:
                    drain_some(block=True)

            # Drain remaining
            while futures:
                drain_some(block=True)

            # Final flush (should be empty)
            while next_to_write in buffer:
                out_f.write(buffer.pop(next_to_write) + "\n")
                next_to_write += 1

    print("=== DONE ===")
    print(f"Read examples: {n_read}")
    print(f"Wrote: {args.out_jsonl}")
    if unknown_lang_count:
        print(f"Unknown ISO mapping count: {unknown_lang_count} (consider installing pycountry)")


if __name__ == "__main__":
    main()
