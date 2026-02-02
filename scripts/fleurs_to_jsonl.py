#!/usr/bin/env python3
"""
fleurs_to_jsonl_local.py

Convert an already-downloaded FLEURS-style directory into per-split metadata.jsonl.

Expected on-disk layout:

<root_fl_dir>/
  <lang>/
    train.tsv
    dev.tsv
    test.tsv
    train/            (may or may not exist yet)
      audio/<wavname>
    dev/
      audio/<wavname>
    test/
      audio/<wavname>
    train.tar.gz      (optional; extract if train/ missing)
    dev.tar.gz        (optional; extract if dev/ missing)
    test.tar.gz       (optional; extract if test/ missing)

TSV columns (tab-separated):
  rawid, wavname, text, text_norm, text_char, spk, gender

Output JSONL schema per line:
{
  "id": "<lang>_<rawid>_<spk>",
  "rawid": "<rawid>",
  "wavname": "<wavname>",
  "file_name": "<root_fl_dir>/<lang>/<split>/audio/<wavname>",
  "text": "<text>",
  "text_norm": "<text_norm>",
  "text_char": "<text_char>",
  "speaker": "<spk>",
  "gender": "<gender>",
  "language": "<lang>",
  "split": "<split>"
}

Writes:
  <out_dir>/
    train/metadata.jsonl
    dev/metadata.jsonl
    test/metadata.jsonl

Usage:
  python fleurs_to_jsonl_local.py \
    --root_fl_dir /path/to/fleurs_root \
    --out_dir /path/to/out

Notes:
- If <root_fl_dir>/<lang>/<split>/ does not exist, we try to extract:
    <root_fl_dir>/<lang>/<split>.tar.gz
  into <root_fl_dir>/<lang>/.
- We do NOT modify audio; we just point file_name to where it should be.
"""

import argparse
import csv
import json
import os
import tarfile
from typing import Dict, List, Optional, Tuple


SPLITS = ["train", "dev", "test"]


def list_lang_dirs(root_fl_dir: str) -> List[str]:
    langs = []
    for name in os.listdir(root_fl_dir):
        p = os.path.join(root_fl_dir, name)
        if os.path.isdir(p):
            langs.append(name)
    return sorted(langs)


def ensure_split_extracted(root_fl_dir: str, lang: str, split: str) -> str:
    """
    Ensure <root_fl_dir>/<lang>/<split>/ exists.
    If not, try extracting <root_fl_dir>/<lang>/<split>.tar.gz into <root_fl_dir>/<lang>/.
    Return the split directory path.
    """
    lang_dir = os.path.join(root_fl_dir, lang)
    split_dir = os.path.join(lang_dir, split)

    if os.path.isdir(split_dir):
        return split_dir

    tgz = os.path.join(lang_dir, f"{split}.tar.gz")
    if not os.path.isfile(tgz):
        # Nothing to extract; just return expected path.
        return split_dir

    print(f"[extract] Missing {split_dir}, extracting {tgz} -> {lang_dir}")
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(path=lang_dir)

    return split_dir


def tsv_path(root_fl_dir: str, lang: str, split: str) -> str:
    return os.path.join(root_fl_dir, lang, f"{split}.tsv")


def read_tsv_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue

            # Split into exactly 7 fields:
            # rawid, wavname, text, text_norm, text_char, spk, gender
            parts = line.split("\t", maxsplit=6)

            if len(parts) != 7:
                raise RuntimeError(
                    f"Bad TSV row (expected 7 cols after split) at {path}:{ln}: {parts}"
                )

            rows.append(parts)
    return rows



def make_out_writers(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    writers = {}
    for split in SPLITS:
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        writers[split] = open(os.path.join(split_dir, "metadata.jsonl"), "w", encoding="utf-8")
    return writers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_fl_dir", required=True, help="Root directory containing 102 language subdirs.")
    ap.add_argument("--out_dir", required=True, help="Where to write per-split metadata.jsonl files.")
    ap.add_argument("--require_audio", action="store_true",
                    help="If set, error if referenced audio file does not exist after extraction attempt.")
    ap.add_argument("--audio_subdir", default="audio",
                    help="Audio directory name under each split dir (default: audio).")
    args = ap.parse_args()

    root_fl_dir = os.path.abspath(args.root_fl_dir)
    out_dir = os.path.abspath(args.out_dir)

    langs = list_lang_dirs(root_fl_dir)
    if not langs:
        raise RuntimeError(f"No language directories found under: {root_fl_dir}")

    writers = make_out_writers(out_dir)

    total = 0
    missing_tsv = 0
    missing_audio = 0

    try:
        for lang in langs:
            for split in SPLITS:
                tsv = tsv_path(root_fl_dir, lang, split)
                if not os.path.isfile(tsv):
                    missing_tsv += 1
                    continue

                # Ensure audio dir exists (extract if needed)
                split_dir = ensure_split_extracted(root_fl_dir, lang, split)
                audio_dir = os.path.join(split_dir, args.audio_subdir)

                rows = read_tsv_rows(tsv)
                for row in rows:
                    rawid, wavname, text, text_norm, text_char, spk, gender = row

                    utt_id = f"{lang}_{rawid}_{spk}"
                    wav_path = os.path.join(root_fl_dir, lang, args.audio_subdir, split, wavname)
                    wav_path = os.path.abspath(wav_path)

                    if args.require_audio and not os.path.isfile(wav_path):
                        missing_audio += 1
                        raise RuntimeError(f"Missing audio: {wav_path}")

                    meta = {
                        "id": utt_id,
                        "rawid": rawid,
                        "wavname": wavname,
                        "file_name": wav_path,
                        "text": text,
                        "text_norm": text_norm,
                        "text_char": text_char,
                        "speaker": spk,
                        "gender": gender,
                        "language": lang,
                        "split": split,
                    }
                    writers[split].write(json.dumps(meta, ensure_ascii=False) + "\n")
                    total += 1

        print("=== DONE ===")
        print(f"root_fl_dir: {root_fl_dir}")
        print(f"out_dir:     {out_dir}")
        print(f"langs:       {len(langs)}")
        print(f"total_utts:  {total}")
        if missing_tsv:
            print(f"missing_tsv_files: {missing_tsv} (some langs may not have all splits)")
        if missing_audio:
            print(f"missing_audio: {missing_audio}")

        for split in SPLITS:
            print(f"wrote: {os.path.join(out_dir, split, 'metadata.jsonl')}")

    finally:
        for f in writers.values():
            f.close()


if __name__ == "__main__":
    main()
