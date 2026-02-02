#!/usr/bin/env python3
"""
extract_fleurs_tars.py

Extract all <split>.tar.gz files under a FLEURS-style root directory.

Expected layout:
<root_fl_dir>/
  <lang>/
    train.tar.gz
    dev.tar.gz
    test.tar.gz
    train/
    dev/
    test/

Behavior:
- For each language dir:
  - If <split>/ already exists, skip extraction
  - Else if <split>.tar.gz exists, extract it into <lang>/
- Safe to re-run (idempotent)

Usage:
  python extract_fleurs_tars.py --root_fl_dir /path/to/fleurs_root
"""

import argparse
import os
import tarfile

SPLITS = ["train", "dev", "test"]


def is_lang_dir(path: str) -> bool:
    return os.path.isdir(path)


def extract_if_needed(lang_dir: str, split: str):
    split_dir = os.path.join(lang_dir, f"audio/{split}")
    tar_path = os.path.join(lang_dir, f"audio/{split}.tar.gz")
    dst_dir = os.path.join(lang_dir, f"audio")

    if os.path.isdir(split_dir):
        # Already extracted
        return False

    if not os.path.isfile(tar_path):
        # Nothing to extract
        return False

    print(f"[extract] {tar_path} -> {dst_dir}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=dst_dir)

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_fl_dir", required=True, help="Root FLEURS directory (contains language subdirs).")
    args = ap.parse_args()

    root = os.path.abspath(args.root_fl_dir)
    if not os.path.isdir(root):
        raise RuntimeError(f"Not a directory: {root}")

    langs = sorted(
        d for d in os.listdir(root)
        if is_lang_dir(os.path.join(root, d))
    )

    if not langs:
        raise RuntimeError(f"No language directories found under: {root}")

    extracted = 0
    skipped = 0

    for lang in langs:
        lang_dir = os.path.join(root, lang)
        for split in SPLITS:
            did = extract_if_needed(lang_dir, split)
            if did:
                extracted += 1
            else:
                skipped += 1

    print("=== DONE ===")
    print(f"Languages checked: {len(langs)}")
    print(f"Archives extracted: {extracted}")
    print(f"Skipped (already present or missing tar): {skipped}")


if __name__ == "__main__":
    main()
