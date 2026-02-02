#!/usr/bin/env python3
"""
gather_dicow_outputs.py

Collect DiCoW outputs from the new directory structure:

  <root>/<id>/tcp_wer_hyp.json
  <root>/<id>/ref.json

For each <id>:
- Load both JSON files
- Combine into a single object
- Add:
    - "id"
    - "language" = <id>.split("_")[0] + "-eng"
- Write one JSON object per line (JSONL)
- UTF-8 output (ensure_ascii=False)

Usage:
  python gather_dicow_outputs.py \
    --root /path/to/root \
    --out out.jsonl
"""

import argparse
import json
import os
from typing import Dict


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing <id>/ subdirs")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip ids missing hyp or ref instead of erroring",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)

    if not os.path.isdir(root):
        raise RuntimeError(f"Root is not a directory: {root}")

    ids = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )

    if not ids:
        raise RuntimeError(f"No id directories found under {root}")

    written = 0
    skipped = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for id_ in ids:
            dpath = os.path.join(root, id_)
            hyp_path = os.path.join(dpath, "tcp_wer_hyp.json")
            ref_path = os.path.join(dpath, "ref.json")

            if not os.path.isfile(hyp_path) or not os.path.isfile(ref_path):
                if args.skip_missing:
                    skipped += 1
                    continue
                raise RuntimeError(
                    f"Missing files for id={id_}: "
                    f"{'tcp_wer_hyp.json' if not os.path.isfile(hyp_path) else ''} "
                    f"{'ref.json' if not os.path.isfile(ref_path) else ''}"
                )

            hyp = load_json(hyp_path)
            ref = load_json(ref_path)

            # language: <id>.split("_")[0] + "-eng"
            base_lang = id_.split("_", 1)[0]
            language = f"{base_lang}-eng"

            out_obj = {
                "id": id_,
                "language": language,
                "ref": ref,
                "hyp": hyp,
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

    print("=== DONE ===")
    print(f"Root: {root}")
    print(f"Wrote: {written} examples -> {args.out}")
    if skipped:
        print(f"Skipped (missing files): {skipped}")


if __name__ == "__main__":
    main()
