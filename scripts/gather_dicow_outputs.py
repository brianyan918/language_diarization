#!/usr/bin/env python3
import argparse
import fnmatch
import json
import os
from typing import Dict, List, Any


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_json_file(path: str, json_key: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    val = obj
    for k in json_key.split("."):
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            raise KeyError(f"Key '{json_key}' not found in {path}")
    return "" if val is None else str(val).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root output dir")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--pattern", default="*", help="Glob for part filenames (default: *)")
    ap.add_argument(
        "--format",
        choices=["txt", "json"],
        default="txt",
        help="How to read part files",
    )
    ap.add_argument(
        "--json_key",
        default="text",
        help="When --format json, which key to extract",
    )
    ap.add_argument("--skip_empty", action="store_true", help="Skip empty part texts")
    ap.add_argument("--absolute_paths", action="store_true", help="Store absolute paths for parts")
    args = ap.parse_args()

    root = os.path.abspath(args.root)

    with open(args.out, "w", encoding="utf-8") as out_f:
        # <root>/* = languages
        for language in sorted(os.listdir(root)):
            lang_dir = os.path.join(root, language)
            if not os.path.isdir(lang_dir):
                continue

            # <root>/<language>/* = ids
            for sample_id in sorted(os.listdir(lang_dir)):
                id_dir = os.path.join(lang_dir, sample_id)
                if not os.path.isdir(id_dir):
                    continue

                parts: List[Dict[str, Any]] = []

                # <root>/<language>/<id>/* = parts
                for part_name in sorted(os.listdir(id_dir)):
                    if not fnmatch.fnmatch(part_name, args.pattern):
                        continue

                    part_path = os.path.join(id_dir, part_name)
                    if not os.path.isfile(part_path):
                        continue

                    try:
                        if args.format == "txt":
                            text = read_text_file(part_path)
                        else:
                            text = read_json_file(part_path, args.json_key)
                    except Exception as e:
                        parts.append(
                            {
                                "part": part_name,
                                "text": None,
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )
                        continue

                    if args.skip_empty and not text:
                        continue

                    parts.append(
                        {
                            "part": part_name,
                            "text": text,
                            **(
                                {
                                    "path": os.path.abspath(part_path)
                                    if args.absolute_paths
                                    else os.path.relpath(part_path, root)
                                }
                                if args.absolute_paths
                                else {}
                            ),
                        }
                    )

                if not parts:
                    continue

                row = {
                    "language": language,
                    "id": sample_id,
                    "parts": parts,
                }

                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote grouped ASR outputs to {args.out}")


if __name__ == "__main__":
    main()
