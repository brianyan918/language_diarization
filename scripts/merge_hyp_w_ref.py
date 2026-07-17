#!/usr/bin/env python3
import argparse
import json
import re
from typing import Dict, Any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Source JSONL (contains id + text)")
    ap.add_argument("--target", required=True, help="Target JSONL to update")
    ap.add_argument("--output", required=True, help="Output JSONL")
    ap.add_argument("--text_field", default="text", help="Field name to copy (default: text)")
    ap.add_argument("--id_field", default="id", help="ID field name (default: id)")
    ap.add_argument(
        "--on_missing",
        choices=["keep", "empty", "error"],
        default="keep",
        help="What to do if id not found in source (default: keep target as-is)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite text even if target already has it",
    )
    ap.add_argument(
        "--ignore_id_prefix",
        action="store_true",
    )
    args = ap.parse_args()

    # Load source id -> text map
    src_text_by_id: Dict[str, Any] = {}
    with open(args.source, "r", encoding="utf-8") as sf:
        for line in sf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if args.id_field in rec and args.text_field in rec:
                src_text_by_id[str(rec[args.id_field])] = rec[args.text_field]

    # Process target
    with open(args.target, "r", encoding="utf-8") as tf, \
         open(args.output, "w", encoding="utf-8") as out_f:

        for line in tf:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            rid = str(rec.get(args.id_field))
            if args.ignore_id_prefix:
                # Strip pattern like "test-0-" or "test-123-"
                rid = re.sub(r'^[a-z]+-\d+-', '', rid)

            has_text = args.text_field in rec and rec[args.text_field] not in (None, "")

            if rid in src_text_by_id:
                if args.overwrite or not has_text:
                    rec[args.text_field] = src_text_by_id[rid]
            else:
                if args.on_missing == "empty":
                    rec[args.text_field] = ""
                elif args.on_missing == "error":
                    raise KeyError(f"id '{rid}' not found in source")

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
