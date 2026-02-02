#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Set, Optional


def read_rec_list(path: str) -> Set[str]:
    """
    Reads a file containing rec IDs, one per line.
    Ignores empty lines and comment lines starting with '#'.
    """
    recs: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # allow extra columns, take first token as rec id
            rec = s.split()[0]
            recs.add(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, help="Dir containing train/dev/test files of rec IDs")
    ap.add_argument("--manifest", required=True, help="Input JSONL manifest with a 'rec' field")
    ap.add_argument("--out_dir", required=True, help="Output dir; will create <split>/metadata.jsonl")
    ap.add_argument("--rec_field", default="rec", help="Field name in JSONL for recording id (default: rec)")
    ap.add_argument("--write_missing", action="store_true", help="Write unmatched lines to missing/metadata.jsonl")
    ap.add_argument("--error_on_overlap", action="store_true",
                    help="Error if a rec id appears in more than one split")
    args = ap.parse_args()

    split_dir = os.path.abspath(args.split_dir)
    out_dir = os.path.abspath(args.out_dir)

    split_files = {
        "train": os.path.join(split_dir, "train"),
        "dev": os.path.join(split_dir, "dev"),
        "test": os.path.join(split_dir, "test"),
    }

    for name, p in split_files.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}")

    split_recs: Dict[str, Set[str]] = {k: read_rec_list(v) for k, v in split_files.items()}

    # Optional overlap check
    if args.error_on_overlap:
        seen: Dict[str, str] = {}
        for split, recs in split_recs.items():
            for r in recs:
                if r in seen:
                    raise ValueError(f"rec '{r}' appears in both '{seen[r]}' and '{split}'")
                seen[r] = split

    # Create output dirs and open writers
    os.makedirs(out_dir, exist_ok=True)
    writers = {}
    counts = {k: 0 for k in ["train", "dev", "test"]}
    missing_count = 0

    for split in ["train", "dev", "test"]:
        split_out = os.path.join(out_dir, split)
        os.makedirs(split_out, exist_ok=True)
        writers[split] = open(os.path.join(split_out, "metadata.jsonl"), "w", encoding="utf-8")

    missing_writer = None
    if args.write_missing:
        miss_out = os.path.join(out_dir, "missing")
        os.makedirs(miss_out, exist_ok=True)
        missing_writer = open(os.path.join(miss_out, "metadata.jsonl"), "w", encoding="utf-8")

    def assign_split(rec_id: str) -> Optional[str]:
        for split in ["train", "dev", "test"]:
            if rec_id in split_recs[split]:
                return split
        return None

    # Stream manifest and write to split file
    with open(args.manifest, "r", encoding="utf-8") as fin:
        for ln, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec_id = obj.get(args.rec_field)
            if rec_id is None:
                raise ValueError(f"Line {ln}: missing '{args.rec_field}' in manifest")

            rec_id = str(rec_id)
            split = assign_split(rec_id)

            if split is None:
                if missing_writer is not None:
                    missing_writer.write(json.dumps(obj, ensure_ascii=False) + "\n")
                missing_count += 1
            else:
                writers[split].write(json.dumps(obj, ensure_ascii=False) + "\n")
                counts[split] += 1

    # Close writers
    for w in writers.values():
        w.close()
    if missing_writer is not None:
        missing_writer.close()

    print("Done.")
    print(f"train:   {counts['train']}")
    print(f"dev:     {counts['dev']}")
    print(f"test:    {counts['test']}")
    if args.write_missing:
        print(f"missing: {missing_count}  (written to {os.path.join(out_dir, 'missing', 'metadata.jsonl')})")
    else:
        print(f"missing: {missing_count}  (not written; use --write_missing)")


if __name__ == "__main__":
    main()
