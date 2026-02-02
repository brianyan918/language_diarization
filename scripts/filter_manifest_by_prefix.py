#!/usr/bin/env python3
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input JSONL")
    ap.add_argument("-o", "--output", required=True, help="Output JSONL")
    ap.add_argument(
        "--field",
        default="id",
        help="Field to check prefix on (default: id)",
    )
    ap.add_argument(
        "--prefixes",
        nargs="+",
        default=["sp0.9", "sp1.1"],
        help="Prefixes to filter on (default: sp0.9 sp1.1)",
    )
    ap.add_argument(
        "--mode",
        choices=["keep", "drop"],
        default="keep",
        help="keep = keep matching lines, drop = drop matching lines",
    )
    args = ap.parse_args()

    prefixes = tuple(args.prefixes)

    kept = 0
    dropped = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for ln, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            val = obj.get(args.field, "")
            val = "" if val is None else str(val)

            matches = val.startswith(prefixes)

            if (matches and args.mode == "keep") or (not matches and args.mode == "drop"):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1

    print(f"Done.")
    print(f"Kept:    {kept}")
    print(f"Dropped: {dropped}")


if __name__ == "__main__":
    main()
