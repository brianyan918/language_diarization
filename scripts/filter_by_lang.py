#!/usr/bin/env python3
"""
Filter a JSONL manifest by language.

Examples:

Keep only English + Arabic:
    python filter_by_language.py \
        --input metadata.jsonl \
        --output filtered.jsonl \
        --lang-field language \
        --keep eng ara

Remove Spanish:
    python filter_by_language.py \
        --input metadata.jsonl \
        --output filtered.jsonl \
        --lang-field lang \
        --remove spa

Handle code-switched labels like "ara-eng":
    python filter_by_language.py \
        --input metadata.jsonl \
        --output filtered.jsonl \
        --lang-field language \
        --keep ara \
        --split-delimiter -
"""

import argparse
import json
from collections import Counter
from typing import Dict, Any, Iterable, Set, Optional


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")

    parser.add_argument(
        "--lang-field",
        required=True,
        help="Name of the language field in each JSON object"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--keep",
        nargs="+",
        help="List of languages to KEEP"
    )
    group.add_argument(
        "--remove",
        nargs="+",
        help="List of languages to REMOVE"
    )

    parser.add_argument(
        "--split-delimiter",
        default=None,
        help="Optional delimiter for multi-language strings (e.g., '-' for 'ara-eng')"
    )

    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Match languages case-insensitively"
    )

    return parser.parse_args()


def normalize_set(values: Iterable[str], lower: bool) -> Set[str]:
    if lower:
        return {v.lower() for v in values}
    return set(values)


def extract_langs(value: str, delimiter: Optional[str], lower: bool) -> Set[str]:
    if delimiter:
        parts = value.split(delimiter)
    else:
        parts = [value]

    if lower:
        return {p.lower() for p in parts}
    return set(parts)


def main():
    args = parse_args()

    keep_set = normalize_set(args.keep, args.case_insensitive) if args.keep else None
    remove_set = normalize_set(args.remove, args.case_insensitive) if args.remove else None

    total = 0
    kept = 0
    lang_counter = Counter()

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            item: Dict[str, Any] = json.loads(line)

            if args.lang_field not in item:
                raise KeyError(f"Missing language field '{args.lang_field}'")

            lang_value = str(item[args.lang_field])
            langs = extract_langs(
                lang_value,
                args.split_delimiter,
                args.case_insensitive
            )

            keep_example = True

            if keep_set is not None:
                # Keep if ANY language matches
                keep_example = len(langs & keep_set) > 0

            if remove_set is not None:
                # Remove if ANY language matches
                if len(langs & remove_set) > 0:
                    keep_example = False

            if keep_example:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
                for l in langs:
                    lang_counter[l] += 1

    print("===== Filtering Summary =====")
    print(f"Total examples: {total}")
    print(f"Kept examples:  {kept}")
    print(f"Removed:        {total - kept}")
    print("\nLanguage counts in kept data:")
    for lang, count in lang_counter.most_common():
        print(f"{lang}: {count}")


if __name__ == "__main__":
    main()
