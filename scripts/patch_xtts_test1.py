#!/usr/bin/env python3
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str)
    parser.add_argument("--raw_text", type=str)
    args = parser.parse_args()

    lookup = {}
    for line in open(args.raw_text, "r").readlines():
        line = line.strip()
        lookup[line.replace("**", "")] = line

    patched = []
    for line in open(args.manifest, "r").readlines():
        example = json.loads(line)
        example["text"] = lookup[example["text"]]
        patched.append(json.dumps(example, ensure_ascii=False)+"\n")

    with open(args.manifest+".patch", "w") as f:
        f.writelines(patched)

if __name__ == "__main__":
    main()
