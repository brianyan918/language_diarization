#!/usr/bin/env python3
import argparse
import os

def split_file(input_path, num_parts, output_postfix):
    # Read all lines
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    chunk_size = (total + num_parts - 1) // num_parts  # ceiling division

    for i in range(num_parts):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunk = lines[start:end]

        if not chunk:
            break

        out_path = f"{input_path}.{output_postfix}_{i}"
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.writelines(chunk)

        print(f"wrote {out_path}: {len(chunk)} lines")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input text file")
    parser.add_argument("--num_parts", "-n", type=int, required=True, help="Number of output parts")
    parser.add_argument("--output_postfix", "-o", default="split", help="Postfix for file names")
    args = parser.parse_args()

    split_file(args.input, args.num_parts, args.output_postfix)

if __name__ == "__main__":
    main()
