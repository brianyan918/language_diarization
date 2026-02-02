#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Optional


def read_kaldi_map(path: str) -> Dict[str, str]:
    """
    Reads a Kaldi-style mapping where:
      key value...
    Returns dict[key] = "value..." (rest of line, stripped).
    """
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                key, val = parts[0], ""
            else:
                key, val = parts[0], parts[1]
            if key in out:
                raise ValueError(f"Duplicate key '{key}' in {path} at line {ln}")
            out[key] = val.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaldi_dir", required=True, help="Dir containing wav.scp, text, utt2spk")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--language", required=True, help="Language tag to set in every line (e.g., cmn-eng)")
    ap.add_argument("--require_all", action="store_true",
                    help="If set, skip any utterance missing wav/text/utt2spk (default: allow missing fields)")
    ap.add_argument("--wav_field", default="file_name", help="Output field name for wav.scp (default: file_name)")
    ap.add_argument("--text_field", default="text", help="Output field name for text (default: text)")
    ap.add_argument("--spk_field", default="speaker", help="Output field name for utt2spk (default: speaker)")
    args = ap.parse_args()

    kdir = os.path.abspath(args.kaldi_dir)
    wav_path = os.path.join(kdir, "wav.scp")
    txt_path = os.path.join(kdir, "text")
    u2s_path = os.path.join(kdir, "utt2spk")

    for p in [wav_path, txt_path, u2s_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    wav = read_kaldi_map(wav_path)
    txt = read_kaldi_map(txt_path)
    u2s = read_kaldi_map(u2s_path)

    # union of all ids encountered
    all_ids = sorted(set(wav.keys()) | set(txt.keys()) | set(u2s.keys()))

    n_written = 0
    n_skipped = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for uid in all_ids:
            has_w = uid in wav
            has_t = uid in txt
            has_s = uid in u2s

            if args.require_all and not (has_w and has_t and has_s):
                n_skipped += 1
                continue

            obj = {
                "id": uid,
                "language": args.language,
            }
            if has_w:
                obj[args.wav_field] = wav[uid]
            else:
                obj[args.wav_field] = None

            if has_t:
                obj[args.text_field] = txt[uid]
            else:
                obj[args.text_field] = None

            if has_s:
                obj[args.spk_field] = u2s[uid]
            else:
                obj[args.spk_field] = None

            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} JSONL lines to {args.out}")
    if n_skipped:
        print(f"Skipped {n_skipped} (missing one or more fields due to --require_all)")


if __name__ == "__main__":
    main()
