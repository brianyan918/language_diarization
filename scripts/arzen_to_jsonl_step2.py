#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from typing import Dict, Any

# splits audio into segments

def run_ffmpeg(in_wav: str, start: float, end: float, out_wav: str):
    """
    Cut [start, end] seconds from in_wav and write to out_wav.
    Preserves original sample rate and channels.
    """
    dur = max(0.0, end - start)

    cmd = [
        "ffmpeg",
        "-y",                     # overwrite
        "-loglevel", "error",
        "-ss", f"{start:.6f}",
        "-t", f"{dur:.6f}",
        "-i", in_wav,
        "-acodec", "pcm_s16le",   # keep PCM WAV
        out_wav,
    ]

    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input JSONL")
    ap.add_argument("-o", "--output", required=True, help="Output JSONL (updated)")
    ap.add_argument("--out_dir", required=True, help="Directory for cut wavs")
    ap.add_argument("--skip_existing", action="store_true", help="Skip cutting if wav already exists")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for ln, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            rec: Dict[str, Any] = json.loads(line)

            utt_id = rec.get("id")
            in_wav = rec.get("file_name")
            start = rec.get("audio_start_sec")
            end = rec.get("audio_end_sec")

            if utt_id is None or in_wav is None or start is None or end is None:
                raise ValueError(f"Missing required fields at line {ln}")

            out_wav = os.path.join(args.out_dir, f"{utt_id}.wav")

            if not (args.skip_existing and os.path.exists(out_wav)):
                try:
                    run_ffmpeg(in_wav, float(start), float(end), out_wav)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"ffmpeg failed for id={utt_id} at line {ln}") from e

            # ---- update JSON object ----
            rec["file_name"] = out_wav

            # remove segmentation / recording-level metadata
            for k in ["audio_start_sec", "audio_end_sec"]:
                rec.pop(k, None)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done.")
    print(f"Cut audio written to: {args.out_dir}")
    print(f"Updated JSONL written to: {args.output}")


if __name__ == "__main__":
    main()
