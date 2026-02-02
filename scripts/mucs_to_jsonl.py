#!/usr/bin/env python3
"""
prep_mucs.py

Prepare MUCS (Kaldi-style) into:
  <dst>/
    metadata.jsonl
    audio/<uttid>.wav

Input layout (per split_dir):
  <split_dir>/
    transcripts/
      segments   (uttid recid start end)
      text       (uttid <free text...>)
      utt2spk    (uttid spk)
      wav.scp    (recid rec_file_name)
    <wav files live here> (as referenced by wav.scp)

This script:
- reads Kaldi files
- cuts each utterance segment from its recording audio
- writes segmented wav per uttid into <dst>/audio/
- writes one JSONL line per uttid into <dst>/metadata.jsonl

Output JSONL schema (minimal, you can add fields easily):
{
  "id": "<uttid>",
  "file_name": "audio/<uttid>.wav",
  "text": "...",
  "duration": 3.0,
  "language": "<your_lang>",
  "speaker": "<spk>",
  "recid": "<recid>",
  "start": 18.0,
  "end": 21.0
}

Usage:
  python prep_mucs.py \
    --split_dir /path/to/train \
    --dst /path/to/out_train \
    --language ben \
    --audio_ext wav

Notes:
- Uses soundfile to read only the needed frames (no full-file decode if the format supports seek).
- Writes PCM_16 WAV.
- If you have non-wav sources in wav.scp, you can still point to them if libsndfile supports them.
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm


def read_kaldi_map(path: str) -> Dict[str, str]:
    """
    Read Kaldi 'key value...' where value may contain spaces.
    Returns dict[key] = rest_of_line (str).
    Skips empty lines.
    """
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise RuntimeError(f"Bad kaldi map line at {path}:{ln}: {line}")
            out[parts[0]] = parts[1]
    return out


def read_segments(path: str) -> Dict[str, Tuple[str, float, float]]:
    """
    segments: uttid recid start end
    """
    out: Dict[str, Tuple[str, float, float]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) != 4:
                raise RuntimeError(f"Bad segments line at {path}:{ln}: {line}")
            uttid, recid, s, e = cols
            start = float(s)
            end = float(e)
            if end <= start:
                raise RuntimeError(f"Non-positive segment at {path}:{ln}: {line}")
            out[uttid] = (recid, start, end)
    return out


def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def resolve_wav_path(split_dir: str, rec_file: str) -> str:
    # wav.scp gives something like: "072Wvm62KcQqRBNa.wav"
    # wavs are contained in the split dir (per your note)
    p = rec_file
    if not os.path.isabs(p):
        p = os.path.join(split_dir, p)
    return os.path.abspath(p)


def cut_segment_to_wav(src_wav: str, start_sec: float, end_sec: float, dst_wav: str) -> float:
    """
    Cut [start_sec, end_sec] from src_wav and write to dst_wav.
    Returns duration seconds (based on written frames / sr).
    """
    with sf.SoundFile(src_wav, "r") as f:
        sr = f.samplerate
        start_frame = int(round(start_sec * sr))
        end_frame = int(round(end_sec * sr))
        if start_frame < 0:
            start_frame = 0
        if end_frame > len(f):
            end_frame = len(f)
        if end_frame <= start_frame:
            raise RuntimeError(f"Empty after clamping: {src_wav} [{start_sec},{end_sec}]")

        f.seek(start_frame)
        frames = end_frame - start_frame
        audio = f.read(frames, dtype="float32", always_2d=True)

    # Write PCM16 wav
    safe_makedirs(os.path.dirname(dst_wav))
    sf.write(dst_wav, audio, sr, subtype="PCM_16")
    return float(frames) / float(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, help="e.g., /.../mucs/train (contains transcripts/ and wavs)")
    ap.add_argument("--dst", required=True, help="Output dir to create metadata.jsonl and audio/")
    ap.add_argument("--language", required=True, help="Language label to write into metadata (e.g., ben)")
    ap.add_argument("--audio_dirname", default="audio", help="Subdir under dst for segmented wavs")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing segmented wavs")
    args = ap.parse_args()

    split_dir = os.path.abspath(args.split_dir)
    dst = os.path.abspath(args.dst)
    audio_out = os.path.join(dst, args.audio_dirname)
    safe_makedirs(dst)
    safe_makedirs(audio_out)

    tr = os.path.join(split_dir, "transcripts")
    seg_path = os.path.join(tr, "segments")
    txt_path = os.path.join(tr, "text")
    u2s_path = os.path.join(tr, "utt2spk")
    wavscp_path = os.path.join(tr, "wav.scp")

    for p in [seg_path, txt_path, u2s_path, wavscp_path]:
        if not os.path.isfile(p):
            raise RuntimeError(f"Missing required file: {p}")

    segments = read_segments(seg_path)
    text = read_kaldi_map(txt_path)
    utt2spk = read_kaldi_map(u2s_path)
    wavscp = read_kaldi_map(wavscp_path)  # recid -> rec_file_name (may contain spaces but usually doesn't)

    # Resolve recid -> absolute wav path
    rec2wav: Dict[str, str] = {}
    for recid, rec_file in wavscp.items():
        rec2wav[recid] = resolve_wav_path(split_dir, rec_file)

    out_jsonl = os.path.join(dst, "metadata.jsonl")
    n = 0

    # Deterministic order
    uttids = sorted(segments.keys())

    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for uttid in tqdm(uttids, desc="Cutting segments", unit="utt"):
            recid, start, end = segments[uttid]
            if recid not in rec2wav:
                raise RuntimeError(f"recid {recid} not found in wav.scp (uttid={uttid})")
            src_wav = rec2wav[recid]
            if not os.path.isfile(src_wav):
                raise RuntimeError(f"Recording wav missing: {src_wav} (recid={recid})")

            utt_text = text.get(uttid, "")
            spk = utt2spk.get(uttid, "UNK")

            abs_wav = os.path.join(dst, args.audio_dirname, f"{uttid}.wav")
            abs_wav = os.path.abspath(abs_wav)

            if (not args.overwrite) and os.path.isfile(abs_wav):
                info = sf.info(abs_wav)
                dur = float(info.frames) / float(info.samplerate)
            else:
                dur = cut_segment_to_wav(src_wav, start, end, abs_wav)

            meta = {
                "id": uttid,
                "file_name": abs_wav,   # <-- FULL PATH
                "text": utt_text,
                "duration": round(dur, 8),
                "language": args.language,
                "speaker": spk,
                "recid": recid,
                "start": float(start),
                "end": float(end),
            }
            out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            n += 1

    print("=== DONE ===")
    print(f"wrote: {out_jsonl}")
    print(f"audio: {audio_out}")
    print(f"num_utts: {n}")


if __name__ == "__main__":
    main()
