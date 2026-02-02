#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Tuple, Any, Optional


def read_text_map(path: str) -> Dict[str, str]:
    """
    Kaldi-style: key value...
    Returns {key: "value..."} (rest of line).
    """
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            key = parts[0]
            val = parts[1].strip() if len(parts) > 1 else ""
            if key in out:
                raise ValueError(f"Duplicate key '{key}' in {path} at line {ln}")
            out[key] = val
    return out


def read_wav_scp(path: str) -> Dict[str, str]:
    """
    wav.scp: rec rec_path_or_command...
    We keep the rest of line as the recording 'file_name' (path or command).
    """
    return read_text_map(path)


def read_segments(path: str) -> Dict[str, Tuple[str, float, float]]:
    """
    segments line format:
      utt_id rec start end
    Returns {utt_id: (rec, start, end)}
    """
    out: Dict[str, Tuple[str, float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Expected 4 columns in segments, got {len(parts)} at {path}:{ln}: {line}")
            utt_id, rec, start_s, end_s = parts
            try:
                start = float(start_s)
                end = float(end_s)
            except ValueError:
                raise ValueError(f"Non-float start/end at {path}:{ln}: {line}")
            if utt_id in out:
                raise ValueError(f"Duplicate utt_id '{utt_id}' in {path} at line {ln}")
            out[utt_id] = (rec, start, end)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing segments, text, utt2spk, wav.scp")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--language", default=None, help="Optional: set language field (e.g., arz-eng)")
    ap.add_argument("--require_all", action="store_true",
                    help="Skip utterances missing any field (text/utt2spk/wav.scp/segments)")
    ap.add_argument("--audio_field", default="file_name", help="Output field for recording path (default: file_name)")
    args = ap.parse_args()

    base = os.path.abspath(args.dir)
    seg_path = os.path.join(base, "segments")
    txt_path = os.path.join(base, "text")
    u2s_path = os.path.join(base, "utt2spk")
    wav_path = os.path.join(base, "wav.scp")

    for p in [seg_path, txt_path, u2s_path, wav_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    segs = read_segments(seg_path)           # utt -> (rec, start, end)
    texts = read_text_map(txt_path)          # utt -> text
    utt2spk = read_text_map(u2s_path)        # utt -> spk
    wavscp = read_wav_scp(wav_path)          # rec -> path/command

    utt_ids = sorted(set(segs.keys()) | set(texts.keys()) | set(utt2spk.keys()))

    n_written = 0
    n_skipped = 0
    n_missing_wav = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for utt in utt_ids:
            rec: Optional[str] = None
            start = end = None

            if utt in segs:
                rec, start, end = segs[utt]
            text = texts.get(utt)
            spk = utt2spk.get(utt)

            wav_path_or_cmd = None
            if rec is not None:
                wav_path_or_cmd = wavscp.get(rec)

            missing_any = (
                (utt not in segs) or
                (text is None) or
                (spk is None) or
                (wav_path_or_cmd is None)
            )
            if args.require_all and missing_any:
                n_skipped += 1
                continue

            if rec is not None and wav_path_or_cmd is None:
                n_missing_wav += 1

            obj: Dict[str, Any] = {
                "id": utt,
                "rec": rec,
                args.audio_field: wav_path_or_cmd,
                "audio_start_sec": start,
                "audio_end_sec": end,
                "duration": (end - start) if (start is not None and end is not None) else None,
                "text": text,
                "speaker": spk,
            }
            if args.language is not None:
                obj["language"] = args.language

            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} utterances to {args.out}")
    if n_skipped:
        print(f"Skipped {n_skipped} due to --require_all")
    if n_missing_wav:
        print(f"Warning: {n_missing_wav} utterances had rec IDs not found in wav.scp")


if __name__ == "__main__":
    main()
