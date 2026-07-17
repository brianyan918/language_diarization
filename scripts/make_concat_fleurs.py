#!/usr/bin/env python3
"""
Create concat-fleurs with REAL audio concatenation and STRICT 30-sec cap.

Parallelized version:
- Forms groups serially (deterministic given seed)
- Processes each concatenation in parallel (audio read + concat + write)
- Uses per-example deterministic RNG seeds so results are reproducible
  regardless of worker count / scheduling.

Features:
- 2-utt or 30-sec mode
- Strict duration cap including silence (silences are sampled during grouping)
- Without replacement
- Physically concatenates wav files
- Inserts random silence (< max_silence_sec)
- tqdm progress bar
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm
import concurrent.futures as cf
from functools import partial


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Audio helpers
# ------------------------------------------------------------

def load_audio(path: str):
    path = os.path.abspath(path)
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # mono
    return audio.astype(np.float32), sr


def silence(sr: int, seconds: float):
    length = int(round(sr * seconds))
    return np.zeros(length, dtype=np.float32)


# ------------------------------------------------------------
# Grouping logic (STRICT 30-sec) with silences sampled now
# ------------------------------------------------------------

def form_groups_strict_with_silences(
    items: List[Dict[str, Any]],
    mode: str,
    rng: random.Random,
    target_sec: float,
    max_silence_sec: float,
) -> List[Dict[str, Any]]:
    """
    Returns a list of group dicts:
      {"utts": [utt0, utt1, ...], "silences": [sil0, sil1, ...]}  # silences length = len(utts)-1

    STRICT:
      In 30-sec mode, we ensure sum(durations) + sum(sampled_silences) <= target_sec.
    """
    items = list(items)
    rng.shuffle(items)
    groups: List[Dict[str, Any]] = []

    if mode == "2-utt":
        i = 0
        while i + 1 < len(items):
            # still insert random silence between the two
            sil = rng.uniform(0.0, max_silence_sec)
            groups.append({"utts": [items[i], items[i + 1]], "silences": [sil]})
            i += 2
        return groups

    if mode == "30-sec":
        i = 0
        while i < len(items):
            utts: List[Dict[str, Any]] = []
            silences: List[float] = []
            cur_total = 0.0

            while i < len(items):
                utt = items[i]
                utt_dur = float(utt["duration"])

                if not utts:
                    # Always allow first utterance even if > target
                    utts.append(utt)
                    cur_total += utt_dur
                    i += 1
                    continue

                sil = rng.uniform(0.0, max_silence_sec)
                proposed = cur_total + sil + utt_dur
                if proposed > target_sec:
                    break

                silences.append(sil)
                utts.append(utt)
                cur_total = proposed
                i += 1

            if utts:
                groups.append({"utts": utts, "silences": silences})

        return groups

    raise ValueError("Unknown mode")


# ------------------------------------------------------------
# Build concat example (worker-safe)
# ------------------------------------------------------------

def build_example_from_group(
    group: Dict[str, Any],
    new_id: str,
    out_audio_dir: str,
) -> Dict[str, Any]:
    """
    group = {"utts": [...], "silences": [...]}
    """
    utts: List[Dict[str, Any]] = group["utts"]
    silences: List[float] = group["silences"]

    segments_out: List[Dict[str, Any]] = []
    audio_chunks: List[np.ndarray] = []
    cur_time = 0.0

    source_ids: List[str] = []
    languages: List[str] = []
    sr_global: Optional[int] = None

    for idx, utt in enumerate(utts):
        audio, sr = load_audio(utt["file_name"])
        if sr_global is None:
            sr_global = sr
        else:
            if sr != sr_global:
                raise RuntimeError(f"Sampling rate mismatch in group {new_id}: {sr} vs {sr_global}")

        if idx > 0:
            sil_sec = float(silences[idx - 1])
            audio_chunks.append(silence(sr, sil_sec))
            cur_time += sil_sec

        audio_chunks.append(audio)

        # Adjust segments
        for seg in utt["segments"]:
            seg_new = dict(seg)
            seg_new["audio_start_sec"] = float(seg["audio_start_sec"]) + cur_time
            seg_new["audio_end_sec"] = float(seg["audio_end_sec"]) + cur_time
            seg_new["source_utt_id"] = utt["id"]
            segments_out.append(seg_new)

        cur_time += float(utt["duration"])
        source_ids.append(utt["id"])
        languages.append(utt["language"])

    assert sr_global is not None

    full_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
    duration = float(len(full_audio) / sr_global)

    os.makedirs(out_audio_dir, exist_ok=True)
    out_wav = os.path.abspath(os.path.join(out_audio_dir, f"{new_id}.wav"))
    sf.write(out_wav, full_audio, sr_global)

    # stable unique list in order
    langs_unique = list(dict.fromkeys(languages))

    return {
        "id": new_id,
        "file_name": out_wav,
        "duration": duration,
        "text": " ".join([x.get("text", "") for x in utts]).strip(),
        "text_norm": " ".join([x.get("text_norm", "") for x in utts]).strip(),
        "languages": langs_unique,
        "source_ids": source_ids,
        "segments": segments_out,
    }


def _worker(job: Tuple[int, Dict[str, Any], str, str]) -> Tuple[int, Dict[str, Any]]:
    """
    job = (index, group, new_id, out_audio_dir)
    Returns (index, example) so we can restore order deterministically.
    """
    idx, group, new_id, out_audio_dir = job
    ex = build_example_from_group(group, new_id, out_audio_dir)
    return idx, ex


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-audio-dir", required=True)

    p.add_argument("--mode", choices=["2-utt", "30-sec"], required=True)
    p.add_argument("--target-sec", type=float, default=30.0)
    p.add_argument("--max-silence-sec", type=float, default=1.5)

    p.add_argument("--seed", type=int, default=0)

    # parallelism
    p.add_argument("--num-workers", type=int, default=4,
                   help="Number of parallel workers for audio concat/writes.")
    p.add_argument("--backend", choices=["process", "thread"], default="process",
                   help="process is usually faster/safer for CPU+IO; thread can be okay for pure IO.")
    p.add_argument("--chunksize", type=int, default=1,
                   help="Task chunksize for executor.map (process backend benefits from >1 sometimes).")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    args.output_audio_dir = os.path.abspath(args.output_audio_dir)
    args.output_jsonl = os.path.abspath(args.output_jsonl)

    # Ensure output dirs exist
    os.makedirs(args.output_audio_dir, exist_ok=True)
    out_jsonl_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
    os.makedirs(out_jsonl_dir, exist_ok=True)

    items = list(read_jsonl(args.input))

    # Form groups serially (deterministic) AND sample silences here (strict cap includes silences)
    groups = form_groups_strict_with_silences(
        items,
        args.mode,
        rng,
        args.target_sec,
        args.max_silence_sec,
    )

    # Prepare jobs with deterministic ids
    jobs: List[Tuple[int, Dict[str, Any], str, str]] = []
    for i, group in enumerate(groups):
        new_id = f"concat_{i:08d}"
        jobs.append((i, group, new_id, args.output_audio_dir))

    outputs: List[Optional[Dict[str, Any]]] = [None] * len(jobs)

    Executor = cf.ProcessPoolExecutor if args.backend == "process" else cf.ThreadPoolExecutor

    with Executor(max_workers=args.num_workers) as ex:
        # executor.map preserves input order, but we also return idx to be extra safe
        it = ex.map(_worker, jobs, chunksize=max(1, args.chunksize))
        for idx, example in tqdm(it, total=len(jobs), desc="Building concatenations"):
            outputs[idx] = example

    # Sanity check
    if any(x is None for x in outputs):
        raise RuntimeError("Some outputs were not produced (unexpected None).")

    write_jsonl(args.output_jsonl, outputs)  # type: ignore[arg-type]

    print("Done.")
    print("Input utts:", len(items))
    print("Output concats:", len(outputs))
    print("Workers:", args.num_workers, "Backend:", args.backend)


if __name__ == "__main__":
    main()
