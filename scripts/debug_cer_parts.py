#!/usr/bin/env python3
"""
debug_cer_parts.py

Debug script to trace through the CER parts scoring pipeline step-by-step.
Shows all intermediate transformations for a single example.
"""

import re
import unicodedata
from typing import Tuple, Dict

BOLD_MARK_RE = re.compile(r"\*\*")
WS_RE = re.compile(r"\s+")
EPS_SYM = "<eps>"


def strip_unicode_punct(s: str) -> str:
    """Remove Unicode punctuation characters."""
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def normalize_text(text: str, *, ignore_ws: bool, remove_punct: bool, lower: bool) -> str:
    """Normalize text for scoring."""
    if remove_punct:
        text = strip_unicode_punct(text)

    if ignore_ws:
        text = WS_RE.sub("", text)

    if lower:
        text = text.lower()

    return text


def extract_eng_and_matrix(ref_text: str) -> Tuple[str, str]:
    """
    Extract ENG part (inside **...**) and MATRIX part (rest) from ref_text.
    Returns (eng_part, matrix_part)
    
    IMPORTANT:
    - eng_part: text inside ** markers (concatenated with spaces)
    - matrix_part: text with ** markers AND their contents removed
    """
    # Find text inside ** markers
    eng_matches = re.findall(r"\*\*([^*]*)\*\*", ref_text)
    eng_part = " ".join(eng_matches)

    # Remove ** markers AND their contents to get matrix part
    # Replace **...** with nothing (not just the markers)
    matrix_part = re.sub(r"\*\*[^*]*\*\*", "", ref_text)
    # Clean up extra spaces
    matrix_part = WS_RE.sub(" ", matrix_part).strip()

    return eng_part, matrix_part


def combine_hyp_by_speaker(hyp_segments: Dict) -> Dict[str, str]:
    """
    Combine hyp segments per speaker (temporal order within each speaker).
    Returns dict: speaker -> combined_text
    """
    from collections import defaultdict
    
    by_speaker = defaultdict(list)
    for seg in hyp_segments:
        speaker = seg.get("speaker", "unknown")
        words = seg.get("words", "")
        start_time = seg.get("start_time", float("inf"))
        by_speaker[speaker].append((start_time, words))

    # For each speaker, combine their segments in temporal order
    result = {}
    for speaker in sorted(by_speaker.keys()):
        items = sorted(by_speaker[speaker], key=lambda x: x[0])
        combined_words = [words for _, words in items]
        result[speaker] = " ".join(combined_words)

    return result


def levenshtein_distance(ref: str, hyp: str) -> int:
    """Simple Levenshtein distance."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[n][m]


def main():
    # Actual example from the user
    ref_text = "**Plants** प्राकृतिक वातावरण में सबसे अच्छे दिखते हैं **so** सिर्फ एक **specimen** को भी हटाने के प्रलोभन को **resist** करें"
    
    hyp_segments = [
        {
            "speaker": "67_lbl67",
            "words": "प्लांट्स प्राकृतिक क्वातावरण में सबसे अच्छे दिखते हैं, सो सिर्फ एक स्पेसिमेन को भी हटाने के प्रलोभन",
            "start_time": 0.0,
        },
        {
            "speaker": "hi_lbl8",
            "words": "todact",
            "start_time": 8.34,
        },
        {
            "speaker": "67_lbl67",
            "words": "रेजिस्ट करें",
            "start_time": 8.42,
        },
    ]

    print("=" * 80)
    print("DEBUG: CER Parts Scoring Pipeline")
    print("=" * 80)

    print("\n### STEP 1: Input Data ###")
    print(f"\nref_text:\n  {ref_text}")
    print(f"\nhyp_segments ({len(hyp_segments)} segments):")
    for i, seg in enumerate(hyp_segments):
        print(f"  [{i}] speaker={seg['speaker']}, start={seg['start_time']}, words={seg['words']}")

    # Step 2: Extract eng and matrix from ref_text
    print("\n### STEP 2: Extract ENG and MATRIX from ref_text ###")
    eng_raw, matrix_raw = extract_eng_and_matrix(ref_text)
    print(f"\neng_raw (inside **...**):\n  {eng_raw}")
    print(f"\nmatrix_raw (rest with ** removed):\n  {matrix_raw}")

    # Step 3: Combine hyp by speaker
    print("\n### STEP 3: Combine HYP segments PER SPEAKER (temporal order within each) ###")
    hyp_by_speaker = combine_hyp_by_speaker(hyp_segments)
    print(f"\nhyp_by_speaker:")
    for speaker, combined_text in hyp_by_speaker.items():
        print(f"  {speaker}: {combined_text}")

    # Step 4: Normalize
    print("\n### STEP 4: Normalize (no options in this example) ###")
    ignore_ws = False
    remove_punct = False
    lower = False
    
    eng_norm = normalize_text(eng_raw, ignore_ws=ignore_ws, remove_punct=remove_punct, lower=lower)
    matrix_norm = normalize_text(matrix_raw, ignore_ws=ignore_ws, remove_punct=remove_punct, lower=lower)
    
    print(f"\neng_norm:\n  {eng_norm}")
    print(f"\nmatrix_norm:\n  {matrix_norm}")

    # Step 5: Compute distances and select ENG/MATRIX speaker assignment
    print("\n### STEP 5: Compute distances and assign ENG/MATRIX to speakers ###")

    per_speaker = {}
    for speaker, hyp_text in hyp_by_speaker.items():
        hyp_norm = normalize_text(hyp_text, ignore_ws=ignore_ws, remove_punct=remove_punct, lower=lower)
        print(f"\n--- Speaker: {speaker} ---")
        print(f"hyp_norm: {hyp_norm}")

        eng_dist = levenshtein_distance(eng_norm, hyp_norm) if eng_norm else 0
        eng_cer = eng_dist / max(1, len(eng_norm))
        print(f"\nvs ENG:")
        print(f"  ref_len: {len(eng_norm)}")
        print(f"  distance: {eng_dist}")
        print(f"  CER: {eng_cer:.6f}")

        matrix_dist = levenshtein_distance(matrix_norm, hyp_norm) if matrix_norm else 0
        matrix_cer = matrix_dist / max(1, len(matrix_norm))
        print(f"\nvs MATRIX:")
        print(f"  ref_len: {len(matrix_norm)}")
        print(f"  distance: {matrix_dist}")
        print(f"  CER: {matrix_cer:.6f}")

        per_speaker[speaker] = {
            "eng_cer": eng_cer,
            "matrix_cer": matrix_cer,
        }

    if len(per_speaker) == 1:
        missing_eng = 0.0 if len(eng_norm) == 0 else 1.0
        missing_matrix = 0.0 if len(matrix_norm) == 0 else 1.0
        per_speaker["__MISSING__"] = {
            "eng_cer": missing_eng,
            "matrix_cer": missing_matrix,
        }

    speakers = list(per_speaker.keys())
    best_pair = None
    best_score = None

    for eng_speaker in speakers:
        for matrix_speaker in speakers:
            if eng_speaker == matrix_speaker:
                continue
            total_cer = per_speaker[eng_speaker]["eng_cer"] + per_speaker[matrix_speaker]["matrix_cer"]
            if best_score is None or total_cer < best_score:
                best_score = total_cer
                best_pair = (eng_speaker, matrix_speaker)

    if best_pair:
        print(f"\n*** SELECTED ASSIGNMENT ***")
        print(f"  ENG <- {best_pair[0]}")
        print(f"  MATRIX <- {best_pair[1]}")
        print(f"  total CER: {best_score:.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
