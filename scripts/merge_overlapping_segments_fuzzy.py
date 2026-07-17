#!/usr/bin/env python3
"""
merge_overlapping_segments_fuzzy.py

Merge overlapping segments using temporal window + fuzzy text matching.

Implements two strategies:
1. Word-level matching (difflib SequenceMatcher on word tokens)
2. Character-level fuzzy matching (difflib on concatenated text with word boundary preservation)

Input JSONL: segments with overlaps attribute
Output JSONL: same with added merge_info and merged_hyp_text

Usage:
  python merge_overlapping_segments_fuzzy.py \
    -i input.jsonl -o output.jsonl \
    --strategy word_level \
    --min_overlap_duration 0.1 \
    --search_buffer_ratio 0.2 \
    --match_threshold 0.7
"""

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import re

EPS = 1e-9


def calculate_overlap_duration(seg_a: Dict, seg_b: Dict) -> float:
    """Calculate temporal overlap between two segments."""
    overlap_start = max(seg_a["start_time"], seg_b["start_time"])
    overlap_end = min(seg_a["end_time"], seg_b["end_time"])
    return max(0.0, overlap_end - overlap_start)


def calculate_temporal_window(
    seg_a: Dict, seg_b: Dict, overlap_duration: float
) -> Dict[str, Any]:
    """Calculate expected text regions based on temporal positioning."""
    
    dur_a = seg_a["end_time"] - seg_a["start_time"]
    dur_b = seg_b["end_time"] - seg_b["start_time"]
    
    if dur_a <= EPS or dur_b <= EPS:
        return {}
    
    # Position of overlap within each segment
    overlap_start = max(seg_a["start_time"], seg_b["start_time"])
    
    # Relative position in segment A (0.0 to 1.0)
    pos_a_start = (overlap_start - seg_a["start_time"]) / dur_a
    pos_a_end = (min(seg_a["end_time"], seg_b["end_time"]) - seg_a["start_time"]) / dur_a
    
    # Relative position in segment B
    pos_b_start = (overlap_start - seg_b["start_time"]) / dur_b
    pos_b_end = (min(seg_a["end_time"], seg_b["end_time"]) - seg_b["start_time"]) / dur_b
    
    return {
        "overlap_duration": overlap_duration,
        "pos_a_range": (pos_a_start, pos_a_end),
        "pos_b_range": (pos_b_start, pos_b_end),
        "dur_a": dur_a,
        "dur_b": dur_b,
    }


def calculate_search_window(
    pos_range: Tuple[float, float], num_words: int, buffer_ratio: float = 0.2
) -> Tuple[int, int]:
    """
    Calculate search window indices based on temporal position and buffer.
    
    Returns: (start_idx, end_idx)
    """
    if num_words == 0:
        return (0, 0)
    
    start_pos, end_pos = pos_range
    
    # Expected indices
    expected_start = max(0, int(start_pos * num_words))
    expected_end = min(num_words, int(end_pos * num_words) + 1)
    
    # Buffer expansion
    buffer = max(1, int(buffer_ratio * num_words))
    
    window_start = max(0, expected_start - buffer)
    window_end = min(num_words, expected_end + buffer)
    
    return (window_start, window_end)


def find_overlap_word_level(
    words_a: List[str],
    words_b: List[str],
    window_a: Tuple[int, int],
    window_b: Tuple[int, int],
    min_match_length: int = 1,
) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Find overlap using word-level difflib matching.
    
    Returns: (a_start, a_end, b_start, b_end, similarity_ratio)
    or None if no match found.
    """
    
    subset_a = words_a[window_a[0] : window_a[1]]
    subset_b = words_b[window_b[0] : window_b[1]]
    
    if not subset_a or not subset_b:
        return None
    
    matcher = SequenceMatcher(None, subset_a, subset_b)
    matches = matcher.get_matching_blocks()
    
    # Find longest match (excluding terminal sentinel)
    best_match = None
    best_size = 0
    for match in matches:
        if match.size > best_size:
            best_match = match
            best_size = match.size
    
    if best_match is None or best_size < min_match_length:
        return None
    
    # Convert to global indices
    global_a_start = window_a[0] + best_match.a
    global_a_end = global_a_start + best_match.size
    global_b_start = window_b[0] + best_match.b
    global_b_end = global_b_start + best_match.size
    
    # Calculate similarity ratio
    similarity = best_match.size / max(len(subset_a), len(subset_b))
    
    return (global_a_start, global_a_end, global_b_start, global_b_end, similarity)


def find_overlap_char_level(
    words_a: List[str],
    words_b: List[str],
    window_a: Tuple[int, int],
    window_b: Tuple[int, int],
    char_sim_threshold: float = 0.7,
    min_match_length: int = 1,
) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Find overlap using character-level fuzzy matching.
    
    Works by joining words with space delimiter, doing character-level matching,
    then converting back to word indices.
    
    Returns: (a_start, a_end, b_start, b_end, char_similarity)
    or None if no match found.
    """
    
    subset_a = words_a[window_a[0] : window_a[1]]
    subset_b = words_b[window_b[0] : window_b[1]]
    
    if not subset_a or not subset_b:
        return None
    
    # Join with space delimiter to preserve word boundaries
    text_a = " ".join(subset_a)
    text_b = " ".join(subset_b)
    
    matcher = SequenceMatcher(None, text_a, text_b)
    matches = matcher.get_matching_blocks()
    
    # Find longest match
    best_match = None
    best_size = 0
    for match in matches:
        if match.size > best_size:
            best_match = match
            best_size = match.size
    
    if best_match is None or best_size == 0:
        return None
    
    # Convert character indices to word indices
    def char_to_word_idx(text: str, char_idx: int) -> int:
        """Convert character index to word index (count spaces before)."""
        return text[:char_idx].count(" ")
    
    word_a_start = char_to_word_idx(text_a, best_match.a)
    word_a_end = char_to_word_idx(text_a, best_match.a + best_match.size)
    word_b_start = char_to_word_idx(text_b, best_match.b)
    word_b_end = char_to_word_idx(text_b, best_match.b + best_match.size)
    
    # Ensure we have at least min_match_length words
    match_length = word_a_end - word_a_start
    if match_length < min_match_length:
        return None
    
    # Calculate character-level similarity
    char_similarity = best_match.size / max(len(text_a), len(text_b))
    
    if char_similarity < char_sim_threshold:
        return None
    
    # Convert back to global indices
    global_a_start = window_a[0] + word_a_start
    global_a_end = window_a[0] + word_a_end
    global_b_start = window_b[0] + word_b_start
    global_b_end = window_b[0] + word_b_end
    
    return (global_a_start, global_a_end, global_b_start, global_b_end, char_similarity)


def choose_overlap_text(
    overlap_a: str,
    overlap_b: str,
    a_pos: float,
    b_pos: float,
    expected_pos: float,
    overlap_strategy: str = "longer",
) -> Tuple[str, str]:
    """
    Choose which overlap text to use based on strategy.
    
    Args:
        overlap_a: Text from segment A in overlap region
        overlap_b: Text from segment B in overlap region
        a_pos: Normalized position of overlap in segment A (0.0-1.0)
        b_pos: Normalized position of overlap in segment B (0.0-1.0)
        expected_pos: Expected normalized position from temporal info
        overlap_strategy: Strategy for choosing text
    
    Returns: (chosen_text, strategy_used)
    """
    
    if overlap_strategy == "first":
        # Always use segment A (earlier segment)
        return overlap_a, "first"
    
    elif overlap_strategy == "longer":
        # Use the longer text (more information)
        chosen = overlap_a if len(overlap_a) >= len(overlap_b) else overlap_b
        return chosen, "longer"
    
    elif overlap_strategy == "temporal_position":
        # Prefer text whose position is closer to expected temporal position
        a_error = abs(a_pos - expected_pos)
        b_error = abs(b_pos - expected_pos)
        chosen = overlap_a if a_error <= b_error else overlap_b
        reason = f"temporal_position(a_err={a_error:.2f}, b_err={b_error:.2f})"
        return chosen, reason
    
    elif overlap_strategy == "identical_or_longer":
        # If identical, use either; if different, use longer
        if overlap_a == overlap_b:
            return overlap_a, "identical"
        chosen = overlap_a if len(overlap_a) >= len(overlap_b) else overlap_b
        reason = f"identical_or_longer(used={'a' if len(overlap_a) >= len(overlap_b) else 'b'})"
        return chosen, reason
    
    else:
        return overlap_a, "default_first"


def merge_two_segments(
    seg_a: Dict[str, Any],
    seg_b: Dict[str, Any],
    overlap_duration: float,
    match_strategy: str = "word_level",
    overlap_strategy: str = "longer",
    search_buffer_ratio: float = 0.2,
    match_threshold: float = 0.7,
    min_match_length: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Merge two overlapping segments.
    
    Returns: (merged_segment, merge_info)
    """
    
    words_a = seg_a.get("words", "").split()
    words_b = seg_b.get("words", "").split()
    
    if not words_a or not words_b:
        # Can't merge if no words
        return (
            {
                "start_time": seg_a["start_time"],
                "end_time": seg_b["end_time"],
                "words": (seg_a.get("words", "") + " " + seg_b.get("words", "")).strip(),
                "speaker": seg_b.get("speaker", seg_a.get("speaker", "")),
                "session_id": seg_a.get("session_id", ""),
            },
            {
                "status": "no_words",
                "reason": f"Empty text: a={len(words_a)}, b={len(words_b)}",
                "overlap_duration": overlap_duration,
            },
        )
    
    # Step 1: Calculate temporal window
    temporal_info = calculate_temporal_window(seg_a, seg_b, overlap_duration)
    
    if not temporal_info:
        return (
            {
                "start_time": seg_a["start_time"],
                "end_time": seg_b["end_time"],
                "words": (seg_a.get("words", "") + " " + seg_b.get("words", "")).strip(),
                "speaker": seg_b.get("speaker", seg_a.get("speaker", "")),
                "session_id": seg_a.get("session_id", ""),
            },
            {"status": "invalid_times", "overlap_duration": overlap_duration},
        )
    
    # Step 2: Calculate search windows
    window_a = calculate_search_window(
        temporal_info["pos_a_range"], len(words_a), buffer_ratio=search_buffer_ratio
    )
    window_b = calculate_search_window(
        temporal_info["pos_b_range"], len(words_b), buffer_ratio=search_buffer_ratio
    )
    
    # Step 3: Find overlap using selected strategy
    match = None
    if match_strategy == "word_level":
        match = find_overlap_word_level(words_a, words_b, window_a, window_b, min_match_length=min_match_length)
    elif match_strategy == "char_level":
        match = find_overlap_char_level(
            words_a, words_b, window_a, window_b, char_sim_threshold=match_threshold, min_match_length=min_match_length
        )
    
    # Step 4: Build merged text or fallback
    if match is None:
        # Fallback: concatenate
        merged_text = (seg_a.get("words", "") + " " + seg_b.get("words", "")).strip()
        return (
            {
                "start_time": seg_a["start_time"],
                "end_time": seg_b["end_time"],
                "words": merged_text,
                "speaker": seg_b.get("speaker", seg_a.get("speaker", "")),
                "session_id": seg_a.get("session_id", ""),
            },
            {
                "status": "fallback_concatenate",
                "reason": "No match found",
                "match_strategy": match_strategy,
                "overlap_strategy": overlap_strategy,
                "window_a": window_a,
                "window_b": window_b,
                "overlap_duration": overlap_duration,
            },
        )
    
    # Extract match info
    a_start, a_end, b_start, b_end, similarity = match
    
    # Build text components
    text_before = " ".join(words_a[:a_start]) if a_start > 0 else ""
    overlap_a = " ".join(words_a[a_start:a_end])
    overlap_b = " ".join(words_b[b_start:b_end])
    text_after = " ".join(words_b[b_end:]) if b_end < len(words_b) else ""
    
    # Step 5: Choose which overlap text to use
    expected_pos = (temporal_info["pos_a_range"][0] + temporal_info["pos_a_range"][1]) / 2
    a_overlap_pos = (a_start + a_end) / 2 / len(words_a) if words_a else 0.5
    b_overlap_pos = (b_start + b_end) / 2 / len(words_b) if words_b else 0.5
    
    text_overlap, strategy_decision = choose_overlap_text(
        overlap_a, overlap_b, a_overlap_pos, b_overlap_pos, expected_pos, overlap_strategy
    )
    
    merged_parts = [p for p in [text_before, text_overlap, text_after] if p]
    merged_text = " ".join(merged_parts)
    
    return (
        {
            "start_time": seg_a["start_time"],
            "end_time": seg_b["end_time"],
            "words": merged_text,
            "speaker": seg_b.get("speaker", seg_a.get("speaker", "")),
            "session_id": seg_a.get("session_id", ""),
        },
        {
            "status": "merged",
            "match_strategy": match_strategy,
            "overlap_strategy": overlap_strategy,
            "strategy_decision": strategy_decision,
            "overlap_duration": overlap_duration,
            "match_indices": {"a": [a_start, a_end], "b": [b_start, b_end]},
            "similarity": similarity,
            "text_overlap": text_overlap,
            "text_overlap_a": overlap_a,
            "text_overlap_b": overlap_b,
            "window_a": window_a,
            "window_b": window_b,
        },
    )


def merge_segment_list(
    segments: List[Dict[str, Any]],
    min_overlap_duration: float = 0.1,
    match_strategy: str = "word_level",
    overlap_strategy: str = "longer",
    search_buffer_ratio: float = 0.2,
    match_threshold: float = 0.7,
    min_match_length: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Merge all overlapping segments in a list.
    
    Returns: (merged_segments, merge_history)
    """
    
    if len(segments) < 2:
        return segments, []
    
    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: (s["start_time"], s["end_time"]))
    
    merged = []
    history = []
    
    i = 0
    while i < len(sorted_segs):
        current = sorted_segs[i].copy()
        
        # Check for overlaps with subsequent segments
        j = i + 1
        merge_chain = []
        
        while j < len(sorted_segs):
            overlap_dur = calculate_overlap_duration(current, sorted_segs[j])
            
            if overlap_dur < min_overlap_duration:
                break  # No more overlaps with this chain
            
            # Merge current with next segment
            merged_seg, merge_info = merge_two_segments(
                current,
                sorted_segs[j],
                overlap_dur,
                match_strategy=match_strategy,
                overlap_strategy=overlap_strategy,
                search_buffer_ratio=search_buffer_ratio,
                match_threshold=match_threshold,
                min_match_length=min_match_length,
            )
            
            merge_info["pair"] = [i, j]
            merge_chain.append(merge_info)
            
            current = merged_seg
            j += 1
        
        merged.append(current)
        history.extend(merge_chain)
        
        i = j if j > i + 1 else i + 1
    
    return merged, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Input JSONL file")
    ap.add_argument("--output", "-o", required=True, help="Output JSONL file")
    ap.add_argument(
        "--match_strategy",
        choices=["word_level", "char_level"],
        default="word_level",
        help="Matching strategy: word_level (fast) or char_level (fuzzy)",
    )
    ap.add_argument(
        "--overlap_strategy",
        choices=["first", "longer", "temporal_position", "identical_or_longer"],
        default="longer",
        help=(
            "Strategy for choosing which overlap text to keep:\n"
            "  first: Always use segment A (earlier)\n"
            "  longer: Use the longer text (more information)\n"
            "  temporal_position: Prefer text closer to expected position\n"
            "  identical_or_longer: Use identical if they match, else longer"
        ),
    )
    ap.add_argument(
        "--min_overlap_duration",
        type=float,
        default=0.1,
        help="Minimum overlap duration to trigger merge (seconds)",
    )
    ap.add_argument(
        "--search_buffer_ratio",
        type=float,
        default=0.2,
        help="Buffer ratio for search window expansion (0.0-1.0)",
    )
    ap.add_argument(
        "--match_threshold",
        type=float,
        default=0.7,
        help="Minimum similarity threshold (0.0-1.0)",
    )
    ap.add_argument(
        "--min_match_length",
        type=int,
        default=1,
        help="Minimum match length in words for word_level or chars for char_level",
    )
    ap.add_argument(
        "--write_merge_report",
        default="",
        help="Optional path to write detailed merge report",
    )
    args = ap.parse_args()
    
    stats = {
        "processed": 0,
        "with_hyp_merges": 0,
        "with_ref_merges": 0,
        "total_hyp_merges": 0,
        "total_ref_merges": 0,
    }
    
    merge_report_lines = []
    
    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for ln, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            entry = json.loads(line)
            utt_id = entry.get("id", f"LINE{ln}")
            
            stats["processed"] += 1
            
            # Initialize history variables
            ref_history = []
            hyp_history = []
            
            # Merge ref segments
            ref_segs = entry.get("ref", [])
            if ref_segs:
                ref_merged, ref_history = merge_segment_list(
                    ref_segs,
                    min_overlap_duration=args.min_overlap_duration,
                    match_strategy=args.match_strategy,
                    overlap_strategy=args.overlap_strategy,
                    search_buffer_ratio=args.search_buffer_ratio,
                    match_threshold=args.match_threshold,
                    min_match_length=args.min_match_length,
                )
                entry["ref_merged"] = ref_merged
                entry["ref_merge_info"] = ref_history
                
                if ref_history:
                    stats["with_ref_merges"] += 1
                    stats["total_ref_merges"] += len(ref_history)
            
            # Merge hyp segments
            hyp_segs = entry.get("hyp", [])
            if hyp_segs:
                hyp_merged, hyp_history = merge_segment_list(
                    hyp_segs,
                    min_overlap_duration=args.min_overlap_duration,
                    match_strategy=args.match_strategy,
                    overlap_strategy=args.overlap_strategy,
                    search_buffer_ratio=args.search_buffer_ratio,
                    match_threshold=args.match_threshold,
                    min_match_length=args.min_match_length,
                )
                entry["hyp_merged"] = hyp_merged
                entry["hyp_merge_info"] = hyp_history
                
                if hyp_history:
                    stats["with_hyp_merges"] += 1
                    stats["total_hyp_merges"] += len(hyp_history)
                
                # Build merged hyp text
                merged_hyp_text = " ".join(seg.get("words", "") for seg in hyp_merged)
                merged_hyp_text = " ".join(merged_hyp_text.split()).strip()
                entry["hyp_text_merged"] = merged_hyp_text
            
            # Build merged ref text
            if ref_segs:
                merged_ref_text = " ".join(seg.get("words", "") for seg in ref_merged)
                merged_ref_text = " ".join(merged_ref_text.split()).strip()
                entry["ref_text_merged"] = merged_ref_text
            
            # Prepare report
            if hyp_history or ref_history:
                merge_report_lines.append(f"\n=== {utt_id} ===")
                for info in hyp_history:
                    merge_report_lines.append(
                        f"HYP merge: {info['pair']} | dur={info['overlap_duration']:.3f}s | "
                        f"status={info['status']} | overlap_text='{info.get('text_overlap', '')}'"
                    )
                for info in ref_history:
                    merge_report_lines.append(
                        f"REF merge: {info['pair']} | dur={info['overlap_duration']:.3f}s | "
                        f"status={info['status']}"
                    )
            
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Print statistics
    print("=== MERGE STATISTICS ===")
    print(f"Processed: {stats['processed']}")
    print(f"With HYP merges: {stats['with_hyp_merges']} ({100*stats['with_hyp_merges']/max(1, stats['processed']):.1f}%)")
    print(f"With REF merges: {stats['with_ref_merges']} ({100*stats['with_ref_merges']/max(1, stats['processed']):.1f}%)")
    print(f"Total HYP merges: {stats['total_hyp_merges']}")
    print(f"Total REF merges: {stats['total_ref_merges']}")
    print(f"\nMatch strategy: {args.match_strategy}")
    print(f"Overlap strategy: {args.overlap_strategy}")
    print(f"Min overlap duration: {args.min_overlap_duration}s")
    print(f"Search buffer ratio: {args.search_buffer_ratio}")
    print(f"Match threshold: {args.match_threshold}")
    print(f"\nOutput: {args.output}")
    
    if args.write_merge_report:
        with open(args.write_merge_report, "w", encoding="utf-8") as f:
            f.write("\n".join(merge_report_lines))
        print(f"Merge report: {args.write_merge_report}")


if __name__ == "__main__":
    main()
