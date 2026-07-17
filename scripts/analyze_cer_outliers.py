#!/usr/bin/env python3
"""
analyze_cer_outliers.py

Identify utterances with high CER that negatively impact overall and per-language scores.
Ranks utterances by impact and analyzes distribution to identify outliers.

Key features:
- Per-utterance CER scoring with detailed breakdown (INS/DEL/SUB)
- Ranking by absolute CER and by impact on global/per-language scores
- Statistical analysis: mean, median, std dev, percentiles
- Outlier detection using IQR and z-score methods
- Worst examples per language ranked by impact
- Distribution visualization via histogram bins
- Output: per-utterance JSONL + readable report

Normalization options (same as score_cer_confusions.py):
  --ignore_ws         : remove all whitespace chars from ref/hyp
  --remove_punct      : remove Unicode punctuation from ref/hyp
  --lower             : lowercase both ref/hyp

Usage:
  python analyze_cer_outliers.py -i in.jsonl \
    --ref_field text --hyp_field whisper_pred_text --lang_field language \
    --ignore_ws --remove_punct --lower \
    --output_jsonl per_utt_scores.jsonl \
    --output_report outlier_report.txt \
    --top_k 50 --percentile_thresholds 75,90,95,99
"""

import argparse
import json
import re
import unicodedata
import statistics
from collections import defaultdict
from typing import Any, Dict, Tuple, List, Optional

BOLD_MARK_RE = re.compile(r"\*\*")
WS_RE = re.compile(r"\s+")


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def strip_unicode_punct(s: str) -> str:
    """Remove Unicode punctuation characters."""
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def normalize_text(text: str, *, is_ref: bool, ignore_ws: bool, remove_punct: bool, lower: bool) -> str:
    """Shared normalization for ref/hyp."""
    if is_ref:
        text = BOLD_MARK_RE.sub("", text)

    if remove_punct:
        text = strip_unicode_punct(text)

    if ignore_ws:
        text = WS_RE.sub("", text)

    if lower:
        text = text.lower()

    return text


def levenshtein_counts_and_edits(ref: str, hyp: str) -> Tuple[int, int, int, int, List[Tuple[str, str, str]]]:
    """
    Full DP Levenshtein with backtrace to identify individual edits.
    
    Returns:
      edits, ins, del, sub, edit_list
      
    edit_list: [(type, ref_char, hyp_char), ...] where type in ['SUB', 'DEL', 'INS', 'MATCH']
    """
    n = len(ref)
    m = len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]  # 'I','D','S','M'

    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"

    for i in range(1, n + 1):
        rc = ref[i - 1]
        for j in range(1, m + 1):
            hc = hyp[j - 1]

            cost_sub = 0 if rc == hc else 1
            v_del = dp[i - 1][j] + 1
            v_ins = dp[i][j - 1] + 1
            v_sub = dp[i - 1][j - 1] + cost_sub

            best = v_sub
            best_op = "M" if cost_sub == 0 else "S"

            if v_del < best:
                best = v_del
                best_op = "D"
            if v_ins < best:
                best = v_ins
                best_op = "I"

            dp[i][j] = best
            op[i][j] = best_op

    i, j = n, m
    ins = dele = sub = 0
    edit_list = []

    while i > 0 or j > 0:
        cur_op = op[i][j]
        if cur_op == "M":
            edit_list.append(("MATCH", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif cur_op == "S":
            sub += 1
            edit_list.append(("SUB", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif cur_op == "D":
            dele += 1
            edit_list.append(("DEL", ref[i - 1], ""))
            i -= 1
        elif cur_op == "I":
            ins += 1
            edit_list.append(("INS", "", hyp[j - 1]))
            j -= 1
        else:
            if i > 0 and j > 0:
                if ref[i - 1] == hyp[j - 1]:
                    edit_list.append(("MATCH", ref[i - 1], hyp[j - 1]))
                    i -= 1
                    j -= 1
                else:
                    sub += 1
                    edit_list.append(("SUB", ref[i - 1], hyp[j - 1]))
                    i -= 1
                    j -= 1
            elif i > 0:
                dele += 1
                edit_list.append(("DEL", ref[i - 1], ""))
                i -= 1
            else:
                ins += 1
                edit_list.append(("INS", "", hyp[j - 1]))
                j -= 1

    edit_list.reverse()
    edits = dp[n][m]
    return edits, ins, dele, sub, edit_list


def cer_from_counts(edits: int, ref_len: int) -> float:
    return edits / max(1, ref_len) if ref_len > 0 else 0.0


def calculate_percentiles(values: List[float]) -> Dict[int, float]:
    """Calculate percentiles for a list of values."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    result = {}
    for p in [25, 50, 75, 90, 95, 99]:
        idx = int(len(sorted_vals) * p / 100.0)
        idx = max(0, min(idx, len(sorted_vals) - 1))
        result[p] = sorted_vals[idx]
    return result


def detect_outliers_iqr(values: List[float]) -> Tuple[List[float], List[float]]:
    """Detect outliers using Interquartile Range method."""
    if len(values) < 4:
        return [], []
    sorted_vals = sorted(values)
    q1_idx = len(sorted_vals) // 4
    q3_idx = 3 * len(sorted_vals) // 4
    q1 = sorted_vals[q1_idx]
    q3 = sorted_vals[q3_idx]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mild_outliers = [v for v in values if (v < lower_bound or v > upper_bound) and v <= upper_bound + 0.5 * iqr]
    extreme_outliers = [v for v in values if v > upper_bound + 0.5 * iqr or v < lower_bound - 0.5 * iqr]
    return mild_outliers, extreme_outliers


def strip_id_prefix(s: str) -> str:
    """Strip prefix from id/language."""
    return re.sub(r'^[a-z]+-\d+-', '', s)


def format_edit_sequence(edits: List[Tuple[str, str, str]], max_edits: int = 50) -> str:
    """Format edit sequence for display."""
    result = []
    for i, (edit_type, ref_char, hyp_char) in enumerate(edits):
        if i >= max_edits:
            result.append(f"... +{len(edits) - max_edits} more")
            break
        
        # Format characters nicely
        ref_char = ref_char if ref_char else "∅"
        hyp_char = hyp_char if hyp_char else "∅"
        
        if edit_type == "MATCH":
            result.append(f"{ref_char}")
        elif edit_type == "SUB":
            result.append(f"[{ref_char}→{hyp_char}]")
        elif edit_type == "DEL":
            result.append(f"[{ref_char}→∅]")
        elif edit_type == "INS":
            result.append(f"[∅→{hyp_char}]")
    
    return "".join(result)


def main():
    ap = argparse.ArgumentParser(
        description="Identify CER outliers and problematic utterances."
    )
    ap.add_argument("-i", "--input", required=True, help="Input JSONL")
    ap.add_argument("--ref_field", default="text", help="Reference field")
    ap.add_argument("--hyp_field", default="whisper_pred_text", help="Hypothesis field")
    ap.add_argument("--lang_field", default="language", help="Language field")
    ap.add_argument("--id_field", default="id", help="ID field")
    ap.add_argument("--exclude_langs", nargs="*", default=[], help="Languages to exclude")
    ap.add_argument("--include_langs", nargs="*", default=None, help="Only score these languages")

    ap.add_argument("--ignore_ws", action="store_true", help="Remove all whitespace")
    ap.add_argument("--remove_punct", action="store_true", help="Remove punctuation")
    ap.add_argument("--lower", action="store_true", help="Lowercase")

    ap.add_argument("--output_jsonl", required=True, help="Output JSONL with per-utt scores")
    ap.add_argument("--output_report", required=True, help="Output text report")
    ap.add_argument("--top_k", type=int, default=50, help="Top-K worst utterances to report")
    ap.add_argument("--percentile_thresholds", default="75,90,95,99",
                    help="Percentiles to highlight (comma-separated)")
    ap.add_argument("--ignore_id_prefix", action="store_true", help="Strip ID prefix")

    args = ap.parse_args()

    exclude = set(args.exclude_langs)
    include = set(args.include_langs) if args.include_langs else None
    percentiles = [int(p) for p in args.percentile_thresholds.split(",")]

    # Data structures
    all_scores = []  # List of per-utterance score dicts
    per_lang_scores = defaultdict(list)  # lang -> list of cer values
    overall_stats = {"edits": 0, "ref_len": 0, "count": 0, "skipped": 0}

    # Process input
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            lang = safe_str(rec.get(args.lang_field, "UNKNOWN"))
            if args.ignore_id_prefix:
                lang = strip_id_prefix(lang)

            if lang in exclude or (include is not None and lang not in include):
                continue

            ref_raw = safe_str(rec.get(args.ref_field, ""))
            hyp_raw = safe_str(rec.get(args.hyp_field, ""))

            if not ref_raw or not hyp_raw:
                overall_stats["skipped"] += 1
                continue

            ref = normalize_text(
                ref_raw,
                is_ref=True,
                ignore_ws=args.ignore_ws,
                remove_punct=args.remove_punct,
                lower=args.lower,
            )
            hyp = normalize_text(
                hyp_raw,
                is_ref=False,
                ignore_ws=args.ignore_ws,
                remove_punct=args.remove_punct,
                lower=args.lower,
            )

            edits, ins, dele, sub, edit_seq = levenshtein_counts_and_edits(ref, hyp)
            rlen = len(ref)
            cer = cer_from_counts(edits, rlen)

            utt_id = rec.get(args.id_field, "UNKNOWN")

            score_dict = {
                "id": utt_id,
                "language": lang,
                "cer": cer,
                "edits": edits,
                "ins": ins,
                "del": dele,
                "sub": sub,
                "ref_len": rlen,
                "ref": ref[:200] if len(ref) <= 200 else ref[:197] + "...",
                "hyp": hyp[:200] if len(hyp) <= 200 else hyp[:197] + "...",
                "edit_sequence": format_edit_sequence(edit_seq, max_edits=30),
            }

            all_scores.append(score_dict)
            per_lang_scores[lang].append(cer)

            overall_stats["edits"] += edits
            overall_stats["ref_len"] += rlen
            overall_stats["count"] += 1

    # Calculate statistics
    if not all_scores:
        print("No utterances to analyze.")
        return

    all_cer_values = [s["cer"] for s in all_scores]
    overall_cer = cer_from_counts(overall_stats["edits"], overall_stats["ref_len"])

    # Sort by CER descending (worst first)
    all_scores_sorted = sorted(all_scores, key=lambda x: x["cer"], reverse=True)

    # Calculate per-language statistics
    lang_stats = {}
    for lang, cer_values in per_lang_scores.items():
        if not cer_values:
            continue
        lang_stats[lang] = {
            "count": len(cer_values),
            "cer_mean": statistics.mean(cer_values),
            "cer_median": statistics.median(cer_values),
            "cer_std": statistics.stdev(cer_values) if len(cer_values) > 1 else 0.0,
            "cer_min": min(cer_values),
            "cer_max": max(cer_values),
            "percentiles": calculate_percentiles(cer_values),
        }

    global_stats = {
        "overall_cer": overall_cer,
        "count": overall_stats["count"],
        "cer_mean": statistics.mean(all_cer_values),
        "cer_median": statistics.median(all_cer_values),
        "cer_std": statistics.stdev(all_cer_values) if len(all_cer_values) > 1 else 0.0,
        "cer_min": min(all_cer_values),
        "cer_max": max(all_cer_values),
        "percentiles": calculate_percentiles(all_cer_values),
    }

    # Save per-utterance JSONL
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for score_dict in all_scores_sorted:
            f.write(json.dumps(score_dict, ensure_ascii=False) + "\n")

    # Generate report
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("CER OUTLIER ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")

        # Global statistics
        f.write("=== GLOBAL STATISTICS ===\n")
        f.write(f"Total utterances: {global_stats['count']}\n")
        f.write(f"Overall CER: {global_stats['overall_cer']:.6f}\n")
        f.write(f"  Total edits: {overall_stats['edits']}\n")
        f.write(f"  Total ref chars: {overall_stats['ref_len']}\n")
        f.write(f"  Skipped: {overall_stats['skipped']}\n\n")

        f.write(f"CER Distribution:\n")
        f.write(f"  Mean:    {global_stats['cer_mean']:.6f}\n")
        f.write(f"  Median:  {global_stats['cer_median']:.6f}\n")
        f.write(f"  Std Dev: {global_stats['cer_std']:.6f}\n")
        f.write(f"  Min:     {global_stats['cer_min']:.6f}\n")
        f.write(f"  Max:     {global_stats['cer_max']:.6f}\n\n")

        f.write(f"Percentiles:\n")
        for p in sorted(global_stats['percentiles'].keys()):
            f.write(f"  {p}th:  {global_stats['percentiles'][p]:.6f}\n")
        f.write("\n")

        # Percentile thresholds
        f.write("=== PERCENTILE THRESHOLDS ===\n")
        f.write("Utterances at each percentile threshold:\n\n")
        for p in percentiles:
            if p in global_stats['percentiles']:
                threshold = global_stats['percentiles'][p]
                count_above = sum(1 for s in all_scores if s['cer'] >= threshold)
                f.write(f"{p}th percentile (CER >= {threshold:.6f}): {count_above} utterances\n")
        f.write("\n")

        # Outlier detection
        f.write("=== OUTLIER DETECTION (IQR Method) ===\n")
        mild_outliers, extreme_outliers = detect_outliers_iqr(all_cer_values)
        f.write(f"Mild outliers: {len(mild_outliers)}\n")
        f.write(f"Extreme outliers: {len(extreme_outliers)}\n\n")

        if extreme_outliers:
            f.write(f"Extreme outlier CER values:\n")
            for val in sorted(set(extreme_outliers), reverse=True)[:10]:
                f.write(f"  {val:.6f}\n")
            f.write("\n")

        # CER without outliers
        f.write("=== CER IMPACT OF OUTLIERS ===\n")
        
        # Remove extreme outliers
        extreme_outlier_set = set(extreme_outliers)
        scores_no_extreme = [s for s in all_scores if s['cer'] not in extreme_outlier_set]
        if scores_no_extreme:
            edits_no_extreme = sum(s['edits'] for s in scores_no_extreme)
            ref_len_no_extreme = sum(s['ref_len'] for s in scores_no_extreme)
            cer_no_extreme = cer_from_counts(edits_no_extreme, ref_len_no_extreme)
            f.write(f"Overall CER (excluding extreme outliers):\n")
            f.write(f"  CER: {cer_no_extreme:.6f} (was {overall_cer:.6f}, change: {overall_cer - cer_no_extreme:+.6f})\n")
            f.write(f"  Edits: {edits_no_extreme} (removed {overall_stats['edits'] - edits_no_extreme})\n")
            f.write(f"  Ref chars: {ref_len_no_extreme} (removed {overall_stats['ref_len'] - ref_len_no_extreme})\n")
            f.write(f"  Utterances: {len(scores_no_extreme)} (removed {len(all_scores) - len(scores_no_extreme)})\n\n")
        
        # Remove mild + extreme outliers
        all_outlier_set = set(mild_outliers + extreme_outliers)
        scores_no_outliers = [s for s in all_scores if s['cer'] not in all_outlier_set]
        if scores_no_outliers:
            edits_no_outliers = sum(s['edits'] for s in scores_no_outliers)
            ref_len_no_outliers = sum(s['ref_len'] for s in scores_no_outliers)
            cer_no_outliers = cer_from_counts(edits_no_outliers, ref_len_no_outliers)
            f.write(f"Overall CER (excluding all outliers):\n")
            f.write(f"  CER: {cer_no_outliers:.6f} (was {overall_cer:.6f}, change: {overall_cer - cer_no_outliers:+.6f})\n")
            f.write(f"  Edits: {edits_no_outliers} (removed {overall_stats['edits'] - edits_no_outliers})\n")
            f.write(f"  Ref chars: {ref_len_no_outliers} (removed {overall_stats['ref_len'] - ref_len_no_outliers})\n")
            f.write(f"  Utterances: {len(scores_no_outliers)} (removed {len(all_scores) - len(scores_no_outliers)})\n\n")

        # Top-K worst utterances
        f.write("=" * 100 + "\n")
        f.write(f"=== TOP {args.top_k} WORST UTTERANCES (by CER) ===\n")
        f.write("=" * 100 + "\n\n")

        for rank, score_dict in enumerate(all_scores_sorted[:args.top_k], 1):
            f.write(f"Rank {rank}: {score_dict['id']} [{score_dict['language']}]\n")
            f.write(f"  CER: {score_dict['cer']:.6f} (edits={score_dict['edits']}, ins={score_dict['ins']}, del={score_dict['del']}, sub={score_dict['sub']}, ref_len={score_dict['ref_len']})\n")
            f.write(f"  REF: {score_dict['ref']}\n")
            f.write(f"  HYP: {score_dict['hyp']}\n")
            f.write(f"  Edits: {score_dict['edit_sequence']}\n")
            f.write("\n")

        # Per-language statistics
        f.write("=" * 100 + "\n")
        f.write("=== PER-LANGUAGE STATISTICS ===\n")
        f.write("=" * 100 + "\n\n")

        for lang in sorted(lang_stats.keys(), key=lambda x: lang_stats[x]['cer_mean'], reverse=True):
            stats = lang_stats[lang]
            f.write(f"Language: {lang}\n")
            f.write(f"  Utterances: {stats['count']}\n")
            f.write(f"  CER: mean={stats['cer_mean']:.6f}, median={stats['cer_median']:.6f}, std={stats['cer_std']:.6f}\n")
            f.write(f"  Range: {stats['cer_min']:.6f} - {stats['cer_max']:.6f}\n")
            f.write(f"  Percentiles:\n")
            for p in sorted(stats['percentiles'].keys()):
                f.write(f"    {p}th: {stats['percentiles'][p]:.6f}\n")

            # Top-3 worst for this language
            lang_worst = [s for s in all_scores_sorted if s['language'] == lang][:3]
            if lang_worst:
                f.write(f"  Top 3 worst:\n")
                for i, s in enumerate(lang_worst, 1):
                    f.write(f"    {i}. {s['id']}: CER={s['cer']:.6f}\n")
            f.write("\n")

        # Distribution histogram
        f.write("=" * 100 + "\n")
        f.write("=== CER DISTRIBUTION (Histogram) ===\n")
        f.write("=" * 100 + "\n\n")

        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, float('inf')]
        bin_labels = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.75", "0.75-1.0", "1.0-2.0", "2.0+"]

        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            count = sum(1 for s in all_scores if lower <= s['cer'] < upper)
            pct = 100.0 * count / len(all_scores)
            bar_width = int(pct / 2)
            bar = "█" * bar_width
            f.write(f"{bin_labels[i]:10s}: {count:5d} ({pct:5.1f}%) {bar}\n")

        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write(f"Output JSONL: {args.output_jsonl}\n")
        f.write(f"Output report: {args.output_report}\n")

    print(f"Analysis complete.")
    print(f"  Per-utterance scores: {args.output_jsonl}")
    print(f"  Report: {args.output_report}")
    print(f"  Total utterances: {len(all_scores)}")
    print(f"  Overall CER: {overall_stats['edits']}/{overall_stats['ref_len']} = {overall_cer:.6f}")


if __name__ == "__main__":
    main()
