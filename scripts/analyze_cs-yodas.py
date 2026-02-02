#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional


def as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = int(math.ceil((p / 100.0) * len(sorted_vals))) - 1
    k = max(0, min(k, len(sorted_vals) - 1))
    return sorted_vals[k]


def stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "std": None, "p90": None, "p95": None, "max": None}
    vals = sorted(values)
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1)) if n >= 2 else 0.0
    return {"count": n, "min": vals[0], "mean": mean, "median": median, "std": std, "p90": percentile(vals, 90), "p95": percentile(vals, 95), "max": vals[-1]}


def fmt(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.6g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input JSONL")
    ap.add_argument("-o", "--output", required=True, help="Output JSONL")
    ap.add_argument("--languages", nargs="+", required=True, help="Languages to filter (exact match on rec['language'])")
    ap.add_argument("-n", "--top_n", type=int, required=True, help="Keep top-N by score per selected language")
    ap.add_argument("--score_field", default="score", help="Score field name (default: score)")
    ap.add_argument("--ascending", action="store_true", help="Lower score is better (default: higher is better)")
    ap.add_argument("--stats_out", default=None, help="Optional: write stats JSON to this path")
    ap.add_argument("--keep_missing_score", action="store_true", help="Keep lines missing score for filtered langs (default: drop)")
    args = ap.parse_args()

    higher_is_better = not args.ascending
    target_langs = set(args.languages)

    records: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    all_scores_by_lang = defaultdict(list)
    missing_score_by_lang = defaultdict(int)

    for r in records:
        lang = r.get("language", "UNKNOWN")
        v = as_float(r.get(args.score_field))
        if v is None:
            missing_score_by_lang[lang] += 1
        else:
            all_scores_by_lang[lang].append(v)

    by_lang = defaultdict(list)
    passthrough = []
    for r in records:
        lang = r.get("language")
        if lang in target_langs:
            by_lang[lang].append(r)
        else:
            passthrough.append(r)

    kept_filtered = []
    kept_scores_by_lang = defaultdict(list)
    dropped_missing_score = defaultdict(int)

    for lang, items in by_lang.items():
        scored_items = []
        missing_items = 0

        for x in items:
            v = as_float(x.get(args.score_field))
            if v is None:
                missing_items += 1
                if args.keep_missing_score:
                    kept_filtered.append(x)
            else:
                scored_items.append((v, x))

        if missing_items and not args.keep_missing_score:
            dropped_missing_score[lang] += missing_items

        scored_items.sort(key=lambda t: t[0], reverse=higher_is_better)
        top = scored_items[: args.top_n]
        kept_filtered.extend([x for _, x in top])
        kept_scores_by_lang[lang].extend([s for s, _ in top])

    final_records = kept_filtered + passthrough
    with open(args.output, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print stats
    langs_all = sorted(set(all_scores_by_lang.keys()) | set(missing_score_by_lang.keys()))
    print("\nScore statistics by language (ALL records)")
    print("\t".join(["language", "count", "missing", "min", "mean", "median", "std", "p90", "p95", "max"]))
    for lang in langs_all:
        st = stats(all_scores_by_lang.get(lang, []))
        row = [
            lang,
            str(st["count"]),
            str(missing_score_by_lang.get(lang, 0)),
            fmt(st["min"]), fmt(st["mean"]), fmt(st["median"]), fmt(st["std"]),
            fmt(st["p90"]), fmt(st["p95"]), fmt(st["max"]),
        ]
        print("\t".join(row))

    print("\nScore statistics by language (KEPT top-N for selected languages)")
    print("\t".join(["language", "kept", "dropped_missing", "min", "mean", "median", "std", "p90", "p95", "max"]))
    for lang in sorted(target_langs):
        st = stats(kept_scores_by_lang.get(lang, []))
        row = [
            lang,
            str(st["count"]),
            str(dropped_missing_score.get(lang, 0)),
            fmt(st["min"]), fmt(st["mean"]), fmt(st["median"]), fmt(st["std"]),
            fmt(st["p90"]), fmt(st["p95"]), fmt(st["max"]),
        ]
        print("\t".join(row))

    if args.stats_out:
        out_obj = {
            "score_field": args.score_field,
            "higher_is_better": higher_is_better,
            "top_n": args.top_n,
            "filtered_languages": sorted(target_langs),
            "all": {
                lang: {**stats(all_scores_by_lang.get(lang, [])), "missing": missing_score_by_lang.get(lang, 0)}
                for lang in langs_all
            },
            "kept_filtered": {
                lang: {**stats(kept_scores_by_lang.get(lang, [])), "dropped_missing": dropped_missing_score.get(lang, 0)}
                for lang in sorted(target_langs)
            },
        }
        with open(args.stats_out, "w", encoding="utf-8") as sf:
            json.dump(out_obj, sf, ensure_ascii=False, indent=2)
        print(f"\nWrote stats JSON to: {args.stats_out}")


if __name__ == "__main__":
    main()
