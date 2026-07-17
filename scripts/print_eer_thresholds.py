#!/usr/bin/env python3
"""
Extract and print EER thresholds from DET JSON output(s).

Supports JSON formats produced by:
- det_9modes_priors.py  (out["curves"][key]["eer_threshold"])
- det_sweep_smoothing.py (out["settings"][key]["eer_threshold"])
- det_from_posteriors.py (out["methods"][key]["eer_threshold"])

Usage:
  python print_eer_thresholds.py det_9modes.json
  python print_eer_thresholds.py det_smoothing.json
  python print_eer_thresholds.py det_methods.json

Optional:
  --sort eer         sort by eer (ascending)
  --sort name        sort by curve name (default)
  --topk 20          print only top-k (after sorting)
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="Path to DET output JSON.")
    ap.add_argument("--sort", choices=["name", "eer", "thr"], default="name",
                    help="Sort output by curve name, EER, or threshold.")
    ap.add_argument("--topk", type=int, default=0, help="If >0, print only top-k rows.")
    return ap.parse_args()


def _extract_rows(det: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    """
    Returns list of (curve_name, eer, eer_threshold).
    """
    rows: List[Tuple[str, float, float]] = []

    if isinstance(det.get("curves"), dict):
        # 9-modes priors script format
        for name, obj in det["curves"].items():
            eer = float(obj.get("eer"))
            thr = float(obj.get("eer_threshold"))
            rows.append((name, eer, thr))
        return rows

    if isinstance(det.get("settings"), dict):
        # smoothing sweep format
        for name, obj in det["settings"].items():
            eer = float(obj.get("eer"))
            thr = float(obj.get("eer_threshold"))
            rows.append((name, eer, thr))
        return rows

    if isinstance(det.get("methods"), dict):
        # 3-method format
        for name, obj in det["methods"].items():
            eer = float(obj.get("eer"))
            thr = float(obj.get("eer_threshold"))
            rows.append((name, eer, thr))
        return rows

    raise ValueError("Unrecognized DET JSON format. Expected keys: 'curves', 'settings', or 'methods'.")


def main():
    args = parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        det = json.load(f)

    rows = _extract_rows(det)

    if args.sort == "name":
        rows.sort(key=lambda r: r[0])
    elif args.sort == "eer":
        rows.sort(key=lambda r: r[1])
    elif args.sort == "thr":
        rows.sort(key=lambda r: r[2])

    if args.topk and args.topk > 0:
        rows = rows[: args.topk]

    # Print a neat table
    print(f"File: {args.json_path}")
    print(f"{'curve':60s}  {'EER':>10s}  {'EER_threshold':>16s}")
    print("-" * 92)
    for name, eer, thr in rows:
        # thresholds can be inf/-inf if something went weird; print as-is
        print(f"{name:60s}  {eer:10.6f}  {thr:16.8f}")

    # Helpful reminder
    print("\nNote: EER_threshold is in the same units as the curve's score.")
    print("  - For A_log_p: log(prob)")
    print("  - For B_one_vs_rest_lse: log-space target-vs-rest score")
    print("  - For C_minus_max_other: log-space margin vs best competitor")


if __name__ == "__main__":
    main()