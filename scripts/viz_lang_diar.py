#!/usr/bin/env python3
"""
viz_lang_diar.py

Visualize language diarization as horizontal colored bars (REF vs HYP) per utterance.

Changes from your version:
- Instead of random/reservoir sampling, this selects the FIRST N valid utterances
  in file order (and line order within each file).

Everything else is unchanged.
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl


@dataclass
class Seg:
    start: float
    end: float
    label: int


# -------------------------
# I/O + parsing
# -------------------------
def load_vocab_id_token(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Parse vocab lines like '2 eng' -> lang->id and id->lang."""
    tok2id: Dict[str, int] = {}
    id2tok: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Bad vocab line at {path}:{ln}: {line}")
            idx = int(parts[0])
            tok = parts[1]
            tok2id[tok] = idx
            id2tok[idx] = tok
    if not tok2id:
        raise ValueError(f"Empty vocab: {path}")
    return tok2id, id2tok


def iter_jsonl_inputs(
    jsonl_path: Optional[str] = None,
    jsonl_glob: Optional[str] = None,
) -> Iterator[Tuple[str, int, dict]]:
    if (jsonl_path is None) == (jsonl_glob is None):
        raise ValueError("Provide exactly one of --input_jsonl or --input_jsonl_glob")

    if jsonl_path is not None:
        paths = [jsonl_path]
    else:
        paths = sorted(glob.glob(jsonl_glob or ""))
        if not paths:
            raise RuntimeError(f"No JSONL files matched glob: {jsonl_glob}")

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield p, ln, json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"JSON decode error in {p}:{ln}: {e}") from e


def merge_adjacent(segs: List[Seg]) -> List[Seg]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: (s.start, s.end))
    out = [Seg(segs[0].start, segs[0].end, segs[0].label)]
    for s in segs[1:]:
        prev = out[-1]
        if abs(s.start - prev.end) <= 1e-9 and s.label == prev.label:
            prev.end = max(prev.end, s.end)
        else:
            out.append(Seg(s.start, s.end, s.label))
    return out


def build_ref_segments(passthrough: dict, tok2id: Dict[str, int]) -> List[Seg]:
    ts = passthrough.get("segment_timestamps", [])
    langs = passthrough.get("segment_langs", [])
    if len(ts) != len(langs):
        raise ValueError(f"segment_timestamps and segment_langs mismatch: {len(ts)} vs {len(langs)}")
    ref: List[Seg] = []
    for (s, e), lang in zip(ts, langs):
        lab = tok2id.get(lang, -1)
        s = float(s)
        e = float(e)
        if e > s:
            ref.append(Seg(s, e, lab))
    return merge_adjacent(ref)


def build_hyp_segments(pred_list: List[dict]) -> List[Seg]:
    hyp: List[Seg] = []
    for d in pred_list:
        s = float(d["start"])
        e = float(d["end"])
        lab = int(d["label"])
        if e > s:
            hyp.append(Seg(s, e, lab))
    return merge_adjacent(hyp)


def get_utt_title(utt_key: str, passthrough: dict) -> str:
    utt_id = passthrough.get("utt_id", "")
    fn = passthrough.get("file_name", "")
    if utt_id:
        return f"{utt_key} | {utt_id}"
    if fn:
        return f"{utt_key} | {os.path.basename(fn)}"
    return utt_key


# -------------------------
# Plotting
# -------------------------
def label_name(label: int, id2tok: Dict[int, str]) -> str:
    if label == -1:
        return "UNK"
    return id2tok.get(label, str(label))


def color_for_label(label: int, cmap) -> Tuple[float, float, float, float]:
    if label == -1:
        return (0.5, 0.5, 0.5, 1.0)  # gray for unknown
    x = (abs(label) * 2654435761) % 2**32
    v = (x / 2**32)
    return cmap(v)


def clamp_segments(segs: List[Seg], tmin: float, tmax: float) -> List[Seg]:
    out = []
    for s in segs:
        a = max(tmin, s.start)
        b = min(tmax, s.end)
        if b > a:
            out.append(Seg(a, b, s.label))
    return out


def plot_utt(
    ref: List[Seg],
    hyp: List[Seg],
    title: str,
    id2tok: Dict[int, str],
    out_path: str,
    max_duration: Optional[float],
):
    t0 = 0.0
    t1 = 0.0
    for s in ref + hyp:
        t1 = max(t1, s.end)

    if max_duration is not None and max_duration > 0:
        t1 = min(t1, max_duration)
        ref = clamp_segments(ref, t0, t1)
        hyp = clamp_segments(hyp, t0, t1)

    fig, ax = plt.subplots(figsize=(14, 2.2), dpi=150)
    cmap = mpl.cm.get_cmap("tab20")

    bar_h = 0.35
    y_ref = 0.65
    y_hyp = 0.15

    def draw_row(segs: List[Seg], y: float):
        for s in segs:
            ax.broken_barh(
                [(s.start, s.end - s.start)],
                (y, bar_h),
                facecolors=color_for_label(s.label, cmap),
                edgecolors="none",
            )

    draw_row(ref, y_ref)
    draw_row(hyp, y_hyp)

    ax.set_xlim(t0, max(t1, 0.01))
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(title, fontsize=9)

    present = []
    seen = set()
    for s in ref + hyp:
        if s.label not in seen:
            seen.add(s.label)
            present.append(s.label)

    present = present[:12]
    handles = [
        mpl.patches.Patch(color=color_for_label(lab, cmap), label=label_name(lab, id2tok))
        for lab in present
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.35),
            ncol=min(len(handles), 6),
            frameon=False,
            fontsize=8,
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", help="Single JSONL file.")
    ap.add_argument("--input_jsonl_glob", help='Glob for sharded JSONLs (e.g., "/path/*.jsonl").')
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num", type=int, default=20, help="Number of utterances to visualize (first N after sorting).")
    ap.add_argument("--max_duration", type=float, default=0.0)
    ap.add_argument("--prefix", type=str, default="utt")
    args = ap.parse_args()

    tok2id, id2tok = load_vocab_id_token(args.vocab)

    # ------------------------------------------------------------------
    # Collect all valid utterances
    # ------------------------------------------------------------------
    all_utts: List[Tuple[str, dict]] = []

    for _src, _ln, obj in iter_jsonl_inputs(
        jsonl_path=args.input_jsonl,
        jsonl_glob=args.input_jsonl_glob,
    ):
        if not isinstance(obj, dict) or len(obj) != 1:
            continue

        utt_key = next(iter(obj.keys()))  # e.g. "1330"
        entry = obj[utt_key]
        all_utts.append((utt_key, entry))

    if not all_utts:
        raise RuntimeError("No valid utterances found in input.")

    # ------------------------------------------------------------------
    # Sort by utterance key
    #   - numeric if possible
    #   - otherwise lexicographic
    # ------------------------------------------------------------------
    def sort_key(x):
        k = x[0]
        try:
            return int(k)
        except ValueError:
            return k

    all_utts.sort(key=sort_key)

    # Take first N
    selected = all_utts[: args.num]

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    max_dur = args.max_duration if args.max_duration and args.max_duration > 0 else None

    for i, (utt_key, entry) in enumerate(selected):
        pred_list = entry.get("pred", [])
        passthrough = entry.get("passthrough", {})

        ref = build_ref_segments(passthrough, tok2id)
        hyp = build_hyp_segments(pred_list)

        title = get_utt_title(utt_key, passthrough)
        out_path = os.path.join(args.out_dir, f"{args.prefix}.{i:04d}.{utt_key}.png")

        plot_utt(ref, hyp, title, id2tok, out_path, max_duration=max_dur)

    print(f"Wrote {len(selected)} figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
