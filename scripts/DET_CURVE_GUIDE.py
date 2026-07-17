#!/usr/bin/env python3
"""
DET CURVE PLOTTING GUIDE
========================

This guide explains how to use plot_det_curves.py to analyze embedded language
detection performance by varying the confidence threshold.

WORKFLOW
========

1. Run inference with dump_posteriors=True to get frame-level posteriors:
   
   python -m src.main \
     experiment=inference/langdiar_mms_multi \
     run_folder=my_det_analysis \
     inference.inference_runner.ckpt_path=path/to/checkpoint.ckpt \
     inference.inference_runner.dump_posteriors=true

2. This generates output JSONL files with posteriors and timing info

3. Run DET curve plotting script:
   
   python scripts/plot_det_curves.py \
     --input_jsonl_glob "exp/runs/my_det_analysis/*.jsonl" \
     --output_plot det_curve.png \
     --collar 0.0 \
     --title "DET Curve: Embedded Language Detection"

UNDERSTANDING THE SCRIPT
========================

Input Format:
  Each JSONL line contains:
  {
    "utt_id": {
      "posteriors": {
        "values": [[0.1, 0.9], [0.2, 0.8], ...],  # Frame posteriors (frames x vocab_size)
        "frame_times": [0.0, 0.04, 0.08, ...],    # Start time of each frame
        "frame_duration": 0.04                     # Duration per frame
      },
      "passthrough": {
        "segment_timestamps": [[0, 2.5], [2.5, 5]],
        "segment_langs": ["ara", "eng"],
        ...
      }
    }
  }

Threshold Variation:
  - For each threshold T in [0, 1]:
    1. Extract frame-level posterior for embedded language (class_idx=1)
    2. Threshold: if posterior[frame][1] >= T, label as embedded (2), else matrix (1)
    3. Convert frame labels to time-based segments
    4. Compare against reference labels
    5. Compute Miss Rate and FA Rate

Metrics (all time-weighted):
  - Miss Rate = Miss_seconds / RefSpeech_seconds
    (when frame should be embedded but predicted matrix)
  - FA Rate = FA_seconds / RefSpeech_seconds (or NonRefSpeech_seconds)
    (when frame should be matrix but predicted embedded)
  - All computed in reference time using boundary collar

DET Curve:
  - X-axis: False Alarm Rate (FA) - lower is better
  - Y-axis: Miss Rate (FNR) - lower is better
  - Each point represents one threshold value
  - Curve shows tradeoff between FA and Miss as threshold varies

COMMAND-LINE OPTIONS
====================

Required (one of):
  --input_jsonl FILE              Single JSONL file with posteriors
  --input_jsonl_glob PATTERN      Glob pattern for sharded JSONLs

Optional:
  --collar SECONDS                Boundary collar to ignore transitions (default: 0.0)
  --english_token STR             Token name for embedded language (default: "eng")
  --non_speech_token STR          Optional token to treat as non-speech
  --output_plot FILE              Output plot filename (default: "det_curve.png")
  --thresholds STR                Comma-separated thresholds to test (default: 0,0.1,...,1.0)
  --fa_normalized {ref_speech|non_ref_speech}
                                  How to normalize FA (default: "ref_speech")
  --title STR                     Plot title
  --per_utt                       Print per-utterance results

EXAMPLES
========

Example 1: Basic DET curve
  python scripts/plot_det_curves.py \
    --input_jsonl exp/runs/inference_output.jsonl \
    --output_plot det_curve.png

Example 2: With custom thresholds
  python scripts/plot_det_curves.py \
    --input_jsonl exp/runs/inference_output.jsonl \
    --output_plot det_curve.png \
    --thresholds "0.3,0.4,0.5,0.6,0.7,0.8"

Example 3: With boundary collar (ignore ±250ms around transitions)
  python scripts/plot_det_curves.py \
    --input_jsonl exp/runs/inference_output.jsonl \
    --collar 0.25 \
    --output_plot det_curve_collar.png

Example 4: Globbed JSONL files (sharded inference)
  python scripts/plot_det_curves.py \
    --input_jsonl_glob "exp/runs/my_exp/*.*.jsonl" \
    --output_plot det_curve.png \
    --per_utt

INTERPRETING THE RESULTS
========================

The DET curve shows the relationship between False Alarm Rate and Miss Rate:

- Threshold = 0.0: All frames labeled as embedded (high FA, low Miss)
- Threshold = 0.5: Balanced threshold
- Threshold = 1.0: All frames labeled as matrix (low FA, high Miss)

The "knee" of the curve (minimum combined error) indicates optimal threshold.

Equal Error Rate (EER):
  - Where Miss Rate ≈ FA Rate
  - Often considered a good operating point
  - Find by looking for intersection with y=x line

Common Operating Points:
  - High recall (low Miss): Use low threshold (~0.3-0.4)
  - High precision (low FA): Use high threshold (~0.8-0.9)
  - Balanced: Look at EER or knee of curve

INTEGRATION WITH LDER SCRIPT
============================

The plot_det_curves.py and score_lder_detection.py are complementary:

score_lder_detection.py:
  - Computes LDER using fixed argmax predictions
  - Breaks down errors by reference language token
  - Good for evaluating overall system

plot_det_curves.py:
  - Analyzes confidence-based threshold tradeoffs
  - Shows potential performance with optimized threshold
  - Good for tuning embedded language detection threshold

Workflow:
  1. Run inference with dump_posteriors=true
  2. Plot DET curve to find optimal threshold
  3. Apply optimal threshold in downstream system (e.g., for embedding vs matrix selection)
  4. Evaluate final LDER with that threshold

TROUBLESHOOTING
===============

"No posteriors found for ...":
  - Make sure inference was run with dump_posteriors=true
  - Check JSONL format matches expected structure

NaN values in metrics:
  - Can occur if RefSpeech is 0 or very small
  - Check if reference segments are properly loaded
  - May need to check segment_timestamps and segment_langs in passthrough

Points not labeled on plot:
  - Use smaller font or different annotation strategy for many points
  - Check --thresholds parameter
