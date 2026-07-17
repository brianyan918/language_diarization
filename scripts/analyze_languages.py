#!/usr/bin/env python3
"""
Analyze languages across JSONL.GZ cutset files.

Summarizes:
- Unique languages found in supervisions
- Language distribution across cuts
- Anomalies (invalid language codes, unusual patterns)
- Speaker-language mappings

Usage:
    python analyze_languages.py path/to/file1.jsonl.gz [path/to/file2.jsonl.gz ...]
    python analyze_languages.py --supervisions path/to/sups.jsonl.gz --predictions path/to/preds.jsonl.gz
"""

import gzip
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl_gz(filepath: str) -> Dict:
    """Load JSONL.GZ file and return dict of cuts indexed by ID."""
    cuts = {}
    logger.info(f"Loading {filepath}...")
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                cut = json.loads(line)
                cut_id = cut.get('id')
                if not cut_id:
                    logger.warning(f"Line {line_num}: Missing 'id' field, skipping")
                    continue
                cuts[cut_id] = cut
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Failed to parse JSON - {e}")
                continue
    
    logger.info(f"Loaded {len(cuts)} cuts")
    return cuts


def extract_language_from_supervision(supervision: Dict) -> tuple:
    """
    Extract language from a supervision entry without filtering.

    Order of preference:
    1) supervision['language'] if present
    2) speaker suffix after last underscore
    3) speaker itself if no underscore

    Returns:
        (language, issue)
        - language: extracted language string or None if unavailable
        - issue: optional issue tag for diagnostics
    """
    if not isinstance(supervision, dict):
        return None, "non_dict_supervision"

    language = supervision.get("language")
    if language is not None and language != "":
        return str(language), None

    speaker = supervision.get("speaker")
    if speaker:
        if "_" in speaker:
            return speaker.split("_")[-1], None
        return str(speaker), "no_language_suffix"

    return None, "missing_language"


def analyze_supervisions(cuts: Dict, name: str = "Supervisions") -> None:
    """Analyze languages in supervisions across all cuts."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing {name}")
    logger.info(f"{'='*80}")
    
    languages = defaultdict(int)
    language_speakers = defaultdict(set)  # lang -> set of speakers
    anomalies = defaultdict(list)  # anomaly type -> list of (cut_id, value)
    cuts_with_multiple_langs = 0
    
    for cut_id, cut in cuts.items():
        if 'supervisions' not in cut:
            continue
        
        supervisions = cut['supervisions']
        cut_langs = set()
        
        for sup_idx, sup in enumerate(supervisions):
            lang, issue = extract_language_from_supervision(sup)
            if issue:
                anomalies[issue].append((cut_id, sup_idx))

            if lang is None:
                continue

            cut_langs.add(lang)
            languages[lang] += 1

            speaker = sup.get('speaker') if isinstance(sup, dict) else None
            if speaker:
                language_speakers[lang].add(speaker)
            else:
                anomalies['missing_speaker'].append((cut_id, sup_idx))
        
        if len(cut_langs) > 1:
            cuts_with_multiple_langs += 1
    
    # Print summary
    logger.info(f"\nTotal supervisions analyzed: {sum(languages.values())}")
    logger.info(f"Unique languages: {len(languages)}")
    logger.info(f"Cuts with multiple languages: {cuts_with_multiple_langs}")
    
    logger.info(f"\nLanguage distribution:")
    for lang in sorted(languages.keys(), key=lambda x: languages[x], reverse=True):
        count = languages[lang]
        num_speakers = len(language_speakers[lang])
        logger.info(f"  {lang:6s}: {count:6d} supervisions, {num_speakers:3d} speakers")
    
    logger.info(f"\nSpeaker-Language mappings:")
    for lang in sorted(languages.keys()):
        speakers = sorted(language_speakers[lang])
        for speaker in speakers:
            logger.info(f"  {lang} -> {speaker}")
    
    if anomalies:
        logger.warning(f"\nAnomalies detected:")
        for anomaly_type, items in sorted(anomalies.items()):
            logger.warning(f"\n  {anomaly_type}: {len(items)} occurrences")
            # Show first 5 examples
            for cut_id, value in items[:5]:
                logger.warning(f"    - Cut {cut_id}: {value}")
            if len(items) > 5:
                logger.warning(f"    ... and {len(items) - 5} more")


def compare_files(sups_file: str, pred_file: str) -> None:
    """Compare languages between supervisions and predictions files."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparing supervisions vs predictions")
    logger.info(f"{'='*80}")
    
    sups_cuts = load_jsonl_gz(sups_file)
    pred_cuts = load_jsonl_gz(pred_file)
    
    analyze_supervisions(sups_cuts, "Supervisions")
    
    logger.info(f"\n{'-'*80}\n")
    analyze_supervisions(pred_cuts, "Predictions")
    
    # Compare language sets
    sups_langs = set()
    pred_langs = set()
    
    for cut in sups_cuts.values():
        if 'supervisions' in cut:
            for sup in cut['supervisions']:
                lang, _ = extract_language_from_supervision(sup)
                if lang is not None:
                    sups_langs.add(lang)
    
    for cut in pred_cuts.values():
        if 'supervisions' in cut:
            for sup in cut['supervisions']:
                lang, _ = extract_language_from_supervision(sup)
                if lang is not None:
                    pred_langs.add(lang)
    
    logger.info(f"\n{'-'*80}")
    logger.info(f"Language comparison:")
    logger.info(f"  Supervisions only: {sups_langs - pred_langs}")
    logger.info(f"  Predictions only: {pred_langs - sups_langs}")
    logger.info(f"  Common: {sups_langs & pred_langs}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze languages across JSONL.GZ cutset files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  python analyze_languages.py data.jsonl.gz

  # Analyze multiple files
  python analyze_languages.py data1.jsonl.gz data2.jsonl.gz data3.jsonl.gz

  # Compare supervisions and predictions
  python analyze_languages.py --supervisions data.jsonl.gz --predictions predictions.jsonl.gz

  # Verbose output
  python analyze_languages.py --verbose data.jsonl.gz
        """
    )
    
    parser.add_argument(
        'files',
        nargs='*',
        help='JSONL.GZ files to analyze'
    )
    parser.add_argument(
        '--supervisions',
        help='Path to supervisions file (for comparison with predictions)'
    )
    parser.add_argument(
        '--predictions',
        help='Path to predictions file (for comparison with supervisions)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Mode 1: Compare supervisions and predictions
    if args.supervisions and args.predictions:
        if not Path(args.supervisions).exists():
            logger.error(f"Supervisions file not found: {args.supervisions}")
            return 1
        if not Path(args.predictions).exists():
            logger.error(f"Predictions file not found: {args.predictions}")
            return 1
        
        try:
            compare_files(args.supervisions, args.predictions)
            logger.info("\nAnalysis complete!")
            return 0
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return 1
    
    # Mode 2: Analyze individual files
    if args.files:
        for filepath in args.files:
            if not Path(filepath).exists():
                logger.error(f"File not found: {filepath}")
                return 1
            
            try:
                cuts = load_jsonl_gz(filepath)
                analyze_supervisions(cuts, f"File: {filepath}")
            except Exception as e:
                logger.error(f"Analysis failed for {filepath}: {e}", exc_info=True)
                return 1
        
        logger.info("\nAnalysis complete!")
        return 0
    
    # No files specified
    parser.print_help()
    return 1


if __name__ == '__main__':
    exit(main())
