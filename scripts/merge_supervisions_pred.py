#!/usr/bin/env python3
"""
Merge two JSONL.GZ files: one with supervisions and one with supervisions_pred.

The script matches cuts by ID, handling prefix removal in the supervisions_pred file.
Example: train-15740-952_ita-spk0_sample0 (pred) matches ita-spk0_sample0 (supervisions)

Usage:
    python merge_supervisions_pred.py \
        --supervisions-file path/to/supervisions.jsonl.gz \
        --pred-file path/to/supervisions_pred.jsonl.gz \
        --output path/to/output.jsonl.gz \
        [--prefix-pattern PATTERN]
"""

import gzip
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Set
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_lang_speaker_mapping_from_cut(cut: Dict) -> Dict[str, str]:
    """
    Extract mapping from language to speaker ID from a single cut's supervisions.
    
    Validates that each language maps to exactly one speaker within this cut.
    
    Args:
        cut: Single cut dict with 'supervisions' field
        
    Returns:
        Dict mapping language code to speaker ID (e.g., {'ar': 'ara-spk0_ar', 'en': 'eng-spk0_en'})
        
    Raises:
        ValueError: If language-to-speaker mapping is not one-to-one within the cut
    """
    lang_to_speaker = {}
    
    if 'supervisions' not in cut:
        return lang_to_speaker
    
    supervisions = cut['supervisions']
    for sup in supervisions:
        if not isinstance(sup, dict):
            continue
        
        speaker = sup.get('speaker')
        if not speaker:
            continue
        
        # Extract language from speaker ID (last part after underscore)
        # e.g., "ara-spk0_ar" -> language is "ar"
        parts = speaker.split('_')
        if len(parts) < 2:
            logger.warning(f"Speaker ID '{speaker}' doesn't have language suffix (format: name_lang)")
            continue
        
        lang = parts[-1]
        
        # Check if this language already has a mapping within this cut
        if lang in lang_to_speaker:
            existing_speaker = lang_to_speaker[lang]
            if existing_speaker != speaker:
                raise ValueError(
                    f"Language '{lang}' maps to multiple speakers in this cut: "
                    f"'{existing_speaker}' vs '{speaker}'. "
                    f"Language-to-speaker mapping must be one-to-one."
                )
        else:
            lang_to_speaker[lang] = speaker
    
    return lang_to_speaker


def apply_speaker_mapping_to_pred(pred_supervision: Dict, lang_to_speaker: Dict[str, str]) -> Dict:
    """
    Update speaker ID in a predicted supervision using language-to-speaker mapping.
    
    Args:
        pred_supervision: Predicted supervision dict (with 'speaker' and 'language' fields)
        lang_to_speaker: Mapping from language to speaker ID
        
    Returns:
        Updated supervision dict with mapped speaker ID
    """
    language = pred_supervision.get('language')
    
    if not language:
        logger.debug(f"Supervision missing 'language' field, skipping speaker mapping")
        return pred_supervision
    
    if language not in lang_to_speaker:
        logger.warning(f"Language '{language}' not found in mapping, keeping original speaker")
        return pred_supervision
    
    original_speaker = pred_supervision.get('speaker')
    mapped_speaker = lang_to_speaker[language]
    
    pred_supervision['speaker'] = mapped_speaker
    
    logger.debug(f"Mapped speaker for lang={language}: '{original_speaker}' -> '{mapped_speaker}'")
    
    return pred_supervision


def extract_clean_id(cut_id: str, prefix_pattern: str = None) -> str:
    """
    Extract clean ID by removing prefix.
    
    Args:
        cut_id: Original cut ID
        prefix_pattern: Regex pattern to match prefix to remove. If None, removes "train-XXXXX-" prefix
                       Example: train-15740-952_ita-spk0_sample0 --> 952_ita-spk0_sample0
    
    Returns:
        Clean ID after prefix removal
    """
    if prefix_pattern:
        # Use custom regex pattern
        match = re.search(prefix_pattern, cut_id)
        if match:
            return match.group(1)
    else:
        # Default: remove "train-XXXXX-" prefix, keeping everything from first digit sequence onwards
        # Pattern: train-<digits>-<rest> --> <rest>
        match = re.match(r'^train-\d+-(.+)$', cut_id)
        if match:
            return match.group(1)
    
    return cut_id


def load_jsonl_gz(filepath: str) -> Dict:
    """
    Load JSONL.GZ file and return dict of cuts indexed by ID.
    
    Args:
        filepath: Path to JSONL.GZ file
        
    Returns:
        Dict mapping cut ID to cut object
    """
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


def save_jsonl_gz(cuts: Dict, filepath: str) -> None:
    """
    Save dict of cuts to JSONL.GZ file.
    
    Args:
        cuts: Dict of cuts indexed by ID
        filepath: Output JSONL.GZ file path
    """
    logger.info(f"Writing {len(cuts)} cuts to {filepath}...")
    
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        for cut_id, cut in cuts.items():
            f.write(json.dumps(cut) + '\n')
    
    logger.info(f"Successfully wrote {len(cuts)} cuts")


def merge_files(supervisions_file: str, pred_file: str, output_file: str, prefix_pattern: str = None) -> None:
    """
    Merge supervisions file with supervisions_pred file.
    
    Args:
        supervisions_file: Path to main JSONL.GZ file with supervisions
        pred_file: Path to JSONL.GZ file with supervisions_pred
        output_file: Path to output JSONL.GZ file
        prefix_pattern: Optional regex pattern for ID extraction
    """
    # Load both files
    supervisions_cuts = load_jsonl_gz(supervisions_file)
    pred_cuts = load_jsonl_gz(pred_file)
    
    logger.info(f"Supervisions file: {len(supervisions_cuts)} cuts")
    logger.info(f"Predictions file: {len(pred_cuts)} cuts")
    
    # Build mapping: clean_pred_id -> pred_cut
    pred_mapping = {}
    for pred_id, pred_cut in pred_cuts.items():
        clean_id = extract_clean_id(pred_id, prefix_pattern)
        if clean_id in pred_mapping:
            logger.warning(f"Duplicate clean ID '{clean_id}' from '{pred_id}', keeping first occurrence")
            continue
        pred_mapping[clean_id] = pred_cut
    
    logger.info(f"Created mapping with {len(pred_mapping)} unique clean IDs")
    
    # Merge: supervisions + supervisions_pred
    merged_cuts = {}
    matched_count = 0
    unmatched_count = 0
    speaker_mapping_failures = 0
    
    for sup_id, sup_cut in supervisions_cuts.items():
        if sup_id in pred_mapping:
            pred_cut = pred_mapping[sup_id]
            
            # Extract language-to-speaker mapping from THIS cut's supervisions
            try:
                lang_to_speaker = extract_lang_speaker_mapping_from_cut(sup_cut)
            except ValueError as e:
                logger.warning(f"Cut {sup_id}: Failed to extract language-to-speaker mapping: {e}")
                lang_to_speaker = {}
            
            # Merge supervisions_pred into supervisions cut
            if 'supervisions' in pred_cut:
                supervisions_pred = pred_cut['supervisions']
                # Apply speaker mapping to predicted supervisions
                mapped_supervisions = []
                for sup in supervisions_pred:
                    try:
                        mapped_sup = apply_speaker_mapping_to_pred(sup, lang_to_speaker)
                        # Check if mapping actually happened (language was found)
                        if sup.get('language') and sup.get('language') not in lang_to_speaker:
                            logger.warning(f"Cut {sup_id}: Language '{sup.get('language')}' not found in ground truth supervisions")
                            speaker_mapping_failures += 1
                        mapped_supervisions.append(mapped_sup)
                    except Exception as e:
                        logger.warning(f"Cut {sup_id}: Failed to map speaker in supervision: {e}")
                        speaker_mapping_failures += 1
                        mapped_supervisions.append(sup)  # Keep original if mapping fails
                
                sup_cut['supervisions_pred'] = mapped_supervisions
            else:
                logger.warning(f"Cut {sup_id}: 'supervisions' field missing in predictions, skipping")
                sup_cut['supervisions_pred'] = None
            merged_cuts[sup_id] = sup_cut
            matched_count += 1
        else:
            unmatched_count += 1
            logger.debug(f"No prediction found for supervision ID: {sup_id}")
    
    logger.info(f"Matched: {matched_count}/{len(supervisions_cuts)} cuts")
    logger.info(f"Unmatched: {unmatched_count}/{len(supervisions_cuts)} cuts")
    
    if speaker_mapping_failures > 0:
        logger.warning(f"WARNING: {speaker_mapping_failures} speaker mapping failures detected!")
    
    if unmatched_count > 0:
        unmatched_pct = (unmatched_count / len(supervisions_cuts)) * 100
        logger.warning(f"Warning: {unmatched_pct:.1f}% of supervisions have no matching predictions")
    
    # Save merged file
    save_jsonl_gz(merged_cuts, output_file)
    logger.info(f"Merge complete: {len(merged_cuts)} cuts in output")


def main():
    parser = argparse.ArgumentParser(
        description='Merge two JSONL.GZ files: supervisions + supervisions_pred',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default prefix removal
  python merge_supervisions_pred.py \\
    --supervisions-file data.jsonl.gz \\
    --pred-file predictions.jsonl.gz \\
    --output merged.jsonl.gz

  # Custom prefix pattern (keep everything after "lang-" prefix)
  python merge_supervisions_pred.py \\
    --supervisions-file data.jsonl.gz \\
    --pred-file predictions.jsonl.gz \\
    --output merged.jsonl.gz \\
    --prefix-pattern 'lang-(.+)'
        """
    )
    
    parser.add_argument(
        '--supervisions-file',
        required=True,
        help='Path to JSONL.GZ file with supervisions (main file)'
    )
    parser.add_argument(
        '--pred-file',
        required=True,
        help='Path to JSONL.GZ file with supervisions_pred'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output JSONL.GZ file'
    )
    parser.add_argument(
        '--prefix-pattern',
        default=None,
        help='Regex pattern to extract clean ID (default: auto-detect language code)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input files exist
    if not Path(args.supervisions_file).exists():
        logger.error(f"Supervisions file not found: {args.supervisions_file}")
        return 1
    if not Path(args.pred_file).exists():
        logger.error(f"Predictions file not found: {args.pred_file}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        merge_files(
            args.supervisions_file,
            args.pred_file,
            args.output,
            args.prefix_pattern
        )
        logger.info("Merge completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Merge failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
