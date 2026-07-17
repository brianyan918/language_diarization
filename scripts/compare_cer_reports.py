#!/usr/bin/env python3
"""
Compare CER reports across multiple runs/models.
Outputs a tab-separated table suitable for pasting into Google Sheets.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def read_cer_report(filepath: str) -> Dict:
    """Read a CER report JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_language_data(report: Dict) -> Dict[str, Dict]:
    """Extract per-language CER metrics from report."""
    per_lang = report.get('per_language', {})
    
    # Convert to alphabetically sorted dict
    sorted_langs = {}
    for lang in sorted(per_lang.keys()):
        lang_data = per_lang[lang]
        sorted_langs[lang] = {
            'cer': lang_data.get('cer', 0),
            'edits': lang_data.get('edits', 0),
            'ref_chars': lang_data.get('ref_chars', 0),
            'count': lang_data.get('count', 0),
            'ins': lang_data.get('ins', 0),
            'del': lang_data.get('del', 0),
            'sub': lang_data.get('sub', 0),
        }
    
    return sorted_langs


def format_cer_pct(cer: float) -> str:
    """Format CER as percentage."""
    return f"{cer * 100:.2f}%"


def format_num(n: int) -> str:
    """Format number with thousands separator."""
    return f"{n:,}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_cer_reports.py <report1.json> [report2.json] ...")
        print("\nExample:")
        print("  python compare_cer_reports.py model_a.json model_b.json model_c.json")
        sys.exit(1)
    
    report_paths = sys.argv[1:]
    
    # Load all reports and extract data
    reports = {}
    report_to_path = {}  # Map model name to original path
    all_languages = set()
    
    for idx, path in enumerate(report_paths, 1):
        if not Path(path).exists():
            print(f"Warning: File not found: {path}", file=sys.stderr)
            continue
        
        try:
            report = read_cer_report(path)
            # Use parent directory name + index if multiple reports have same filename
            parent_name = Path(path).parent.name
            file_name = Path(path).stem
            model_name = parent_name if parent_name else file_name
            
            # If model_name already exists, append index to make it unique
            if model_name in reports:
                model_name = f"{model_name}_{idx}"
            
            reports[model_name] = extract_language_data(report)
            report_to_path[model_name] = path
            all_languages.update(reports[model_name].keys())
            print(f"Loaded: {model_name} from {path}", file=sys.stderr)
        except Exception as e:
            print(f"Error reading {path}: {e}", file=sys.stderr)
            continue
    
    if not reports:
        print("No valid reports loaded.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(reports)} report(s): {', '.join(sorted(reports.keys()))}", file=sys.stderr)
    
    # Sort languages alphabetically
    sorted_langs = sorted(all_languages)
    sorted_models = sorted(reports.keys())
    
    # Build header row
    header = ["Language"]
    for model in sorted_models:
        header.append(f"{model} (CER)")
        header.append(f"{model} (Edits)")
        header.append(f"{model} (Ref Chars)")
        header.append(f"{model} (Count)")
    
    # Print header
    print("\t".join(header))
    
    # Print data rows
    for lang in sorted_langs:
        row = [lang]
        for model in sorted_models:
            if lang in reports[model]:
                data = reports[model][lang]
                row.append(format_cer_pct(data['cer']))
                row.append(format_num(data['edits']))
                row.append(format_num(data['ref_chars']))
                row.append(format_num(data['count']))
            else:
                row.extend(["N/A", "N/A", "N/A", "N/A"])
        
        print("\t".join(row))
    
    # Print summary row
    print()
    summary_header = ["OVERALL"]
    for model in sorted_models:
        summary_header.append(f"{model} (CER)")
        summary_header.append(f"{model} (Edits)")
        summary_header.append(f"{model} (Ref Chars)")
        summary_header.append(f"{model} (Count)")
    
    print("\t".join(summary_header))
    
    overall_data = defaultdict(dict)
    for model_name in sorted_models:
        path = report_to_path[model_name]
        try:
            report = read_cer_report(path)
            overall = report.get('overall', {})
            overall_data[model_name] = {
                'cer': overall.get('cer', 0),
                'edits': overall.get('edits', 0),
                'ref_chars': overall.get('ref_chars', 0),
                'count': overall.get('scored_utterances', 0),
            }
        except:
            pass
    
    summary_row = [""]
    for model in sorted_models:
        if model in overall_data:
            data = overall_data[model]
            summary_row.append(format_cer_pct(data['cer']))
            summary_row.append(format_num(data['edits']))
            summary_row.append(format_num(data['ref_chars']))
            summary_row.append(format_num(data['count']))
        else:
            summary_row.extend(["N/A", "N/A", "N/A", "N/A"])
    
    print("\t".join(summary_row))
    
    # Print easy copy-paste section
    print("\n" + "="*80)
    print("EASY COPY-PASTE FORMAT")
    print("="*80 + "\n")
    
    for model in sorted_models:
        # Get the full path for context
        path = report_to_path[model]
        print(f"<{path}>; cer in alpha order")
        
        if model in reports:
            for lang in sorted_langs:
                if lang in reports[model]:
                    cer_pct = format_cer_pct(reports[model][lang]['cer'])
                    print(cer_pct)
        print()


if __name__ == "__main__":
    main()
