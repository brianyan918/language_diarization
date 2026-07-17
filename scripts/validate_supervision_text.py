#!/usr/bin/env python3
"""
Validate supervision text fields in gzipped JSONL data.

Checks that:
1. Each record has a "supervisions" array
2. Each supervision in the array has a "text" field
3. The "text" field is not empty or just whitespace

Usage:
  python validate_supervision_text.py -i data.jsonl.gz
  python validate_supervision_text.py -i data.jsonl.gz --fix-missing output.jsonl.gz
"""

import gzip
import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict


def validate_file(input_path, verbose=False, output_path=None, fix_missing=False):
    """
    Validate supervision text fields in gzipped JSONL.
    
    Args:
        input_path: Path to gzipped JSONL file
        verbose: Print details for all issues
        output_path: Write filtered/fixed data here if specified
        fix_missing: Remove supervisions with missing/empty text
    
    Returns:
        Dictionary with statistics
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)
    
    stats = {
        'total_records': 0,
        'total_supervisions': 0,
        'records_with_issues': 0,
        'supervisions_with_empty_text': 0,
        'supervisions_removed': 0,
        'records_removed': 0,
        'issues_by_type': defaultdict(int),
    }
    
    problematic_records = []
    fixed_records = []
    
    # Read and validate
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: Line {line_no} - Invalid JSON: {e}")
                stats['issues_by_type']['invalid_json'] += 1
                continue
            
            stats['total_records'] += 1
            record_id = record.get('id', f'line_{line_no}')
            
            # Check supervisions field
            if 'supervisions' not in record:
                print(f"ERROR: Record {record_id} (line {line_no}) - Missing 'supervisions' field")
                stats['issues_by_type']['missing_supervisions'] += 1
                stats['records_with_issues'] += 1
                problematic_records.append({
                    'line': line_no,
                    'record_id': record_id,
                    'issue': 'missing_supervisions'
                })
                if not fix_missing:
                    continue
                else:
                    stats['records_removed'] += 1
                    continue
            
            supervisions = record.get('supervisions', [])
            if not isinstance(supervisions, list):
                print(f"ERROR: Record {record_id} (line {line_no}) - 'supervisions' is not a list")
                stats['issues_by_type']['supervisions_not_list'] += 1
                stats['records_with_issues'] += 1
                problematic_records.append({
                    'line': line_no,
                    'record_id': record_id,
                    'issue': 'supervisions_not_list'
                })
                if not fix_missing:
                    continue
                else:
                    stats['records_removed'] += 1
                    continue
            
            stats['total_supervisions'] += len(supervisions)
            
            # Check each supervision
            has_empty_text = False
            fixed_supervisions = []
            
            for seg_idx, sup in enumerate(supervisions):
                sup_id = sup.get('id', f'{record_id}_seg{seg_idx}')
                
                # Check text field exists
                if 'text' not in sup:
                    print(f"WARNING: Record {record_id}, supervision {seg_idx} ({sup_id}) - Missing 'text' field")
                    stats['issues_by_type']['missing_text'] += 1
                    stats['supervisions_with_empty_text'] += 1
                    stats['records_with_issues'] += 1
                    has_empty_text = True
                    if not fix_missing:
                        problematic_records.append({
                            'line': line_no,
                            'record_id': record_id,
                            'sup_idx': seg_idx,
                            'sup_id': sup_id,
                            'issue': 'missing_text'
                        })
                    continue
                
                # Check text is not empty
                text = sup.get('text', '').strip()
                if not text:
                    print(f"WARNING: Record {record_id}, supervision {seg_idx} ({sup_id}) - Empty 'text' field")
                    stats['issues_by_type']['empty_text'] += 1
                    stats['supervisions_with_empty_text'] += 1
                    stats['records_with_issues'] += 1
                    has_empty_text = True
                    if not fix_missing:
                        problematic_records.append({
                            'line': line_no,
                            'record_id': record_id,
                            'sup_idx': seg_idx,
                            'sup_id': sup_id,
                            'issue': 'empty_text',
                            'text': repr(sup.get('text', ''))
                        })
                    continue
                
                # Check duration is valid
                duration = sup.get('duration')
                if duration is None:
                    print(f"WARNING: Record {record_id}, supervision {seg_idx} ({sup_id}) - Missing 'duration' field")
                    stats['issues_by_type']['missing_duration'] += 1
                    stats['supervisions_with_empty_text'] += 1
                    stats['records_with_issues'] += 1
                    has_empty_text = True
                    if not fix_missing:
                        problematic_records.append({
                            'line': line_no,
                            'record_id': record_id,
                            'sup_idx': seg_idx,
                            'sup_id': sup_id,
                            'issue': 'missing_duration'
                        })
                    continue
                
                if not isinstance(duration, (int, float)) or duration <= 0:
                    print(f"WARNING: Record {record_id}, supervision {seg_idx} ({sup_id}) - Invalid duration: {duration}")
                    stats['issues_by_type']['invalid_duration'] += 1
                    stats['supervisions_with_empty_text'] += 1
                    stats['records_with_issues'] += 1
                    has_empty_text = True
                    if not fix_missing:
                        problematic_records.append({
                            'line': line_no,
                            'record_id': record_id,
                            'sup_idx': seg_idx,
                            'sup_id': sup_id,
                            'issue': 'invalid_duration',
                            'duration': duration
                        })
                    continue
                
                # Check start time is valid
                start = sup.get('start')
                if start is None:
                    print(f"WARNING: Record {record_id}, supervision {seg_idx} ({sup_id}) - Missing 'start' field")
                    stats['issues_by_type']['missing_start'] += 1
                    stats['supervisions_with_empty_text'] += 1
                    stats['records_with_issues'] += 1
                    has_empty_text = True
                    if not fix_missing:
                        problematic_records.append({
                            'line': line_no,
                            'record_id': record_id,
                            'sup_idx': seg_idx,
                            'sup_id': sup_id,
                            'issue': 'missing_start'
                        })
                    continue
                
                if not isinstance(start, (int, float)) or start < 0:
                    print(f"WARNING: Record {record_id}, supervision {seg_idx} ({sup_id}) - Invalid start time: {start}")
                    stats['issues_by_type']['invalid_start'] += 1
                    stats['supervisions_with_empty_text'] += 1
                    stats['records_with_issues'] += 1
                    has_empty_text = True
                    if not fix_missing:
                        problematic_records.append({
                            'line': line_no,
                            'record_id': record_id,
                            'sup_idx': seg_idx,
                            'sup_id': sup_id,
                            'issue': 'invalid_start',
                            'start': start
                        })
                    continue
                
                # This supervision is valid
                if fix_missing:
                    fixed_supervisions.append(sup)
            
            # Handle fixed records
            if fix_missing:
                if len(fixed_supervisions) == 0:
                    # All supervisions were bad, remove this record
                    stats['records_removed'] += 1
                elif len(fixed_supervisions) < len(supervisions):
                    # Some supervisions were removed
                    stats['supervisions_removed'] += len(supervisions) - len(fixed_supervisions)
                    record['supervisions'] = fixed_supervisions
                    fixed_records.append(record)
                else:
                    # No changes needed
                    fixed_records.append(record)
            elif not has_empty_text:
                fixed_records.append(record)  # Just for output if needed
    
    # Print summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total records processed: {stats['total_records']}")
    print(f"Total supervisions: {stats['total_supervisions']}")
    print(f"Records with issues: {stats['records_with_issues']}")
    print(f"Supervisions with empty/missing text: {stats['supervisions_with_empty_text']}")
    
    if stats['issues_by_type']:
        print(f"\nIssues by type:")
        for issue_type, count in sorted(stats['issues_by_type'].items()):
            print(f"  {issue_type}: {count}")
    
    # Write fixed data if requested
    if output_path and fix_missing:
        output_path = Path(output_path)
        print(f"\nWriting fixed data to: {output_path}")
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for record in fixed_records:
                f.write(json.dumps(record) + '\n')
        
        print(f"\nFixed data stats:")
        print(f"  Records written: {len(fixed_records)}")
        print(f"  Supervisions removed: {stats['supervisions_removed']}")
        print(f"  Records removed: {stats['records_removed']}")
    
    # Report problematic records
    if problematic_records:
        print(f"\n{'='*80}")
        print("FIRST 20 PROBLEMATIC RECORDS")
        print(f"{'='*80}\n")
        for i, prob in enumerate(problematic_records[:20], 1):
            if 'sup_idx' in prob:
                print(f"[{i}] Line {prob['line']}: Record {prob['record_id']}, "
                      f"Supervision {prob['sup_idx']} ({prob['sup_id']})")
                print(f"     Issue: {prob['issue']}")
                if 'text' in prob:
                    print(f"     Text value: {prob['text']}")
            else:
                print(f"[{i}] Line {prob['line']}: Record {prob['record_id']}")
                print(f"     Issue: {prob['issue']}")
        
        if len(problematic_records) > 20:
            print(f"\n... and {len(problematic_records) - 20} more")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Validate supervision text fields in gzipped JSONL data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s -i data.jsonl.gz                      # Just validate
  %(prog)s -i data.jsonl.gz -v                   # Verbose mode
  %(prog)s -i data.jsonl.gz --fix-missing -o fixed.jsonl.gz  # Fix and save
        '''
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to gzipped JSONL file')
    parser.add_argument('-o', '--output', type=str,
                        help='Output path for fixed data (use with --fix-missing)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--fix-missing', action='store_true',
                        help='Remove supervisions with missing/empty text')
    
    args = parser.parse_args()
    
    stats = validate_file(
        args.input,
        verbose=args.verbose,
        output_path=args.output,
        fix_missing=args.fix_missing
    )
    
    # Exit with error code if issues found
    if stats['records_with_issues'] > 0 and not args.fix_missing:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
