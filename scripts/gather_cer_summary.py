import argparse
import json
import os
from pathlib import Path
from tabulate import tabulate

def extract_cer_stats(jsonl_path):
    """Extract CER stats from a json or jsonl file."""
    if not os.path.exists(jsonl_path):
        return None
    try:
        with open(jsonl_path, 'r') as f:
            content = f.read().strip()
            if not content:
                return None
            # Try to parse as JSON first (single object)
            try:
                data = json.loads(content)
                return data.get('overall', {}).get('cer', None)
            except json.JSONDecodeError:
                # If that fails, try first line as JSONL
                f.seek(0)
                first_line = f.readline().strip()
                if not first_line:
                    return None
                data = json.loads(first_line)
                return data.get('overall', {}).get('cer', None)
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Gather and summarize CER statistics from all datasets.")
    parser.add_argument('--base_dir', default='asr_exp/base_v3-turbo/pred-lid', 
                        help='Base directory containing datasets')
    parser.add_argument('--output_json', default='cer_summary.json',
                        help='Output JSON file with summary')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    
    # List of datasets
    datasets = [d for d in sorted(base_dir.iterdir()) if d.is_dir()]
    
    summary = {
        "datasets": [],
        "comparison": {}
    }
    
    table_data = []
    
    for dataset_dir in datasets:
        dataset_name = dataset_dir.name
        
        cer_path = dataset_dir / 'cer.jsonl'
        norm_cer_path = dataset_dir / 'norm_cer.jsonl'
        
        cer = extract_cer_stats(cer_path)
        norm_cer = extract_cer_stats(norm_cer_path)
        
        dataset_info = {
            "name": dataset_name,
            "cer": cer,
            "norm_cer": norm_cer
        }
        
        if cer is not None and norm_cer is not None:
            improvement = cer - norm_cer
            improvement_pct = (improvement / cer * 100) if cer > 0 else 0
            dataset_info["improvement"] = improvement
            dataset_info["improvement_pct"] = improvement_pct
        
        summary["datasets"].append(dataset_info)
        
        # Add to table
        if cer is not None and norm_cer is not None:
            table_data.append([
                dataset_name,
                f"{cer:.4f}",
                f"{norm_cer:.4f}",
                f"{cer - norm_cer:.4f}",
                f"{(cer - norm_cer) / cer * 100:.2f}%"
            ])
        else:
            table_data.append([
                dataset_name,
                f"{cer}" if cer else "N/A",
                f"{norm_cer}" if norm_cer else "N/A",
                "N/A",
                "N/A"
            ])
    
    # Calculate overall stats
    valid_cers = [d["cer"] for d in summary["datasets"] if d["cer"] is not None]
    valid_norm_cers = [d["norm_cer"] for d in summary["datasets"] if d["norm_cer"] is not None]
    
    if valid_cers:
        summary["comparison"]["avg_cer"] = sum(valid_cers) / len(valid_cers)
        summary["comparison"]["avg_norm_cer"] = sum(valid_norm_cers) / len(valid_norm_cers)
        summary["comparison"]["avg_improvement"] = summary["comparison"]["avg_cer"] - summary["comparison"]["avg_norm_cer"]
        summary["comparison"]["avg_improvement_pct"] = (summary["comparison"]["avg_improvement"] / summary["comparison"]["avg_cer"] * 100) if summary["comparison"]["avg_cer"] > 0 else 0
    
    # Print table
    print("\n" + "="*100)
    print("CER SUMMARY: Original vs Normalized Text")
    print("="*100)
    print(tabulate(table_data, headers=["Dataset", "CER (Original)", "CER (Normalized)", "Improvement", "Improvement %"], 
                   tablefmt="grid"))
    
    # Print overall stats
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    if valid_cers:
        print(f"Average CER (Original):     {summary['comparison']['avg_cer']:.4f}")
        print(f"Average CER (Normalized):   {summary['comparison']['avg_norm_cer']:.4f}")
        print(f"Average Improvement:        {summary['comparison']['avg_improvement']:.4f}")
        print(f"Average Improvement %:      {summary['comparison']['avg_improvement_pct']:.2f}%")
    
    # Save to JSON
    with open(args.output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed summary saved to: {args.output_json}")

if __name__ == '__main__':
    main()
