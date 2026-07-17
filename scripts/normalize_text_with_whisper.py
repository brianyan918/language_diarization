import argparse
import json
from tqdm import tqdm

try:
    from whisper_normalizer import WhisperNormalizer
    normalizer = WhisperNormalizer()
except ImportError:
    normalizer = None
    print("Warning: whisper-normalizer not installed. Run 'pip install whisper-normalizer' for text normalization.")

def main():
    parser = argparse.ArgumentParser(description="Normalize text in JSONL using whisper-normalizer.")
    parser.add_argument('--input_jsonl', required=True, help='Path to input JSONL file')
    parser.add_argument('--output_jsonl', required=True, help='Path to output JSONL file')
    args = parser.parse_args()

    with open(args.input_jsonl, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    records = [json.loads(line) for line in lines]

    with open(args.output_jsonl, 'w') as outf:
        for rec in tqdm(records, desc="Normalizing text"):
            if normalizer:
                rec['whisper_norm_text'] = normalizer.normalize(rec['text'])
            else:
                rec['whisper_norm_text'] = rec['text']
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"Wrote normalized records to {args.output_jsonl}")

if __name__ == '__main__':
    main()
