import argparse
import json
import re
import os
from tqdm import tqdm

# Suppress ONNX Runtime threading issues
os.environ['OMP_NUM_THREADS'] = '1'

# Check for GPU availability
try:
    import torch
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print("GPU detected. Using GPU for inference.")
    else:
        print("No GPU detected. Using CPU.")
except ImportError:
    use_gpu = False
    print("PyTorch not available. Using CPU.")

def remove_markers(text):
    """Remove ** markers from text."""
    return text.replace('**', '')

def main():
    parser = argparse.ArgumentParser(description="Apply punctuation and capitalization restoration to text using multilingual ONNX model.")
    parser.add_argument('--input_jsonl', required=True, help='Path to input JSONL file')
    parser.add_argument('--output_jsonl', required=True, help='Path to output JSONL file')
    parser.add_argument('--model_name', default='1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase',
                        help='HuggingFace model name for punctuation/capitalization restoration')
    args = parser.parse_args()

    try:
        from punctuators.models import PunctCapSegModelONNX
    except ImportError:
        print("Error: punctuators not installed. Run 'pip install punctuators'")
        return

    # Load model
    print(f"Loading model: {args.model_name}")
    try:
        model = PunctCapSegModelONNX.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with open(args.input_jsonl, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    records = [json.loads(line) for line in lines]

    # Collect texts to process
    texts_to_process = []
    record_indices = []
    
    for idx, rec in enumerate(records):
        text = rec.get('text', '')
        # Remove ** markers
        text = remove_markers(text)
        if text:
            texts_to_process.append(text)
            record_indices.append(idx)

    # Process in batches
    print(f"Processing {len(texts_to_process)} records...")
    batch_size = 32
    results_map = {}
    
    for batch_start in tqdm(range(0, len(texts_to_process), batch_size), desc="Punctuation/Capitalization Restoration"):
        batch_end = min(batch_start + batch_size, len(texts_to_process))
        batch_texts = texts_to_process[batch_start:batch_end]
        batch_indices = record_indices[batch_start:batch_end]
        
        try:
            batch_results = model.infer(texts=batch_texts, apply_sbd=True)
            for idx, result_list in zip(batch_indices, batch_results):
                # result_list is a list of sentences, join them with space
                results_map[idx] = ' '.join(result_list)
        except Exception as e:
            print(f"Warning: Model inference failed for batch: {e}")
            for idx in batch_indices:
                results_map[idx] = texts_to_process[record_indices.index(idx)]

    # Write output
    with open(args.output_jsonl, 'w') as outf:
        for idx, rec in enumerate(records):
            if idx in results_map:
                rec['punct_cap_text'] = results_map[idx]
            else:
                # No processing done, use original text without markers
                rec['punct_cap_text'] = remove_markers(rec.get('text', ''))
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"Wrote punctuation/capitalization restored records to {args.output_jsonl}")

if __name__ == '__main__':
    main()
