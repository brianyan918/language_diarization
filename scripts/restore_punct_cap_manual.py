import argparse
import json
import os
from typing import List
import numpy as np
from tqdm import tqdm

# Suppress ONNX Runtime threading issues
os.environ['OMP_NUM_THREADS'] = '1'

def load_model_manual(model_name):
    """Load ONNX model manually for better control and performance."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from sentencepiece import SentencePieceProcessor
    
    # Download models from HF hub
    spe_path = hf_hub_download(repo_id=model_name, filename="sp.model")
    onnx_path = hf_hub_download(repo_id=model_name, filename="model.onnx")
    config_path = hf_hub_download(repo_id=model_name, filename="config.yaml")
    
    # Load components
    tokenizer = SentencePieceProcessor(spe_path)
    
    # Create session options to prefer GPU
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Try to use GPU providers first
    providers = ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    
    # Print which provider is being used
    print(f"ONNX Runtime Available Providers: {ort.get_available_providers()}")
    print(f"ONNX Runtime Session Providers: {ort_session.get_providers()}")
    print(f"Using Provider: {ort_session.get_providers()[0]}")
    
    config = OmegaConf.load(config_path)
    
    return tokenizer, ort_session, config

def process_text_batch(texts: List[str], tokenizer, ort_session, config):
    """Process a batch of texts and return punctuation/capitalization restored versions."""
    pre_labels = config.pre_labels
    post_labels = config.post_labels
    null_token = config.get("null_token", "<NULL>")
    acronym_token = config.get("acronym_token", "<ACRONYM>")
    
    # Encode all texts
    encoded_texts = []
    for text in texts:
        input_ids = [tokenizer.bos_id()] + tokenizer.EncodeAsIds(text) + [tokenizer.eos_id()]
        encoded_texts.append(input_ids)
    
    # Find max length for padding
    max_len_batch = max(len(ids) for ids in encoded_texts)
    
    # Pad to max length
    padded_ids = []
    for ids in encoded_texts:
        padded = ids + [tokenizer.pad_id()] * (max_len_batch - len(ids))
        padded_ids.append(padded)
    
    # Convert to numpy array [B, T]
    input_ids_arr = np.array(padded_ids, dtype=np.int64)
    
    # Run inference
    pre_preds, post_preds, cap_preds, sbd_preds = ort_session.run(None, {"input_ids": input_ids_arr})
    
    # Decode outputs for each text
    results = []
    for batch_idx, (input_ids, text) in enumerate(zip(encoded_texts, texts)):
        output_texts = []
        current_chars = []
        
        for token_idx in range(1, len(input_ids) - 1):
            token = tokenizer.IdToPiece(input_ids[token_idx])
            
            # Simple SP decoding
            if token.startswith("▁") and current_chars:
                current_chars.append(" ")
            
            # Token-level predictions
            pre_label = pre_labels[pre_preds[batch_idx][token_idx]]
            post_label = post_labels[post_preds[batch_idx][token_idx]]
            
            # Pre-punctuation
            if pre_label != null_token:
                current_chars.append(pre_label)
            
            # Process characters
            char_start = 1 if token.startswith("▁") else 0
            for token_char_idx, char in enumerate(token[char_start:], start=char_start):
                # Apply capitalization
                if cap_preds[batch_idx][token_idx][token_char_idx]:
                    char = char.upper()
                current_chars.append(char)
                
                # Handle acronyms
                if post_label == acronym_token:
                    current_chars.append(".")
            
            # Post-punctuation
            if post_label != null_token and post_label != acronym_token:
                current_chars.append(post_label)
            
            # Sentence boundary
            if sbd_preds[batch_idx][token_idx]:
                output_texts.append("".join(current_chars))
                current_chars.clear()
        
        # Push final sentence
        if current_chars:
            output_texts.append("".join(current_chars))
        
        results.append(" ".join(output_texts))
    
    return results

def remove_markers(text):
    """Remove ** markers from text."""
    return text.replace('**', '')

def main():
    parser = argparse.ArgumentParser(description="Apply punctuation and capitalization restoration using manual ONNX inference.")
    parser.add_argument('--input_jsonl', required=True, help='Path to input JSONL file')
    parser.add_argument('--output_jsonl', required=True, help='Path to output JSONL file')
    parser.add_argument('--model_name', default='1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase',
                        help='HuggingFace model name')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    args = parser.parse_args()

    # Check for dependencies
    try:
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf
        from sentencepiece import SentencePieceProcessor
    except ImportError as e:
        print(f"Error: Missing required package. Install with: pip install onnxruntime huggingface-hub omegaconf sentencepiece")
        return

    # Load model
    print(f"Loading model: {args.model_name}")
    try:
        tokenizer, ort_session, config = load_model_manual(args.model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Read input
    with open(args.input_jsonl, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    records = [json.loads(line) for line in lines]

    # Collect texts
    texts_to_process = []
    record_indices = []
    
    for idx, rec in enumerate(records):
        text = rec.get('text', '')
        text = remove_markers(text)
        if text:
            texts_to_process.append(text)
            record_indices.append(idx)

    # Process in batches
    print(f"Processing {len(texts_to_process)} records with batch size {args.batch_size}...")
    results_map = {}
    
    for batch_start in tqdm(range(0, len(texts_to_process), args.batch_size), desc="Punctuation/Capitalization"):
        batch_end = min(batch_start + args.batch_size, len(texts_to_process))
        batch_texts = texts_to_process[batch_start:batch_end]
        batch_indices = record_indices[batch_start:batch_end]
        
        try:
            batch_results = process_text_batch(batch_texts, tokenizer, ort_session, config)
            for idx, result in zip(batch_indices, batch_results):
                results_map[idx] = result
        except Exception as e:
            print(f"Warning: Inference failed for batch: {e}")
            for idx, text in zip(batch_indices, batch_texts):
                results_map[idx] = text

    # Write output
    with open(args.output_jsonl, 'w') as outf:
        for idx, rec in enumerate(records):
            if idx in results_map:
                rec['punct_cap_text'] = results_map[idx]
            else:
                rec['punct_cap_text'] = remove_markers(rec.get('text', ''))
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"Wrote punctuation/capitalization restored records to {args.output_jsonl}")

if __name__ == '__main__':
    main()
