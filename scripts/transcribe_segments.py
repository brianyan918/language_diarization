#!/usr/bin/env python3
"""
transcribe_segments.py

Transcribe individual segments from diarized audio without masking.
Each segment is extracted and transcribed independently with forced language ID.

Input JSONL format:
{
  "id": "...",
  "file_name": "path/to/audio.wav",
  "language": "ara-eng",
  "segments": [
    {
      "audio_start_sec": 0.0,
      "audio_end_sec": 2.8,
      "duration": 2.8,
      "text": "...",
      "lang": "eng"
    },
    ...
  ],
  "text": "...",
  ...
}

Output: JSONL with added "segment_transcriptions" field containing per-segment Whisper output.
"""

import argparse
import json
import os
import sys
import tempfile
import numpy as np
import soundfile as sf
import torch
from typing import List, Dict, Tuple
from tqdm import tqdm
from contextlib import contextmanager
from io import StringIO

try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
except ImportError:
    raise ImportError("transformers library not found. Install with: pip install transformers")

# Mapping from OpenAI model names to HuggingFace model IDs
OPENAI_TO_HF_MODEL_MAP = {
    'tiny': 'openai/whisper-tiny',
    'base': 'openai/whisper-base',
    'small': 'openai/whisper-small',
    'medium': 'openai/whisper-medium',
    'large': 'openai/whisper-large-v3',
    'large-v3': 'openai/whisper-large-v3',
    'large-v3-turbo': 'openai/whisper-large-v3-turbo',
}


# Mapping from 3-letter language codes to 2-letter ISO 639-1 codes for Whisper
LANG_CODE_MAP = {
    'ara': 'ar',
    'eng': 'en',
    'fra': 'fr',
    'spa': 'es',
    'deu': 'de',
    'ita': 'it',
    'por': 'pt',
    'rus': 'ru',
    'jpn': 'ja',
    'zho': 'zh',
    'cmn': 'zh',
    'hin': 'hi',
    'ben': 'bn',
    'tam': 'ta',
    'tel': 'te',
    'tur': 'tr',
    'pol': 'pl',
    'nld': 'nl',
    'swe': 'sv',
    'nor': 'no',
    'dan': 'da',
    'fin': 'fi',
    'heb': 'he',
    'tha': 'th',
    'kor': 'ko',
    'vie': 'vi',
    'ind': 'id',
    'fil': 'fil',
    'mya': 'my',
    'khm': 'km',
    'lao': 'lo',
    'cat': 'ca',
    'eus': 'eu',
    'gle': 'ga',
    'glg': 'gl',
    'ell': 'el',
    'hun': 'hu',
    'ron': 'ro',
    'slk': 'sk',
    'slv': 'sl',
    'ces': 'cs',
    'hrv': 'hr',
    'srp': 'sr',
    'mkd': 'mk',
    'bul': 'bg',
    'ukr': 'uk',
    'kat': 'ka',
    'fas': 'fa',
    'urd': 'ur',
    'pus': 'ps',
    'kur': 'ku',
    'amh': 'am',
    'orm': 'om',
    'swa': 'sw',
    'som': 'so',
}


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr


def _track_debug_examples(enriched: Dict, tracker: Dict, max_per_pair: int) -> None:
    """Track examples per language pair for debug output."""
    # Extract language pair from record
    language_field = enriched.get('language', 'unknown')
    
    # Initialize tracker for this pair if needed
    if language_field not in tracker:
        tracker[language_field] = []
    
    # Only track if we haven't reached the limit
    if len(tracker[language_field]) < max_per_pair:
        tracker[language_field].append({
            'id': enriched.get('id'),
            'file_name': enriched.get('file_name'),
            'segment_transcriptions': enriched.get('segment_transcriptions', []),
            'whisper_outputs': enriched.get('whisper_outputs', {}),
        })


def _generate_debug_report(tracker: Dict, output_dir: str, input_jsonl: str, sr: int) -> None:
    """Generate readable debug report with segment examples."""
    debug_dir = os.path.join(output_dir, 'debug_examples')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load original records for reference access
    with open(input_jsonl, 'r') as f:
        records_by_id = {json.loads(line)['id']: json.loads(line) for line in f if line.strip()}
    
    # Generate report for each language pair
    for lang_pair, examples in tracker.items():
        pair_dir = os.path.join(debug_dir, lang_pair.replace('-', '_'))
        os.makedirs(pair_dir, exist_ok=True)
        
        # Create index HTML
        with open(os.path.join(pair_dir, 'index.txt'), 'w', encoding='utf-8') as idx_file:
            idx_file.write(f"Debug Examples for Language Pair: {lang_pair}\n")
            idx_file.write(f"Total examples: {len(examples)}\n")
            idx_file.write("=" * 100 + "\n\n")
            
            for ex_idx, example in enumerate(examples):
                rec_id = example['id']
                seg_trans = example['segment_transcriptions']
                
                idx_file.write(f"[{ex_idx + 1}/{len(examples)}] Record ID: {rec_id}\n")
                idx_file.write(f"Audio file: {example['file_name']}\n")
                idx_file.write(f"Total segments: {len(seg_trans)}\n")
                idx_file.write("-" * 100 + "\n")
                
                # Get original record for reference
                orig_rec = records_by_id.get(rec_id, {})
                orig_segments = orig_rec.get('segments', [])
                
                for seg_idx, seg in enumerate(seg_trans):
                    start = seg.get('audio_start_sec', 0)
                    end = seg.get('audio_end_sec', 0)
                    lang = seg.get('lang', 'unknown')
                    ref_text = seg.get('ref_text', '')
                    whisper_text = seg.get('whisper_text', '')
                    
                    idx_file.write(f"\n  Segment {seg_idx}: [{start:.2f}s - {end:.2f}s] ({lang})\n")
                    idx_file.write(f"    Duration: {end - start:.2f}s\n")
                    idx_file.write(f"    Reference:  {ref_text}\n")
                    idx_file.write(f"    Whisper:    {whisper_text}\n")
                    
                    # Save segment audio
                    if rec_id in records_by_id:
                        try:
                            audio_path = orig_rec.get('file_name')
                            if audio_path and os.path.exists(audio_path):
                                audio, file_sr = sf.read(audio_path)
                                if file_sr != sr:
                                    audio = np.interp(
                                        np.linspace(0, 1, int(len(audio) * sr / file_sr)),
                                        np.linspace(0, 1, len(audio)),
                                        audio
                                    )
                                
                                segment_audio = extract_segment_audio(audio, sr, start, end)
                                seg_audio_path = os.path.join(
                                    pair_dir,
                                    f"{rec_id}_seg{seg_idx:03d}_{lang}.wav"
                                )
                                sf.write(seg_audio_path, segment_audio, sr)
                                idx_file.write(f"    Audio saved: {os.path.basename(seg_audio_path)}\n")
                        except Exception as e:
                            idx_file.write(f"    Audio error: {e}\n")
                
                idx_file.write("\n" + "=" * 100 + "\n\n")
        
        print(f"Debug report for {lang_pair}: {pair_dir}")


def extract_segment_audio(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    """Extract audio segment by time range."""
    start = int(start_sec * sr)
    end = int(end_sec * sr)
    end = min(end, len(audio))
    return audio[start:end]


def transcribe_audio_array(
    model,
    processor,
    device,
    audio: np.ndarray,
    sr: int,
    lang: str = None,
    timestamps: bool = False
) -> Tuple[str, list]:
    """
    Transcribe audio array using HuggingFace Whisper (single, not batched).
    
    Args:
        model: HF Whisper model
        processor: HF Whisper processor
        device: torch device
        audio: Audio array
        sr: Sample rate
        lang: 3-letter language code (e.g., 'eng', 'ara') - will be mapped to 2-letter
        timestamps: If True, include segment timestamps in output (not supported yet)
    
    Returns:
        (text, segments_with_timestamps) tuple
    """
    audio = audio.astype(np.float32)
    
    # Map 3-letter code to 2-letter code if provided
    whisper_lang = LANG_CODE_MAP.get(lang, lang) if lang else None
    
    # Process single audio
    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Ensure correct dtype
    model_dtype = next(model.parameters()).dtype
    inputs = {k: v.to(model_dtype) if v.dtype in [torch.float32, torch.float16] else v
              for k, v in inputs.items()}
    
    with suppress_stdout_stderr():
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                task="transcribe",
                language=whisper_lang,
                max_new_tokens=256,
            )
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # TODO: Add timestamp support if needed
    ts_segments = []
    
    return text, ts_segments


def transcribe_audio_batch(
    model,
    processor,
    device,
    audio_list: List[np.ndarray],
    sr: int,
    lang: str = None,
    timestamps: bool = False
) -> List[Tuple[str, list]]:
    """
    Transcribe a batch of audio arrays using HuggingFace Whisper with true GPU batching.
    
    Args:
        model: HF Whisper model
        processor: HF Whisper processor
        device: torch device
        audio_list: List of audio arrays
        sr: Sample rate
        lang: 3-letter language code (applies to all in batch)
        timestamps: If True, include segment timestamps in output
    
    Returns:
        List of (text, ts_segments) tuples
    """
    if not audio_list:
        return []
    
    # Map 3-letter code to 2-letter code if provided
    whisper_lang = LANG_CODE_MAP.get(lang, lang) if lang else None
    
    # Batch process all audio
    inputs = processor(
        audio_list,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device and ensure correct dtype
    model_dtype = next(model.parameters()).dtype
    inputs = {k: v.to(device).to(model_dtype) if v.dtype in [torch.float32, torch.float16] else v.to(device) 
              for k, v in inputs.items()}
    
    with suppress_stdout_stderr():
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                task="transcribe",
                language=whisper_lang,
                max_new_tokens=256,
            )
    
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    results = [(text, []) for text in texts]  # TODO: Add timestamp support
    
    return results


def process_line(
    line: Dict,
    model,
    sr: int,
    out_dir: str,
    debug: bool = False,
    timestamps: bool = False,
    batch_size: int = 4
) -> Dict:
    """
    Process a single JSONL line: extract segments and return metadata.
    Actual transcription is done in batch processing.
    
    Args:
        line: JSONL record
        model: Whisper model (unused, kept for compatibility)
        sr: Sample rate
        out_dir: Output directory (for debug audio)
        debug: Save segment audio files
        timestamps: Include timestamps in output
        batch_size: Unused in this version
    
    Returns:
        Record with extracted segment metadata and audio
    """
    audio_path = line['file_name']
    segments = line.get('segments', [])
    base_id = line['id']
    
    # Load audio once
    audio, file_sr = sf.read(audio_path)
    if file_sr != sr:
        raise ValueError(f"Sample rate mismatch: {file_sr} != {sr}")
    
    # Extract all segment audio upfront
    segment_data = []
    for idx, seg in enumerate(segments):
        start_sec = seg.get('audio_start_sec', 0)
        end_sec = seg.get('audio_end_sec', 0)
        lang_code = seg.get('lang', 'eng')
        ref_text = seg.get('text', '')
        
        segment_audio = extract_segment_audio(audio, sr, start_sec, end_sec)
        
        segment_data.append({
            'record_id': base_id,
            'idx': idx,
            'audio': segment_audio,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'lang_code': lang_code,
            'ref_text': ref_text
        })
    
    return {
        'record': line,
        'segments': segment_data
    }


def process_batch(batch_segments, model, processor, device, sr: int, timestamps: bool = False):
    """
    Transcribe a batch of segments (potentially from different utterances) using true GPU batching.
    
    Args:
        batch_segments: List of segment dicts with audio, lang_code, etc.
        model: HF Whisper model
        processor: HF Whisper processor
        device: torch device
        sr: Sample rate
        timestamps: Include timestamps in output
    
    Returns:
        List of transcription results in same order as input
    """
    if not batch_segments:
        return []
    
    # Extract audio and language (assumes all same language in batch)
    audio_list = [seg['audio'].astype(np.float32) for seg in batch_segments]
    lang = batch_segments[0]['lang_code']  # All should be same language
    
    # Batch transcribe
    batch_results = transcribe_audio_batch(
        model,
        processor,
        device,
        audio_list,
        sr,
        lang=lang,
        timestamps=timestamps
    )
    
    # Map results back to segment metadata
    results = []
    for seg_data, (transcription, ts_segments) in zip(batch_segments, batch_results):
        seg_out = {
            'record_id': seg_data['record_id'],
            'index': seg_data['idx'],
            'audio_start_sec': seg_data['start_sec'],
            'audio_end_sec': seg_data['end_sec'],
            'duration': seg_data['end_sec'] - seg_data['start_sec'],
            'lang': seg_data['lang_code'],
            'ref_text': seg_data['ref_text'],
            'whisper_text': transcription
        }
        
        if timestamps:
            seg_out['whisper_segments'] = ts_segments
        
        results.append(seg_out)
    
    return results


def process_records_with_global_batching(
    records,
    model,
    processor,
    device,
    sr: int,
    out_dir: str,
    debug: bool = False,
    timestamps: bool = False,
    batch_size: int = 32
):
    """
    Process all records with global batching across utterances.
    Groups segments by language and processes in large batches for GPU efficiency.
    
    Args:
        records: List of JSONL records
        model: HF Whisper model
        processor: HF Whisper processor
        device: torch device
        sr: Sample rate
        out_dir: Output directory
        debug: Save segment audio files
        timestamps: Include timestamps
        batch_size: Number of segments per batch
    
    Yields:
        Enriched records with transcriptions
    """
    from itertools import groupby
    from operator import itemgetter
    
    # First pass: extract all segments from all records
    print("Extracting segments from all records...")
    all_segments = []
    record_map = {}  # Map record_id to original record
    
    for record in tqdm(records, desc="Extracting", leave=False):
        extracted = process_line(record, model, sr, out_dir, debug, timestamps, batch_size)
        all_segments.extend(extracted['segments'])
        record_map[record['id']] = {
            'record': record,
            'segments_by_idx': {}
        }
    
    # Group segments by language for batching
    sorted_segments = sorted(all_segments, key=itemgetter('lang_code'))
    grouped_by_lang = {}
    for lang, group in groupby(sorted_segments, key=itemgetter('lang_code')):
        grouped_by_lang[lang] = list(group)
    
    # Process each language group in global batches
    print("Transcribing segments in batches...")
    segment_results = {}  # Map (record_id, seg_idx) -> transcription
    
    total_segments = sum(len(segs) for segs in grouped_by_lang.values())
    with tqdm(total=total_segments, desc="Transcribing", leave=False) as pbar:
        for lang_code, lang_segments in grouped_by_lang.items():
            for batch_start in range(0, len(lang_segments), batch_size):
                batch_end = min(batch_start + batch_size, len(lang_segments))
                batch = lang_segments[batch_start:batch_end]
                
                # Transcribe entire batch with GPU batching
                batch_results = process_batch(batch, model, processor, device, sr, timestamps)
                
                # Map results back to records
                for seg_data, result in zip(batch, batch_results):
                    key = (seg_data['record_id'], seg_data['idx'])
                    segment_results[key] = result
                    
                    # Debug: save segment audio
                    if debug:
                        seg_audio_path = os.path.join(
                            out_dir,
                            f"{seg_data['record_id']}_seg{seg_data['idx']:03d}_{lang_code}.wav"
                        )
                        sf.write(seg_audio_path, seg_data['audio'], sr)
                
                pbar.update(len(batch))
    
    # Reconstruct enriched records
    print("Assembling output records...")
    for record_id, metadata in tqdm(record_map.items(), desc="Assembling", leave=False):
        original_record = metadata['record']
        num_segments = len(original_record.get('segments', []))
        
        # Collect transcriptions for this record
        segment_transcriptions = []
        whisper_outputs = {}
        
        for seg_idx in range(num_segments):
            key = (record_id, seg_idx)
            if key in segment_results:
                seg_result = segment_results[key]
                segment_transcriptions.append(seg_result)
                
                # Aggregate by language
                lang = seg_result['lang']
                text = seg_result['whisper_text']
                if lang not in whisper_outputs:
                    whisper_outputs[lang] = []
                whisper_outputs[lang].append(text)
        
        # Build enriched record
        enriched = dict(original_record)
        enriched['segment_transcriptions'] = segment_transcriptions
        enriched['whisper_outputs'] = {
            lang: ' '.join(texts).strip()
            for lang, texts in whisper_outputs.items()
        }
        
        yield enriched


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe individual diarized audio segments without masking."
    )
    parser.add_argument('--input_jsonl', required=True, help='Path to input JSONL file')
    parser.add_argument('--output_dir', required=True, help='Output directory for JSONL and debug audio')
    parser.add_argument('--whisper_model', default='large-v3-turbo', help='Whisper model name (default: large-v3-turbo)')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate (default: 16000)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for transcription (default: 4)')
    parser.add_argument('--debug', action='store_true', help='Save segment audio files for debugging')
    parser.add_argument('--timestamps', action='store_true', help='Include timestamp segments in output')
    parser.add_argument('--debug_examples', type=int, default=0, help='Debug mode: process only N examples from start of file for testing (0=off, process all)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device and load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Convert model name if needed
    model_id = OPENAI_TO_HF_MODEL_MAP.get(args.whisper_model, args.whisper_model)
    print(f"Loading Whisper model: {args.whisper_model} ({model_id})")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    
    # Load input
    with open(args.input_jsonl, 'r') as f:
        records = [json.loads(line.strip()) for line in f if line.strip()]
    
    # If debug_examples is set, only process those records for full transcription
    if args.debug_examples > 0:
        print(f"Debug mode: processing only {args.debug_examples} examples total (not full dataset)")
        records_to_process = records[:args.debug_examples]
    else:
        print(f"Processing {len(records)} records...")
        records_to_process = records
    
    # Track examples per language pair for debug
    debug_examples_tracker = {} if args.debug_examples > 0 else None
    
    # Process all records with global batching
    output_jsonl = os.path.join(args.output_dir, 'segment_transcriptions.jsonl')
    with open(output_jsonl, 'w') as outf:
        try:
            for enriched in process_records_with_global_batching(
                records_to_process,
                model,
                processor,
                device,
                args.sample_rate,
                args.output_dir,
                debug=args.debug,
                timestamps=args.timestamps,
                batch_size=args.batch_size
            ):
                # Track examples for debug
                if debug_examples_tracker is not None:
                    _track_debug_examples(enriched, debug_examples_tracker, args.debug_examples)
                
                outf.write(json.dumps(enriched, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error during batch processing: {e}")
            raise
    
    print(f"\nWrote output to: {output_jsonl}")
    
    # Generate debug examples report
    if debug_examples_tracker is not None and debug_examples_tracker:
        _generate_debug_report(
            debug_examples_tracker,
            args.output_dir,
            args.input_jsonl,
            args.sample_rate
        )


if __name__ == '__main__':
    main()
