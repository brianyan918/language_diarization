import argparse
import json
import os
import tempfile
import numpy as np
import soundfile as sf
from typing import List, Tuple
from tqdm import tqdm

try:
    import whisper
except ImportError:
    try:
        from openai import whisper
    except ImportError:
        raise ImportError("whisper module not found. Install with: pip install openai-whisper")

def extract_languages(language_str: str) -> List[str]:
    """Extract languages from hyphenated string (e.g., 'ara-eng' -> ['ara', 'eng'])."""
    return language_str.split('-')

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

def mask_audio(audio: np.ndarray, sr: int, segments: List[dict], target_lang: str) -> np.ndarray:
    """Zero out all segments except those matching target_lang."""
    masked = np.zeros_like(audio)
    for seg in segments:
        if seg['lang'] == target_lang:
            start = int(seg['audio_start_sec'] * sr)
            end = int(seg['audio_end_sec'] * sr)
            end = min(end, len(audio))  # Prevent index out of bounds
            masked[start:end] = audio[start:end]
    return masked

def transcribe_audio_array(model, audio: np.ndarray, sr: int, lang: str = None, timestamps: bool = False) -> Tuple[str, list]:
    """
    Transcribe audio array using Whisper. Saves to temp file temporarily.
    
    Returns:
        If timestamps=False: (text, [])
        If timestamps=True: (text, segments_with_timestamps)
    """
    # Convert to float32 to avoid dtype issues
    audio = audio.astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio, sr)
        # Map 3-letter code to 2-letter code if provided
        whisper_lang = LANG_CODE_MAP.get(lang, lang) if lang else None
        result = model.transcribe(tmp_path, language=whisper_lang if whisper_lang != 'en' else None)
        
        if timestamps:
            # Return text and segments with timestamps
            segments = result.get('segments', [])
            timestamp_segments = [
                {
                    'start': seg.get('start'),
                    'end': seg.get('end'),
                    'text': seg.get('text')
                }
                for seg in segments
            ]
            return result['text'], timestamp_segments
        else:
            return result['text'], []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def process_line(line: dict, model, sr: int, out_dir: str, debug: bool = False, timestamps: bool = False):
    """Process a single JSONL line: mask audio and transcribe for each language."""
    audio_path = line['file_name']
    segments = line['segments']
    base_id = line['id']
    language_str = line['language']
    
    # Extract languages from 'language' field
    target_langs = extract_languages(language_str)
    
    # Load audio
    audio, file_sr = sf.read(audio_path)
    if file_sr != sr:
        raise ValueError(f"Sample rate mismatch: {file_sr} != {sr}")
    
    whisper_outputs = {}
    for lang in tqdm(target_langs, desc=f"Transcribing {base_id}", leave=False):
        masked = mask_audio(audio, sr, segments, lang)
        masked = masked.astype(np.float32)
        text, ts_segments = transcribe_audio_array(model, masked, sr, lang=lang, timestamps=timestamps)
        
        if timestamps:
            whisper_outputs[lang] = {
                'text': text,
                'segments': ts_segments
            }
        else:
            whisper_outputs[lang] = text
        
        # If debug mode, save masked audio
        if debug:
            masked_path = os.path.join(out_dir, f"{base_id}_{lang}_masked.wav")
            sf.write(masked_path, masked, sr)
    
    # Return enriched record
    enriched = dict(line)
    enriched['whisper_outputs'] = whisper_outputs
    return enriched

def main():
    parser = argparse.ArgumentParser(description="Mask and transcribe diarized audio by language.")
    parser.add_argument('--input_jsonl', required=True, help='Path to input JSONL file')
    parser.add_argument('--output_dir', required=True, help='Directory to save transcriptions (and masked audio if --debug)')
    parser.add_argument('--whisper_model', default='base', help='Whisper model name')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate (default 16kHz)')
    parser.add_argument('--debug', action='store_true', help='Save masked audio files for debugging')
    parser.add_argument('--timestamps', action='store_true', help='Include timestamp segments in output')
    args = parser.parse_args()

    import concurrent.futures
    os.makedirs(args.output_dir, exist_ok=True)
    model = whisper.load_model(args.whisper_model)

    with open(args.input_jsonl, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    records = [json.loads(line) for line in lines]

    def mask_job(line):
        audio_path = line['file_name']
        segments = line['segments']
        base_id = line['id']
        language_str = line['language']
        target_langs = extract_languages(language_str)
        try:
            audio, file_sr = sf.read(audio_path)
        except Exception as e:
            return (line, None, f"AUDIO_LOAD_ERROR: {e}")
        if file_sr != args.sample_rate:
            return (line, None, f"SAMPLE_RATE_MISMATCH: {file_sr} != {args.sample_rate}")
        masked_audios = {}
        for lang in target_langs:
            masked = mask_audio(audio, args.sample_rate, segments, lang)
            masked = masked.astype(np.float32)
            masked_audios[lang] = masked
            if args.debug:
                masked_path = os.path.join(args.output_dir, f"{base_id}_{lang}_masked.wav")
                sf.write(masked_path, masked, args.sample_rate)
        return (line, masked_audios, None)

    # Parallel masking
    masked_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(mask_job, records), total=len(records), desc="Masking audio"):
            masked_results.append(result)

    # Sequential transcription (GPU safe)
    output_jsonl = os.path.join(args.output_dir, 'whisper_outputs.jsonl')
    with open(output_jsonl, 'w') as outf:
        for (line, masked_audios, error) in tqdm(masked_results, desc="Transcribing"):
            enriched = dict(line)
            whisper_outputs = {}
            if error:
                enriched['whisper_outputs'] = {"error": error}
            elif masked_audios is not None:
                for lang, masked in masked_audios.items():
                    text, ts_segments = transcribe_audio_array(model, masked, args.sample_rate, lang=lang, timestamps=args.timestamps)
                    if args.timestamps:
                        whisper_outputs[lang] = {
                            'text': text,
                            'segments': ts_segments
                        }
                    else:
                        whisper_outputs[lang] = text
                enriched['whisper_outputs'] = whisper_outputs
            else:
                enriched['whisper_outputs'] = {"error": "Unknown error"}
            outf.write(json.dumps(enriched, ensure_ascii=False) + '\n')
    print(f"Wrote all outputs to {output_jsonl}")

if __name__ == '__main__':
    main()
