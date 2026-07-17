import argparse
import json
import os
import tempfile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import whisper
except ImportError:
    try:
        from openai import whisper
    except ImportError:
        raise ImportError("whisper module not found. Install with: pip install openai-whisper")

def extract_languages(language_str: str):
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

def mask_audio(audio: np.ndarray, sr: int, segments, target_lang: str) -> np.ndarray:
    """Zero out all segments except those matching target_lang."""
    masked = np.zeros_like(audio)
    for seg in segments:
        if seg['lang'] == target_lang:
            start = int(seg['audio_start_sec'] * sr)
            end = int(seg['audio_end_sec'] * sr)
            end = min(end, len(audio))  # Prevent index out of bounds
            masked[start:end] = audio[start:end]
    return masked

def plot_waveform_array(audio, sr, title):
    """Plot waveform array."""
    plt.figure(figsize=(12, 3))
    plt.plot(audio)
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Debug masked audio and transcriptions with Whisper.")
    parser.add_argument('--json_example', required=True, help='JSON example (string or file path)')
    parser.add_argument('--whisper_model', default='base', help='Whisper model name')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate (default 16kHz)')
    parser.add_argument('--output_dir', default=None, help='Optional directory to save masked audio files')
    args = parser.parse_args()

    # Load JSON example
    if os.path.exists(args.json_example):
        with open(args.json_example, 'r') as f:
            data = json.load(f)
    else:
        data = json.loads(args.json_example)

    audio_path = data['file_name']
    segments = data['segments']
    base_id = data['id']
    language_str = data['language']
    
    # Extract languages from 'language' field
    target_langs = extract_languages(language_str)
    
    # Load audio
    audio, file_sr = sf.read(audio_path)
    if file_sr != args.sample_rate:
        raise ValueError(f"Sample rate mismatch: {file_sr} != {args.sample_rate}")

    try:
        model = whisper.load_model(args.whisper_model)
    except AttributeError:
        raise RuntimeError("Whisper model loading failed. Ensure openai-whisper is installed: pip install openai-whisper")
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for lang in tqdm(target_langs, desc="Processing languages"):
        masked = mask_audio(audio, args.sample_rate, segments, lang)
        # Convert to float32 to avoid dtype issues
        masked = masked.astype(np.float32)
        print(f"\n--- {lang.upper()} ---")
        
        # Plot waveform
        plot_waveform_array(masked, args.sample_rate, f"Waveform: {base_id} [{lang}]")
        
        # Save masked audio if output_dir is provided
        if args.output_dir:
            masked_path = os.path.join(args.output_dir, f"{base_id}_{lang}_masked.wav")
            sf.write(masked_path, masked, args.sample_rate)
            print(f"Saved masked audio to: {masked_path}")
            # Transcribe from saved file
            whisper_lang = LANG_CODE_MAP.get(lang, lang)
            result = model.transcribe(masked_path, language=whisper_lang if whisper_lang != 'en' else None)
        else:
            # Create temp file for transcription
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sf.write(tmp_path, masked, args.sample_rate)
                whisper_lang = LANG_CODE_MAP.get(lang, lang)
                result = model.transcribe(tmp_path, language=whisper_lang if whisper_lang != 'en' else None)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        print("Transcription:")
        print(result['text'])

if __name__ == '__main__':
    main()
