import argparse
import json
from tqdm import tqdm

# 3-letter to 2-letter language code mapping
LANG_CODE_MAP = {
    'ara': 'ar', 'eng': 'en', 'fra': 'fr', 'spa': 'es', 'deu': 'de', 'ita': 'it', 'por': 'pt', 'rus': 'ru',
    'jpn': 'ja', 'zho': 'zh', 'hin': 'hi', 'ben': 'bn', 'tam': 'ta', 'tel': 'te', 'tur': 'tr', 'pol': 'pl',
    'nld': 'nl', 'swe': 'sv', 'nor': 'no', 'dan': 'da', 'fin': 'fi', 'heb': 'he', 'tha': 'th', 'kor': 'ko',
    'vie': 'vi', 'ind': 'id', 'fil': 'fil', 'mya': 'my', 'khm': 'km', 'lao': 'lo', 'cat': 'ca', 'eus': 'eu',
    'gle': 'ga', 'glg': 'gl', 'ell': 'el', 'hun': 'hu', 'ron': 'ro', 'slk': 'sk', 'slv': 'sl', 'ces': 'cs',
    'hrv': 'hr', 'srp': 'sr', 'mkd': 'mk', 'bul': 'bg', 'ukr': 'uk', 'kat': 'ka', 'fas': 'fa', 'urd': 'ur',
    'pus': 'ps', 'kur': 'ku', 'amh': 'am', 'orm': 'om', 'swa': 'sw', 'som': 'so',
}

# Cache for InverseNormalizers by language
normalizers_cache = {}

def get_inverse_normalizer(lang_code):
    """Get or create an InverseNormalizer for the given language."""
    if lang_code not in normalizers_cache:
        try:
            from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
            normalizers_cache[lang_code] = InverseNormalizer(lang=lang_code)
        except Exception as e:
            print(f"Warning: Could not load InverseNormalizer for {lang_code}: {e}")
            normalizers_cache[lang_code] = None
    return normalizers_cache[lang_code]

def inverse_normalize_segments(record):
    """Apply inverse text normalization to segments and join them."""
    segs = record.get('segments', [])
    norm_texts = []
    for seg in segs:
        text = seg.get('text', '')
        lang3 = seg.get('lang', '')
        lang2 = LANG_CODE_MAP.get(lang3, lang3)
        
        normalizer = get_inverse_normalizer(lang2)
        if normalizer:
            try:
                norm = normalizer.inverse_normalize(text, verbose=False)
            except Exception as e:
                print(f"Warning: ITN failed for '{text}' ({lang2}): {e}")
                norm = text
        else:
            norm = text
        
        seg['inverse_normalized_text'] = norm
        if lang3 == 'eng':
            norm = f"**{norm}**"
        norm_texts.append(norm)
    return ' '.join(norm_texts)

def main():
    parser = argparse.ArgumentParser(description="Apply inverse text normalization to segments using NeMo.")
    parser.add_argument('--input_jsonl', required=True, help='Path to input JSONL file')
    parser.add_argument('--output_jsonl', required=True, help='Path to output JSONL file')
    args = parser.parse_args()

    try:
        from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
    except ImportError:
        print("Error: nemo_text_processing not installed. Run 'pip install nemo_text_processing'")
        return

    with open(args.input_jsonl, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    records = [json.loads(line) for line in lines]

    with open(args.output_jsonl, 'w') as outf:
        for rec in tqdm(records, desc="Applying inverse text normalization"):
            rec['inverse_normalized_text'] = inverse_normalize_segments(rec)
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"Wrote inverse normalized records to {args.output_jsonl}")

if __name__ == '__main__':
    main()
