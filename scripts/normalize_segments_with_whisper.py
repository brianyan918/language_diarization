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

try:
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
    basic_normalizer = BasicTextNormalizer()
    english_normalizer = EnglishTextNormalizer()
except ImportError:
    basic_normalizer = None
    english_normalizer = None
    print("Warning: OpenAI Whisper not installed. Run 'pip install whisper' for text normalization.")

def normalize_segments(record):
    segs = record.get('segments', [])
    norm_texts = []
    for seg in segs:
        text = seg.get('text', '')
        lang3 = seg.get('lang', '')
        if lang3 == 'eng' and english_normalizer:
            norm = english_normalizer(text)
        elif basic_normalizer:
            norm = basic_normalizer(text)
        else:
            norm = text
        seg['whisper_normalized_text'] = norm
        if lang3 == 'eng':
            norm = f"**{norm}**"
        norm_texts.append(norm)
    return ' '.join(norm_texts)

def main():
    parser = argparse.ArgumentParser(description="Normalize segment texts in JSONL using whisper-normalizer.")
    parser.add_argument('--input_jsonl', required=True, help='Path to input JSONL file')
    parser.add_argument('--output_jsonl', required=True, help='Path to output JSONL file')
    args = parser.parse_args()

    with open(args.input_jsonl, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    records = [json.loads(line) for line in lines]

    with open(args.output_jsonl, 'w') as outf:
        for rec in tqdm(records, desc="Normalizing segments"):
            rec['whisper_normalized_text'] = normalize_segments(rec)
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"Wrote normalized records to {args.output_jsonl}")

if __name__ == '__main__':
    main()
