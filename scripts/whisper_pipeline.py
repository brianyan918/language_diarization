#!/usr/bin/env python3
"""
whisper_pipeline.py

Build language diarization baseline from Whisper ASR output using:
1. Whisper hypothesis text extraction
2. Text-level language ID (fastText word-level LID)
3. Forced alignment (fairseq MMS alignment)
4. Language segment boundaries from alignment

Pipeline:
  - Read Whisper output JSONL (with hypothesis text)
  - Match IDs with reference JSONL (for passthrough/audio info)
  - Segment hypothesis by language using word-level LID
  - Run forced alignment to get frame-level boundaries
  - Map alignment to language segments with timestamps
  - Output in reference format with pred segments

Usage:
  python scripts/whisper_pipeline.py \
    --whisper_jsonl asr_exp/base_v3-turbo/pred-lid/arzen/out.jsonl \
    --reference_jsonl model/spoken-language-diarization/exp/runs/.../langdiar_whisper_multi.0.jsonl \
    --output predictions.jsonl \
    --fasttext_model models/lid.176.bin \
    --uroman_path uroman/bin
"""

import argparse
import json
import torch
import torchaudio
import sox
import subprocess
import tempfile
import importlib.util
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import sys
import os

# Add workspace root to path
workspace_root = '/data/group_data/swl/old_home/byan/lang_diar'
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, 'fairseq'))

import torchaudio.functional as F

# Load align_utils directly
def _load_module(filepath, module_name):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load with fairseq path setup
old_path = sys.path.copy()
sys.path.insert(0, os.path.join(workspace_root, 'fairseq'))

align_utils = _load_module(
    os.path.join(workspace_root, 'fairseq/examples/mms/data_prep/align_utils.py'),
    'align_utils'
)
time_to_frame = align_utils.time_to_frame
load_model_dict = align_utils.load_model_dict
merge_repeats = align_utils.merge_repeats
get_spans = align_utils.get_spans
get_uroman_tokens = align_utils.get_uroman_tokens

text_norm = _load_module(
    os.path.join(workspace_root, 'fairseq/examples/mms/data_prep/text_normalization.py'),
    'text_normalization'
)
text_normalize = text_norm.text_normalize

sys.path = old_path

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ISO-639-3 -> fastText ISO-2-ish labels
ISO3_TO_ISO2 = {
    "eng": "en", "ara": "ar", "cmn": "zh", "hin": "hi", "ben": "bn",
    "spa": "es", "fra": "fr", "deu": "de", "ita": "it", "por": "pt",
    "rus": "ru", "jpn": "ja", "kor": "ko", "tur": "tr", "vie": "vi",
    "tha": "th", "ces": "cs", "ron": "ro", "hun": "hu", "pol": "pl",
    "ukr": "uk", "swe": "sv", "nor": "no", "dan": "da", "fin": "fi",
    "ell": "el", "heb": "he", "ind": "id", "mal": "ms", "tgl": "tl",
    "nld": "nl", "zho": "zh",
}


# ========================
# Language ID Utilities (via subprocess)
# ========================
def run_fasttext_lid_batch(
    text_list: List[str],
    lang_list: List[str],
    fasttext_model: str,
    fasttext_env: str,
) -> List[List[Tuple[str, str]]]:
    """
    Run fastText word-level LID on multiple texts in batch to amortize subprocess overhead.
    
    Args:
        text_list: List of texts to process
        lang_list: List of language pairs for each text
        fasttext_model: Path to fasttext model
        fasttext_env: Path to fasttext environment
    
    Returns:
        List of (word, language) lists, one per input text
    """
    if not text_list:
        return []
    
    # Convert to absolute paths
    fasttext_model = os.path.abspath(fasttext_model)
    fasttext_env = os.path.abspath(fasttext_env)
    
    # Create temporary JSONL file for fasttext script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f_in:
        input_file = f_in.name
        for text, langs in zip(text_list, lang_list):
            rec = {
                'text': text,
                'language': langs,
            }
            f_in.write(json.dumps(rec) + '\n')
    
    with tempfile.NamedTemporaryFile(mode='r', suffix='.jsonl', delete=False) as f_out:
        output_file = f_out.name
    
    # Run fasttext_word_lid_v-tok.py on batch
    cmd = [
        'bash', '-c',
        f'source {fasttext_env} && python {workspace_root}/scripts/fasttext_word_lid_v-tok.py '
        f'--input {input_file} --output {output_file} --fasttext_model {fasttext_model} '
        f'--merge_same_script_runs'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, cwd=workspace_root, 
                       capture_output=True, timeout=300, text=True)
    except subprocess.TimeoutExpired:
        print(f"WARNING: fasttext_word_lid batch timed out (300s).")
        try:
            os.unlink(input_file)
            os.unlink(output_file)
        except:
            pass
        return [[] for _ in text_list]
    except subprocess.CalledProcessError as e:
        print(f"ERROR running fasttext_word_lid batch:")
        print(f"  stderr: {e.stderr}")
        return [[] for _ in text_list]
    except Exception as e:
        print(f"ERROR: {e}")
        return [[] for _ in text_list]
    
    # Parse output
    all_segments = [[] for _ in text_list]
    try:
        if os.path.exists(output_file):
            idx = 0
            with open(output_file, 'r') as f:
                for line in f:
                    if idx >= len(text_list):
                        break
                    line = line.strip()
                    if not line:
                        continue
                    
                    rec = json.loads(line)
                    lid_words = rec.get('fasttext_word_lid', {}).get('words', [])
                    for word_entry in lid_words:
                        word = word_entry.get('word')
                        lang = word_entry.get('pred_lang')
                        if word and lang:
                            all_segments[idx].append((word, lang))
                    idx += 1
    except Exception as e:
        print(f"ERROR parsing fasttext batch output: {e}")
    finally:
        # Cleanup
        try:
            os.unlink(input_file)
        except:
            pass
        try:
            os.unlink(output_file)
        except:
            pass
    
    return all_segments


def run_fasttext_lid_subprocess(
    text: str,
    fasttext_model: str,
    fasttext_env: str,
    candidates: List[str],
) -> List[Tuple[str, str]]:
    """
    Run fastText word-level LID on single text (fallback for non-batched calls).
    """
    result = run_fasttext_lid_batch([text], ['-'.join(candidates)], fasttext_model, fasttext_env)
    return result[0] if result else []


def segment_by_language(
    text: str,
    fasttext_model: str,
    fasttext_env: str,
    candidates: List[str],
) -> List[Tuple[str, str]]:
    """Segment text into (word, language) tuples using fastText word-level LID."""
    return run_fasttext_lid_subprocess(text, fasttext_model, fasttext_env, candidates)


# ========================
# Forced Alignment Utilities
# ========================
def generate_emissions(model, audio_file: str) -> Tuple[torch.Tensor, float]:
    """Generate acoustic emissions for alignment."""
    waveform, _ = torchaudio.load(audio_file)
    waveform = waveform.to(DEVICE)
    total_duration = sox.file_info.duration(audio_file)
    audio_sf = sox.file_info.sample_rate(audio_file)
    
    if audio_sf != SAMPLING_FREQ:
        print(f"WARNING: Audio sample rate {audio_sf} != {SAMPLING_FREQ}, resampling")
        waveform = F.resample(waveform, audio_sf, SAMPLING_FREQ)
    
    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time = i
            segment_end_time = min(i + EMISSION_INTERVAL, total_duration)
            
            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            
            waveform_split = waveform[
                :,
                int(SAMPLING_FREQ * input_start_time) : int(SAMPLING_FREQ * input_end_time),
            ]
            
            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)
            
            emissions_ = emissions_[
                emission_start_frame - offset : emission_end_frame - offset, :
            ]
            emissions_arr.append(emissions_)
            i = segment_end_time
    
    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)
    
    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)
    
    return emissions, stride


def get_alignments(
    audio_file: str,
    tokens: List[str],
    model,
    dictionary: Dict[str, int],
    use_star: bool = False,
) -> Tuple[List, float, Optional[float]]:
    """Run forced alignment and return segments with frame boundaries."""
    try:
        emissions, stride = generate_emissions(model, audio_file)
    except Exception as e:
        print(f"ERROR generating emissions: {e}")
        return [], 0.0, None
    
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(DEVICE)], dim=1)
    
    # Build token indices
    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        return [], stride, None
    
    if not token_indices:
        return [], stride, None
    
    blank = dictionary["<blank>"]
    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)
    
    input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1)
    target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)
    
    if input_lengths < target_lengths:
        return [], stride, None
    
    try:
        path, score = F.forced_align(
            emissions.unsqueeze(0), targets.unsqueeze(0), input_lengths, target_lengths, blank=blank
        )
    except Exception as e:
        print(f"ERROR in forced align: {e}")
        return [], stride, None
    
    path = path.squeeze().to("cpu").tolist()
    score = (score.sum() / T).item() if T > 0 else 0.0
    
    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride, score


# ========================
# Main Pipeline
# ========================
def load_reference_by_id(reference_jsonl: str) -> Dict[str, Dict]:
    """Load reference data indexed by utt_id from passthrough."""
    ref_data = {}
    
    with open(reference_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            obj = json.loads(line)
            utt_key = next(iter(obj.keys()))
            entry = obj[utt_key]
            passthrough = entry.get('passthrough', {})
            utt_id = passthrough.get('utt_id')
            
            if utt_id:
                ref_data[utt_id] = {
                    'key': utt_key,
                    'entry': entry,
                    'passthrough': passthrough,
                }
    
    return ref_data


def build_output_entry(
    whisper_obj: Dict,
    ref_data: Dict,
    pred_segments: List[Dict],
    vocab_id_to_token: Dict[int, str],
) -> Optional[Dict]:
    """
    Build output entry in reference format.
    
    Maps pred_segments (with language strings) to reference format (with language IDs).
    """
    whisper_id = whisper_obj.get('id')
    
    # Try to find matching reference
    # In practice, may need more sophisticated matching
    matched_ref = None
    for ref_id, ref_info in ref_data.items():
        # Simple heuristic: match if whisper_id is substring or vice versa
        if whisper_id and (whisper_id in ref_id or ref_id in whisper_id):
            matched_ref = ref_info
            break
    
    if not matched_ref:
        return None
    
    # Convert pred_segments with language strings to language IDs
    output_pred = []
    vocab_token_to_id = {v: k for k, v in vocab_id_to_token.items()}
    
    for seg in pred_segments:
        lang_str = seg.get('lang')
        lang_id = vocab_token_to_id.get(lang_str, None)
        
        if lang_id is not None:
            output_pred.append({
                'start': seg['start'],
                'end': seg['end'],
                'label': lang_id,
                'score': None,
            })
    
    output_obj = {
        matched_ref['key']: {
            'pred': output_pred,
            'passthrough': matched_ref['passthrough'],
        }
    }
    
    return output_obj


def main():
    ap = argparse.ArgumentParser(description="Whisper-based language diarization baseline")
    ap.add_argument('--whisper_jsonl', required=True, help='Whisper output JSONL')
    ap.add_argument('--reference_jsonl', required=True, help='Reference JSONL with passthrough')
    ap.add_argument('--output', required=True, help='Output JSONL file')
    ap.add_argument('--fasttext_model', required=True, help='Path to lid.176.bin')
    ap.add_argument('--fasttext_env', required=True, help='Path to fasttext environment activate script')
    ap.add_argument('--uroman_path', default='uroman/bin', help='Path to uroman bin')
    ap.add_argument('--use_star', action='store_true', help='Use <star> token at start')
    ap.add_argument('--vocab', default='data/vocab_102.txt', help='Vocab file for language IDs')
    args = ap.parse_args()
    
    # Load models and vocabulary
    print("Loading models...")
    align_model, dictionary = load_model_dict()
    align_model = align_model.to(DEVICE)
    
    # Load vocab for language ID mapping
    vocab_id_to_token = {}
    with open(args.vocab, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                vocab_id_to_token[int(parts[0])] = parts[1]
    
    # Load reference data
    print(f"Loading reference data from {args.reference_jsonl}...")
    ref_data = load_reference_by_id(args.reference_jsonl)
    print(f"  Loaded {len(ref_data)} reference entries")
    
    # Process Whisper output
    print(f"Processing {args.whisper_jsonl}...")
    
    # First pass: collect all whisper objects
    whisper_objects = []
    with open(args.whisper_jsonl, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line:
                whisper_objects.append(json.loads(line))
    
    print(f"  Loaded {len(whisper_objects)} utterances")
    
    # Batch process LID (amortize subprocess overhead)
    batch_size = 100  # Process 10 utterances per fasttext call
    hyp_texts_batch = []
    candidates_batch = []
    indices = []
    
    with open(args.output, 'w') as fout:
        for idx, whisper_obj in enumerate(tqdm(whisper_objects, desc="Processing")):
            # Extract hypothesis text
            hyp_text = whisper_obj.get('whisper_pred_text') or whisper_obj.get('ref_text', '')
            if not hyp_text:
                continue
            
            # Get candidate languages
            language_pair = whisper_obj.get('language', 'eng')
            candidates = [x.strip() for x in language_pair.split('-') if x.strip()]
            
            hyp_texts_batch.append(hyp_text)
            candidates_batch.append('-'.join(candidates))
            indices.append(idx)
            
            # Process batch when full or at end
            if len(hyp_texts_batch) == batch_size or idx == len(whisper_objects) - 1:
                # Batch LID
                batch_segments = run_fasttext_lid_batch(
                    hyp_texts_batch,
                    candidates_batch,
                    args.fasttext_model,
                    args.fasttext_env
                )
                
                # Process each utterance in batch and write output immediately
                for batch_idx, (whisper_idx, lang_segments) in enumerate(zip(indices, batch_segments)):
                    whisper_obj = whisper_objects[whisper_idx]
                    
                    if not lang_segments:
                        continue
                    
                    # Reconstruct segmented text
                    seg_texts = []
                    seg_langs = []
                    for text, lang in lang_segments:
                        seg_texts.append(text)
                        seg_langs.append(lang)
                    
                    # Get audio file
                    audio_file = whisper_obj.get('file_name')
                    if not audio_file:
                        continue
                    
                    try:
                        if not sox.file_info.duration(audio_file):
                            continue
                    except Exception:
                        continue
                    
                    # Normalize and tokenize for alignment
                    language_pair = whisper_obj.get('language', 'eng')
                    candidates = [x.strip() for x in language_pair.split('-') if x.strip()]
                    
                    norm_texts = [text_normalize(t, seg_langs[i]) for i, t in enumerate(seg_texts)]
                    tokens = get_uroman_tokens(norm_texts, args.uroman_path, candidates[0] if candidates else 'eng')
                    
                    # Run forced alignment
                    segments, stride, score = get_alignments(
                        audio_file,
                        tokens,
                        align_model,
                        dictionary,
                        args.use_star,
                    )
                    
                    if not segments:
                        continue
                    
                    # Map alignment to language segments
                    try:
                        spans = get_spans(tokens, segments)
                    except Exception as e:
                        continue
                    
                    # Build output pred segments
                    pred_segments = []
                    for i, text in enumerate(seg_texts):
                        if i < len(spans):
                            span = spans[i]
                            seg_start_idx = span[0].start
                            seg_end_idx = span[-1].end
                            
                            audio_start_sec = seg_start_idx * stride / 1000
                            audio_end_sec = seg_end_idx * stride / 1000
                            
                            pred_segments.append({
                                'start': audio_start_sec,
                                'end': audio_end_sec,
                                'lang': seg_langs[i],
                            })
                    
                    # Build output entry
                    output_obj = build_output_entry(whisper_obj, ref_data, pred_segments, vocab_id_to_token)
                    
                    # Write immediately instead of buffering
                    if output_obj:
                        fout.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                        fout.flush()  # Flush to disk
                
                # Reset batch
                hyp_texts_batch = []
                candidates_batch = []
                indices = []
    
    print(f"Output written to {args.output}")


if __name__ == '__main__':
    main()
