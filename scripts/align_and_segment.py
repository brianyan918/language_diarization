import os
import torch
import torchaudio
import sox
import json
import argparse
from tqdm import tqdm


from examples.mms.data_prep.text_normalization import text_normalize
from examples.mms.data_prep.align_utils import (
    get_uroman_tokens,
    time_to_frame,
    load_model_dict,
    merge_repeats,
    get_spans,
)
import torchaudio.functional as F

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_emissions(model, audio_file):
    waveform, _ = torchaudio.load(audio_file)  # waveform: channels X T
    waveform = waveform.to(DEVICE)
    total_duration = sox.file_info.duration(audio_file)

    audio_sf = sox.file_info.sample_rate(audio_file)
    assert audio_sf == SAMPLING_FREQ

    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)

            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[
                :,
                int(SAMPLING_FREQ * input_start_time) : int(
                    SAMPLING_FREQ * (input_end_time)
                ),
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
            i += EMISSION_INTERVAL

    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)

    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)

    return emissions, stride


def get_alignments(
    audio_file,
    tokens,
    model,
    dictionary,
    use_star,
):
    # Generate emissions
    emissions, stride = generate_emissions(model, audio_file)
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(DEVICE)], dim=1)

    # Force Alignment
    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        print(f"Empty transcript!!!!! for audio file {audio_file}")
        token_indices = []

    blank = dictionary["<blank>"]
    
    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)
    
    input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1)
    target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)
    if input_lengths < target_lengths:
        return [], stride, None

    # return path and score
    try:
        path, score = F.forced_align(
            emissions.unsqueeze(0), targets.unsqueeze(0), input_lengths, target_lengths, blank=blank
        )
    except:
        return [], stride, None
    path = path.squeeze().to("cpu").tolist()
    score = (score.sum() / T).item()
    
    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride, score

def filter_blank_segs(tokens, transcripts, norm_transcripts, langs):
    new_tokens = []
    new_transcripts = []
    new_norm_transcripts = []
    new_langs = []
    for i, t in enumerate(tokens):
        if t != '':
            new_tokens.append(t)
            new_transcripts.append(transcripts[i])
            new_norm_transcripts.append(norm_transcripts[i])
            new_langs.append(langs[i])
    return new_tokens, new_transcripts, new_norm_transcripts, new_langs

def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    model, dictionary = load_model_dict()
    model = model.to(DEVICE)

    with open( f"{args.outdir}/manifest.json", "w") as f:
        pass
    
    for example in tqdm(open(args.manifest).readlines()):
        example = json.loads(example)

        transcripts = [seg["text"] for seg in example["segments"]]
        langs = [seg["lang"] for seg in example["segments"]]
        norm_transcripts = [text_normalize(seg["text"], seg["lang"]) for seg in example["segments"]]
        tokens = get_uroman_tokens(norm_transcripts, args.uroman_path, example["language"].split("-")[0])

        if args.use_star:
            dictionary["<star>"] = len(dictionary)
            tokens = ["<star>"] + tokens
            transcripts = ["<star>"] + transcripts
            norm_transcripts = ["<star>"] + norm_transcripts

        tokens, transcripts, norm_transcripts, langs = filter_blank_segs(tokens, transcripts, norm_transcripts, langs)

        segments, stride, score = get_alignments(
            example["file_name"],
            tokens,
            model,
            dictionary,
            args.use_star,
        )
        # Get spans of each line in input text file
        try:
            spans = get_spans(tokens, segments)
        except:
            continue

        if len(segments) == 0:
            # filter out failed forced alignments
            continue

        with open( f"{args.outdir}/manifest.json", "a") as f:
            segments = []
            for i, t in enumerate(transcripts):
                span = spans[i]
                seg_start_idx = span[0].start
                seg_end_idx = span[-1].end

                audio_start_sec = seg_start_idx * stride / 1000
                audio_end_sec = seg_end_idx * stride / 1000 
                
                sample = {
                    "audio_start_sec": audio_start_sec,
                    "audio_end_sec": audio_end_sec,
                    "duration": audio_end_sec - audio_start_sec,
                    "text": t,
                    "normalized_text":norm_transcripts[i],
                    "uroman_tokens": tokens[i],
                    "lang": langs[i]
                }
                segments.append(sample)
            output = {
                "id": example["id"],
                "file_name": example["file_name"],
                "language": example["language"],
                "text": example["text"],
                "speaker": example["speaker"],
                "segments": segments,
                "score": str(score),
            }
            f.write(json.dumps(output) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and segment long audio files")
    parser.add_argument(
        "--manifest", type=str, help="Path to input text file "
    )
    parser.add_argument(
        "-u", "--uroman_path", type=str, default="eng", help="Location to uroman/bin"
    )
    parser.add_argument(
        "-s",
        "--use_star",
        action="store_true",
        help="Use star at the start of transcript",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Output directory to store segmented audio files",
    )
    print("Using torch version:", torch.__version__)
    print("Using torchaudio version:", torchaudio.__version__)
    print("Using device: ", DEVICE)
    args = parser.parse_args()
    main(args)
