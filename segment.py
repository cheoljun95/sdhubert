from model.segmenter import SDHuBERTSegmenter, MincutWrapper
import numpy as np
import argparse
from pathlib import Path
import tqdm
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment audio files into a specified output directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="The input directory containing audio files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="The output directory to store the processed files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of files to process in each batch.",
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default='cuda', 
        help="The device to run the model on."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="ckpts/sdhubert_base.pt",
        help="The path to the model checkpoint"
    )
    parser.add_argument(
        "--normcut_threshold",
        type=float,
        default=0.1,
        help="The threshold for NormCut. Higher the value, more conservative in removing non-speech segments."
    )
    parser.add_argument(
        "--silence_threshold",
        type=float,
        default=0.02,
        help="Somtimes, NormCut is not accurate for some noise or respiratory sounds like inhaling. We can remove such by thresholding the segment by the amplitude of waveform."
    )
    parser.add_argument(
        "--syllable_duration",
        type=float,
        default=0.1,
        help="The heuristic duration of syllable in seconds. Not to miss some fast spoken syllables, we put a number (0.1s) that is shorter than the regular English syllable length"
    )
    parser.add_argument(
        "--merge_threshold",
        type=float,
        default=0.4,
        help="The threshold of the similarity for merging oversegmented syllables. The cosine similarity is used."
    )
    args = parser.parse_args()
    
    device = args.device
    
    if 'cuda' in device and not torch.cuda.is_available():
        print("CUDA is not available! Using CPU instead")
        device = 'cpu'
    segmenter = SDHuBERTSegmenter(args.ckpt_path, 
                                  layer=9, 
                                  normcut_strategy="relative",
                                  normcut_threshold=args.normcut_threshold,
                                  silence_threshold=args.silence_threshold,
                                  device=device)
    
    mincut = MincutWrapper(syl_dur=args.syllable_duration,
                           merge_threshold=args.merge_threshold,
                           ft_sr=50) 
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    wav_files = [f for f in input_dir.glob("*.wav")] + \
                [f for f in input_dir.glob("*.flac")] + \
                [f for f in input_dir.glob("*.ogg")]
    wav_files.sort()
    
    batch_size = args.batch_size
    
    for batch_i in tqdm.tqdm(range(0, len(wav_files), batch_size)):
        wav_file_batch = wav_files[batch_i:batch_i+batch_size]
        outputs = mincut(segmenter(wav_file_batch))
        for wf, output in zip(wav_file_batch, outputs):
            np.save(output_dir/f"{wf.stem}_feature.npy", 
                    output["features"])
            np.save(output_dir/f"{wf.stem}_segmentfeature.npy", 
                    output["segment_features"])
            with open(output_dir/f"{wf.stem}_segment.txt", "w") as f:
                for s,e in output["segments"]:
                    line = f"{s:.03f}, {e:.03f}\n"
                    f.write(line)
                    
            
            