import numpy as np
from pathlib import Path
from model.segmenter import SDHuBERTSegmenter, MincutWrapper
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default='ckpts/sdhubert_base.pt')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--librispeech_dataroot", type=str, default='/data/common/LibriSpeech')
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--save_name", type=str, default=None)
parser.add_argument("--split", type=str, default="all")

if __name__ == '__main__':
    args = parser.parse_args()
    save_name = args.save_name if args.save_name is not None else Path(args.ckpt_path).stem
    segmenter = SDHuBERTSegmenter(args.ckpt_path, layer=9, normcut_layer=11, normcut_threshold=2, device=args.device)
    mincut = MincutWrapper(syl_dur=0.2, ft_sr=50) #, min_segment_len=5, pre_merge=False, min_cut_minimum=5)
    librispeech_dataroot = Path(args.librispeech_dataroot)
    save_dir = args.save_dir if args.save_dir is not None else librispeech_dataroot/'segments'
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir/save_name
    save_dir.mkdir(exist_ok=True)
    
    tag_files = {'train':['files/librispeech_test.txt'],
                 'val': ['files/librispeech_val.txt'],
                 'test': ['files/librispeech_train_10Ksubset.txt'],
                'all': ['files/librispeech_test.txt',
                        'files/librispeech_val.txt',
                        'files/librispeech_train_10Ksubset.txt']}
    
    for tag_file in tag_files[args.split]:
        with open(tag_file, 'r') as f:
            tags = [t.rstrip() for t in f.readlines()]
            for tag in tqdm.tqdm(tags):
                wav_file = librispeech_dataroot/tag
                parent_1 = wav_file.parent.stem
                parent_2 = wav_file.parent.parent.stem
                parent_3 = wav_file.parent.parent.parent.stem
                (save_dir/parent_3).mkdir(exist_ok=True)
                (save_dir/parent_3/parent_2).mkdir(exist_ok=True)
                (save_dir/parent_3/parent_2/parent_1).mkdir(exist_ok=True)
                file_name = save_dir/parent_3/parent_2/parent_1/f'{wav_file.stem}.npy'
                outputs = segmenter(str(wav_file))
                outputs = mincut(**outputs)
                np.save(file_name,outputs)