import numpy as np
from pathlib import Path
from model.segmenter import SDHuBERTSegmenter, MincutWrapper
import tqdm

if __name__ == '__main__':
    output_path = '/home/cheoljun/linguistic-structure-discovery/outputs/2023-12-26/19-33-51'
    #output_path = '/data/cheoljun/sdhubert_outputs_20231018/2023-07-21/02-47-01' 
    model_name = 'sdhubertv2'
    segmenter = SDHuBERTSegmenter(output_path)
    mincut = MincutWrapper(syl_dur=0.2, ft_sr=50) #, min_segment_len=5, pre_merge=False, min_cut_minimum=5)
    librispeech_dataroot= Path('/data/common/LibriSpeech')
    save_dir = Path(f'/data/common/LibriSpeech/{model_name}_outputs')
    save_dir.mkdir(exist_ok=True)
    tag_files = ['files/librispeech_test.txt', 'files/librispeech_val.txt', 'files/librispeech_train_10Ksubset.txt']
    
    for tag_file in tag_files:
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
                
                if file_name.exists():
                    continue
                outputs = segmenter(str(wav_file))
                outputs = mincut(outputs['segments'], outputs['features'])
                np.save(file_name,outputs)