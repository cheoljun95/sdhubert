import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from pathlib import Path
import random
import soundfile as sf
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

class SpeechDataset(Dataset):
    
    def __init__(self, data, sample_len=None):
        super().__init__()
        self.data = data
        self.sample_len = sample_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i):
        wav_path = self.data[i]
        wav,sr = sf.read(wav_path)
        assert sr ==16000
        if self.sample_len is not None:
            p = int(np.random.uniform(0,max(1,len(wav)-int(self.sample_len*sr))))
            wav = wav[p:p+int(self.sample_len*sr)]
            
        return {'wav':wav}
    
    @staticmethod
    def collate(batch):
        data = {}
        wav_input = processor([d['wav'] for d in batch],
                              sampling_rate=16000, return_tensors="pt",
                              padding=True)
        
        data['wavs'] = wav_input.input_values.detach()
        data['wav_lens'] = [len(d['wav']) for d in batch]
        output = {'wav': data['wavs'], 
                  'wavlen': data['wav_lens']}
        return output

class SpeechDataModule(LightningDataModule):
    def __init__(self,
                 root_dir,
                 sample_len=None,
                 batch_size=64,
                 val_batch_size=None,
                 num_workers=4,
                 drop_last=True,
                 pin_memory=True,
                 
                 ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.batch_size=batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.sample_len = sample_len
        
    def _load_data(self, split):
        split_dirs={'train':  ['train-clean-100', 'train-clean-360', 'train-other-500'],
                    'valid': ['dev-clean'],
                    'test':['test-clean','test-other']}[split]
        data = []
        for split_dir in split_dirs:
            data += [f for f in (self.root_dir/split_dir).glob('**/*.flac')]
        return data
    
    def train_dataloader(self):
        data = self._load_data('train')
        dataset = SpeechDataset(data, sample_len=self.sample_len)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechDataset.collate
        )
        return loader
    
    def val_dataloader(self):
        data = self._load_data('valid')
        dataset = SpeechDataset(data, sample_len=self.sample_len)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechDataset.collate
        )
        return loader
    
    def test_dataloader(self):
        data = self._load_data('test')
        dataset = SpeechDataset(data, sample_len=self.sample_len)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechDataset.collate
        )
        return loader
    