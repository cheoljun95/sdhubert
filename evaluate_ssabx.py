#from model.sdhubert import XModel
from model_v7 import SAggModel
from utils.utils import load_model
from utils.ssabx import SSABXEvaluator
import soundfile as sf
from transformers import Wav2Vec2Processor
import argparse
import torch

class SentembWrapper(object):
    def __init__(self, output_path, layer=8, mode='latest', device='cuda'):
        '''
            output_path: path created by torch lightning
            mode: if 'latest' load the latest epoch result, if 'best' load the best val_loss ckpt
        '''
        
        model,cfg = load_model(output_path, SAggModel, mode=mode,verbose=True)
        self.model = model.net.eval().to(device)
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.layer = layer
        
        
    def __call__(self, wav):
        wav_input = self.processor([wav],
                                   sampling_rate=16000, return_tensors="pt",
                                   padding=True)
        
        wav_input = wav_input.input_values.detach().to(self.device)
        
        with torch.no_grad():
            states, att = self.model.encode(wav_input)
            cls_token = self.model.final_lin(self.model.final_proj(states[-1][:,0,:]))
            
        
            
        #sentemb = states[self.layer][0,0].detach().cpu().numpy()
        sentemb = states[self.layer][0,1:].mean(0).detach().cpu().numpy()
        #sentemb = cls_token[0].detach().cpu().numpy()
        return sentemb
    
    

if __name__ == '__main__':
    output_path = '{OUTPUT_DIR}/{DATE}/{TIME}/{EXP_NAME}' # created by torch-lightning
    model = SentembWrapper(output_path)
    librispeech_dataroot= '/data/common/LibriSpeech'
    ssabx_triplets='files/ssabx.json'
    ssabx=SSABXEvaluator(librispeech_dataroot, ssabx_triplets, model)
    acc = ssabx.evaluate()
    print(f'ACC: {acc}')