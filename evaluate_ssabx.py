#from model.sdhubert import XModel
from utils.misc import load_model
from utils.ssabx import SSABXEvaluator
import soundfile as sf
from transformers import Wav2Vec2Processor
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default='ckpts/sdhubert_base.pt')
parser.add_argument("--layer", type=int, default=9)
parser.add_argument("--mode", type=str, default='favg')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--librispeech_dataroot", type=str, default='/data/common/LibriSpeech')
parser.add_argument("--ssabx_triplets", type=str, default='files/ssabx.json')

class SentembWrapper(object):
    def __init__(self, model, layer=9, mode='favg', device='cuda'):
        '''
        '''
        self.model = model.eval().to(device)
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.layer = layer
        self.mode = mode
        
        
    def __call__(self, wav):
        wav_input = self.processor([wav],
                                   sampling_rate=16000, return_tensors="pt",
                                   padding=True)
        
        wav_input = wav_input.input_values.detach().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(wav_input, inference_mode=True)
            cls_token = outputs['cls']
            states = outputs['hidden_states']
            
        if self.mode =='favg':
            sentemb = states[self.layer][0,1:].mean(0).detach().cpu().numpy()
        elif self.mode == 'agg':
            if self.layer != 'final':
                sentemb = states[self.layer][0,0].detach().cpu().numpy()
            else:
                sentemb = cls_token[0].detach().cpu().numpy()
        else:
            raise NotImplementedError
        return sentemb
    
    

if __name__ == '__main__':
    args = parser.parse_args()
    model,_ = load_model(args.ckpt_path)
    sentemb_extractor = SentembWrapper(model, layer=args.layer, mode=args.mode, device=args.device)
    ssabx=SSABXEvaluator(args.librispeech_dataroot, args.ssabx_triplets, sentemb_extractor)
    acc = ssabx.evaluate()
    print(f'SSABX ACC: {acc}')