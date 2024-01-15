import torch
import numpy as np
from pathlib import Path
import tqdm
import json
import soundfile as sf
from sklearn.linear_model import LogisticRegression

class SSABXEvaluator(object):
    def __init__(self, librispeech_dataroot, ssabx_triplets, model=None):
        '''
            librispeech_dataroot: the root directory of LibriSpeech
            ssabx_triplets: ssabx.json file path
            model: wav -> sentence embedding
        '''
        self.dataroot = Path(librispeech_dataroot)
        self.triplets = json.load(open(ssabx_triplets,'r'))
        self.model = model
        
    def _get_sentemb(self, sentinfo, model):
        file_name = sentinfo['file_name']
        wav,sr = sf.read(self.dataroot/file_name)
        s,e = sentinfo['segment']
        s = int(np.floor(s*sr))
        e = int(np.ceil(e*sr))
        
        sent = wav[s:e]
        sentemb = model(sent)
        return sentemb
        
    def evaluate(self, model=None, average=True):
        '''Evaluate SSABX task
            model: wav -> sentence embedding
            avergage: if true, averaged ABX accuracy is output.
        '''
        if model is None:
            model = self.model    
        assert model is not None, "Model should be set by 'model' parameter."
        
        pred = []
        for _, triplet in tqdm.tqdm(self.triplets.items(), leave=False):
            x = self._get_sentemb(triplet['X'], model)
            p = self._get_sentemb(triplet['P'], model)
            n = self._get_sentemb(triplet['N'], model)
            x = x/np.linalg.norm(x)
            p = p/np.linalg.norm(p)
            n = n/np.linalg.norm(n)
            asim=(x*p).sum()
            bsim=(x*n).sum()
            pred.append(asim>bsim)
            
        pred=np.array(pred)
        acc=(pred*1.0)
        
        if average:
            return acc.mean()
        else:
            return acc
        