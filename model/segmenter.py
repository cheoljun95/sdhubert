import torch.nn as nn
import torch
import numpy as np
from utils.misc import load_model
from transformers import Wav2Vec2Processor
import soundfile as sf
import librosa

class SDHuBERTSegmenter(nn.Module):
    
    def __init__(self, ckpt_path, layer=9, normcut_layer=11, normcut_threshold = 2,device='cuda',
                 min_segment_len=0,  **kwargs):
        super().__init__()
        model,_ = load_model(ckpt_path)
        self.model = model.eval().to(device)
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        # 0 indexed at the cnn output, the transformer layers are 1-12 
        self.layer=layer
        self.normcut_layer=normcut_layer
        self.normcut_threshold = normcut_threshold
        self.wav_sr = 16000
        self.min_segment_len = min_segment_len
            
    
    def mask_to_segment(self, mask):
        # mask: 1d-array of boolean mask
        valid_mask_ext = np.concatenate([np.zeros(1),mask*1.0,np.zeros(1)],0)

        turning = valid_mask_ext[1:]-valid_mask_ext[:-1]

        turn_on = np.nonzero(turning>0)[0]
        turn_off =np.nonzero((-turning)>0)[0]

        segments = np.array([[turn_on[i],turn_off[i]] for i in range(len(turn_on))])

        return segments

    def trim_segment(self, segments):
        trimmed_segments=[]
        for si,ei in segments:
            if ei-si>=self.min_segment_len:
                trimmed_segments.append([si,ei])
        trimmed_segments=np.array(trimmed_segments)
        return trimmed_segments

    def forward(self, wav):
        if isinstance(wav, str):
            wav, sr = sf.read(wav)
            if sr != self.wav_sr:
                wav = librosa.resample(wav,orig_sr=sr,target_sr=self.wav_sr)
        inputs=self.processor([wav],sampling_rate=self.wav_sr, return_tensors="pt",padding=True,return_attention_mask=True)
        inputs=inputs.input_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs, inference_mode=True)
            hidden_states = outputs['hidden_states']
        
        # first state is cls_token
        states=hidden_states[self.layer].squeeze(0).cpu().numpy()[1:]
        norm=hidden_states[self.normcut_layer].squeeze(0).cpu().numpy()[1:]
        valid_mask=np.linalg.norm(norm,axis=1)>self.normcut_threshold
        states[~valid_mask] = 0
        segments = self.trim_segment(self.mask_to_segment(valid_mask))
        segment_features = np.stack([states[s:e].mean(0) for s,e in segments])
        outputs = {'segments':segments,
                   'features':states,
                   'segment_features': segment_features,
                   'mask':valid_mask}
        return outputs
    

class MincutWrapper(nn.Module):
    
    def __init__(self, syl_dur=0.2, ft_sr=50, merge_threshold=0.3, min_segment_len=0, 
                 min_cut_minimum=5, pre_merge=True, **kwargs):
        super().__init__()
        from mincut import mincut
        self.mincut = mincut
        self.syl_dur = syl_dur
        self.ft_sr = ft_sr
        self.merge_threshold = merge_threshold
        self.min_segment_len = min_segment_len
        self.min_cut_minimum = min_cut_minimum
        self.pre_merge = pre_merge
    
    def _merge(self, feat, seg_boundary_frame_pairs):
        seg_boundary_frame_pairs_orig= seg_boundary_frame_pairs.copy()
        if len(seg_boundary_frame_pairs) >= 3:
            seg_boundary_frame_pairs = seg_boundary_frame_pairs_orig
            all_feat = [feat[round(l):round(r)].mean(0) for l,r in seg_boundary_frame_pairs]
            all_sim = [np.dot(l,r)/(np.linalg.norm(l)*np.linalg.norm(r)) for l,r in zip(all_feat[:-1], all_feat[1:])]
            min_id = np.argmax(all_sim)
            while all_sim[min_id] >= self.merge_threshold and len(seg_boundary_frame_pairs) >= 3:
                l_merge, r_merge = seg_boundary_frame_pairs[min_id], seg_boundary_frame_pairs[min_id+1]
                seg_boundary_frame_pairs = [pair for i, pair in enumerate(seg_boundary_frame_pairs) if i != min_id and i != min_id+1]
                seg_boundary_frame_pairs.insert(min_id, [l_merge[0], r_merge[1]])
                all_feat = [feat[round(l):round(r)].mean(0) for l,r in seg_boundary_frame_pairs]
                all_sim = [np.dot(l,r)/(np.linalg.norm(l)*np.linalg.norm(r)) for l,r in zip(all_feat[:-1], all_feat[1:])]
                min_id = np.argmax(all_sim)
        feat = np.array([feat[round(l):round(r)].mean(0) for l,r in seg_boundary_frame_pairs])
        seg_boundary_frame_pairs=np.array(seg_boundary_frame_pairs)
        return feat, seg_boundary_frame_pairs
        
    def _run_mincut(self, feat):
        # feat: (T, d)
        num_syllable = int(np.ceil(len(feat)/self.ft_sr/self.syl_dur))
        
        ssm = feat@feat.transpose(1,0)
        ssm = ssm - np.min(ssm) + 1e-7 # make it non-negative
        seg_boundary_frame = self.mincut.min_cut(ssm, num_syllable+1) # +1 for the algo

        seg_boundary_frame_pairs_orig = [[l,r] for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])] # 
        seg_boundary_frame_pairs = [item for item in seg_boundary_frame_pairs_orig if item[1]-item[0] > 2]
        if len(seg_boundary_frame_pairs)==0: # this shouldn't happen though
            seg_boundary_frame_pairs = seg_boundary_frame_pairs_orig
        
        if self.pre_merge:
            feat, seg_boundary_frame_pairs = self._merge(feat, seg_boundary_frame_pairs)
        return feat, seg_boundary_frame_pairs


    def forward(self, segments, features, **kwargs):
        
        boundaries=[]
        pooled_feat=[]
        for segment in segments:
            if (segment[1]-segment[0])<self.min_cut_minimum:
                boundaries_=[(segment-segment[0])]
                pooled_feat_=[features[segment[0]:segment[1]].mean(0)]
            else:
                pooled_feat_, boundaries_ = self._run_mincut(features[segment[0]:segment[1]])
            for bi,(bd,ft_) in enumerate(zip(boundaries_,pooled_feat_)):
                if np.isnan(np.sum(ft_)):
                    continue
                if (bd[1]-bd[0])<self.min_segment_len:
                    continue
                boundaries.append(bd+segment[0])
                pooled_feat.append(ft_)
                
        pooled_feat=np.stack(pooled_feat)
        boundaries = np.stack(boundaries)
        if not self.pre_merge:
            pooled_feat, boundaries = self._merge(pooled_feat,boundaries)
        
        boundaries=boundaries*1.0/self.ft_sr
        #assert ((boundaries[:,1]-boundaries[:,0])>=(1.0/self.min_segment_len)).all()
        outputs={'segments':boundaries,
                 'features':features,
                 'segment_features':pooled_feat,
                }
        return outputs
    