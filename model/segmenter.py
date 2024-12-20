import torch.nn as nn
import torch
import numpy as np
from utils.misc import load_model
from transformers import Wav2Vec2Processor
import soundfile as sf
import librosa


class SDHuBERTSegmenter(nn.Module):

    def __init__(
        self,
        ckpt_path,
        layer=9,
        normcut_layer=10,
        normcut_strategy="relative",
        normcut_threshold=0.1,
        silence_threshold=0.02,
        device="cuda",
        min_segment_len=2,
        zero_pad=0,
        **kwargs,
    ):
        super().__init__()
        model, _ = load_model(ckpt_path)
        self.model = model.eval().to(device)
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.layer = layer
        self.normcut_layer = normcut_layer
        self.normcut_threshold = normcut_threshold
        self.silence_threshold = silence_threshold
        
        self.min_segment_len = min_segment_len
        self.zero_pad = zero_pad
        self.normcut_strategy = normcut_strategy
        assert normcut_strategy in ["absolute", "relative"]
        
        ### hard-coded config
        self.wav_sr = 16000
        self.ft_sr = 50
        self.wav_ft_sr = self.wav_sr//self.ft_sr
        self.buffer_pad = self.wav_ft_sr//2


    def preprocess_wav(self, wav_paths):
        lengths = []
        wavs = []
        for wav_path in wav_paths:
            wav, sr = sf.read(wav_path)
            if len(wav.shape)==2:
                wav = wav[...,0]
            if sr != self.wav_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.wav_sr)
                
            if len(wav)<self.wav_sr*0.1:
                print(f"WARNING:{str(wav_path)} has too short length {len(wav)/self.wav_sr:.02f}") 
                continue
            wav = (wav-wav.mean())/wav.std()
            lengths.append(len(wav))
            if self.zero_pad >0:
                wav = np.concatenate([np.zeros(self.buffer_pad),wav, np.zeros(self.buffer_pad)])
            wavs.append(wav)
        return wavs, lengths
        

    def mask_to_segment(self, mask):
        segments_list = []
        for m in mask:
            valid_mask_ext = np.concatenate([np.zeros(1), m * 1.0, np.zeros(1)], 0)
            turning = valid_mask_ext[1:] - valid_mask_ext[:-1]
            turn_on = np.nonzero(turning > 0)[0]
            turn_off = np.nonzero((-turning) > 0)[0]
            segments = np.array(
                [[turn_on[i], turn_off[i]] for i in range(len(turn_on))]
            )
            segments_list.append(segments)
        return segments_list

    def trim_segment(self, segments, wavs):
        trimmed_segments_list = []
        for seg, wav in zip(segments, wavs):
            trimmed_segments = []
            for si, ei in seg:
                wav_trimmed = wav[self.zero_pad+self.buffer_pad:]
                if (ei - si >= self.min_segment_len and 
                    wav_trimmed[self.wav_ft_sr*si:self.wav_ft_sr*ei].abs().mean() > self.silence_threshold):
                    trimmed_segments.append([si, ei])
            trimmed_segments = np.array(trimmed_segments)
            trimmed_segments_list.append(trimmed_segments)
        return trimmed_segments_list

    def forward(self, wav_paths):
        if not isinstance(wav_paths, list):
            wav_paths = [wav_paths]
            single_input = True
        else:
            single_input = False
            
        wavs, lengths = self.preprocess_wav(wav_paths)

        wavs =  nn.utils.rnn.pad_sequence([torch.from_numpy(wav).float() for wav in wavs], batch_first=True, padding_value=0.0)
        #inputs = inputs.input_values.to(self.device)
        inputs = wavs.to(self.device)
        wavlen = np.array(lengths)/self.wav_sr
        features = []
        norm_features = []
        with torch.no_grad():
            outputs = self.model(inputs,
                                 wavlen=wavlen,
                                 inference_mode=True)
            hidden_states = outputs["hidden_states"]
            features=hidden_states[self.layer][:,1:]
            norm_features=hidden_states[self.normcut_layer][:,1:]
        
        results = [] 

        
        for idx in range(len(wav_paths)):
            states = features[idx][: lengths[idx]//320].cpu().numpy()
            norm = norm_features[idx][: lengths[idx]//320] #.cpu().numpy()
            #norm = np.linalg.norm(norm, axis=1)
            norm =torch.linalg.vector_norm(norm, ord=2, dim=-1).cpu().numpy()
            if self.normcut_strategy == "relative":
                norm_min = np.percentile(norm,1)
                norm_max = np.percentile(norm,99)
                norm = (norm-norm_min)/(norm_max-norm_min)

            valid_mask = norm > self.normcut_threshold
            states[~valid_mask] = 0

            segments = self.mask_to_segment([valid_mask])[
                0
            ]  # mask_to_segment expects a list
            trimmed_segments = self.trim_segment([segments], [wavs[idx]])[
                0
            ]  # trim_segment expects a list

            result = {
                "segments": trimmed_segments,
                "features": states,
                "mask": valid_mask,
                "norm":norm,
            }
            results.append(result)
        if single_input:
            results = results[0]
        return results

from mincut import mincut

        
class MincutWrapper(object):

    def __init__(
        self,
        syl_dur=0.1,
        ft_sr=50,
        merge_threshold=0.4,
        min_segment_len=1,
        min_cut_minimum=1,
        pre_merge=True,
        **kwargs,
    ):
        from mincut import mincut

        self.mincut = mincut
        self.syl_dur = syl_dur
        self.ft_sr = ft_sr
        self.merge_threshold = merge_threshold
        self.min_segment_len = min_segment_len
        self.min_cut_minimum = min_cut_minimum
        self.pre_merge = pre_merge

    def _merge(self, feat, seg_boundary_frame_pairs):
        seg_boundary_frame_pairs_orig = seg_boundary_frame_pairs.copy()
        if len(seg_boundary_frame_pairs) >= 2:
            seg_boundary_frame_pairs = seg_boundary_frame_pairs_orig
            all_feat = [
                feat[l : r].mean(0) for l, r in seg_boundary_frame_pairs
            ]
            all_sim = [
                np.dot(l, r) / (np.linalg.norm(l) * np.linalg.norm(r))
                for l, r in zip(all_feat[:-1], all_feat[1:])
            ]
            min_id = np.argmax(all_sim)
            while (
                all_sim[min_id] >= self.merge_threshold
                and len(seg_boundary_frame_pairs) >= 2
            ):
                l_merge, r_merge = (
                    seg_boundary_frame_pairs[min_id],
                    seg_boundary_frame_pairs[min_id + 1],
                )
                seg_boundary_frame_pairs = [
                    pair
                    for i, pair in enumerate(seg_boundary_frame_pairs)
                    if i != min_id and i != min_id + 1
                ]
                seg_boundary_frame_pairs.insert(min_id, [l_merge[0], r_merge[1]])
                if len(seg_boundary_frame_pairs) >= 2:
                    all_feat = [
                        feat[round(l) : round(r)].mean(0)
                        for l, r in seg_boundary_frame_pairs
                    ]
                    all_sim = [
                        np.dot(l, r) / (np.linalg.norm(l) * np.linalg.norm(r))
                        for l, r in zip(all_feat[:-1], all_feat[1:])
                    ]
                    min_id = np.argmax(all_sim)
                else:
                    break
        feat = np.array(
            [feat[l : r].mean(0) for l, r in seg_boundary_frame_pairs]
        )
        seg_boundary_frame_pairs = np.array(seg_boundary_frame_pairs)
        return feat, seg_boundary_frame_pairs

    def _run_mincut(self, feat):
        # feat: (T, d)
        num_syllable = int(np.ceil(len(feat) / self.ft_sr / self.syl_dur))

        ssm = feat @ feat.transpose(1, 0)
        ssm = ssm - np.min(ssm) + 1e-7  # make it non-negative
        seg_boundary_frame = self.mincut.min_cut(
            ssm, num_syllable + 1
        )  # +1 for the algo
        seg_boundary_frame = np.array(seg_boundary_frame)
        seg_boundary_frame[-1] = seg_boundary_frame[-1]+1
        seg_boundary_frame = np.unique(seg_boundary_frame)
        seg_boundary_frame_pairs_orig = [
            [l, r] for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])
        ]  #
        seg_boundary_frame_pairs = [
            item for item in seg_boundary_frame_pairs_orig # if item[1] - item[0] > 2
        ]
        if len(seg_boundary_frame_pairs) == 0:  # this shouldn't happen though
            seg_boundary_frame_pairs = seg_boundary_frame_pairs_orig

        if self.pre_merge:
            feat, seg_boundary_frame_pairs = self._merge(feat, seg_boundary_frame_pairs)
        return feat, seg_boundary_frame_pairs

    def process(self, segments, features, output_in_second=True, **kwargs):
        
        boundaries = []
        pooled_feat = []
        
        for segment in segments:
            if (segment[1] - segment[0]) < self.min_cut_minimum:
                boundaries_ = [(segment - segment[0])]
                pooled_feat_ = [features[segment[0] : segment[1]].mean(0)] #.cpu().numpy()]
            else:
                pooled_feat_, boundaries_ = self._run_mincut(
                    features[segment[0] : segment[1]] #.cpu().numpy()
                )
            for bi, (bd, ft_) in enumerate(zip(boundaries_, pooled_feat_)):
                if np.isnan(np.sum(ft_)):
                    continue
                if (bd[1] - bd[0]) < self.min_segment_len:
                    continue
                boundaries.append(bd + segment[0])
                pooled_feat.append(ft_)
        if len(pooled_feat) >0:
            pooled_feat = np.stack(pooled_feat)
            boundaries = np.stack(boundaries)
            if not self.pre_merge:
                pooled_feat, boundaries = self._merge(pooled_feat, boundaries)
        else:
            pooled_feat =np.zeros((0,768))
            boundaries = np.zeros((0,2))
        if output_in_second:
            boundaries = boundaries/self.ft_sr
        outputs = {
            "segments": boundaries,
            "features": features,
            "length": len(features),
            "segment_features": pooled_feat,
        }
        return outputs
    
    def __call__(self, input_dict=None, segments=None, features=None, output_in_second=True, **kwargs):
        if input_dict is not None:
            # in the case the dictionary outputs from the SD-HuBERT are given
            if isinstance(input_dict, list):
                # given as a list of dictionary
                return [self.process(**d,output_in_second=output_in_second) for d in input_dict]
            else:
                return self.process(**input_dict, output_in_second=output_in_second)
        else:
            assert segments is not None, "Segments should be input!"
            assert features is not None, "Features should be input!"
            return self.process(segments, features, output_in_second=output_in_second)
