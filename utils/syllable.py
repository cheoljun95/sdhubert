'''
Adapted from https://github.com/jasonppy/syllable-discovery.git
'''
import numpy as np
from pathlib import Path
import json 
import random
import tqdm
from scipy.optimize import linear_sum_assignment

def match_boundary(gt, pred, tolerance):
    """
    gt: list of ground truth boundaries
    pred: list of predicted boundaries
    all in seconds
    """
    gt_pointer = 0
    pred_pointer = 0
    gt_len = len(gt)
    pred_len = len(pred)
    match_pred = 0
    match_gt = 0
    while gt_pointer < gt_len and pred_pointer < pred_len:
        if np.abs(gt[gt_pointer] - pred[pred_pointer]) <= tolerance:
            match_gt += 1
            match_pred += 1
            gt_pointer += 1
            pred_pointer += 1
        elif gt[gt_pointer] > pred[pred_pointer]:
            pred_pointer += 1
        else:
            gt_pointer += 1
    return match_gt, match_pred, gt_len, pred_len


def f1_score(prec,recall):
    return 2*prec*recall/(prec+recall)

def os_score(prec, recall):
    return recall/prec -1

def r1_score(recall, os):
    return np.sqrt((1-recall)**2 + os**2)

def r2_score(recall, os):
    return (-os + recall - 1) / np.sqrt(2)

def r_value(r1, r2):
    return 1. - (np.abs(r1) + np.abs(r2))/2.

class BoundaryDetectionEvaluator(object):
    
    def __init__(self, segment_path, test_syllables, val_syllables, tolerance=0.05,
                max_val_num = 500):
        self.dataroot = Path(segment_path)
        self.test_syllables = json.load(open(test_syllables,'r'))
        self.val_syllables = json.load(open(val_syllables,'r'))
        self.tolerance = tolerance
        self.best_shift = None
        self.shift_range=np.arange(-0.05, 0.05, 0.005)
        if max_val_num is not None:
            val_keys = list(self.val_syllables.keys())
            random.shuffle(val_keys)
            val_keys = val_keys[:max_val_num]
            val_keys.sort()
            self.val_syllables = {key:self.val_syllables[key] in val_keys}
        
    def _find_best_shift(self):
        best_shift = -999
        best_r_val = -999
        print('Searching best shift...')
        for shift in tqdm.tqdm(self.shift_range,leave=False):
            r_val = self.evaluate(is_val=True, shift=shift)['r_val']
            if r_val > best_r_val:
                best_shift = shift
                best_r_val = r_val
        print("Best shift:", best_shift)
        self.best_shift = best_shift
            
    def evaluate(self, is_val=False, shift=0):
        if not is_val and self.best_shift is None:
            self._find_best_shift()
        
        if is_val:
            best_shift = shift
        else:
            best_shift = self.best_shift
        
        if is_val:
            syllables=self.val_syllables
        else:
            syllables=self.test_syllables
        results = {'match_gt':0,
                   'match_gt':0,
                   'match_pred':0,
                   'gt_len':0,
                   'pred_len':0,}
        
        for _, syls in syllables.items():
            file_name = syls['file_name']
            gt_segments = [[float(segment['start']),float(segment['end'])] for segment in syls['syllables']]
            gt_segments = np.array(gt_segments)
            pred_segments = np.load(self.dataroot/file_name.replace('.flac','.npy'), allow_pickle=True)[()]['segments']
            gt_boundaries = np.unique(gt_segments)+best_shift
            gt_boundaries.sort()
            pred_boundaries = np.unique(pred_segments[:,0])
            pred_boundaries.sort()
            match_gt, match_pred, gt_len, pred_len = match_boundary(gt_boundaries,
                                                                           pred_boundaries,
                                                                           self.tolerance)
            results['match_gt'] += match_gt
            results['match_pred'] += match_pred
            results['gt_len'] += gt_len
            results['pred_len'] += pred_len
            
        results['prec'] = results['match_pred'] / results['pred_len']
        results['recall'] = results['match_gt'] / results['gt_len']
        results['f1'] = f1_score(results['prec'], results['recall'])
        results['os'] = os_score(results['prec'], results['recall'])
        results['r1'] = r1_score(results['recall'], results['os'])
        results['r2'] = r2_score(results['recall'], results['os'])
        results['r_val'] =r_value(results['r1'], results['r2'])
        
        return results
    
            
            
def temporal_iou_mat(bd,aln):
    bd_exp=np.repeat(bd[:,None,:],len(aln),axis=1)
    aln_exp=np.repeat(aln[None,:,:],len(bd),axis=0)
    start_mat =np.concatenate([bd_exp[:,:,0:1],aln_exp[:,:,0:1]],-1)
    end_mat =np.concatenate([bd_exp[:,:,1:],aln_exp[:,:,1:]],-1)

    inter_mat = end_mat.min(-1)-start_mat.max(-1)
    union_mat = end_mat.max(-1)-start_mat.min(-1)
    iou_mat = inter_mat/(union_mat+0.0001)
    return iou_mat

def match_cluster(gt_segments, pred_segments):
    iou_mat = temporal_iou_mat(gt_segments, pred_segments)
    gt_idxs, pred_idxs = linear_sum_assignment(iou_mat, maximize=True)
    return gt_idxs, pred_idxs

def trim_label(label):
    return label.replace('0','').replace('1','').replace('2','').replace('3','')

def add_count_dict(dict_, elm):
    if elm not in dict_.keys():
        dict_[elm]=0
    dict_[elm]+=1
    
def append_set_dict(dict_, key,elm,):
    if key not in dict_.keys():
        dict_[key]=[]
    if elm not in dict_[key]:
        dict_[key].append(elm)