import argparse
from pathlib import Path
import numpy as np
import joblib
import json
import tqdm
from collections import Counter
from utils.syllable import trim_label, match_cluster, add_count_dict, append_set_dict

parser = argparse.ArgumentParser()
parser.add_argument("--segment_path", type=str, default='/data/common/LibriSpeech/segments/sdhubert_base')
parser.add_argument("--km_path", type=str, default='km/sdhubert_base_16384.pt')
parser.add_argument("--reducer_path", type=str, default='km/sdhubert_base_16384to4096.npy')
parser.add_argument("--test_syllables", type=str, default='files/librispeech_syllable_test.json')
parser.add_argument("--save_name", type=str, default=None)
parser.add_argument("--result_path", type=str, default='results')

if __name__=='__main__':
    args = parser.parse_args()
    segment_path = Path(args.segment_path)
    test_syllables = json.load(open(args.test_syllables, 'r'))
    
    km_model=joblib.load(args.km_path)
    if args.reducer_path is None:
        reducer = None
    else:
        reducer = np.load(args.reducer_path)
    
    save_name = args.save_name if args.save_name is not None else Path(args.segment_path).stem
    
    print(f'{save_name} - Counting statistics...')
    co_occurence_matrix={}
    syl_counts={}
    cluster_counts={}
    syl_to_cluster={}
    cluster_to_syl={}
    
    for _, syls in tqdm.tqdm(test_syllables.items(), leave=False):
        file_name = syls['file_name']
        gt_segments = [[float(segment['start']),float(segment['end'])] for segment in syls['syllables']]
        if len(syls['syllables'])==0:
            continue
        labels=[trim_label(segment['label']) for segment in syls['syllables']]
        gt_segments = np.array(gt_segments)
    
        preds = np.load(segment_path/file_name.replace('.flac','.npy'), allow_pickle=True)[()]
        
        pred_segments = preds['segments']
        pred_units = km_model.predict(preds['segment_features'])
        if reducer is not None:
            pred_units = reducer[pred_units]
        gt_idxs, pred_idxs = match_cluster(gt_segments, pred_segments)
        
        for gt_i, pred_i in zip(gt_idxs, pred_idxs):
            add_count_dict(co_occurence_matrix, (labels[gt_i], pred_units[pred_i]))
            add_count_dict(syl_counts, labels[gt_i])
            add_count_dict(cluster_counts, pred_units[pred_i])
            append_set_dict(syl_to_cluster, labels[gt_i], pred_units[pred_i])
            append_set_dict(cluster_to_syl, pred_units[pred_i], labels[gt_i])
        
    counts={}
    counts['co_occurence_matrix']=co_occurence_matrix
    counts['syl_counts']=syl_counts
    counts['cluster_counts']=cluster_counts
    counts['syl_to_cluster']=syl_to_cluster
    counts['cluster_to_syl']=cluster_to_syl
    
    # Cluster Purity
    cluster_purity =[]
    total_cnt = np.sum([cnt for _, cnt in syl_counts.items()])
    assert np.sum([cnt for _, cnt in syl_counts.items()])==np.sum([cnt for _, cnt in cluster_counts.items()])
    assert np.sum([cnt for _, cnt in syl_counts.items()])==np.sum([cnt for _, cnt in co_occurence_matrix.items()])
    
    for syl,syl_cnt in syl_counts.items():
        cluster = syl_to_cluster[syl]
        counter = Counter({lab:co_occurence_matrix[(syl,lab)] for lab in cluster}) 
        argmax_lab, max_cnt = counter.most_common()[0]
        cluster_purity.append(max_cnt/total_cnt)

    cluster_purity = np.sum(cluster_purity)
    
    # Syllable Purity
    syllable_purity =[]
    for lab,lab_cnt in cluster_counts.items():
        syllabe = cluster_to_syl[lab]
        counter = Counter({syl:co_occurence_matrix[(syl,lab)] for syl in syllabe}) 
        argmax_syl, max_cnt = counter.most_common()[0]
        syllable_purity.append(max_cnt/total_cnt)

    syllable_purity = np.sum(syllable_purity)
    
    print(f'{save_name} - Cluster Purity: {cluster_purity:.04f}')
    print(f'{save_name} - Syllable Purity: {syllable_purity:.04f}')
    
    Path(args.result_path).mkdir(exist_ok=True)
    result_path = Path(args.result_path)/f'{save_name}.json'
    if result_path.exists():
        try:
            result = json.load(open(result_path,'r'))
        except:
            print("Can't open preexisting result")
            result = {}
    else:
        result = {}
        
    result['clustering_quality'] = {'cluster_purity':cluster_purity,
                                    'syllable_purity':syllable_purity}
    
    with open(result_path, 'w') as f:
        json.dump(result, f)