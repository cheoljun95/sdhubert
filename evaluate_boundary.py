from pathlib import Path
from utils.syllable import BoundaryDetectionEvaluator
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--segment_path", type=str, default='/data/common/LibriSpeech/segments/sdhubert_base')
parser.add_argument("--test_syllables", type=str, default='files/librispeech_syllable_test.json')
parser.add_argument("--val_syllables", type=str, default='files/librispeech_syllable_val.json')
parser.add_argument("--save_name", type=str, default=None)
parser.add_argument("--result_path", type=str, default='results')


if __name__ == '__main__':
    args = parser.parse_args()
    save_name = args.save_name if args.save_name is not None else Path(args.segment_path).stem
    evaluator = BoundaryDetectionEvaluator(args.segment_path, 
                                           args.test_syllables,
                                           args.val_syllables,
                                           tolerance=0.05, max_val_num=None)
    boundary_results = evaluator.evaluate()
    print("Precision:", boundary_results['prec'])
    print("Recall: ", boundary_results['recall'])
    print("F1:", boundary_results['f1'])
    print("Over-segmentation: ", boundary_results['os'])
    print("R value: ", boundary_results['r_val'])
    
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
        
    result['boundary_detection'] = boundary_results
    
    with open(result_path, 'w') as f:
        json.dump(result, f)
        