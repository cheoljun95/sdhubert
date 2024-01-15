#from model.sdhubert import XModel
from utils.syllable import BoundaryDetectionEvaluator
import argparse


if __name__ == '__main__':
    librispeech_dataroot= '/data/common/LibriSpeech'
    test_syllables = 'files/librispeech_syllable_test.json'
    val_syllables = 'files/librispeech_syllable_val.json'
    output_name = 'sdhubertv2_outputs'
    
    evaluator = BoundaryDetectionEvaluator(librispeech_dataroot, output_name, test_syllables,val_syllables,
                                           tolerance=0.05, max_val_num=None)
    results = evaluator.evaluate()
    print("Precision:", results['prec'])
    print("Recall: ", results['recall'])
    print("F1:",results['f1'])
    print("Over-segmentation: ", results['os'])
    print("R value: ", results['r_val'])