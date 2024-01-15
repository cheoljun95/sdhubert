# SD-HuBERT: Sentence-Level Self-Distillation Induces Syllabic Organization In HuBERT


In [this paper](https://arxiv.org/abs/2310.10803), we demonstrate that a syllabic organization emerges in learning sentence-level representation of speech. In particular, we adopt "self-distillation" objective to fine-tune the pretrained HuBERT with an aggregator token that summarizes the entire sentence. Without any supervision, the resulting model draws definite boundaries in speech, and the representations across frames show salient syllabic structures. We demonstrate that this emergent structure largely corresponds to the ground truth syllables. Furthermore, we propose a new benchmark task, Spoken Speech ABX, for evaluating sentence-level representation of speech. When compared to previous models, our model outperforms in both unsupervised syllable discovery and learning sentence-level representation. Together, we demonstrate that the self-distillation of HuBERT gives rise to syllabic organization without relying on external labels or modalities, and potentially provides novel data-driven units for spoken language modeling. 
![SD-HuBERT](figures/main_figure.jpg)

## Environment

1. We recommend to set up a conda environment. We trained/tested the model on Python 3.9.
```
conda create -n sdhubert python=3.9
conda activate rnnt
```
2. Please install a working version of PyTorch which fits your computing resources (tested on torch=1.13.1, CUDA==11.7, Linux).
3. Then install dependency packages through `pip install -r requirements.txt`.
4. We use the segmentation algorithm suggested by [Peng et al., 2023](https://arxiv.org/abs/2305.11435). We are using the implementation shared by the author, so please check the original code/installation [here](https://github.com/jasonppy/syllable-discovery/tree/master). You need Cython for this.
```
cd ./mincut
python setup.py build_ext --inplace
```

## Apply SD-HuBERT to get syllabic tokens

Download [the pretrained model] (https://drive.google.com/file/d/1u2jTdAck8qD6ZEb5bqHfvUNsN-9DgGfg/view?usp=drive_link) and put under the `ckpts/`. The following code will provide segment boundaries and the pooled feature per segment.

```python
from model.segmenter import SDHuBERTSegmenter, MincutWrapper

device = "cuda:0"
ckpt_path = "ckpts/sdhubert_base.pt" # or your own path
segmenter = SDHuBERTSegmenter(ckpt_path, layer=9, normcut_layer=11, normcut_threshold=2, device=device)
mincut = MincutWrapper(syl_dur=0.2, ft_sr=50) 

wav_file = WAV_FILE
outputs = mincut(**segmenter(wav_file))
```

The output should look like
```
{'segments': array of boundaries,
 'features': original feature of frames,
 'segment_features': average feature per segment,
 }
```

To get unit category, you can apply pretrained clustering model as follows. Please download the assets ([km](https://drive.google.com/file/d/14zdEttya2X8PdjDMUt4lyHWOOY2OS3Zr/view?usp=drive_link) and [reducer](https://drive.google.com/file/d/19XisepDAfULOKFY147RDYT5UAk2ZnCr-/view?usp=drive_link)) and place under `km/`.

```python
import numpy as np
import joblib

km_path = "km/sdhubert_base_16384.pt"
reducer_path = "km/sdhubert_base_16384to4096.npy"
km_model = joblib.load(km_path)
reducer = np.load(reducer_path)

# Unit prediction
units = [reducer[km_model.predict(segment_feature)] for segment_feature in outputs['segment_features']]
```

## Training SD-HuBERT

First, download the [LibriSpeech](https://www.openslr.org/12) data under some data directory, let's say `LIBRISPEECH_ROOT`. The directory should look like 
```
LIBRISPEECH_ROOT
├── train-clean-100
├── train-clean-360
├── train-other-500
├── dev-clean
├── dev-other
├── test-clean
└── test-other
```

The trainer is implemented using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), so please download the package through `pip install lightning` (we used lightning==2.1.2).

You can train with the following command. Also please check `configs/sdhubert_base` for detailed configurations.
```
python train.py --config-name=sdhubert_base data.root_dir=LIBRISPEECH_ROOT
```

After the model training is finished, export model to more handy checkpoint file. The `ckpt_path` should be pointed to the path that is created by running the training script.
```
python export_model.py --ckpt_path=outputs/DATE/TIME/NAME
```

## Evaluation

Please run through the following commands to extract segments and evaluate syllable boundary detection, purity, and SSABX task. Also, please check the arguments in the scripts to get full control of experiment.

### Extract segments

```
python extract_segments.py --ckpt_path={CKPT: e.g., ckpts/sdhubert_base.pt} --librispeech_dataroot=LIBRISPEECH_ROOT
```
This will extract segments under `SEGMENT_PATH=LIBRISPEECH_ROOT/segments/NAME`. The `NAME` is `sdhubert_base` by default.

### Evaluate syllable boundary detection

```
python evaluate_boundary.py --segment_path=SEGMENT_PATH
```

### Train clustering model

```
python train_km.py --segment_path=SEGMENT_PATH --n_clusters=16384 --n_clusters_agglomerative=4096
```

### Evaluate syllable clustering quality

```
python evaluate_purity.py --segment_path=SEGMENT_PATH --km_path=km/sdhubert_base_16384.pt --reducer_path=km/sdhubert_base_16384to4096.npy
```

### Evaluate Spoken Sentence ABX (SSABX) task

Also, check `files/ssabx.json` for the SSABX triplets mined from LibriSpeech (more detail can be found in the paper).
```
python evaluate_ssabx.py --ckpt_path={CKPT: e.g., ckpts/sdhubert_base.pt} --librispeech_dataroot=LIBRISPEECH_ROOT
```

## Acknowledgements

Thanks to Puyuan Peng for sharing the [code and resources](https://github.com/jasonppy/syllable-discovery/tree/master). 

## Citation

```
@inproceedings{cho2023sd,
  title={SD-HuBERT: Sentence-Level Self-Distillation Induces Syllabic Organization in HuBERT},
  author={Cho, Cheol Jun and Mohamed, Abdelrahman and Li, Shang-Wen and Black, Alan W and Anumanchipalli, Gopala K},
  journal={ICASSP},
  year={2024}
}
```