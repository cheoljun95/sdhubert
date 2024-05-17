import numpy as np
import torch
import yaml
from pathlib import Path
from model.sdhubert import SDHuBERT
from model.sdwavlm import SDHuBERT as SDWavLM
from collections import OrderedDict

def load_model(ckpt_path, load_wavlm=False, **kwargs):
    try:
        ckpt = torch.load(ckpt_path)
        cfg = ckpt['config']
        state_dict = ckpt['state_dict']
    except:
        cfg, ckpt_path = load_cfg_and_ckpt_path(version_dir=ckpt_path,**kwargs)
        ckpt = torch.load(ckpt_path)
        cfg = cfg['model']
        state_dict = OrderedDict()
        for module_name, state in ckpt['state_dict'].items():
            if f'net.' in module_name:
                new_name = module_name.split(f'net.')[-1]
                state_dict[new_name] = state
    if load_wavlm:
        model = SDWavLM(**cfg)
    else:
        model = SDHuBERT(**cfg)
    model.load_state_dict(state_dict, strict=False)
    model = model.eval()

    return model, cfg

def load_cfg_and_ckpt_path(version_dir, mode='latest'):
    version_dir = Path(version_dir)
    try:
        cfg = yaml.load(open(version_dir / '.hydra' / 'config.yaml'), Loader=yaml.FullLoader)
    except:
        cfg = yaml.load(open(version_dir.parent / '.hydra' / 'config.yaml'), Loader=yaml.FullLoader)
    version_name = [f for f in (version_dir / 'lightning_logs').glob('version_*')][0].stem
    if mode == 'best':
        checkpoint_path = [f for f in Path(version_dir / 'lightning_logs' / version_name / 'checkpoints').glob('*.ckpt')
                           if 'best' in f.name]
        checkpoint_path = checkpoint_path[-1]
    else:
        checkpoint_path = [f for f in
                           Path(version_dir / 'lightning_logs' / version_name / 'checkpoints').glob('*.ckpt')]

        def get_epoch(fileName):
            epoch = [n for n in fileName.split('-') if 'epoch' in n][0]
            return int(epoch.split('=')[-1])

        checkpoint_path.sort(key=lambda f: get_epoch(f.name))
        checkpoint_path = checkpoint_path[-1]
        
    print(f"{str(checkpoint_path)} is located.")
    return cfg, str(checkpoint_path)