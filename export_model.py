import torch
from collections import OrderedDict
from utils.misc import load_cfg_and_ckpt_path
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--output_path", type=str, default='ckpts')
parser.add_argument("--mode", type=str, default='latest')

if __name__ == '__main__':
    args = parser.parse_args()
    version_dir = args.ckpt_path
    output_dir = Path(args.output_path)
    output_dir.mkdir(exist_ok=True)
    cfg, ckpt_path = load_cfg_and_ckpt_path(version_dir=version_dir, mode=args.mode)
    state_dict= torch.load(ckpt_path,map_location='cpu')['state_dict']
    name = cfg['=name']

    trimmed_state_dict = OrderedDict()
    for module_name, state in state_dict.items():
        if f'net.' in module_name:
            trimmed_name = module_name.split(f'net.')[-1]
            trimmed_state_dict[trimmed_name] = state

    module_ckpt = {'config': cfg['model'],
                   'state_dict': trimmed_state_dict}
    torch.save(module_ckpt, output_dir/f'{name}.pt')