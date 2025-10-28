#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.serialization import add_safe_globals

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.register as R
from api.pdb_dataset import PDBDataset
from models.IterVAE.model import CondIterAutoEncoder
from models.LDM.ldm import LDMMolDesign


def parse_args():
    parser = argparse.ArgumentParser(description='Extract IterVAE latents from a single complex')
    parser.add_argument('--config', type=str, default='api/demo/config.yaml', help='Demo config file')
    parser.add_argument('--ckpt', type=str, default='checkpoints/model.ckpt', help='Checkpoint path')
    parser.add_argument('--template', type=str, default=None, help='Template class name to use')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index, -1 for CPU')
    parser.add_argument('--latent_out', type=str, default='api/demo/generations/latent_example.pt', help='File to store extracted latents')
    parser.add_argument('--seed', type=int, default=12, help='Random seed for reproducibility')
    return parser.parse_args()


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(data, device):
    if isinstance(data, dict):
        return {key: to_device(val, device) for key, val in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(val, device) for val in data)
    if hasattr(data, 'to'):
        return data.to(device)
    return data


def load_autoencoder(ckpt_path: str, device):
    add_safe_globals([LDMMolDesign, CondIterAutoEncoder])
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, CondIterAutoEncoder):
        autoencoder = obj
    elif hasattr(obj, 'autoencoder'):
        autoencoder = obj.autoencoder
    else:
        raise TypeError(f'Unsupported checkpoint object type: {type(obj)}')
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    return autoencoder


def pick_template(config_templates, template_name=None):
    if template_name is None:
        return config_templates[0]
    for tpl in config_templates:
        if tpl['class'] == template_name:
            return tpl
    raise ValueError(f'Template {template_name} not found in config')


def main():
    args = parse_args()
    setup_seed(args.seed)

    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    config = yaml.safe_load(open(args.config, 'r'))
    template_cfg = pick_template(config['templates'], args.template)
    template = R.construct(template_cfg)

    dataset = PDBDataset(**config['dataset'], template_config=template, n_samples=1)
    sample = dataset[0]
    batch = dataset.collate_fn([sample])
    cplx_desc = batch.pop('cplx_desc')[0]

    autoencoder = load_autoencoder(args.ckpt, device)
    batch = to_device(batch, device)

    with torch.no_grad():
        Zh, Zx, Zx_mu, signed_Zx_log_var, _, _, _, _ = autoencoder.encode(
            X=batch['X'],
            S=batch['S'],
            A=batch['A'],
            bonds=batch['bonds'],
            chain_ids=batch['chain_ids'],
            generate_mask=batch['generate_mask'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            deterministic=True
        )

    latent = {
        'Zh': Zh.cpu(),
        'Zx': Zx.cpu(),
        'Zx_mu': Zx_mu.cpu(),
        'signed_Zx_log_var': signed_Zx_log_var.cpu(),
        'metadata': {
            'config': args.config,
            'ckpt': args.ckpt,
            'template': template_cfg['class'],
            'pdb_paths': config['dataset']['pdb_paths'],
            'cplx_id': cplx_desc.id,
        }
    }

    out_path = Path(args.latent_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latent, out_path)

    info_path = out_path.with_suffix(out_path.suffix + '.json')
    info_path.write_text(json.dumps(latent['metadata'], indent=2))

    print(f'Latents saved to {out_path}')
    print(f'Metadata saved to {info_path}')
    print(f'Zh shape: {tuple(latent["Zh"].shape)}, Zx shape: {tuple(latent["Zx"].shape)}')


if __name__ == '__main__':
    main()
