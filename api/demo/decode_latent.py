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
from api.generate import data_to_cplx
from api.pdb_dataset import PDBDataset
from models.IterVAE.model import CondIterAutoEncoder
from models.LDM.ldm import LDMMolDesign


def parse_args():
    parser = argparse.ArgumentParser(description='Decode IterVAE latents into a binder structure')
    parser.add_argument('--config', type=str, default='api/demo/config.yaml', help='Demo config file')
    parser.add_argument('--ckpt', type=str, default='checkpoints/model.ckpt', help='Checkpoint path')
    parser.add_argument('--latent', type=str, default='api/demo/generations/latent_example.pt', help='Latent tensor file')
    parser.add_argument('--template', type=str, default=None, help='Template class name to use')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index, -1 for CPU')
    parser.add_argument('--out_dir', type=str, default='api/demo/generations/latent_decode', help='Directory to store decoded structures')
    parser.add_argument('--vae_steps', type=int, default=10, help='Number of VAE decoding iterations')
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

    latent = torch.load(args.latent, map_location=device)
    Zh = latent['Zh'].to(device)
    Zx = latent['Zx'].to(device)

    gen_kwargs = dict(
        X=batch['X'],
        S=batch['S'],
        A=batch['A'],
        bonds=batch['bonds'],
        position_ids=batch['position_ids'],
        chain_ids=batch['chain_ids'],
        generate_mask=batch['generate_mask'],
        block_lengths=batch['block_lengths'],
        lengths=batch['lengths'],
        is_aa=batch['is_aa'],
        given_latent=(Zh, Zx, None),
        n_iter=args.vae_steps,
    )
    if 'topo_generate_mask' in batch:
        gen_kwargs['topo_generate_mask'] = batch['topo_generate_mask']

    with torch.no_grad():
        batch_S, batch_X, batch_A, batch_ll, batch_inter_bonds, batch_intra_bonds = autoencoder.generate(**gen_kwargs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{cplx_desc.id}_latent_decode.pdb'

    cplx_desc, details = data_to_cplx(
        cplx_desc,
        template,
        batch_S[0],
        batch_X[0],
        batch_A[0],
        batch_ll[0],
        batch_inter_bonds[0],
        batch_intra_bonds[0],
        str(out_path)
    )

    if cplx_desc is None:
        raise RuntimeError('Failed to convert decoded results back to complex')

    info = {
        'out_path': str(out_path),
        'smiles': details['smiles'],
        'pmetric': details['pmetric'],
        'generated_sequence': details['gen_seq'],
        'template': template_cfg['class'],
        'latent_path': args.latent,
        'config': args.config,
        'ckpt': args.ckpt,
    }
    (out_dir / 'decode_details.json').write_text(json.dumps(info, indent=2))
    print(f'Decoded binder saved to {out_path}')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
