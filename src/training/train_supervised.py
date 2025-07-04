import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast
import yaml
import numpy as np

from ..models.net_small import ChessNetSmall
from .performance import init_device, get_amp_scaler
from .checkpoint import save
from .logging import init_logging


def load_config(path: Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(cfg_path: Path, log_dir: Path, ckpt_dir: Path, data_path: Path):
    cfg = load_config(cfg_path)
    device = init_device(cfg)
    scaler = get_amp_scaler(cfg['use_amp'])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path)
    dataset = TensorDataset(
        torch.tensor(data['states']),
        torch.tensor(data['policies']),
        torch.tensor(data['values'])
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory'],
        persistent_workers=cfg['num_workers'] > 0,
    )

    model = ChessNetSmall(cfg['model']['blocks'], cfg['model']['channels']).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    writer = init_logging(str(log_dir), cfg)

    for epoch in range(cfg['epochs']):
        model.train()
        for i, (states, policies, values) in enumerate(loader):
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg['use_amp']):
                pred_p, pred_v = model(states)
                loss_p = criterion_policy(pred_p, policies.argmax(dim=1))
                loss_v = criterion_value(pred_v.squeeze(), values)
                loss = loss_p + loss_v
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if i % 100 == 0:
                writer.add_scalar('train/loss', loss.item(), epoch * len(loader) + i)

        save(model, optimizer, epoch + 1, ckpt_dir / f'epoch_{epoch+1}.pt')
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--log_dir', type=Path, required=True)
    parser.add_argument('--ckpt_dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.log_dir, args.ckpt_dir, args.data)
