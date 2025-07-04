import argparse
from pathlib import Path
import chess

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast
import yaml
import numpy as np

from ..models.net_large import load_from_small
from .performance import init_device, get_amp_scaler
from .checkpoint import save
from .logging import init_logging
from .mcts import Node, run_mcts
from data.pgn_parser import board_to_tensor


def load_config(path: Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def selfplay(model, cfg, device):
    games_data = []
    for _ in range(cfg['selfplay_games']):
        board = chess.Board()
        nodes = []
        while not board.is_game_over():
            root = Node(board.copy(), 0)
            run_mcts(model, root, cfg['mcts'], device)
            visits = np.array([child.visit_count for child in root.children.values()])
            moves = list(root.children.keys())
            probs = visits / visits.sum()
            move = np.random.choice(moves, p=probs)
            policy = np.zeros(4672, dtype=np.float32)
            # placeholder index
            policy[0] = 1.0
            state = board_to_tensor(board)
            board.push(move)
            nodes.append((state, policy))
        result = board.result()
        if result == '1-0':
            value = 1.0
        elif result == '0-1':
            value = -1.0
        else:
            value = 0.0
        for state, policy in nodes:
            games_data.append((state, policy, value))
            value = -value
    states, policies, values = zip(*games_data)
    return (
        torch.tensor(np.stack(states)),
        torch.tensor(np.stack(policies)),
        torch.tensor(values, dtype=torch.float32)
    )


def main(cfg_path: Path, log_dir: Path, ckpt_dir: Path, small_ckpt: Path):
    cfg = load_config(cfg_path)
    device = init_device(cfg)
    scaler = get_amp_scaler(cfg['use_amp'])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = load_from_small(str(small_ckpt))
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    writer = init_logging(str(log_dir), cfg)

    states, policies, values = selfplay(model, cfg, device)
    dataset = TensorDataset(states, policies, values)
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory'],
        persistent_workers=cfg['num_workers'] > 0,
    )

    for epoch in range(1):
        model.train()
        for i, (s, p, v) in enumerate(loader):
            s = s.to(device)
            p = p.to(device)
            v = v.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg['use_amp']):
                pred_p, pred_v = model(s)
                loss_p = criterion_policy(pred_p, p.argmax(dim=1))
                loss_v = criterion_value(pred_v.squeeze(), v)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--log_dir', type=Path, required=True)
    parser.add_argument('--ckpt_dir', type=Path, required=True)
    parser.add_argument('--small_ckpt', type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.log_dir, args.ckpt_dir, args.small_ckpt)
