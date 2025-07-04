import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # noqa: E402

import os  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.training import train_supervised, train_selfplay  # noqa: E402
from src.models.net_small import ChessNetSmall  # noqa: E402
from src.models.net_large import ChessNetLarge  # noqa: E402


def test_train_supervised_smoke(tmp_path):
    cfg = {
        'device': 'cpu',
        'cudnn_benchmark': False,
        'use_amp': False,
        'batch_size': 1,
        'num_workers': 0,
        'pin_memory': False,
        'lr': 1e-3,
        'weight_decay': 0.0,
        'epochs': 1,
        'model': {'blocks': 1, 'channels': 16},
    }
    config_path = tmp_path / 'cfg.yaml'
    import yaml
    config_path.write_text(yaml.safe_dump(cfg))
    data = {
        'states': np.zeros((1, 12, 8, 8), dtype=np.float32),
        'policies': np.zeros((1, 4672), dtype=np.float32),
        'values': np.zeros(1, dtype=np.float32),
    }
    data_path = tmp_path / 'data.npz'
    np.savez_compressed(data_path, **data)
    os.environ['WANDB_MODE'] = 'disabled'
    train_supervised.main(config_path, tmp_path / 'log', tmp_path / 'ckpt', data_path)
    assert any(p.suffix == '.pt' for p in (tmp_path / 'ckpt').iterdir())


def test_train_selfplay_smoke(tmp_path, monkeypatch):
    cfg = {
        'device': 'cpu',
        'cudnn_benchmark': False,
        'use_amp': False,
        'batch_size': 1,
        'num_workers': 0,
        'pin_memory': False,
        'mcts': {'simulations': 1, 'cpuct': 1.0},
        'model': {'initial': 'small', 'upgraded': {'blocks': 1, 'channels': 16}},
        'selfplay_games': 1,
    }
    import yaml
    config_path = tmp_path / 'cfg.yaml'
    config_path.write_text(yaml.safe_dump(cfg))

    # create small checkpoint for ChessNetSmall
    small_model = ChessNetSmall()
    ckpt_path = tmp_path / 'small.pt'
    torch.save(small_model.state_dict(), ckpt_path)

    def dummy_selfplay(model, cfg, device):
        states = torch.zeros(1, 12, 8, 8)
        policies = torch.zeros(1, 4672)
        values = torch.zeros(1)
        return states, policies, values

    monkeypatch.setattr(train_selfplay, 'selfplay', dummy_selfplay)
    monkeypatch.setattr(train_selfplay, 'load_from_small', lambda p: ChessNetLarge())

    os.environ['WANDB_MODE'] = 'disabled'
    train_selfplay.main(config_path, tmp_path / 'log', tmp_path / 'ckpt', ckpt_path)
    assert any(p.suffix == '.pt' for p in (tmp_path / 'ckpt').iterdir())
