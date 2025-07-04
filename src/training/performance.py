import torch


def init_device(cfg):
    """Initialize the computation device."""
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', False)
    device = cfg.get('device', 'cpu')
    return torch.device(device if torch.cuda.is_available() else 'cpu')


def get_amp_scaler(use_amp: bool):
    return torch.cuda.amp.GradScaler() if use_amp else None
