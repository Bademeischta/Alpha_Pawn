import torch
from pathlib import Path


def save(model, optimizer, epoch: int, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, path)


def load(model, optimizer, path: Path):
    if not path.exists():
        return 0
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']
