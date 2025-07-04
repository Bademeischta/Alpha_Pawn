import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # noqa: E402

import torch  # noqa: E402
from src.models.net_small import ChessNetSmall  # noqa: E402
from src.models.net_large import ChessNetLarge  # noqa: E402


def test_chessnet_small_forward():
    model = ChessNetSmall()
    x = torch.zeros(1, 12, 8, 8)
    policy, value = model(x)
    assert policy.shape == (1, 4672)
    assert value.shape == (1, 1)


def test_chessnet_large_forward():
    model = ChessNetLarge()
    x = torch.zeros(1, 12, 8, 8)
    policy, value = model(x)
    assert policy.shape == (1, 4672)
    assert value.shape == (1, 1)
