import torch
import torch.nn as nn
from .base import ResidualBlock, PolicyHead, ValueHead
from .net_small import ChessNetSmall


class ChessNetLarge(nn.Module):

    """Bigger network used after scaling with self-play."""

    def __init__(self, blocks: int = 16, channels: int = 128):
        super().__init__()
        self.input_conv = nn.Conv2d(12, channels, 3, padding=1)
        blocks_list = [ResidualBlock(channels) for _ in range(blocks)]
        self.res_blocks = nn.Sequential(*blocks_list)
        self.policy_head = PolicyHead(channels, move_size=4672)
        self.value_head = ValueHead(channels)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


def load_from_small(checkpoint_path: str) -> 'ChessNetLarge':
    small = ChessNetSmall()
    small.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    large = ChessNetLarge()
    large.input_conv.weight.data[:64] = small.input_conv.weight.data
    large.res_blocks[:8].load_state_dict(small.res_blocks.state_dict())
    return large
