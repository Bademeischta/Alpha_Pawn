import torch.nn as nn
from .base import ResidualBlock, PolicyHead, ValueHead


class ChessNetSmall(nn.Module):

    """Compact chess network for supervised pre-training."""

    def __init__(self, blocks: int = 8, channels: int = 64):
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
