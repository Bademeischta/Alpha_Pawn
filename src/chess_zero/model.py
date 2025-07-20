import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.input_conv = nn.Conv2d(18, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.res_layers = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4672),  # 4672 = 8x8x73 moves
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        out = torch.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_layers:
            out = block(out)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, default=None)
    args = parser.parse_args()

    net = PolicyValueNet()
    if args.init:
        torch.save(net.state_dict(), args.init)
