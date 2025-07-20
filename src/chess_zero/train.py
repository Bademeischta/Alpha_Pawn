import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from .model import PolicyValueNet


class SelfPlayDataset(Dataset):
    def __init__(self, data_dir):
        from pathlib import Path
        self.files = list(Path(data_dir).glob('game_*.pt'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data['history'], data['result']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_out', type=str, required=True)
    args = parser.parse_args()

    dataset = SelfPlayDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = PolicyValueNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for history, result in loader:
        # TODO: implement training logic
        pass

    torch.save(model.state_dict(), args.checkpoint_out)


if __name__ == '__main__':
    main()
