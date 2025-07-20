import argparse
import chess
import torch
from .model import PolicyValueNet
from .mcts import MCTS
from .utils import board_to_tensor


def play_game(model: PolicyValueNet, max_moves: int = 160):
    board = chess.Board()
    mcts = MCTS(model)
    history = []

    for _ in range(max_moves):
        if board.is_game_over():
            break
        root = mcts.run(board)
        move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        history.append((board.fen(), move.uci()))
        board.push(move)
    return history, board.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--out', type=str, default='data/selfplay')
    args = parser.parse_args()

    model = PolicyValueNet()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    for i in range(args.games):
        history, result = play_game(model)
        torch.save({'history': history, 'result': result}, f"{args.out}/game_{i:04d}.pt")


if __name__ == '__main__':
    main()
