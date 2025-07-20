import argparse
import chess
import chess.engine
import torch
from .model import PolicyValueNet
from .mcts import MCTS


def play_against_stockfish(model_path: str, stockfish_cmd: str, games: int = 1):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_cmd)
    model = PolicyValueNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    results = []
    for _ in range(games):
        board = chess.Board()
        mcts = MCTS(model)
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                root = mcts.run(board)
                move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
                board.push(move)
            else:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
        results.append(board.result())
    engine.quit()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True)
    parser.add_argument('--uci', action='store_true')
    args = parser.parse_args()
    if args.uci:
        print('UCI mode not yet implemented')
        return
    else:
        res = play_against_stockfish(args.engine, 'stockfish')
        print(res)


if __name__ == '__main__':
    main()
