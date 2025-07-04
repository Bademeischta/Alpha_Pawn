import chess
import chess.pgn
import numpy as np
from pathlib import Path

PIECE_TO_IDX = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

MOVE_SIZE = 4672


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert a board to a 12x8x8 tensor representation."""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for piece_type in chess.PIECE_TYPES:
        for color in (chess.WHITE, chess.BLACK):
            idx = PIECE_TO_IDX[(piece_type, color)]
            for square in board.pieces(piece_type, color):
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                tensor[idx, row, col] = 1.0
    return tensor


def parse_pgn(path: Path) -> None:
    """Parse a PGN file and store dataset as NPZ."""
    boards = []
    policies = []
    values = []
    with open(path, "r", encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
        while game:
            result = game.headers.get("Result")
            if result == "1-0":
                value = 1.0
            elif result == "0-1":
                value = -1.0
            else:
                value = 0.0

            board = game.board()
            for move in game.mainline_moves():
                boards.append(board_to_tensor(board))
                policy = np.zeros(MOVE_SIZE, dtype=np.float32)  # placeholder
                policies.append(policy)
                values.append(value)
                board.push(move)

            game = chess.pgn.read_game(f)

    np.savez_compressed(path.with_suffix(".npz"),
                        states=np.stack(boards),
                        policies=np.stack(policies),
                        values=np.array(values, dtype=np.float32))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn", type=Path)
    args = parser.parse_args()
    parse_pgn(args.pgn)
