import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import chess.pgn  # noqa: E402
from data.pgn_parser import board_to_tensor, parse_pgn  # noqa: E402


def test_board_to_tensor_shape():
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (12, 8, 8)


def test_parse_pgn(tmp_path):
    pgn_text = """[Event \"Test\"]\n1. e4 e5 2. Nf3 Nc6 1/2-1/2\n"""
    pgn_file = tmp_path / "game.pgn"
    pgn_file.write_text(pgn_text)
    parse_pgn(pgn_file)
    npz_path = pgn_file.with_suffix('.npz')
    data = np.load(npz_path)
    assert data['states'].shape[1:] == (12, 8, 8)
    assert data['policies'].shape[-1] == 4672
    assert len(data['values']) == len(data['states'])
