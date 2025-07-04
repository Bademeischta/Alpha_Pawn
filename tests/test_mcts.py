import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # noqa: E402

import chess  # noqa: E402
import numpy as np  # noqa: E402
from src.training.mcts import Node, expand, backup, move_to_index  # noqa: E402


def test_expand_and_backup():
    board = chess.Board()
    node = Node(board, prior=1.0)
    policy = np.zeros(4672, dtype=np.float32)
    legal_moves = list(board.legal_moves)
    indices = [move_to_index(m) for m in legal_moves]
    policy[indices] = 1 / len(indices)
    expand(node, policy)
    assert len(node.children) == len(legal_moves)
    backup([node], 1.0)
    assert node.visit_count == 1
    assert node.value_sum == 1.0
