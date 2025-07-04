import math
import numpy as np
import chess
from data.pgn_parser import board_to_tensor


class Node:
    def __init__(self, board: chess.Board, prior: float):
        self.board = board
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


def select_child(node, cpuct):
    best_score = -float('inf')
    best_move = None
    best_child = None
    for move, child in node.children.items():
        u = cpuct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
        score = child.value + u
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    return best_move, best_child


def expand(node: Node, policy: np.ndarray) -> None:
    """Expand a node using the provided policy probabilities."""
    legal_moves = list(node.board.legal_moves)
    indices = np.array([move_to_index(m) for m in legal_moves])
    priors = policy[indices]
    total = priors.sum()
    if total > 0:
        priors = priors / total
    for move, prior in zip(legal_moves, priors):
        board = node.board.copy()
        board.push(move)
        node.children[move] = Node(board, float(prior))


def move_to_index(move: chess.Move) -> int:
    """Map a move to an action index in a 4672-action space."""
    from_sq = move.from_square
    to_sq = move.to_square
    if move.promotion:
        promo_map = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3,
        }
        direction = 0 if chess.square_rank(to_sq) > chess.square_rank(from_sq) else 1
        return 4096 + from_sq * 9 + direction * 4 + promo_map.get(move.promotion, 0)
    return from_sq * 64 + to_sq


def simulate(model, node, device):
    board_tensor = board_to_tensor(node.board)
    policy_logits, value = model(board_tensor.unsqueeze(0).to(device))
    policy = policy_logits.softmax(-1)[0].detach().cpu().numpy()
    value = value.item()
    return policy, value


def run_mcts(model, root: Node, cfg, device):
    for _ in range(cfg['simulations']):
        node = root
        path = []
        while node.expanded():
            move, node = select_child(node, cfg['cpuct'])
            path.append(node)
        policy, value = simulate(model, node, device)
        if not node.expanded():
            expand(node, policy)
        backup(path + [node], value)


def backup(nodes, value):
    for node in reversed(nodes):
        node.value_sum += value
        node.visit_count += 1
        value = -value
