import math
import random
import chess
import torch
from .model import PolicyValueNet
from .utils import board_to_tensor


class Node:
    def __init__(self, board: chess.Board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, model: PolicyValueNet, simulations: int = 100, cpuct: float = 2.0):
        self.model = model
        self.simulations = simulations
        self.cpuct = cpuct

    def run(self, board: chess.Board):
        root = Node(board)
        for _ in range(self.simulations):
            node = root
            search_path = [node]

            # Selection
            while node.expanded():
                max_ucb, node = max(
                    (
                        self._ucb(child)
                        for child in node.children.values()
                    ),
                    key=lambda x: x[0]
                )
                node = node
                search_path.append(node)

            # Expansion
            if not node.board.is_game_over():
                self._expand(node)
                value = self._evaluate(node)
            else:
                result = self._terminal_value(node.board)
                value = result

            # Backpropagation
            for n in search_path:
                n.visit_count += 1
                n.value_sum += value if n.board.turn == board.turn else -value
        return root

    def _expand(self, node: Node):
        for move in node.board.legal_moves:
            new_board = node.board.copy()
            new_board.push(move)
            node.children[move] = Node(new_board, parent=node, prior=1/len(list(node.board.legal_moves)))

    def _evaluate(self, node: Node):
        tensor = board_to_tensor(node.board).unsqueeze(0)
        with torch.no_grad():
            policy, value = self.model(tensor)
        # simple prior distribution over legal moves
        policy = torch.softmax(policy[0], dim=0)
        for move, p in zip(node.board.legal_moves, policy[: node.board.legal_moves.count()]):
            if move in node.children:
                node.children[move].prior = p.item()
        return value.item()

    def _terminal_value(self, board: chess.Board):
        result = board.result()
        if result == '1-0':
            return 1.0
        if result == '0-1':
            return -1.0
        return 0.0

    def _ucb(self, child: Node):
        prior_score = self.cpuct * child.prior * math.sqrt(child.parent.visit_count) / (1 + child.visit_count)
        value_score = child.value()
        return prior_score + value_score, child
