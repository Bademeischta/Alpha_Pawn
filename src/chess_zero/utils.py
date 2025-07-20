import torch
import chess

PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

AUX_PLANES = 6


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a chess.Board into a (18, 8, 8) tensor."""
    planes = torch.zeros(18, 8, 8, dtype=torch.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 6
            index = PIECE_TO_INDEX[piece.piece_type] + color_offset
            row = 7 - square // 8
            col = square % 8
            planes[index, row, col] = 1.0

    # Aux planes
    planes[12].fill_(1.0 if board.turn == chess.WHITE else 0.0)
    planes[13].fill_(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    planes[14].fill_(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    planes[15].fill_(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    planes[16].fill_(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    planes[17].fill_(min(board.fullmove_number / 200, 1.0))

    return planes
