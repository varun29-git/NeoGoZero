from __future__ import annotations

from myalphago.go.game import GameState, Move
from myalphago.go.types import Player, Point

BoardPlanes = tuple[tuple[tuple[int, ...], ...], ...]
PolicyVector = tuple[float, ...]


def encode_game_state(game_state: GameState) -> BoardPlanes:
    size = game_state.board.size
    current_player = game_state.next_player
    opponent = current_player.other

    current_stones = _stone_plane(game_state, current_player)
    opponent_stones = _stone_plane(game_state, opponent)
    next_player_plane = _filled_plane(size, 1 if current_player is Player.BLACK else 0)

    return (current_stones, opponent_stones, next_player_plane)


def encode_board_snapshot(
    board: tuple[tuple[int, int, str], ...],
    player: Player,
    board_size: int,
) -> BoardPlanes:
    stones = {(row, col): color for row, col, color in board}
    opponent = player.other

    current_stones = tuple(
        tuple(
            1 if stones.get((row, col)) == player.value else 0
            for col in range(1, board_size + 1)
        )
        for row in range(1, board_size + 1)
    )
    opponent_stones = tuple(
        tuple(
            1 if stones.get((row, col)) == opponent.value else 0
            for col in range(1, board_size + 1)
        )
        for row in range(1, board_size + 1)
    )
    next_player_plane = _filled_plane(board_size, 1 if player is Player.BLACK else 0)

    return (current_stones, opponent_stones, next_player_plane)


def encode_policy(
    visit_distribution: dict[Move, float],
    board_size: int,
) -> PolicyVector:
    vector = [0.0] * (board_size * board_size + 1)
    for move, probability in visit_distribution.items():
        vector[move_to_index(move, board_size)] = probability
    return tuple(vector)


def move_to_index(move: Move, board_size: int) -> int:
    if move.is_pass:
        return board_size * board_size
    assert move.point is not None
    if not (1 <= move.point.row <= board_size and 1 <= move.point.col <= board_size):
        raise ValueError(f"move is outside a {board_size}x{board_size} board: {move}")
    return (move.point.row - 1) * board_size + (move.point.col - 1)


def index_to_move(index: int, board_size: int) -> Move:
    pass_index = board_size * board_size
    if index == pass_index:
        return Move.pass_turn()
    if index < 0 or index > pass_index:
        raise ValueError(f"policy index is outside a {board_size}x{board_size} board: {index}")

    row = index // board_size + 1
    col = index % board_size + 1
    return Move.play(Point(row, col))


def _stone_plane(game_state: GameState, player: Player) -> tuple[tuple[int, ...], ...]:
    size = game_state.board.size
    return tuple(
        tuple(
            1 if game_state.board.get(Point(row, col)) is player else 0
            for col in range(1, size + 1)
        )
        for row in range(1, size + 1)
    )


def _filled_plane(size: int, value: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(value for _ in range(size)) for _ in range(size))
