from __future__ import annotations

from go_engine.game import GameState, Move
from go_engine.types import Player, Point

BoardPlanes = tuple[tuple[tuple[int, ...], ...], ...]
PolicyVector = tuple[float, ...]


BoardSnapshot = tuple[tuple[int, int, str], ...]


def encode_game_state(game_state: GameState, history_length: int = 1) -> BoardPlanes:
    history = []
    state: GameState | None = game_state
    while state is not None and len(history) < history_length:
        history.append(state.board.snapshot_key())
        state = state.previous_state

    return encode_board_history(
        board_history=tuple(history),
        player=game_state.next_player,
        board_size=game_state.board.size,
        history_length=history_length,
    )


def encode_board_snapshot(
    board: BoardSnapshot,
    player: Player,
    board_size: int,
    history_length: int = 1,
    board_history: tuple[BoardSnapshot, ...] | None = None,
) -> BoardPlanes:
    history = board_history if board_history is not None else (board,)
    return encode_board_history(
        board_history=history,
        player=player,
        board_size=board_size,
        history_length=history_length,
    )


def encode_board_history(
    board_history: tuple[BoardSnapshot, ...],
    player: Player,
    board_size: int,
    history_length: int = 1,
) -> BoardPlanes:
    if history_length < 1:
        raise ValueError("history_length must be at least 1")

    planes = []
    opponent = player.other
    for index in range(history_length):
        board = board_history[index] if index < len(board_history) else ()
        stones = {(row, col): color for row, col, color in board}
        planes.append(_snapshot_plane(stones, player, board_size))
        planes.append(_snapshot_plane(stones, opponent, board_size))

    planes.append(_filled_plane(board_size, 1 if player is Player.BLACK else 0))
    return tuple(planes)


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


def _snapshot_plane(
    stones: dict[tuple[int, int], str],
    player: Player,
    board_size: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            1 if stones.get((row, col)) == player.value else 0
            for col in range(1, board_size + 1)
        )
        for row in range(1, board_size + 1)
    )


def _filled_plane(size: int, value: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(value for _ in range(size)) for _ in range(size))
