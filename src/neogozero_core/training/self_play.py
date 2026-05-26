from __future__ import annotations

from dataclasses import dataclass

from neogozero_core.bots.mcts_bot import MCTSBot
from neogozero_core.go.game import GameState, Move
from neogozero_core.go.types import Player


BoardSnapshot = tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class TrainingExample:
    board: BoardSnapshot
    player: Player
    visit_distribution: dict[Move, float]
    winner: Player
    board_history: tuple[BoardSnapshot, ...] = ()

    @property
    def value(self) -> float:
        return 1.0 if self.winner is self.player else -1.0


@dataclass(frozen=True)
class SelfPlayGame:
    examples: tuple[TrainingExample, ...]
    winner: Player
    moves: tuple[Move, ...]
    final_state: GameState


@dataclass(frozen=True)
class _PendingExample:
    board: BoardSnapshot
    player: Player
    visit_distribution: dict[Move, float]
    board_history: tuple[BoardSnapshot, ...]


def generate_self_play_game(
    bot: MCTSBot,
    board_size: int = 9,
    max_moves: int | None = None,
    history_length: int = 1,
    temperature: float = 1.0,
    temperature_drop_move: int = 30,
    add_dirichlet_noise: bool = True,
) -> SelfPlayGame:
    game = GameState.new_game(board_size=board_size)
    move_limit = max_moves if max_moves is not None else board_size * board_size * 3
    pending_examples: list[_PendingExample] = []
    moves: list[Move] = []

    for _ in range(move_limit):
        if game.is_over():
            break

        move_temperature = temperature if len(moves) < temperature_drop_move else 0.0
        search_result = bot.search(
            game,
            temperature=move_temperature,
            add_dirichlet_noise=add_dirichlet_noise,
        )
        board_history = _snapshot_history(game, history_length)
        pending_examples.append(
            _PendingExample(
                board=_snapshot_board(game),
                player=game.next_player,
                visit_distribution=search_result.visit_distribution(),
                board_history=board_history,
            )
        )
        moves.append(search_result.selected_move)
        game = game.apply_move(search_result.selected_move)

    if not game.is_over():
        game = game.apply_move(Move.pass_turn())
        if not game.is_over():
            game = game.apply_move(Move.pass_turn())

    winner = game.winner()
    examples = tuple(
        TrainingExample(
            board=example.board,
            player=example.player,
            visit_distribution=example.visit_distribution,
            winner=winner,
            board_history=example.board_history,
        )
        for example in pending_examples
    )

    return SelfPlayGame(
        examples=examples,
        winner=winner,
        moves=tuple(moves),
        final_state=game,
    )


def _snapshot_board(game_state: GameState) -> BoardSnapshot:
    return game_state.board.zobrist_key()


def _snapshot_history(
    game_state: GameState,
    history_length: int,
) -> tuple[BoardSnapshot, ...]:
    history = []
    state: GameState | None = game_state
    while state is not None and len(history) < history_length:
        history.append(_snapshot_board(state))
        state = state.previous_state
    return tuple(history)
