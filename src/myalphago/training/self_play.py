from __future__ import annotations

from dataclasses import dataclass

from myalphago.bots.mcts_bot import MCTSBot
from myalphago.go.game import GameState, Move
from myalphago.go.types import Player


BoardSnapshot = tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class TrainingExample:
    board: BoardSnapshot
    player: Player
    visit_distribution: dict[Move, float]
    winner: Player

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


def generate_self_play_game(
    bot: MCTSBot,
    board_size: int = 9,
    max_moves: int | None = None,
) -> SelfPlayGame:
    game = GameState.new_game(board_size=board_size)
    move_limit = max_moves if max_moves is not None else board_size * board_size * 3
    pending_examples: list[_PendingExample] = []
    moves: list[Move] = []

    for _ in range(move_limit):
        if game.is_over():
            break

        search_result = bot.search(game)
        pending_examples.append(
            _PendingExample(
                board=_snapshot_board(game),
                player=game.next_player,
                visit_distribution=search_result.visit_distribution(),
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
