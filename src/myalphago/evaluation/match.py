from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from myalphago.go.game import GameState, Move, Score
from myalphago.go.types import Player


class Bot(Protocol):
    def select_move(self, game_state: GameState) -> Move:
        """Choose one legal move for the current game state."""


@dataclass(frozen=True)
class GameResult:
    final_state: GameState
    moves: tuple[Move, ...]
    score: Score
    winner: Player
    reached_move_limit: bool = False

    @property
    def num_moves(self) -> int:
        return len(self.moves)


@dataclass(frozen=True)
class MatchResult:
    games: tuple[GameResult, ...]

    @property
    def black_wins(self) -> int:
        return sum(1 for game in self.games if game.winner is Player.BLACK)

    @property
    def white_wins(self) -> int:
        return sum(1 for game in self.games if game.winner is Player.WHITE)

    @property
    def average_moves(self) -> float:
        if not self.games:
            return 0.0
        return sum(game.num_moves for game in self.games) / len(self.games)

    @property
    def average_margin(self) -> float:
        if not self.games:
            return 0.0
        return sum(game.score.margin for game in self.games) / len(self.games)

    def win_rate(self, player: Player) -> float:
        if not self.games:
            return 0.0
        wins = self.black_wins if player is Player.BLACK else self.white_wins
        return wins / len(self.games)


BotFactory = Callable[[int], Bot]


def play_game(
    black_bot: Bot,
    white_bot: Bot,
    board_size: int = 9,
    komi: float = 7.5,
    max_moves: int | None = None,
) -> GameResult:
    game = GameState.new_game(board_size=board_size)
    bots = {
        Player.BLACK: black_bot,
        Player.WHITE: white_bot,
    }
    moves: list[Move] = []
    move_limit = max_moves if max_moves is not None else board_size * board_size * 3
    reached_move_limit = False

    for _ in range(move_limit):
        if game.is_over():
            break

        move = bots[game.next_player].select_move(game)
        if not game.is_valid_move(move):
            raise ValueError(f"{game.next_player.value} bot selected illegal move: {move}")

        moves.append(move)
        game = game.apply_move(move)
    else:
        reached_move_limit = not game.is_over()

    score = game.score(komi=komi)
    winner = Player.BLACK if score.black > score.white else Player.WHITE
    return GameResult(
        final_state=game,
        moves=tuple(moves),
        score=score,
        winner=winner,
        reached_move_limit=reached_move_limit,
    )


def play_match(
    black_bot_factory: BotFactory,
    white_bot_factory: BotFactory,
    num_games: int,
    board_size: int = 9,
    komi: float = 7.5,
    max_moves: int | None = None,
) -> MatchResult:
    if num_games < 1:
        raise ValueError("num_games must be at least 1")

    games = tuple(
        play_game(
            black_bot=black_bot_factory(game_index),
            white_bot=white_bot_factory(game_index),
            board_size=board_size,
            komi=komi,
            max_moves=max_moves,
        )
        for game_index in range(num_games)
    )
    return MatchResult(games=games)
