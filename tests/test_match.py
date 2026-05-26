from __future__ import annotations

import random

import pytest

from neogozero_core.bots.random_bot import RandomBot
from neogozero_core.evaluation.match import play_game, play_match
from neogozero_core.go.game import GameState, Move
from neogozero_core.go.types import Player, Point


class IllegalBot:
    def select_move(self, game_state: GameState) -> Move:
        return Move.play(Point(1, 1))


def test_play_game_returns_a_completed_result() -> None:
    result = play_game(
        RandomBot(rng=random.Random(1)),
        RandomBot(rng=random.Random(2)),
        board_size=3,
        max_moves=100,
    )

    assert result.final_state.is_over()
    assert result.num_moves > 0
    assert result.winner in {Player.BLACK, Player.WHITE}


def test_play_match_aggregates_results() -> None:
    result = play_match(
        black_bot_factory=lambda index: RandomBot(rng=random.Random(index)),
        white_bot_factory=lambda index: RandomBot(rng=random.Random(100 + index)),
        num_games=3,
        board_size=3,
        max_moves=100,
    )

    assert len(result.games) == 3
    assert result.black_wins + result.white_wins == 3
    assert result.average_moves > 0


def test_play_game_rejects_illegal_bot_moves() -> None:
    with pytest.raises(ValueError):
        play_game(
            IllegalBot(),
            RandomBot(rng=random.Random(1)),
            board_size=3,
            max_moves=10,
        )
