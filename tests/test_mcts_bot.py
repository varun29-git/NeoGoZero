from __future__ import annotations

import random

import pytest

from myalphago.bots.mcts_bot import MCTSBot
from myalphago.bots.random_bot import RandomBot
from myalphago.go.game import GameState, Move
from myalphago.go.types import Player


def test_mcts_selects_a_legal_move() -> None:
    game = GameState.new_game(board_size=5)
    bot = MCTSBot(num_rounds=8, max_rollout_moves=20, rng=random.Random(1))

    move = bot.select_move(game)

    assert game.is_valid_move(move)


def test_mcts_can_play_a_small_game_against_random() -> None:
    game = GameState.new_game(board_size=3)
    bots = {
        Player.BLACK: MCTSBot(
            num_rounds=3,
            max_rollout_moves=12,
            rng=random.Random(1),
        ),
        Player.WHITE: RandomBot(rng=random.Random(2)),
    }

    for _ in range(100):
        if game.is_over():
            break
        move = bots[game.next_player].select_move(game)
        assert game.is_valid_move(move)
        game = game.apply_move(move)

    assert game.is_over()


def test_mcts_rejects_move_selection_after_game_over() -> None:
    game = GameState.new_game(board_size=3)
    game = game.apply_move(Move.pass_turn())
    game = game.apply_move(Move.pass_turn())
    bot = MCTSBot(num_rounds=1, rng=random.Random(1))

    with pytest.raises(ValueError):
        bot.select_move(game)
