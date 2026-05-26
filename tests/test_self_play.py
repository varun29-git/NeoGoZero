from __future__ import annotations

import random

from search_players.mcts_bot import MCTSBot
from zero_training_pipeline.self_play import generate_self_play_game


def test_self_play_generates_training_examples() -> None:
    bot = MCTSBot(
        num_rounds=3,
        max_rollout_moves=8,
        rng=random.Random(1),
    )

    game = generate_self_play_game(bot, board_size=3, max_moves=30)

    assert game.final_state.is_over()
    assert len(game.examples) == len(game.moves)
    assert game.examples
    assert game.examples[0].winner is game.winner
    assert game.examples[0].value in {-1.0, 1.0}
    assert sum(game.examples[0].visit_distribution.values()) == 1.0


def test_self_play_records_board_history() -> None:
    bot = MCTSBot(
        num_rounds=2,
        max_rollout_moves=8,
        rng=random.Random(2),
    )

    game = generate_self_play_game(bot, board_size=3, max_moves=12, history_length=2)

    assert game.examples[0].board_history
    assert len(game.examples[-1].board_history) <= 2
