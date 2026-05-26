from __future__ import annotations

import random

from myalphago.bots.mcts_bot import MCTSBot
from myalphago.training.self_play import generate_self_play_game


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
