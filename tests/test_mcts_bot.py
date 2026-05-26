from __future__ import annotations

import random

import pytest

from myalphago.bots.mcts_bot import (
    Evaluation,
    MCTSBot,
    MCTSNode,
    _select_move_from_visit_counts,
)
from myalphago.bots.random_bot import RandomBot
from myalphago.go.game import GameState, Move
from myalphago.go.types import Player, Point


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


def test_puct_selection_uses_prior_probability() -> None:
    game = GameState.new_game(board_size=3)
    parent = MCTSNode(game_state=game)
    parent.num_rollouts = 10

    low_prior_child = MCTSNode(
        game_state=game.apply_move(Move.play(Point(1, 1))),
        parent=parent,
        move=Move.play(Point(1, 1)),
        prior_probability=0.1,
    )
    high_prior_child = MCTSNode(
        game_state=game.apply_move(Move.play(Point(2, 2))),
        parent=parent,
        move=Move.play(Point(2, 2)),
        prior_probability=0.9,
    )
    for child in (low_prior_child, high_prior_child):
        child.num_rollouts = 2
        child.value_sum = 0.0
    parent.children = [low_prior_child, high_prior_child]

    assert parent.select_child(c_puct=1.5) is high_prior_child


def test_evaluation_normalizes_legal_priors_and_clamps_value() -> None:
    game = GameState.new_game(board_size=3)
    move_a = Move.play(Point(1, 1))
    move_b = Move.play(Point(1, 2))
    illegal_move = Move.play(Point(4, 4))

    evaluation = Evaluation.from_priors(
        game,
        {
            move_a: 2.0,
            move_b: 1.0,
            illegal_move: 100.0,
        },
        value=3.0,
    )

    assert evaluation.move_priors == {
        move_a: pytest.approx(2 / 3),
        move_b: pytest.approx(1 / 3),
    }
    assert evaluation.value == 1.0


def test_mcts_can_use_a_custom_evaluator_prior() -> None:
    class CenterPriorEvaluator:
        def evaluate(self, game_state: GameState) -> Evaluation:
            center = Move.play(Point(2, 2))
            return Evaluation.from_priors(game_state, {center: 1.0}, value=0.0)

    game = GameState.new_game(board_size=3)
    bot = MCTSBot(num_rounds=2, evaluator=CenterPriorEvaluator())

    assert bot.select_move(game) == Move.play(Point(2, 2))


def test_temperature_zero_selects_most_visited_move() -> None:
    move_a = Move.play(Point(1, 1))
    move_b = Move.play(Point(1, 2))

    selected = _select_move_from_visit_counts(
        {move_a: 1, move_b: 5},
        temperature=0.0,
        rng=random.Random(1),
    )

    assert selected == move_b


def test_dirichlet_noise_changes_root_priors() -> None:
    game = GameState.new_game(board_size=3)
    root = MCTSNode(game_state=game)
    root.expand(Evaluation.uniform(game, value=0.0).move_priors)
    before = [child.prior_probability for child in root.children]

    root.add_dirichlet_noise(alpha=0.3, epsilon=1.0, rng=random.Random(1))
    after = [child.prior_probability for child in root.children]

    assert after != before
    assert sum(after) == pytest.approx(1.0)
