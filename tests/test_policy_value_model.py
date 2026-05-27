from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from search_players.mcts_bot import MCTSBot
from go_engine.game import GameState
from policy_value_networks.resnet_policy_value.policy_value import (
    PolicyValueNet,
    ResidualBlock,
    TorchPolicyValueEvaluator,
)
from zero_training_pipeline.self_play import generate_self_play_game
from zero_training_pipeline.torch_training import examples_to_tensors, train_step


def test_policy_value_network_forward_shapes() -> None:
    model = PolicyValueNet(board_size=3, input_planes=3, channels=8, num_res_blocks=2)
    boards = torch.zeros((2, 3, 3, 3), dtype=torch.float32)

    policy_logits, values = model(boards)

    assert policy_logits.shape == (2, 10)
    assert values.shape == (2,)


def test_policy_value_network_uses_residual_tower() -> None:
    model = PolicyValueNet(board_size=3, input_planes=3, channels=8, num_res_blocks=3)

    assert len(model.residual_tower) == 3
    assert all(isinstance(block, ResidualBlock) for block in model.residual_tower)


def test_examples_to_tensors_shapes() -> None:
    bot = MCTSBot(num_rounds=2, max_rollout_moves=6, rng=random.Random(1))
    game = generate_self_play_game(bot, board_size=3, max_moves=12)

    boards, policies, values = examples_to_tensors(game.examples[:2], board_size=3)

    assert boards.shape == (2, 3, 3, 3)
    assert policies.shape == (2, 10)
    assert values.shape == (2,)


def test_train_step_returns_finite_loss() -> None:
    bot = MCTSBot(num_rounds=2, max_rollout_moves=6, rng=random.Random(1))
    game = generate_self_play_game(bot, board_size=3, max_moves=12)
    model = PolicyValueNet(board_size=3, input_planes=3, channels=8, num_res_blocks=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss = train_step(model, optimizer, game.examples[:4], board_size=3)

    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))


def test_torch_evaluator_returns_legal_priors() -> None:
    game = GameState.new_game(board_size=3)
    model = PolicyValueNet(board_size=3, input_planes=3, channels=8, num_res_blocks=2)
    evaluator = TorchPolicyValueEvaluator(model)

    evaluation = evaluator.evaluate(game)

    assert set(evaluation.move_priors) == set(game.legal_moves())
    assert sum(evaluation.move_priors.values()) == pytest.approx(1.0)
    assert -1.0 <= evaluation.value <= 1.0


def test_torch_evaluator_batches_multiple_states() -> None:
    game_a = GameState.new_game(board_size=3)
    game_b = game_a.apply_move(next(move for move in game_a.legal_moves() if not move.is_pass))
    model = PolicyValueNet(board_size=3, input_planes=3, channels=8, num_res_blocks=2)
    evaluator = TorchPolicyValueEvaluator(model)

    evaluations = evaluator.evaluate_many((game_a, game_b))

    assert len(evaluations) == 2
    assert set(evaluations[0].move_priors) == set(game_a.legal_moves())
    assert set(evaluations[1].move_priors) == set(game_b.legal_moves())
